"""Owns the model; runs training in a background thread and emits progress events."""
import math
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from . import data
from .config import INPUT_SIZE
from .model import SmallCNN, activations_snapshot, init_weights, weights_snapshot

EMIT_INTERVAL_S = 0.1


@dataclass
class TrainConfig:
    epochs: int = 1
    lr: float = 0.01
    batch_size: int = 64
    optimizer: str = "adam"  # "adam" | "sgd"
    subset: int = 0  # 0 = full training set
    augment: bool = True
    delay_ms: int = 0  # sleep after each batch, to slow training down for watching


class Trainer:
    def __init__(self, on_event: Callable[[dict], None] | None = None):
        self.model = SmallCNN()
        init_weights(self.model, "kaiming", seed=0)
        self.lock = threading.RLock()  # guards self.model
        self._ctl = threading.Condition()  # guards the fields below
        self.status = "idle"  # idle | running | paused
        self._stop = False
        self._step_budget = 0
        self._thread: threading.Thread | None = None
        self.config = TrainConfig()
        self.on_event = on_event or (lambda msg: None)
        self.current_input: np.ndarray | None = None
        self.history: list[dict] = []
        self.epoch_history: list[dict] = []
        self.global_step = 0
        self.test_acc: float | None = None

    # ---------- model state ----------

    def weights(self) -> dict:
        with self.lock:
            return weights_snapshot(self.model)

    def _forward(self, img: np.ndarray) -> dict:
        x = torch.from_numpy(np.asarray(img, dtype=np.float32).reshape(1, 1, INPUT_SIZE, INPUT_SIZE))
        with self.lock, torch.no_grad():
            _, acts = self.model(x, return_activations=True)
        snap = activations_snapshot(acts)
        return {"activations": snap, "probs": snap["probs"]}

    def predict(self, img: np.ndarray) -> dict:
        self.current_input = np.asarray(img, dtype=np.float32).reshape(INPUT_SIZE, INPUT_SIZE)
        return self._forward(self.current_input)

    def predict_current(self) -> dict | None:
        return None if self.current_input is None else self._forward(self.current_input)

    def evaluate(self) -> dict | None:
        if not data.available():
            return None
        test = data.load()["test"]
        with self.lock, torch.no_grad():
            logits = self.model(test.x)
            loss = F.cross_entropy(logits, test.y).item()
            acc = (logits.argmax(1) == test.y).float().mean().item()
        self.test_acc = acc
        return {"test_acc": acc, "test_loss": loss}

    def _model_changed(self, reason: str) -> None:
        self.history.clear()
        self.epoch_history.clear()
        self.global_step = 0
        self.on_event({
            "type": "model",
            "reason": reason,
            "weights": self.weights(),
            "prediction": self.predict_current(),
            "eval": self.evaluate(),
        })

    def _require_idle(self) -> None:
        if self.status != "idle":
            raise RuntimeError("stop training first")

    def reset(self, scheme: str = "kaiming", seed: int | None = None, scale: float = 0.1) -> None:
        self._require_idle()
        with self.lock:
            init_weights(self.model, scheme, seed, scale)
        self._model_changed("reset")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock:
            torch.save({"state_dict": self.model.state_dict()}, path)

    def load(self, path: Path, reason: str = "load") -> None:
        self._require_idle()
        ckpt = torch.load(path)
        with self.lock:
            self.model.load_state_dict(ckpt["state_dict"])
        self._model_changed(reason)

    def status_message(self) -> dict:
        return {
            "type": "state",
            "status": self.status,
            "config": asdict(self.config),
            "global_step": self.global_step,
            "history": self.history,
            "epoch_history": self.epoch_history,
            "test_acc": self.test_acc,
        }

    # ---------- training control ----------

    def _set_status(self, status: str) -> None:
        with self._ctl:
            self.status = status
            self._ctl.notify_all()
        self.on_event(self.status_message())

    def start(self, config: TrainConfig, paused: bool = False) -> None:
        if not data.available():
            raise RuntimeError("MNIST data not prepared (run: python -m app.data --prepare)")
        if config.optimizer not in ("adam", "sgd"):
            raise ValueError("optimizer must be 'adam' or 'sgd'")
        with self._ctl:
            if self.status != "idle":
                raise RuntimeError("training already in progress")
            self.config = config
            self._stop = False
            self._step_budget = 0
        self._set_status("paused" if paused else "running")
        self._thread = threading.Thread(target=self._run, args=(config,), daemon=True)
        self._thread.start()

    def pause(self) -> None:
        if self.status == "running":
            self._set_status("paused")

    def resume(self) -> None:
        if self.status == "paused":
            self._set_status("running")

    def step(self) -> None:
        with self._ctl:
            if self.status != "paused":
                raise RuntimeError("pause training before stepping")
            self._step_budget += 1
            self._ctl.notify_all()

    def stop(self) -> None:
        with self._ctl:
            self._stop = True
            self._ctl.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=10)

    def _gate(self) -> tuple[bool, bool]:
        """Block while paused. Returns (keep_going, is_single_step)."""
        with self._ctl:
            while True:
                if self._stop:
                    return False, False
                if self.status == "running":
                    return True, False
                if self._step_budget > 0:
                    self._step_budget -= 1
                    return True, True
                self._ctl.wait()

    def _run(self, cfg: TrainConfig) -> None:
        try:
            train = data.load()["train"]
            n = min(cfg.subset, len(train.y)) if cfg.subset > 0 else len(train.y)
            idx = torch.randperm(len(train.y))[:n]
            xs, ys = train.x[idx], train.y[idx]
            params = self.model.parameters()
            opt = (torch.optim.Adam(params, lr=cfg.lr) if cfg.optimizer == "adam"
                   else torch.optim.SGD(params, lr=cfg.lr, momentum=0.9))
            steps_per_epoch = math.ceil(n / cfg.batch_size)
            recent = deque(maxlen=20)  # (loss, correct, count) of recent batches
            last_emit = 0.0

            for epoch in range(cfg.epochs):
                perm = torch.randperm(n)
                for b in range(steps_per_epoch):
                    keep_going, single = self._gate()
                    if not keep_going:
                        return
                    bi = perm[b * cfg.batch_size:(b + 1) * cfg.batch_size]
                    xb, yb = xs[bi], ys[bi]
                    if cfg.augment:
                        xb = data.augment(xb)
                    with self.lock:
                        logits = self.model(xb)
                        loss = F.cross_entropy(logits, yb)
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                    self.global_step += 1
                    recent.append((loss.item() * len(yb), (logits.argmax(1) == yb).sum().item(), len(yb)))

                    now = time.monotonic()
                    last_batch = b == steps_per_epoch - 1
                    if single or last_batch or now - last_emit >= EMIT_INTERVAL_S:
                        last_emit = now
                        count = sum(r[2] for r in recent)
                        point = {
                            "step": self.global_step,
                            "epoch": epoch + b / steps_per_epoch,
                            "loss": sum(r[0] for r in recent) / count,
                            "acc": sum(r[1] for r in recent) / count,
                        }
                        self.history.append(point)
                        self.on_event({
                            "type": "progress",
                            **point,
                            "epoch_index": epoch,
                            "epochs": cfg.epochs,
                            "batch": b + 1,
                            "steps_per_epoch": steps_per_epoch,
                            "weights": self.weights(),
                            "prediction": self.predict_current(),
                        })
                    if cfg.delay_ms > 0:
                        time.sleep(cfg.delay_ms / 1000)

                result = self.evaluate() or {}
                entry = {"epoch": epoch + 1, "step": self.global_step, **result}
                self.epoch_history.append(entry)
                self.on_event({"type": "epoch_end", **entry})
        except Exception as exc:  # surface errors to the UI instead of dying silently
            self.on_event({"type": "error", "message": str(exc)})
            raise
        finally:
            self._set_status("idle")
