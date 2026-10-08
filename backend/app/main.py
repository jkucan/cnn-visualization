import asyncio
import json
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import data
from .config import INPUT_SIZE, PRETRAINED_PATH, STATIC_DIR, USER_CHECKPOINT_PATH
from .preprocess import decode_data_url, to_digit_image
from .schemas import CheckpointRequest, PredictRequest, ResetRequest, TrainRequest
from .trainer import TrainConfig, Trainer


class Hub:
    """Fans trainer events out to connected WebSocket clients; safe to call from any thread."""

    def __init__(self):
        self.loop: asyncio.AbstractEventLoop | None = None
        self.clients: set[asyncio.Queue] = set()

    def publish(self, msg: dict) -> None:
        if self.loop is not None:
            self.loop.call_soon_threadsafe(self._fanout, json.dumps(msg))

    def _fanout(self, text: str) -> None:
        for q in self.clients:
            if q.full():  # slow client: drop the oldest update rather than lag behind
                q.get_nowait()
            q.put_nowait(text)


hub = Hub()
trainer = Trainer(on_event=hub.publish)


@asynccontextmanager
async def lifespan(app: FastAPI):
    hub.loop = asyncio.get_running_loop()
    yield
    trainer.stop()


app = FastAPI(title="CNN digit demo", lifespan=lifespan)


def _conflict(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(409, str(exc)) from exc


@app.get("/api/model/info")
def model_info():
    return {
        **trainer.model.info(),
        "data_available": data.available(),
        "pretrained_available": PRETRAINED_PATH.exists(),
        "user_checkpoint_available": USER_CHECKPOINT_PATH.exists(),
        "train_size": len(data.load()["train"].y) if data.available() else 0,
    }


@app.get("/api/model/weights")
def model_weights():
    return trainer.weights()


@app.post("/api/model/reset")
def model_reset(req: ResetRequest):
    _conflict(trainer.reset, req.init, req.seed, req.scale)
    return {"ok": True}


@app.post("/api/model/save")
def model_save(req: CheckpointRequest):
    if req.name != "user":
        raise HTTPException(400, "only the 'user' checkpoint can be written")
    trainer.save(USER_CHECKPOINT_PATH)
    return {"ok": True}


@app.post("/api/model/load")
def model_load(req: CheckpointRequest):
    path = PRETRAINED_PATH if req.name == "pretrained" else USER_CHECKPOINT_PATH
    if not path.exists():
        raise HTTPException(404, f"no {req.name} checkpoint")
    _conflict(trainer.load, path, req.name)
    return {"ok": True}


@app.post("/api/predict")
def predict(req: PredictRequest):
    if req.image is not None:
        try:
            img = to_digit_image(decode_data_url(req.image))
        except Exception as exc:
            raise HTTPException(400, f"could not decode image: {exc}") from exc
    else:
        img = np.clip(np.asarray(req.pixels, dtype=np.float32), 0, 1).reshape(INPUT_SIZE, INPUT_SIZE)
    return trainer.predict(img)


@app.get("/api/samples")
def samples(n: int = 1, split: str = "test"):
    if not data.available():
        raise HTTPException(503, "MNIST data not prepared")
    if split not in ("train", "test"):
        raise HTTPException(400, "split must be 'train' or 'test'")
    s = data.load()[split]
    idx = np.random.randint(0, len(s.y), size=max(1, min(n, 64)))
    return [
        {"index": int(i), "label": int(s.y[i]), "pixels": [round(v, 4) for v in s.x[i].flatten().tolist()]}
        for i in idx
    ]


@app.post("/api/train/start")
def train_start(req: TrainRequest):
    cfg = TrainConfig(**req.model_dump(exclude={"paused"}))
    _conflict(trainer.start, cfg, paused=req.paused)
    return {"ok": True}


@app.post("/api/train/pause")
def train_pause():
    trainer.pause()
    return {"ok": True}


@app.post("/api/train/resume")
def train_resume():
    trainer.resume()
    return {"ok": True}


@app.post("/api/train/step")
def train_step():
    _conflict(trainer.step)
    return {"ok": True}


@app.post("/api/train/stop")
def train_stop():
    trainer.stop()
    return {"ok": True}


class SpeedRequest(BaseModel):
    delay_ms: int = Field(ge=0, le=2000)


@app.post("/api/train/speed")
def train_speed(req: SpeedRequest):
    # The running loop reads this config object on every batch.
    trainer.config.delay_ms = req.delay_ms
    return {"ok": True}


@app.websocket("/ws/train")
async def ws_train(ws: WebSocket):
    await ws.accept()
    q: asyncio.Queue = asyncio.Queue(maxsize=8)
    hub.clients.add(q)
    try:
        await ws.send_text(json.dumps(trainer.status_message()))
        await ws.send_text(json.dumps({"type": "model", "reason": "connect", "weights": trainer.weights(),
                                       "prediction": trainer.predict_current(), "eval": None}))
        while True:
            await ws.send_text(await q.get())
    except WebSocketDisconnect:
        pass
    finally:
        hub.clients.discard(q)


if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
