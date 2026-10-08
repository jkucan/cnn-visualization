"""Train the checkpoint that ships with the image, for the "Load pretrained" button.

    python -m app.pretrain [--if-missing] [--epochs 8]
"""
import argparse
import time

import torch
import torch.nn.functional as F

from . import data
from .config import PRETRAINED_PATH
from .model import SmallCNN, init_weights


def pretrain(epochs: int = 8, lr: float = 3e-3, batch_size: int = 64) -> None:
    torch.manual_seed(0)
    model = SmallCNN()
    init_weights(model, "kaiming", seed=0)
    d = data.load()
    train, test = d["train"], d["test"]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    t0 = time.time()
    for epoch in range(epochs):
        perm = torch.randperm(len(train.y))
        for i in range(0, len(perm), batch_size):
            bi = perm[i:i + batch_size]
            loss = F.cross_entropy(model(data.augment(train.x[bi])), train.y[bi])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        with torch.no_grad():
            acc = (model(test.x).argmax(1) == test.y).float().mean().item()
        print(f"epoch {epoch + 1}/{epochs}  test acc {acc:.4f}  ({time.time() - t0:.0f}s)")
    PRETRAINED_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, PRETRAINED_PATH)
    print(f"saved {PRETRAINED_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--if-missing", action="store_true")
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()
    if args.if_missing and PRETRAINED_PATH.exists():
        print(f"{PRETRAINED_PATH} already exists")
    else:
        pretrain(args.epochs)
