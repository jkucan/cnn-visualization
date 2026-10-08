"""MNIST download, 28x28 -> INPUT_SIZE x INPUT_SIZE resize and caching.

    python -m app.data --prepare
"""
import argparse
from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.nn.functional as F

from .config import DATA_DIR, INPUT_SIZE, MNIST_PATH


@dataclass
class Split:
    x: torch.Tensor  # float32 [N, 1, S, S] with S = INPUT_SIZE, 0..1
    y: torch.Tensor  # int64 [N]


def _resize(images: torch.Tensor) -> torch.Tensor:
    x = images.unsqueeze(1).float() / 255.0
    x = F.interpolate(x, size=(INPUT_SIZE, INPUT_SIZE), mode="bilinear", antialias=True, align_corners=False)
    return (x.clamp(0, 1) * 255).round().to(torch.uint8)


def prepare() -> None:
    if MNIST_PATH.exists():
        print(f"{MNIST_PATH} already exists")
        return
    from torchvision.datasets import MNIST

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for split, train in (("train", True), ("test", False)):
        ds = MNIST(str(DATA_DIR), train=train, download=True)
        out[split] = {"x": _resize(ds.data), "y": ds.targets.clone()}
        print(f"{split}: {len(ds)} images resized to {INPUT_SIZE}x{INPUT_SIZE}")
    torch.save(out, MNIST_PATH)
    print(f"saved {MNIST_PATH}")


@lru_cache(maxsize=1)
def load() -> dict[str, Split]:
    raw = torch.load(MNIST_PATH)
    return {k: Split(v["x"].float() / 255.0, v["y"]) for k, v in raw.items()}


def available() -> bool:
    return MNIST_PATH.exists()


def augment(x: torch.Tensor, max_shift: float = 0.5, max_rot_deg: float = 10.0, scale_range: float = 0.1) -> torch.Tensor:
    """Random small shift / rotation / scale per image, so drawn digits generalize better."""
    n = x.shape[0]
    ang = (torch.rand(n) * 2 - 1) * max_rot_deg * torch.pi / 180
    sc = 1 + (torch.rand(n) * 2 - 1) * scale_range
    # affine_grid works in [-1, 1] coordinates; one pixel = 2 / size.
    shift = (torch.rand(n, 2) * 2 - 1) * max_shift * 2 / x.shape[-1]
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([
        torch.stack([cos, -sin, shift[:, 0]], 1),
        torch.stack([sin, cos, shift[:, 1]], 1),
    ], 1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
