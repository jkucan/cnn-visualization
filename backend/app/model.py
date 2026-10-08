import math

import torch
import torch.nn.functional as F
from torch import nn

from .config import CONV_FILTERS, HIDDEN, INPUT_SIZE

# Normalization applied inside the model, so every caller feeds raw 0..1 pixels.
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

INIT_SCHEMES = ("kaiming", "xavier", "normal", "uniform")


class SmallCNN(nn.Module):
    """input -> conv 3x3 -> ReLU -> max-pool 2x2 -> dense (ReLU) -> dense (10)."""

    def __init__(self, c1: int = CONV_FILTERS, hidden: int = HIDDEN, classes: int = 10):
        super().__init__()
        self.c1, self.hidden, self.classes = c1, hidden, classes
        self.conv1 = nn.Conv2d(1, c1, 3, padding=1)
        pooled = INPUT_SIZE // 2
        self.fc1 = nn.Linear(c1 * pooled * pooled, hidden)
        self.fc2 = nn.Linear(hidden, classes)

    def forward(self, x: torch.Tensor, return_activations: bool = False):
        acts = {"input": x}
        h = (x - MNIST_MEAN) / MNIST_STD
        h = acts["conv1"] = F.relu(self.conv1(h))
        h = acts["pool1"] = F.max_pool2d(h, 2)
        h = acts["fc1"] = F.relu(self.fc1(h.flatten(1)))
        logits = acts["logits"] = self.fc2(h)
        if not return_activations:
            return logits
        acts["probs"] = F.softmax(logits, dim=1)
        return logits, acts

    def info(self) -> dict:
        s = INPUT_SIZE
        return {
            "input_size": s,
            "layers": [
                {"name": "input", "shape": [1, s, s]},
                {"name": "conv1", "kind": "conv", "kernel": 3, "shape": [self.c1, s, s]},
                {"name": "pool1", "kind": "pool", "shape": [self.c1, s // 2, s // 2]},
                {"name": "fc1", "kind": "dense", "shape": [self.hidden]},
                {"name": "fc2", "kind": "dense", "shape": [self.classes]},
            ],
            "params": sum(p.numel() for p in self.parameters()),
        }


def init_weights(model: SmallCNN, scheme: str = "kaiming", seed: int | None = None, scale: float = 0.1) -> None:
    if scheme not in INIT_SCHEMES:
        raise ValueError(f"unknown init scheme {scheme!r}")
    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    with torch.no_grad():
        for layer in (model.conv1, model.fc1, model.fc2):
            w, b = layer.weight, layer.bias
            fan_in = w[0].numel()
            fan_out = w.shape[0] * (w[0, 0].numel() if w.dim() > 2 else 1)
            if scheme == "kaiming":
                # PyTorch's default: kaiming_uniform(a=sqrt(5)) -> U(-1/sqrt(fan_in), 1/sqrt(fan_in)).
                bound = 1 / math.sqrt(fan_in)
                w.uniform_(-bound, bound, generator=gen)
                b.uniform_(-bound, bound, generator=gen)
                continue
            if scheme == "xavier":
                bound = math.sqrt(6 / (fan_in + fan_out))
                w.uniform_(-bound, bound, generator=gen)
            elif scheme == "normal":
                w.normal_(0, scale, generator=gen)
            else:
                w.uniform_(-scale, scale, generator=gen)
            b.zero_()


def _flat(t: torch.Tensor) -> list[float]:
    return [round(v, 5) for v in t.detach().flatten().tolist()]


def weights_snapshot(model: SmallCNN) -> dict:
    return {
        name: {"shape": list(p.shape), "data": _flat(p)}
        for name, p in model.named_parameters()
    }


def activations_snapshot(acts: dict) -> dict:
    """Activations for the first item in the batch, flattened per layer."""
    return {name: _flat(t[0]) for name, t in acts.items()}
