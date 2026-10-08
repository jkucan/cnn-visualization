import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("DATA_DIR", Path(__file__).resolve().parents[1] / "data"))
CHECKPOINT_DIR = Path(os.environ.get("CHECKPOINT_DIR", Path(__file__).resolve().parents[1] / "checkpoints"))
STATIC_DIR = Path(os.environ.get("STATIC_DIR", Path(__file__).resolve().parents[1] / "static"))

# Network size. Kept tiny on purpose: every weight is visible in the UI.
INPUT_SIZE = 8
CONV_FILTERS = 4
HIDDEN = 16
# Longest side of the digit inside the input frame (MNIST's 20/28 ratio, rounded).
DIGIT_BOX = round(INPUT_SIZE * 20 / 28)

# File names carry the architecture so stale caches/checkpoints are never loaded.
ARCH_TAG = f"s{INPUT_SIZE}_c{CONV_FILTERS}_h{HIDDEN}"
MNIST_PATH = DATA_DIR / f"mnist{INPUT_SIZE}.pt"
PRETRAINED_PATH = DATA_DIR / f"pretrained_{ARCH_TAG}.pt"
USER_CHECKPOINT_PATH = CHECKPOINT_DIR / f"user_{ARCH_TAG}.pt"
