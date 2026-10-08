"""Turn a drawing into an MNIST-like INPUT_SIZE x INPUT_SIZE image: crop, scale to fit, center by mass."""
import base64
import io

import numpy as np
from PIL import Image

from .config import DIGIT_BOX, INPUT_SIZE

INK_THRESHOLD = 0.05


def decode_data_url(data: str) -> Image.Image:
    if "," in data:
        data = data.split(",", 1)[1]
    img = Image.open(io.BytesIO(base64.b64decode(data)))
    if img.mode in ("RGBA", "LA"):
        # Transparent canvas pixels count as background.
        bg = Image.new("RGBA", img.size, (0, 0, 0, 255))
        img = Image.alpha_composite(bg, img.convert("RGBA"))
    return img.convert("L")


def to_digit_image(img: Image.Image) -> np.ndarray:
    """Grayscale image (any size) -> float32 [INPUT_SIZE, INPUT_SIZE] in 0..1, white digit on black."""
    arr = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    if arr.mean() > 0.5:  # dark ink on light paper
        arr = 1.0 - arr
    out = np.zeros((INPUT_SIZE, INPUT_SIZE), dtype=np.float32)

    ys, xs = np.nonzero(arr > INK_THRESHOLD)
    if len(ys) == 0:
        return out
    crop = arr[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]

    h, w = crop.shape
    scale = DIGIT_BOX / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    small = Image.fromarray((crop * 255).astype(np.uint8)).resize((nw, nh), Image.BOX)
    small = np.asarray(small, dtype=np.float32) / 255.0
    if small.max() > 0:
        small /= small.max()

    total = small.sum()
    if total > 0:
        cy = (small.sum(1) * np.arange(nh)).sum() / total
        cx = (small.sum(0) * np.arange(nw)).sum() / total
    else:
        cy, cx = (nh - 1) / 2, (nw - 1) / 2
    center = (INPUT_SIZE - 1) / 2
    top = int(np.clip(round(center - cy), 0, INPUT_SIZE - nh))
    left = int(np.clip(round(center - cx), 0, INPUT_SIZE - nw))
    out[top: top + nh, left: left + nw] = small
    return out
