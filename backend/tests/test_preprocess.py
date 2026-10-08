import base64
import io

import numpy as np
from PIL import Image, ImageDraw

from app.config import DIGIT_BOX, INPUT_SIZE as S
from app.preprocess import decode_data_url, to_digit_image


def _canvas(draw_fn, size=280, bg=0, fg=255):
    img = Image.new("L", (size, size), bg)
    draw_fn(ImageDraw.Draw(img), fg)
    return img


def test_empty_canvas_gives_blank_image():
    out = to_digit_image(Image.new("L", (280, 280), 0))
    assert out.shape == (S, S)
    assert out.max() == 0


def test_off_center_digit_is_centered_and_scaled():
    img = _canvas(lambda d, fg: d.line([(20, 20), (20, 80)], fill=fg, width=12))
    out = to_digit_image(img)
    assert out.shape == (S, S) and 0 <= out.min() and out.max() <= 1
    ys, xs = np.nonzero(out > 0.05)
    assert ys.max() - ys.min() + 1 == DIGIT_BOX  # longest side scaled to the digit box
    total = out.sum()
    cy = (out.sum(1) * np.arange(S)).sum() / total
    cx = (out.sum(0) * np.arange(S)).sum() / total
    assert abs(cy - (S - 1) / 2) <= 1 and abs(cx - (S - 1) / 2) <= 1


def test_tiny_dot_does_not_crash():
    img = _canvas(lambda d, fg: d.point((140, 140), fill=fg))
    out = to_digit_image(img)
    assert out.max() > 0


def test_dark_ink_on_white_is_inverted():
    img = _canvas(lambda d, fg: d.ellipse([60, 60, 200, 220], outline=fg, width=15), bg=255, fg=0)
    out = to_digit_image(img)
    assert out[0, 0] == 0 and out.max() > 0.5


def test_decode_transparent_png_data_url():
    img = Image.new("RGBA", (50, 50), (0, 0, 0, 0))
    ImageDraw.Draw(img).line([(25, 5), (25, 45)], fill=(255, 255, 255, 255), width=5)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    out = to_digit_image(decode_data_url(url))
    assert out.max() > 0.5
