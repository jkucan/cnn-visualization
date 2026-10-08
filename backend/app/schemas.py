from typing import Literal

from pydantic import BaseModel, Field, model_validator

from .config import INPUT_SIZE


class ResetRequest(BaseModel):
    init: Literal["kaiming", "xavier", "normal", "uniform"] = "kaiming"
    seed: int | None = None
    scale: float = Field(0.1, gt=0, le=10)


class PredictRequest(BaseModel):
    # Either a canvas data URL (preprocessed server-side) or ready-made pixels in 0..1.
    image: str | None = None
    pixels: list[float] | None = None

    @model_validator(mode="after")
    def one_of(self):
        if (self.image is None) == (self.pixels is None):
            raise ValueError("provide exactly one of 'image' or 'pixels'")
        n = INPUT_SIZE * INPUT_SIZE
        if self.pixels is not None and len(self.pixels) != n:
            raise ValueError(f"'pixels' must have {n} values ({INPUT_SIZE}x{INPUT_SIZE})")
        return self


class TrainRequest(BaseModel):
    epochs: int = Field(1, ge=1, le=50)
    lr: float = Field(0.01, gt=0, le=1)
    batch_size: int = Field(64, ge=1, le=4096)
    optimizer: Literal["adam", "sgd"] = "adam"
    subset: int = Field(0, ge=0)
    augment: bool = True
    delay_ms: int = Field(0, ge=0, le=2000)
    paused: bool = False


class CheckpointRequest(BaseModel):
    name: Literal["user", "pretrained"] = "user"
