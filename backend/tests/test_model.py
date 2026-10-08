import pytest
import torch

from app.config import CONV_FILTERS, HIDDEN, INPUT_SIZE as S
from app.model import INIT_SCHEMES, SmallCNN, init_weights, weights_snapshot


def test_forward_shapes():
    model = SmallCNN()
    logits, acts = model(torch.rand(2, 1, S, S), return_activations=True)
    assert logits.shape == (2, 10)
    expected = {
        "input": (2, 1, S, S), "conv1": (2, CONV_FILTERS, S, S), "pool1": (2, CONV_FILTERS, S // 2, S // 2),
        "fc1": (2, HIDDEN), "logits": (2, 10), "probs": (2, 10),
    }
    assert {k: tuple(v.shape) for k, v in acts.items()} == expected
    assert torch.allclose(acts["probs"].sum(1), torch.ones(2))


def test_param_count():
    conv = CONV_FILTERS * 9 + CONV_FILTERS
    fc1 = CONV_FILTERS * (S // 2) ** 2 * HIDDEN + HIDDEN
    fc2 = HIDDEN * 10 + 10
    assert SmallCNN().info()["params"] == conv + fc1 + fc2


@pytest.mark.parametrize("scheme", INIT_SCHEMES)
def test_init_is_seeded(scheme):
    a, b = SmallCNN(), SmallCNN()
    init_weights(a, scheme, seed=7)
    init_weights(b, scheme, seed=7)
    assert weights_snapshot(a) == weights_snapshot(b)
    init_weights(b, scheme, seed=8)
    assert weights_snapshot(a) != weights_snapshot(b)


def test_random_init_predicts_roughly_uniform():
    model = SmallCNN()
    init_weights(model, "kaiming", seed=0)
    _, acts = model(torch.rand(1, 1, S, S), return_activations=True)
    assert acts["probs"].max().item() < 0.3
