import time

import pytest
from fastapi.testclient import TestClient

from app import data
from app.config import CONV_FILTERS, INPUT_SIZE as S
from app.main import app, trainer

needs_data = pytest.mark.skipif(not data.available(), reason="MNIST not prepared")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_info(client):
    r = client.get("/api/model/info")
    assert r.status_code == 200
    assert r.json()["params"] == trainer.model.info()["params"]


def test_reset_and_predict_pixels(client):
    assert client.post("/api/model/reset", json={"init": "xavier", "seed": 1}).status_code == 200
    r = client.post("/api/predict", json={"pixels": [0.0] * S * S})
    body = r.json()
    assert len(body["probs"]) == 10
    assert len(body["activations"]["conv1"]) == CONV_FILTERS * S * S


def test_predict_rejects_bad_input(client):
    assert client.post("/api/predict", json={}).status_code == 422
    assert client.post("/api/predict", json={"pixels": [0.0] * 10}).status_code == 422


def test_websocket_sends_state_then_model(client):
    with client.websocket_connect("/ws/train") as ws:
        assert ws.receive_json()["type"] == "state"
        msg = ws.receive_json()
        assert msg["type"] == "model" and "conv1.weight" in msg["weights"]


@needs_data
def test_samples(client):
    s = client.get("/api/samples?n=3").json()
    assert len(s) == 3 and len(s[0]["pixels"]) == S * S and 0 <= s[0]["label"] <= 9


@needs_data
def test_one_epoch_trains_above_85_percent(client):
    client.post("/api/model/reset", json={"seed": 0})
    r = client.post("/api/train/start", json={"epochs": 1, "lr": 0.005, "augment": False})
    assert r.status_code == 200
    deadline = time.time() + 120
    while trainer.status != "idle" and time.time() < deadline:
        time.sleep(0.2)
    assert trainer.status == "idle"
    assert trainer.epoch_history[-1]["test_acc"] > 0.85


@needs_data
def test_pause_step_stop(client):
    client.post("/api/model/reset", json={"seed": 0})
    assert client.post("/api/train/start", json={"paused": True}).status_code == 200
    assert trainer.status == "paused"
    assert client.post("/api/train/step").status_code == 200
    deadline = time.time() + 10
    while trainer.global_step < 1 and time.time() < deadline:
        time.sleep(0.05)
    assert trainer.global_step == 1
    assert client.post("/api/model/reset", json={}).status_code == 409  # busy
    client.post("/api/train/stop")
    assert trainer.status == "idle"
