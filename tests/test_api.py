import json
from io import BytesIO
from types import SimpleNamespace

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

import src.api as api


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(api, "STORAGE_DIR", tmp_path / "storage")
    monkeypatch.setattr(api, "RAW_UPLOADS_DIR", tmp_path / "storage" / "raw_uploads")
    monkeypatch.setattr(api, "PREDICTIONS_DIR", tmp_path / "storage" / "predictions")
    monkeypatch.setattr(
        api,
        "PREDICTIONS_LOG_PATH",
        tmp_path / "storage" / "predictions" / "predictions.jsonl",
    )

    class FakeModel:
        def __call__(self, pixel_values):
            return SimpleNamespace(logits=torch.tensor([[0.1, 3.0, 0.2]]))

    def fake_model_bundle():
        return {
            "labels": ["Akita Inu", "Beagle", "Poodle"],
            "model": FakeModel(),
            "transform": lambda image: torch.zeros((3, 224, 224)),
        }

    monkeypatch.setattr(api, "get_model_bundle", fake_model_bundle)
    return TestClient(api.app)


def _jpeg_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (32, 32), color=(120, 80, 40)).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_health_returns_ok(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "device" in response.json()


def test_predict_saves_uploaded_image_and_prediction_metadata(client):
    response = client.post(
        "/predict",
        files={"image": ("dog.jpg", _jpeg_bytes(), "image/jpeg")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == "Beagle"
    assert body["confidence"] == pytest.approx(0.896191, abs=0.000001)
    assert body["model_version"] == "best_model"
    assert body["timestamp"].endswith("+00:00")

    saved_images = list(api.RAW_UPLOADS_DIR.glob("*.jpg"))
    assert len(saved_images) == 1
    assert saved_images[0].read_bytes() == _jpeg_bytes()

    records = api.PREDICTIONS_LOG_PATH.read_text(encoding="utf-8").splitlines()
    assert len(records) == 1
    record = json.loads(records[0])
    assert record["id"] == body["id"]
    assert record["uploaded_filename"] == "dog.jpg"
    assert record["predicted_breed"] == "Beagle"
    assert record["confidence"] == body["confidence"]
    assert record["model_version"] == "best_model"
    assert record["timestamp"] == body["timestamp"]
    assert record["image_path"].endswith(f"{body['id']}.jpg")


def test_predict_rejects_non_image_upload_without_saving(client):
    response = client.post(
        "/predict",
        files={"image": ("notes.txt", b"not an image", "text/plain")},
    )

    assert response.status_code == 415
    assert not api.RAW_UPLOADS_DIR.exists()
    assert not api.PREDICTIONS_LOG_PATH.exists()


def test_predict_rejects_invalid_image_without_saving(client):
    response = client.post(
        "/predict",
        files={"image": ("dog.jpg", b"not really a jpeg", "image/jpeg")},
    )

    assert response.status_code == 400
    assert not api.RAW_UPLOADS_DIR.exists()
    assert not api.PREDICTIONS_LOG_PATH.exists()
