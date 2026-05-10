import json

import pytest

from src.predictions import promote


def _prediction_record(prediction_id: str = "prediction-1") -> dict[str, object]:
    return {
        "id": prediction_id,
        "uploaded_filename": "dog.jpg",
        "predicted_breed": "Beagle",
        "confidence": 0.99,
        "model_version": "best_model",
        "timestamp": "2026-05-07T19:00:00+00:00",
        "image_path": f"storage/raw_uploads/{prediction_id}.jpg",
    }


def test_promote_predictions_copies_images_and_writes_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(promote, "PROJECT_ROOT", tmp_path)
    training_data_dir = tmp_path / "data" / "9Breeds"
    predictions_log_path = tmp_path / "storage" / "predictions" / "predictions.jsonl"
    manifest_path = tmp_path / "storage" / "promotions" / "promoted_predictions.jsonl"
    upload_path = tmp_path / "storage" / "raw_uploads" / "prediction-1.jpg"
    record = _prediction_record()

    (training_data_dir / "Beagle").mkdir(parents=True)
    (training_data_dir / "Poodle").mkdir(parents=True)
    predictions_log_path.parent.mkdir(parents=True)
    predictions_log_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    upload_path.parent.mkdir(parents=True)
    upload_path.write_bytes(b"image")

    promoted_count = promote.promote_predictions_to_training_data(
        predictions_log_path=predictions_log_path,
        training_data_dir=training_data_dir,
        manifest_path=manifest_path,
    )

    target_path = training_data_dir / "Beagle" / "prediction-1.jpg"
    assert promoted_count == 1
    assert target_path.read_bytes() == b"image"

    manifest_record = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_record["prediction_id"] == "prediction-1"
    assert manifest_record["label"] == "Beagle"
    assert manifest_record["source_image_path"] == "storage/raw_uploads/prediction-1.jpg"
    assert manifest_record["training_image_path"] == "data/9Breeds/Beagle/prediction-1.jpg"
    assert manifest_record["source_line_number"] == 1


def test_promote_predictions_is_idempotent_from_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(promote, "PROJECT_ROOT", tmp_path)
    training_data_dir = tmp_path / "data" / "9Breeds"
    predictions_log_path = tmp_path / "storage" / "predictions" / "predictions.jsonl"
    manifest_path = tmp_path / "storage" / "promotions" / "promoted_predictions.jsonl"
    upload_path = tmp_path / "storage" / "raw_uploads" / "prediction-1.jpg"
    record = _prediction_record()

    (training_data_dir / "Beagle").mkdir(parents=True)
    predictions_log_path.parent.mkdir(parents=True)
    predictions_log_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    upload_path.parent.mkdir(parents=True)
    upload_path.write_bytes(b"image")

    first_count = promote.promote_predictions_to_training_data(
        predictions_log_path,
        training_data_dir,
        manifest_path,
    )
    second_count = promote.promote_predictions_to_training_data(
        predictions_log_path,
        training_data_dir,
        manifest_path,
    )

    assert first_count == 1
    assert second_count == 0
    assert len(manifest_path.read_text(encoding="utf-8").splitlines()) == 1


def test_promote_predictions_rejects_unknown_labels(tmp_path, monkeypatch):
    monkeypatch.setattr(promote, "PROJECT_ROOT", tmp_path)
    training_data_dir = tmp_path / "data" / "9Breeds"
    predictions_log_path = tmp_path / "storage" / "predictions" / "predictions.jsonl"
    manifest_path = tmp_path / "storage" / "promotions" / "promoted_predictions.jsonl"
    record = _prediction_record()

    (training_data_dir / "Poodle").mkdir(parents=True)
    predictions_log_path.parent.mkdir(parents=True)
    predictions_log_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unknown label: Beagle"):
        promote.promote_predictions_to_training_data(
            predictions_log_path,
            training_data_dir,
            manifest_path,
        )


def test_promote_predictions_rejects_missing_images(tmp_path, monkeypatch):
    monkeypatch.setattr(promote, "PROJECT_ROOT", tmp_path)
    training_data_dir = tmp_path / "data" / "9Breeds"
    predictions_log_path = tmp_path / "storage" / "predictions" / "predictions.jsonl"
    manifest_path = tmp_path / "storage" / "promotions" / "promoted_predictions.jsonl"
    record = _prediction_record()

    (training_data_dir / "Beagle").mkdir(parents=True)
    predictions_log_path.parent.mkdir(parents=True)
    predictions_log_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Prediction image not found"):
        promote.promote_predictions_to_training_data(
            predictions_log_path,
            training_data_dir,
            manifest_path,
        )
