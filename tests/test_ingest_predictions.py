import json

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from src.ingest_predictions import Prediction, _mask_database_url, ingest_predictions


def _prediction_record(prediction_id: str = "prediction-1") -> dict[str, object]:
    return {
        "id": prediction_id,
        "uploaded_filename": "dog.jpg",
        "predicted_breed": "Beagle",
        "confidence": 0.995838,
        "model_version": "best_model",
        "timestamp": "2026-05-03T00:38:33.150698+00:00",
        "image_path": f"storage/raw_uploads/{prediction_id}.jpg",
    }


def _database_url(tmp_path) -> str:
    return f"sqlite:///{tmp_path / 'predictions.db'}"


def _fetch_predictions(database_url: str) -> list[Prediction]:
    engine = create_engine(database_url)
    session_factory = sessionmaker(bind=engine)
    with session_factory() as session:
        return list(session.scalars(select(Prediction)).all())


def test_ingest_predictions_loads_metadata_with_source_trace(tmp_path):
    source_path = tmp_path / "predictions.jsonl"
    database_url = _database_url(tmp_path)
    raw_json = json.dumps(_prediction_record())
    source_path.write_text(raw_json + "\n", encoding="utf-8")

    ingested_count = ingest_predictions(source_path, database_url)

    assert ingested_count == 1

    row = _fetch_predictions(database_url)[0]
    assert row.prediction_id == "prediction-1"
    assert row.uploaded_filename == "dog.jpg"
    assert row.predicted_breed == "Beagle"
    assert row.confidence == pytest.approx(0.995838)
    assert row.model_version == "best_model"
    assert row.prediction_timestamp == "2026-05-03T00:38:33.150698+00:00"
    assert row.image_path == "storage/raw_uploads/prediction-1.jpg"
    assert row.source_log_path == str(source_path.resolve())
    assert row.source_line_number == 1
    assert row.source_raw_json == raw_json
    assert row.ingested_at.endswith("+00:00")


def test_ingest_predictions_is_idempotent(tmp_path):
    source_path = tmp_path / "predictions.jsonl"
    database_url = _database_url(tmp_path)
    source_path.write_text(json.dumps(_prediction_record()) + "\n", encoding="utf-8")

    first_count = ingest_predictions(source_path, database_url)
    second_count = ingest_predictions(source_path, database_url)

    assert first_count == 1
    assert second_count == 0
    assert len(_fetch_predictions(database_url)) == 1


def test_ingest_predictions_rejects_invalid_jsonl(tmp_path):
    source_path = tmp_path / "predictions.jsonl"
    source_path.write_text("{not valid json}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Line 1 is not valid JSON"):
        ingest_predictions(source_path, _database_url(tmp_path))


def test_ingest_predictions_rejects_missing_required_fields(tmp_path):
    source_path = tmp_path / "predictions.jsonl"
    source_path.write_text(json.dumps({"id": "prediction-1"}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Line 1 is missing required fields"):
        ingest_predictions(source_path, _database_url(tmp_path))


def test_mask_database_url_hides_password():
    masked_url = _mask_database_url(
        "postgresql+psycopg://myapp_user:secret@localhost:5432/myapp"
    )

    assert masked_url == "postgresql+psycopg://myapp_user:***@localhost:5432/myapp"
