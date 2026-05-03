from pathlib import Path

import pytest
from sqlalchemy import create_engine, text

from src.retrain_model import (
    TrainingSample,
    build_imagefolder_dataset,
    load_training_samples,
)


def test_load_training_samples_reads_database_rows(tmp_path):
    database_url = f"sqlite:///{tmp_path / 'training.db'}"
    engine = create_engine(database_url)
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE training_samples (image_path TEXT, label TEXT)"))
        connection.execute(
            text("INSERT INTO training_samples VALUES (:image_path, :label)"),
            {"image_path": "storage/raw_uploads/dog.jpg", "label": "Beagle"},
        )

    samples = load_training_samples(
        database_url,
        "SELECT image_path, label FROM training_samples",
    )

    assert samples == [TrainingSample(Path("storage/raw_uploads/dog.jpg"), "Beagle")]


def test_build_imagefolder_dataset_adds_database_images(tmp_path, monkeypatch):
    project_root = tmp_path
    base_data_dir = tmp_path / "data" / "9Breeds"
    output_dir = tmp_path / "storage" / "retraining" / "imagefolder"
    upload_path = tmp_path / "storage" / "raw_uploads" / "new.jpg"

    (base_data_dir / "Beagle").mkdir(parents=True)
    (base_data_dir / "Poodle").mkdir(parents=True)
    (base_data_dir / "Beagle" / "base.jpg").write_bytes(b"base")
    upload_path.parent.mkdir(parents=True)
    upload_path.write_bytes(b"new")

    monkeypatch.setattr("src.retrain_model.PROJECT_ROOT", project_root)

    result = build_imagefolder_dataset(
        base_data_dir,
        [TrainingSample(Path("storage/raw_uploads/new.jpg"), "Beagle")],
        output_dir,
    )

    assert result == output_dir
    assert (output_dir / "Beagle" / "base.jpg").read_bytes() == b"base"
    assert (output_dir / "Beagle" / "new.jpg").read_bytes() == b"new"
    assert (output_dir / "Poodle").is_dir()


def test_build_imagefolder_dataset_rejects_unknown_labels(tmp_path):
    base_data_dir = tmp_path / "data" / "9Breeds"
    output_dir = tmp_path / "storage" / "retraining" / "imagefolder"
    (base_data_dir / "Beagle").mkdir(parents=True)

    with pytest.raises(ValueError, match="Unknown label"):
        build_imagefolder_dataset(
            base_data_dir,
            [TrainingSample(Path("dog.jpg"), "Unknown")],
            output_dir,
        )
