from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, text

from src.training import retrain
from src.training.data import (
    TrainingSample,
    build_imagefolder_dataset,
    build_retraining_split_dataset,
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

    monkeypatch.setattr("src.training.data.PROJECT_ROOT", project_root)

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


def test_build_retraining_split_dataset_keeps_promoted_images_train_only(
    tmp_path, monkeypatch
):
    project_root = tmp_path
    training_data_dir = tmp_path / "data" / "9Breeds"
    output_dir = tmp_path / "storage" / "retraining" / "imagefolder"
    manifest_path = tmp_path / "storage" / "promotions" / "promoted_predictions.jsonl"

    label_dir = training_data_dir / "Beagle"
    label_dir.mkdir(parents=True)
    for index in range(10):
        (label_dir / f"base-{index}.jpg").write_bytes(b"base")
    promoted_image = label_dir / "prediction-1.jpg"
    promoted_image.write_bytes(b"promoted")

    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        '{"prediction_id": "prediction-1", '
        '"training_image_path": "data/9Breeds/Beagle/prediction-1.jpg"}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr("src.training.data.PROJECT_ROOT", project_root)

    result = build_retraining_split_dataset(
        training_data_dir,
        output_dir,
        manifest_path,
    )

    assert result == output_dir
    assert (output_dir / "train" / "Beagle" / "prediction-1.jpg").read_bytes() == b"promoted"
    assert not (output_dir / "validation" / "Beagle" / "prediction-1.jpg").exists()
    assert not (output_dir / "test" / "Beagle" / "prediction-1.jpg").exists()
    assert len(list((output_dir / "validation" / "Beagle").iterdir())) == 1
    assert len(list((output_dir / "test" / "Beagle").iterdir())) == 1


def test_retrain_main_does_not_promote_predictions_without_opt_in(tmp_path, monkeypatch):
    metrics_output = tmp_path / "metrics.json"
    split_data_dir = tmp_path / "split"

    monkeypatch.setattr(retrain, "parse_args", lambda: SimpleNamespace(
        predictions_log=tmp_path / "predictions.jsonl",
        training_data_dir=tmp_path / "data",
        source_data_dir=tmp_path / "source",
        promotion_manifest=tmp_path / "manifest.jsonl",
        split_data_dir=split_data_dir,
        current_checkpoint=tmp_path / "best_model.pth",
        challenger_checkpoint=tmp_path / "challenger_model.pth",
        metrics_output=metrics_output,
        max_epochs=1,
        database_url=None,
        training_samples_sql="SELECT image_path, label FROM training_samples",
        promote_predictions=False,
    ))
    monkeypatch.setattr(retrain, "seed_everything", lambda: None)
    monkeypatch.setattr(
        retrain,
        "promote_predictions_to_training_data",
        lambda **kwargs: pytest.fail("prediction promotion should be opt-in"),
    )
    monkeypatch.setattr(
        retrain,
        "build_retraining_split_dataset",
        lambda training_data_dir, output_dir, manifest_path: split_data_dir,
    )
    monkeypatch.setattr(
        retrain,
        "train_model",
        lambda *args, **kwargs: {
            "current_best": {},
            "challenger": {"loss": 0.1, "accuracy": 1.0},
            "comparison": "no_current_checkpoint",
            "promoted": True,
        },
    )

    retrain.main()

    assert '"promoted_count": 0' in metrics_output.read_text(encoding="utf-8")
