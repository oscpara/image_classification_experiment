from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import create_engine, text

from src.paths import IMAGE_SUFFIXES, PROJECT_ROOT, PROMOTION_MANIFEST_PATH
from src.training.settings import SEED


DEFAULT_PROMOTION_MANIFEST_PATH = PROMOTION_MANIFEST_PATH


@dataclass(frozen=True)
class TrainingSample:
    """One approved image and its human-reviewed label."""

    image_path: Path
    label: str


def load_training_samples(database_url: str, query: str) -> list[TrainingSample]:
    """Read approved retraining samples from the database."""

    engine = create_engine(database_url)
    with engine.connect() as connection:
        rows = connection.execute(text(query)).mappings().all()

    return [
        TrainingSample(image_path=Path(row["image_path"]), label=row["label"])
        for row in rows
    ]


def build_imagefolder_dataset(
    base_data_dir: Path,
    samples: list[TrainingSample],
    output_dir: Path,
) -> Path:
    """Copy base data plus approved database images into imagefolder layout."""

    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.copytree(base_data_dir, output_dir)

    valid_labels = {path.name for path in output_dir.iterdir() if path.is_dir()}
    for sample in samples:
        if sample.label not in valid_labels:
            raise ValueError(f"Unknown label from database: {sample.label}")

        source_path = sample.image_path
        if not source_path.is_absolute():
            source_path = PROJECT_ROOT / source_path
        if not source_path.exists():
            raise FileNotFoundError(f"Training image not found: {source_path}")

        target_path = output_dir / sample.label / source_path.name
        shutil.copy2(source_path, target_path)

    return output_dir


def _copy_label_dirs(training_data_dir: Path, output_dir: Path) -> None:
    """Create train/validation/test split folders with one directory per label."""

    labels = [path.name for path in training_data_dir.iterdir() if path.is_dir()]
    for split_name in ("train", "validation", "test"):
        for label in labels:
            (output_dir / split_name / label).mkdir(parents=True, exist_ok=True)


def _image_files(label_dir: Path) -> list[Path]:
    """Return direct child image files in stable order."""

    return sorted(
        path
        for path in label_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _split_base_files(files: list[Path]) -> tuple[list[Path], list[Path], list[Path]]:
    """Split original training files while keeping tiny classes usable."""

    shuffled_files = list(files)
    random.Random(SEED).shuffle(shuffled_files)
    if len(shuffled_files) < 3:
        return shuffled_files, [], []

    test_count = max(1, round(len(shuffled_files) * 0.1))
    validation_count = max(1, round(len(shuffled_files) * 0.1))
    if test_count + validation_count >= len(shuffled_files):
        test_count = 1
        validation_count = 1

    test_files = shuffled_files[:test_count]
    validation_files = shuffled_files[test_count : test_count + validation_count]
    train_files = shuffled_files[test_count + validation_count :]
    return train_files, validation_files, test_files


def _load_promoted_training_paths(manifest_path: Path) -> set[Path]:
    """Read promoted training paths so they can be forced into the train split."""

    promoted_paths: set[Path] = set()
    if not manifest_path.exists():
        return promoted_paths

    with manifest_path.open("r", encoding="utf-8") as manifest:
        for line_number, line in enumerate(manifest, start=1):
            raw_line = line.strip()
            if not raw_line:
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Promotion manifest line {line_number} is not valid JSON"
                ) from exc
            training_image_path = record.get("training_image_path")
            if training_image_path:
                promoted_paths.add((PROJECT_ROOT / training_image_path).resolve())

    return promoted_paths


def _copy_files(files: list[Path], target_label_dir: Path) -> None:
    """Copy files into one split and fail on duplicate names."""

    for source_path in files:
        target_path = target_label_dir / source_path.name
        if target_path.exists():
            raise FileExistsError(f"Split image already exists: {target_path}")
        shutil.copy2(source_path, target_path)


def build_retraining_split_dataset(
    training_data_dir: Path,
    output_dir: Path,
    manifest_path: Path = DEFAULT_PROMOTION_MANIFEST_PATH,
) -> Path:
    """Build train/validation/test folders with promoted predictions train-only."""

    if not training_data_dir.exists():
        raise FileNotFoundError(f"Training data directory not found: {training_data_dir}")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    _copy_label_dirs(training_data_dir, output_dir)

    promoted_paths = _load_promoted_training_paths(manifest_path)
    for label_dir in sorted(path for path in training_data_dir.iterdir() if path.is_dir()):
        promoted_files: list[Path] = []
        base_files: list[Path] = []
        for image_file in _image_files(label_dir):
            if image_file.resolve() in promoted_paths:
                promoted_files.append(image_file)
            else:
                base_files.append(image_file)

        train_files, validation_files, test_files = _split_base_files(base_files)
        _copy_files(train_files + promoted_files, output_dir / "train" / label_dir.name)
        _copy_files(validation_files, output_dir / "validation" / label_dir.name)
        _copy_files(test_files, output_dir / "test" / label_dir.name)

    return output_dir
