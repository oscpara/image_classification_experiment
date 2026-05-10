from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.paths import (
    DATA_DIR,
    PREDICTIONS_LOG_PATH,
    PROJECT_ROOT,
    PROMOTION_MANIFEST_PATH,
)


DEFAULT_PREDICTIONS_LOG_PATH = PREDICTIONS_LOG_PATH
DEFAULT_TRAINING_DATA_DIR = DATA_DIR
DEFAULT_PROMOTION_MANIFEST_PATH = PROMOTION_MANIFEST_PATH

REQUIRED_FIELDS = {
    "id",
    "predicted_breed",
    "timestamp",
    "image_path",
}


def _resolve_project_path(path: Path) -> Path:
    """Resolve paths recorded relative to the project root."""

    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _load_promoted_ids(manifest_path: Path) -> set[str]:
    """Read promotion history so reruns do not duplicate training images."""

    promoted_ids: set[str] = set()
    if not manifest_path.exists():
        return promoted_ids

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
            prediction_id = record.get("prediction_id")
            if prediction_id:
                promoted_ids.add(str(prediction_id))

    return promoted_ids


def _validate_prediction_record(record: dict[str, Any], line_number: int) -> None:
    """Fail fast when a prediction cannot be promoted safely."""

    missing_fields = sorted(REQUIRED_FIELDS - record.keys())
    if missing_fields:
        joined_fields = ", ".join(missing_fields)
        raise ValueError(f"Line {line_number} is missing required fields: {joined_fields}")


def _training_image_name(record: dict[str, Any]) -> str:
    """Name promoted files by prediction id to avoid uploaded filename collisions."""

    suffix = Path(record["image_path"]).suffix.lower() or ".jpg"
    return f"{record['id']}{suffix}"


def _project_relative_path(path: Path) -> str:
    """Store manifest paths in the same stable form on every OS."""

    return path.relative_to(PROJECT_ROOT).as_posix()


def _append_manifest_record(
    manifest_path: Path,
    *,
    record: dict[str, Any],
    source_image_path: Path,
    training_image_path: Path,
    source_log_path: Path,
    source_line_number: int,
    source_raw_json: str,
) -> None:
    """Record the source-to-training copy for auditability."""

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_record = {
        "prediction_id": record["id"],
        "label": record["predicted_breed"],
        "prediction_timestamp": record["timestamp"],
        "source_image_path": _project_relative_path(source_image_path),
        "training_image_path": _project_relative_path(training_image_path),
        "source_log_path": str(source_log_path),
        "source_line_number": source_line_number,
        "source_raw_json": source_raw_json,
        "promoted_at": datetime.now(timezone.utc).isoformat(),
    }
    with manifest_path.open("a", encoding="utf-8") as manifest:
        manifest.write(json.dumps(manifest_record) + "\n")


def promote_predictions_to_training_data(
    predictions_log_path: Path = DEFAULT_PREDICTIONS_LOG_PATH,
    training_data_dir: Path = DEFAULT_TRAINING_DATA_DIR,
    manifest_path: Path = DEFAULT_PROMOTION_MANIFEST_PATH,
) -> int:
    """Copy unpromoted prediction images into the training data folders."""

    if not predictions_log_path.exists():
        raise FileNotFoundError(f"Predictions log not found: {predictions_log_path}")
    if not training_data_dir.exists():
        raise FileNotFoundError(f"Training data directory not found: {training_data_dir}")

    valid_labels = {path.name for path in training_data_dir.iterdir() if path.is_dir()}
    promoted_ids = _load_promoted_ids(manifest_path)
    source_log_path = predictions_log_path.resolve()
    promoted_count = 0

    with predictions_log_path.open("r", encoding="utf-8") as predictions_log:
        for line_number, line in enumerate(predictions_log, start=1):
            raw_line = line.strip()
            if not raw_line:
                continue

            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Line {line_number} is not valid JSON") from exc

            _validate_prediction_record(record, line_number)
            prediction_id = str(record["id"])
            if prediction_id in promoted_ids:
                continue

            label = record["predicted_breed"]
            if label not in valid_labels:
                raise ValueError(f"Line {line_number} has unknown label: {label}")

            source_image_path = _resolve_project_path(Path(record["image_path"]))
            if not source_image_path.exists():
                raise FileNotFoundError(f"Prediction image not found: {source_image_path}")

            training_image_path = training_data_dir / label / _training_image_name(record)
            if training_image_path.exists():
                raise FileExistsError(f"Training image already exists: {training_image_path}")

            shutil.copy2(source_image_path, training_image_path)
            _append_manifest_record(
                manifest_path,
                record=record,
                source_image_path=source_image_path,
                training_image_path=training_image_path,
                source_log_path=source_log_path,
                source_line_number=line_number,
                source_raw_json=raw_line,
            )
            promoted_ids.add(prediction_id)
            promoted_count += 1

    return promoted_count


def parse_args() -> argparse.Namespace:
    """Parse command line options for promoting prediction images."""

    parser = argparse.ArgumentParser(
        description="Append prediction images to the training data using predicted labels."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_PREDICTIONS_LOG_PATH,
        help="Path to predictions.jsonl.",
    )
    parser.add_argument(
        "--training-data-dir",
        type=Path,
        default=DEFAULT_TRAINING_DATA_DIR,
        help="Training imagefolder root, with one directory per breed.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_PROMOTION_MANIFEST_PATH,
        help="JSONL manifest tracking promoted predictions.",
    )
    return parser.parse_args()


def main() -> None:
    """Run prediction promotion from the command line."""

    args = parse_args()
    promoted_count = promote_predictions_to_training_data(
        predictions_log_path=args.source,
        training_data_dir=args.training_data_dir,
        manifest_path=args.manifest,
    )
    print(f"Promoted {promoted_count} prediction images into {args.training_data_dir}")
