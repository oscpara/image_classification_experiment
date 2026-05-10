from __future__ import annotations

from src.predictions.promote import (
    DEFAULT_PREDICTIONS_LOG_PATH,
    DEFAULT_PROMOTION_MANIFEST_PATH,
    DEFAULT_TRAINING_DATA_DIR,
    PROJECT_ROOT,
    _append_manifest_record,
    _load_promoted_ids,
    _project_relative_path,
    _resolve_project_path,
    _training_image_name,
    _validate_prediction_record,
    main,
    parse_args,
    promote_predictions_to_training_data,
)


__all__ = [
    "DEFAULT_PREDICTIONS_LOG_PATH",
    "DEFAULT_PROMOTION_MANIFEST_PATH",
    "DEFAULT_TRAINING_DATA_DIR",
    "PROJECT_ROOT",
    "_append_manifest_record",
    "_load_promoted_ids",
    "_project_relative_path",
    "_resolve_project_path",
    "_training_image_name",
    "_validate_prediction_record",
    "main",
    "parse_args",
    "promote_predictions_to_training_data",
]


if __name__ == "__main__":
    main()
