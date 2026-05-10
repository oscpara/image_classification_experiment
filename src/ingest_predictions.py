from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.predictions.ingest import (
    DEFAULT_PREDICTIONS_LOG_PATH,
    Base,
    Prediction,
    _insert_prediction,
    _mask_database_url,
    _prediction_exists,
    _prediction_from_record,
    _validate_record,
    create_schema,
    ingest_predictions,
    main,
    parse_args,
)


__all__ = [
    "DEFAULT_PREDICTIONS_LOG_PATH",
    "Base",
    "Prediction",
    "_insert_prediction",
    "_mask_database_url",
    "_prediction_exists",
    "_prediction_from_record",
    "_validate_record",
    "create_schema",
    "ingest_predictions",
    "main",
    "parse_args",
]


if __name__ == "__main__":
    main()
