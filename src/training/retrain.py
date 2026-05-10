from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from src.config import get_database_url
from src.paths import (
    CHECKPOINT_PATH,
    CHALLENGER_CHECKPOINT_PATH,
    DATA_DIR,
    PREDICTIONS_LOG_PATH,
    PROMOTION_MANIFEST_PATH,
    RETRAINING_DATA_DIR,
    RETRAINING_METRICS_PATH,
    RETRAINING_SOURCE_DATA_DIR,
)
from src.predictions.promote import promote_predictions_to_training_data
from src.training.data import (
    build_imagefolder_dataset,
    build_retraining_split_dataset,
    load_training_samples,
)
from src.training.model import train_model
from src.training.settings import MAX_EPOCHS, seed_everything

TRAINING_SAMPLES_SQL = os.environ.get(
    "TRAINING_SAMPLES_SQL",
    "SELECT image_path, label FROM training_samples WHERE approved = true",
)


def parse_args() -> argparse.Namespace:
    """Parse retraining job settings."""

    parser = argparse.ArgumentParser(description="Retrain the dog breed classifier.")
    parser.add_argument("--predictions-log", type=Path, default=PREDICTIONS_LOG_PATH)
    parser.add_argument("--training-data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--source-data-dir", type=Path, default=RETRAINING_SOURCE_DATA_DIR)
    parser.add_argument("--promotion-manifest", type=Path, default=PROMOTION_MANIFEST_PATH)
    parser.add_argument("--split-data-dir", type=Path, default=RETRAINING_DATA_DIR)
    parser.add_argument("--current-checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--challenger-checkpoint", type=Path, default=CHALLENGER_CHECKPOINT_PATH)
    parser.add_argument("--metrics-output", type=Path, default=RETRAINING_METRICS_PATH)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument(
        "--database-url",
        default=get_database_url(),
        help="SQLAlchemy database URL for approved training samples.",
    )
    parser.add_argument("--training-samples-sql", default=TRAINING_SAMPLES_SQL)
    return parser.parse_args()


def main() -> None:
    """Promote predictions, retrain a challenger, and compare to current best."""

    args = parse_args()

    seed_everything()
    promoted_count = 0
    if args.predictions_log.exists():
        promoted_count = promote_predictions_to_training_data(
            predictions_log_path=args.predictions_log,
            training_data_dir=args.training_data_dir,
            manifest_path=args.promotion_manifest,
        )
        print(f"Promoted {promoted_count} new prediction images")
    else:
        print(f"No predictions log found at {args.predictions_log}; skipping prediction promotion")

    training_data_dir = args.training_data_dir
    if args.database_url:
        samples = load_training_samples(args.database_url, args.training_samples_sql)
        training_data_dir = build_imagefolder_dataset(
            args.training_data_dir,
            samples,
            args.source_data_dir,
        )
        print(f"Loaded {len(samples)} approved database samples")

    data_dir = build_retraining_split_dataset(
        training_data_dir,
        args.split_data_dir,
        args.promotion_manifest,
    )
    metrics = train_model(
        data_dir,
        current_checkpoint_path=args.current_checkpoint,
        challenger_checkpoint_path=args.challenger_checkpoint,
        max_epochs=args.max_epochs,
    )
    metrics_record = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "promoted_count": promoted_count,
        "split_data_dir": str(data_dir),
        "current_checkpoint": str(args.current_checkpoint),
        "challenger_checkpoint": str(args.challenger_checkpoint),
        **metrics,
    }
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(json.dumps(metrics_record, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote retraining metrics to {args.metrics_output}")
