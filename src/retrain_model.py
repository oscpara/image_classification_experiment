from __future__ import annotations

from src.paths import (
    CHECKPOINT_PATH,
    CHALLENGER_CHECKPOINT_PATH,
    DATA_DIR,
    IMAGE_SUFFIXES,
    MODEL_CHECKPOINT,
    PROJECT_ROOT,
    RETRAINING_DATA_DIR,
    RETRAINING_METRICS_PATH,
)
from src.predictions.promote import (
    DEFAULT_PREDICTIONS_LOG_PATH,
    DEFAULT_PROMOTION_MANIFEST_PATH,
    promote_predictions_to_training_data,
)
from src.training.data import (
    TrainingSample,
    _copy_files,
    _copy_label_dirs,
    _image_files,
    _load_promoted_training_paths,
    _split_base_files,
    build_imagefolder_dataset,
    build_retraining_split_dataset,
    load_training_samples,
)
from src.training.model import (
    _build_dataloaders,
    _load_model,
    collate_fn,
    run_epoch,
    train_model,
)
from src.training.retrain import main, parse_args
from src.training.settings import (
    BATCH_SIZE,
    DEVICE,
    LEARNING_RATE,
    MAX_EPOCHS,
    PATIENCE,
    SEED,
    seed_everything,
)


__all__ = [
    "BATCH_SIZE",
    "CHECKPOINT_PATH",
    "CHALLENGER_CHECKPOINT_PATH",
    "DATA_DIR",
    "DEFAULT_PREDICTIONS_LOG_PATH",
    "DEFAULT_PROMOTION_MANIFEST_PATH",
    "DEVICE",
    "IMAGE_SUFFIXES",
    "LEARNING_RATE",
    "MAX_EPOCHS",
    "MODEL_CHECKPOINT",
    "PATIENCE",
    "PROJECT_ROOT",
    "RETRAINING_DATA_DIR",
    "RETRAINING_METRICS_PATH",
    "SEED",
    "TrainingSample",
    "_build_dataloaders",
    "_copy_files",
    "_copy_label_dirs",
    "_image_files",
    "_load_model",
    "_load_promoted_training_paths",
    "_split_base_files",
    "build_imagefolder_dataset",
    "build_retraining_split_dataset",
    "collate_fn",
    "load_training_samples",
    "main",
    "parse_args",
    "promote_predictions_to_training_data",
    "run_epoch",
    "seed_everything",
    "train_model",
]


if __name__ == "__main__":
    main()
