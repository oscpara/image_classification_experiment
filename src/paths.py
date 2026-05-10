from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "9Breeds"
STORAGE_DIR = PROJECT_ROOT / "storage"
RAW_UPLOADS_DIR = STORAGE_DIR / "raw_uploads"
PREDICTIONS_DIR = STORAGE_DIR / "predictions"
PREDICTIONS_LOG_PATH = PREDICTIONS_DIR / "predictions.jsonl"
PROMOTIONS_DIR = STORAGE_DIR / "promotions"
PROMOTION_MANIFEST_PATH = PROMOTIONS_DIR / "promoted_predictions.jsonl"
RETRAINING_DATA_DIR = STORAGE_DIR / "retraining" / "imagefolder"
RETRAINING_SOURCE_DATA_DIR = STORAGE_DIR / "retraining" / "source_imagefolder"
RETRAINING_METRICS_PATH = STORAGE_DIR / "retraining" / "metrics.json"
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "temp"
    / "models"
    / "convnext-dogs-classification"
    / "checkpoints"
    / "best_model.pth"
)
CHALLENGER_CHECKPOINT_PATH = CHECKPOINT_PATH.with_name("challenger_model.pth")
MODEL_CHECKPOINT = "facebook/convnext-base-224"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
