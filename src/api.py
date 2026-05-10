from functools import lru_cache
from datetime import datetime, timezone
from io import BytesIO
import json
import os
from pathlib import Path
from uuid import uuid4
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from transformers import AutoImageProcessor, AutoModelForImageClassification

from src.paths import (
    CHECKPOINT_PATH,
    DATA_DIR,
    IMAGE_SUFFIXES,
    MODEL_CHECKPOINT,
    PREDICTIONS_DIR,
    PREDICTIONS_LOG_PATH,
    PROJECT_ROOT,
    RAW_UPLOADS_DIR,
    STORAGE_DIR,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


app = FastAPI(title="Dog Breed Image Classifier")


def _safe_file_suffix(filename: str | None) -> str:
    """Return a known image suffix so saved uploads have predictable names."""

    suffix = Path(filename or "").suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return suffix
    return ".jpg"


def _save_prediction_record(record: dict[str, Any], image_bytes: bytes, suffix: str) -> None:
    """Persist the uploaded image and append its prediction metadata to JSONL."""

    RAW_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    image_path = RAW_UPLOADS_DIR / f"{record['id']}{suffix}"
    image_path.write_bytes(image_bytes)

    record["image_path"] = str(image_path.relative_to(PROJECT_ROOT))
    with PREDICTIONS_LOG_PATH.open("a", encoding="utf-8") as predictions_log:
        predictions_log.write(json.dumps(record) + "\n")


def _load_labels() -> list[str]:
    """Load class labels from the training-data folder names."""

    if not DATA_DIR.exists():
        raise RuntimeError(f"Could not find label directory: {DATA_DIR}")

    labels = sorted(path.name for path in DATA_DIR.iterdir() if path.is_dir())
    if not labels:
        raise RuntimeError(f"No class folders found in: {DATA_DIR}")

    return labels


@lru_cache(maxsize=1)
def get_model_bundle() -> dict[str, Any]:
    """Load and cache the image processor, transform, labels, and model."""

    if not CHECKPOINT_PATH.exists():
        raise RuntimeError(f"Could not find model checkpoint: {CHECKPOINT_PATH}")

    labels = _load_labels()
    id2label = dict(enumerate(labels))
    label2id = {label: idx for idx, label in id2label.items()}

    image_processor = AutoImageProcessor.from_pretrained(
        MODEL_CHECKPOINT,
        local_files_only=True,
        use_fast=False,
    )
    image_size = image_processor.size.get("shortest_edge", 224)
    transform = Compose(
        [
            Resize(image_size),
            CenterCrop(image_size),
            ToTensor(),
            Normalize(mean=image_processor.image_mean, std=image_processor.image_std),
        ]
    )

    model = AutoModelForImageClassification.from_pretrained(
        MODEL_CHECKPOINT,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
        local_files_only=True,
        use_safetensors=False,
    )
    state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return {
        "labels": labels,
        "model": model,
        "transform": transform,
    }


@app.get("/health")
def health() -> dict[str, str]:
    """Return a lightweight readiness response."""

    return {"status": "ok", "device": DEVICE}


@app.post("/predict")
async def predict(image: UploadFile = File(...)) -> dict[str, Any]:
    """Run inference for one uploaded image and record an audit trail."""

    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload must be an image file.")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    try:
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Could not decode image.") from exc

    try:
        bundle = get_model_bundle()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    pixel_values = bundle["transform"](pil_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = bundle["model"](pixel_values=pixel_values).logits
        probabilities = torch.softmax(logits, dim=-1)[0]
        confidence, label_id = probabilities.max(dim=-1)

    label_index = int(label_id.item())
    prediction = bundle["labels"][label_index]
    confidence_score = round(float(confidence.item()), 6)
    model_version = CHECKPOINT_PATH.stem
    timestamp = datetime.now(timezone.utc).isoformat()

    record = {
        "id": str(uuid4()),
        "uploaded_filename": image.filename,
        "predicted_breed": prediction,
        "confidence": confidence_score,
        "model_version": model_version,
        "timestamp": timestamp,
    }
    _save_prediction_record(record, image_bytes, _safe_file_suffix(image.filename))

    return {
        "id": record["id"],
        "prediction": prediction,
        "confidence": confidence_score,
        "model_version": model_version,
        "timestamp": timestamp,
    }
