from __future__ import annotations

import argparse
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import load_dataset
from sqlalchemy import create_engine, text
from torch.utils.data import DataLoader
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForImageClassification

try:
    from src.config import get_database_url
except ModuleNotFoundError:
    from config import get_database_url


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "9Breeds"
RETRAINING_DATA_DIR = PROJECT_ROOT / "storage" / "retraining" / "imagefolder"
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "temp"
    / "models"
    / "convnext-dogs-classification"
    / "checkpoints"
    / "best_model.pth"
)
MODEL_CHECKPOINT = "facebook/convnext-base-224"
TRAINING_SAMPLES_SQL = os.environ.get(
    "TRAINING_SAMPLES_SQL",
    "SELECT image_path, label FROM training_samples WHERE approved = true",
)

SEED = 42
BATCH_SIZE = 8
LEARNING_RATE = 5e-4
MAX_EPOCHS = 100
PATIENCE = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class TrainingSample:
    """One approved image and its human-reviewed label."""

    image_path: Path
    label: str


def seed_everything() -> None:
    """Make dataset splits and training as repeatable as PyTorch allows."""

    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


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


def collate_fn(examples):
    """Stack transformed images and labels into a PyTorch batch."""

    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def run_epoch(model, dataloader, optimizer=None) -> dict[str, float]:
    """Run one train or evaluation epoch."""

    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    context = torch.enable_grad() if is_training else torch.no_grad()

    with context:
        for batch in tqdm(dataloader, desc="Training" if is_training else "Evaluating"):
            batch = {key: value.to(DEVICE) for key, value in batch.items()}
            if is_training:
                optimizer.zero_grad()

            outputs = model(pixel_values=batch["pixel_values"], labels=batch["labels"])
            loss = outputs.loss
            logits = outputs.logits

            if is_training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_examples += batch["labels"].shape[0]
            total_correct += (logits.argmax(dim=-1) == batch["labels"]).sum().item()

    return {
        "loss": total_loss / len(dataloader),
        "accuracy": total_correct / total_examples,
    }


def train_model(data_dir: Path) -> None:
    """Retrain using the same ConvNeXT method as the notebook."""

    dataset = load_dataset("imagefolder", data_dir=str(data_dir))["train"]
    labels = dataset.features["label"].names
    id2label = dict(enumerate(labels))
    label2id = {label: idx for idx, label in id2label.items()}

    split_train_val_test = dataset.train_test_split(
        test_size=0.2,
        seed=SEED,
        stratify_by_column="label",
    )
    train_val_dataset = split_train_val_test["train"]
    test_sim_dataset = split_train_val_test["test"]
    split_test_sim = test_sim_dataset.train_test_split(
        test_size=0.5,
        seed=SEED,
        stratify_by_column="label",
    )
    test_dataset = split_test_sim["train"]
    split_train_val = train_val_dataset.train_test_split(
        test_size=0.25,
        seed=SEED,
        stratify_by_column="label",
    )
    train_dataset = split_train_val["train"]
    validation_dataset = split_train_val["test"]

    image_processor = AutoImageProcessor.from_pretrained(MODEL_CHECKPOINT)
    image_size = image_processor.size.get("shortest_edge", 224)
    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    train_transform = Compose(
        [RandomResizedCrop(image_size), RandomHorizontalFlip(), ToTensor(), normalize]
    )
    eval_transform = Compose([Resize(image_size), CenterCrop(image_size), ToTensor(), normalize])

    def apply_train_transform(examples):
        examples["pixel_values"] = [
            train_transform(image.convert("RGB")) for image in examples["image"]
        ]
        return examples

    def apply_eval_transform(examples):
        examples["pixel_values"] = [
            eval_transform(image.convert("RGB")) for image in examples["image"]
        ]
        return examples

    train_dataloader = DataLoader(
        train_dataset.with_transform(apply_train_transform),
        collate_fn=collate_fn,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    validation_dataloader = DataLoader(
        validation_dataset.with_transform(apply_eval_transform),
        collate_fn=collate_fn,
        batch_size=BATCH_SIZE,
    )
    test_dataloader = DataLoader(
        test_dataset.with_transform(apply_eval_transform),
        collate_fn=collate_fn,
        batch_size=BATCH_SIZE,
    )

    model = AutoModelForImageClassification.from_pretrained(
        MODEL_CHECKPOINT,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    if CHECKPOINT_PATH.exists():
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))

    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    model.to(DEVICE)
    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=LEARNING_RATE,
    )

    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        print(f"Epoch {epoch}/{MAX_EPOCHS}")
        train_metrics = run_epoch(model, train_dataloader, optimizer=optimizer)
        validation_metrics = run_epoch(model, validation_dataloader)
        print(f"Train: {train_metrics}")
        print(f"Validation: {validation_metrics}")

        if validation_metrics["loss"] < best_validation_loss:
            best_validation_loss = validation_metrics["loss"]
            epochs_without_improvement = 0
            CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"Saved best model to {CHECKPOINT_PATH}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            break

    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    print(f"Test: {run_epoch(model, test_dataloader)}")


def parse_args() -> argparse.Namespace:
    """Parse retraining job settings."""

    parser = argparse.ArgumentParser(description="Retrain the dog breed classifier.")
    parser.add_argument(
        "--database-url",
        default=get_database_url(),
        help="SQLAlchemy database URL. Defaults to DATABASE_URL, then config.ini.",
    )
    parser.add_argument("--training-samples-sql", default=TRAINING_SAMPLES_SQL)
    return parser.parse_args()


def main() -> None:
    """Build the retraining dataset and run training."""

    args = parse_args()
    if not args.database_url:
        raise SystemExit("Set DATABASE_URL, set [database].url in config.ini, or pass --database-url.")

    seed_everything()
    samples = load_training_samples(args.database_url, args.training_samples_sql)
    data_dir = build_imagefolder_dataset(DATA_DIR, samples, RETRAINING_DATA_DIR)
    train_model(data_dir)


if __name__ == "__main__":
    main()
