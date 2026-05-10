from __future__ import annotations

import shutil
from pathlib import Path

import torch
from datasets import load_dataset
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

from src.paths import CHECKPOINT_PATH, CHALLENGER_CHECKPOINT_PATH, MODEL_CHECKPOINT
from src.training.settings import BATCH_SIZE, DEVICE, LEARNING_RATE, MAX_EPOCHS, PATIENCE


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


def _build_dataloaders(data_dir: Path):
    """Load split imagefolder data and attach training/evaluation transforms."""

    dataset = load_dataset("imagefolder", data_dir=str(data_dir))
    labels = dataset["train"].features["label"].names
    id2label = dict(enumerate(labels))
    label2id = {label: idx for idx, label in id2label.items()}

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
        dataset["train"].with_transform(apply_train_transform),
        collate_fn=collate_fn,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    validation_dataloader = DataLoader(
        dataset["validation"].with_transform(apply_eval_transform),
        collate_fn=collate_fn,
        batch_size=BATCH_SIZE,
    )
    test_dataloader = DataLoader(
        dataset["test"].with_transform(apply_eval_transform),
        collate_fn=collate_fn,
        batch_size=BATCH_SIZE,
    )

    return train_dataloader, validation_dataloader, test_dataloader, id2label, label2id


def _load_model(id2label, label2id, checkpoint_path: Path | None = None):
    """Create the ConvNeXT classifier and optionally load a local checkpoint."""

    model = AutoModelForImageClassification.from_pretrained(
        MODEL_CHECKPOINT,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    if checkpoint_path and checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.to(DEVICE)
    return model


def train_model(
    data_dir: Path,
    current_checkpoint_path: Path = CHECKPOINT_PATH,
    challenger_checkpoint_path: Path = CHALLENGER_CHECKPOINT_PATH,
    max_epochs: int = MAX_EPOCHS,
) -> dict[str, dict[str, float] | str | bool]:
    """Train a challenger and promote it only when it beats the current best."""

    (
        train_dataloader,
        validation_dataloader,
        test_dataloader,
        id2label,
        label2id,
    ) = _build_dataloaders(data_dir)

    current_metrics: dict[str, float] | None = None
    if current_checkpoint_path.exists():
        current_model = _load_model(id2label, label2id, current_checkpoint_path)
        current_model.eval()
        current_metrics = run_epoch(current_model, test_dataloader)
        print(f"Current best test: {current_metrics}")

    model = _load_model(id2label, label2id, current_checkpoint_path)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=LEARNING_RATE,
    )

    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    for epoch in range(1, max_epochs + 1):
        print(f"Epoch {epoch}/{max_epochs}")
        train_metrics = run_epoch(model, train_dataloader, optimizer=optimizer)
        validation_metrics = run_epoch(model, validation_dataloader)
        print(f"Train: {train_metrics}")
        print(f"Validation: {validation_metrics}")

        if validation_metrics["loss"] < best_validation_loss:
            best_validation_loss = validation_metrics["loss"]
            epochs_without_improvement = 0
            challenger_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), challenger_checkpoint_path)
            print(f"Saved challenger model to {challenger_checkpoint_path}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            break

    model.load_state_dict(torch.load(challenger_checkpoint_path, map_location=DEVICE))
    challenger_metrics = run_epoch(model, test_dataloader)
    print(f"Challenger test: {challenger_metrics}")

    comparison = "no_current_checkpoint"
    promoted = False
    if current_metrics:
        if challenger_metrics["accuracy"] > current_metrics["accuracy"]:
            comparison = "challenger_better"
            shutil.copy2(challenger_checkpoint_path, current_checkpoint_path)
            promoted = True
            print(f"Promoted challenger to current best: {current_checkpoint_path}")
        elif challenger_metrics["accuracy"] == current_metrics["accuracy"]:
            comparison = "challenger_tied"
        else:
            comparison = "challenger_worse"
    else:
        shutil.copy2(challenger_checkpoint_path, current_checkpoint_path)
        promoted = True
        print(f"Saved first best model: {current_checkpoint_path}")

    return {
        "current_best": current_metrics or {},
        "challenger": challenger_metrics,
        "comparison": comparison,
        "promoted": promoted,
    }
