from __future__ import annotations

import random

import torch


SEED = 42
BATCH_SIZE = 8
LEARNING_RATE = 5e-4
MAX_EPOCHS = 100
PATIENCE = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything() -> None:
    """Make dataset splits and training as repeatable as PyTorch allows."""

    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
