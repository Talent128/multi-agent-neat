"""Project-wide generic helpers."""

import random

import numpy as np
import torch


def seed_everything(seed: int):
    """Seed Python, NumPy, and PyTorch RNGs."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
