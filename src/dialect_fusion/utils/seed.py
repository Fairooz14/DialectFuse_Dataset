
from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Seed python, numpy and torch (CPU + all CUDA devices)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device :", device)
    if device.type == "cuda":
        print("GPU    :", torch.cuda.get_device_name(0))
        print(
            "VRAM   :",
            round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
            "GB",
        )
    return device
