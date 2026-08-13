from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch


def load_npy(path: Path) -> np.ndarray:
    return np.load(path, allow_pickle=False)


def save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def to_tensor(array: np.ndarray) -> torch.Tensor:
    if array.ndim == 2:
        array = array[None, ...]
    elif array.ndim == 3 and array.shape[-1] in (1, 3, 4):
        array = np.moveaxis(array, -1, 0)
    return torch.from_numpy(np.ascontiguousarray(array)).float()


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().numpy()
    if array.ndim == 3 and array.shape[0] in (1, 3, 4):
        return np.moveaxis(array, 0, -1) if array.shape[0] != 1 else array[0]
    return array


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

