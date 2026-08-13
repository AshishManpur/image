"""Memory-mapped paired dataset for SPARC-Base V1.0 (Contract Part 8, step 2).

The raw dataset is 6800 individual ``.npy`` files. On Windows/NTFS the per-file open
cost dominates reading a 64 KiB array, so the training set is packed into three
memory-mapped arrays. Reads then become page faults against the OS cache: the entire
training set is ~0.5 GB in float16 and fits in RAM.

The packer also removes the 6800 AppleDouble ``._*.npy`` decoys under ``__MACOSX``,
which carry a ``.npy`` extension, are 163 bytes each, and raise on ``np.load``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from configs.sparc_config import DataConfig
from datasets.degradation import sample_degradation_params, synthesize_lr
from datasets.transforms import apply_geometric_pair, sample_geometric_ops
from utils.logging_utils import get_logger

_LOGGER = get_logger(__name__)

MANIFEST_NAME = "manifest.json"


def load_manifest(packed_root: Path) -> dict[str, Any]:
    """Load the packing manifest.

    Args:
        packed_root: Directory containing ``manifest.json``.

    Returns:
        The manifest dictionary.

    Raises:
        FileNotFoundError: If the manifest is missing.
    """
    path = Path(packed_root) / MANIFEST_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"No manifest at {path}. Run `python scripts/pack_dataset.py` first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


class PackedRestorationDataset(Dataset[dict[str, torch.Tensor]]):
    """Paired LR/GT dataset backed by memory-mapped arrays.

    Args:
        lr_path: Path to the packed LR array ``(N, 128, 128)``.
        gt_path: Path to the packed GT array ``(N, 256, 256)``, or ``None`` for test.
        indices: Subset of sample indices to expose.
        config: Dataset configuration.
        training: Enables augmentation and LR re-synthesis.
        seed: Base seed; each sample derives a deterministic per-item generator.

    Raises:
        FileNotFoundError: If a packed array is missing.
        ValueError: If the arrays disagree on sample count or have wrong shapes.
    """

    def __init__(
        self,
        lr_path: Path,
        gt_path: Path | None,
        indices: np.ndarray | None = None,
        config: DataConfig | None = None,
        training: bool = False,
        seed: int = 1337,
    ) -> None:
        self.config = config or DataConfig()
        self.training = training
        self.seed = seed

        self.lr_path = Path(lr_path)
        self.gt_path = Path(gt_path) if gt_path is not None else None
        for path in (self.lr_path, self.gt_path):
            if path is not None and not path.exists():
                raise FileNotFoundError(f"Packed array not found: {path}")

        self._lr: np.ndarray | None = None
        self._gt: np.ndarray | None = None

        lr_head = np.load(self.lr_path, mmap_mode="r")
        n_samples = lr_head.shape[0]
        if lr_head.shape[1:] != (self.config.lr_size, self.config.lr_size):
            raise ValueError(
                f"LR array has shape {lr_head.shape}, expected "
                f"(N, {self.config.lr_size}, {self.config.lr_size})."
            )
        if self.gt_path is not None:
            gt_head = np.load(self.gt_path, mmap_mode="r")
            if gt_head.shape[0] != n_samples:
                raise ValueError(
                    f"LR/GT count mismatch: {n_samples} vs {gt_head.shape[0]}."
                )
            if gt_head.shape[1:] != (self.config.gt_size, self.config.gt_size):
                raise ValueError(
                    f"GT array has shape {gt_head.shape}, expected "
                    f"(N, {self.config.gt_size}, {self.config.gt_size})."
                )

        self.indices = (
            np.arange(n_samples) if indices is None else np.asarray(indices, dtype=np.int64)
        )
        if self.indices.size and int(self.indices.max()) >= n_samples:
            raise ValueError(
                f"Index {int(self.indices.max())} out of range for {n_samples} samples."
            )

    # ------------------------------------------------------------------ mmap lazy
    def _arrays(self) -> tuple[np.ndarray, np.ndarray | None]:
        """Open memmaps lazily so that DataLoader workers each get their own handle."""
        if self._lr is None:
            self._lr = np.load(self.lr_path, mmap_mode="r")
            if self.gt_path is not None:
                self._gt = np.load(self.gt_path, mmap_mode="r")
        return self._lr, self._gt

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, position: int) -> dict[str, torch.Tensor]:
        """Return one training sample.

        Args:
            position: Index into ``self.indices``.

        Returns:
            Dict with ``lr`` ``(1,128,128)``, ``index``, and — when ground truth is
            available — ``gt`` ``(1,256,256)`` and ``resynth`` (0/1 flag).
        """
        index = int(self.indices[position])
        lr_array, gt_array = self._arrays()

        lr = torch.from_numpy(np.asarray(lr_array[index], dtype=np.float32)).unsqueeze(0)
        sample: dict[str, torch.Tensor] = {"index": torch.tensor(index, dtype=torch.long)}

        if gt_array is None:
            sample["lr"] = lr
            return sample

        gt = torch.from_numpy(np.asarray(gt_array[index], dtype=np.float32)).unsqueeze(0)
        resynth = False

        if self.training:
            generator = torch.Generator().manual_seed(
                (self.seed * 1_000_003 + index) % (2**63 - 1)
            )
            if torch.rand(1, generator=generator).item() < self.config.resynth_prob:
                params = sample_degradation_params(self.config, generator)
                lr = synthesize_lr(
                    gt.unsqueeze(0), params, generator, self.config.noise_at_lr
                ).squeeze(0)
                resynth = True
            ops = sample_geometric_ops(self.config, generator)
            lr, gt = apply_geometric_pair(lr, gt, ops)

        sample["lr"] = lr
        sample["gt"] = gt
        sample["resynth"] = torch.tensor(int(resynth), dtype=torch.long)
        return sample


def build_datasets(
    config: DataConfig,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    seed: int = 1337,
) -> tuple[PackedRestorationDataset, PackedRestorationDataset]:
    """Construct the training and validation datasets from packed arrays.

    Args:
        config: Dataset configuration.
        train_indices: Indices for the training partition.
        val_indices: Indices for the validation partition.
        seed: Base seed for per-sample augmentation RNGs.

    Returns:
        ``(train_dataset, val_dataset)``. Validation has augmentation disabled.
    """
    root = Path(config.packed_root)
    lr_path = root / "train_lr.npy"
    gt_path = root / "train_gt.npy"
    train = PackedRestorationDataset(
        lr_path, gt_path, train_indices, config, training=True, seed=seed
    )
    val = PackedRestorationDataset(
        lr_path, gt_path, val_indices, config, training=False, seed=seed
    )
    _LOGGER.info("Datasets built: %d train / %d val", len(train), len(val))
    return train, val


def build_test_dataset(config: DataConfig) -> PackedRestorationDataset:
    """Construct the unlabelled test dataset."""
    return PackedRestorationDataset(
        Path(config.packed_root) / "test_lr.npy", None, None, config, training=False
    )
