"""Group-aware train/validation splitting (Contract Part 6).

Phase 1 measured that near-duplicate crops sit at adjacent IDs: 14.1 % of GT images
have a twin with 16x16-signature cosine > 0.99, and 98.9 % of those twins are within
+/-2 IDs. A uniform random split therefore places near-identical images on both sides
and inflates validation PSNR.

The split is by **contiguous ID blocks**: IDs are partitioned into blocks of
``block_size``, and every ``every_n``-th block is assigned to validation. With 3200
IDs, ``block_size=32`` and ``every_n=10`` this yields 2880 train / 320 validation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from utils.logging_utils import get_logger

_LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class SplitIndices:
    """Disjoint train and validation index arrays."""

    train: np.ndarray
    val: np.ndarray

    def __post_init__(self) -> None:
        overlap = np.intersect1d(self.train, self.val)
        if overlap.size:
            raise ValueError(f"Split leakage: {overlap.size} shared indices.")

    @property
    def n_train(self) -> int:
        return int(self.train.size)

    @property
    def n_val(self) -> int:
        return int(self.val.size)


def block_ids(num_samples: int, block_size: int) -> np.ndarray:
    """Map each sample index to its contiguous block id.

    Args:
        num_samples: Total number of samples.
        block_size: Number of consecutive IDs per block.

    Returns:
        Array of shape ``(num_samples,)`` with the block id of each index.

    Raises:
        ValueError: If ``block_size`` is not positive.
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}.")
    return np.arange(num_samples) // block_size


def group_aware_split(
    num_samples: int, block_size: int = 32, every_n: int = 10, offset: int = 0
) -> SplitIndices:
    """Split by contiguous ID blocks so that near-duplicate scenes never straddle.

    Args:
        num_samples: Total number of samples.
        block_size: Consecutive IDs per block.
        every_n: Assign every ``every_n``-th block to validation.
        offset: Index of the first validation block.

    Returns:
        Disjoint train/validation indices.

    Raises:
        ValueError: If ``every_n`` is smaller than 2 (which would leave no training data).
    """
    if every_n < 2:
        raise ValueError(f"every_n must be >= 2, got {every_n}.")
    blocks = block_ids(num_samples, block_size)
    is_val = ((blocks - offset) % every_n) == 0
    train = np.flatnonzero(~is_val)
    val = np.flatnonzero(is_val)
    _LOGGER.info(
        "Group-aware split: %d train / %d val over %d blocks of %d",
        train.size,
        val.size,
        int(blocks.max()) + 1,
        block_size,
    )
    return SplitIndices(train=train, val=val)


def verify_no_group_overlap(
    split: SplitIndices, block_size: int, num_samples: int
) -> None:
    """Assert that no block appears in both partitions.

    Args:
        split: The split to verify.
        block_size: Block size used to create it.
        num_samples: Total number of samples.

    Raises:
        ValueError: If any block id is present in both train and validation.
    """
    blocks = block_ids(num_samples, block_size)
    shared = np.intersect1d(blocks[split.train], blocks[split.val])
    if shared.size:
        raise ValueError(
            f"Group leakage: {shared.size} block(s) appear in both partitions: "
            f"{shared[:10].tolist()}"
        )
