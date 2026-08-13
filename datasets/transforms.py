"""Paired geometric augmentation (Contract Part 6).

**Only** geometric transforms are permitted. Phase 1 measured the LR-to-GT
photometric relationship as gain 1.001 +/- 0.006 and offset -0.0006 +/- 0.0034 —
exact to within noise. Any input-only brightness, contrast or gamma jitter would
therefore *introduce* a train/test mismatch that does not exist in the data and make
the mapping ill-posed. Extra noise or blur on top of the supplied LR is likewise
forbidden: the input already sits at a measured 7.1 dB median SNR.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from configs.sparc_config import DataConfig


@dataclass(frozen=True)
class GeometricOps:
    """The geometric operations chosen for one sample."""

    hflip: bool
    vflip: bool
    rot_k: int

    @property
    def is_identity(self) -> bool:
        return not self.hflip and not self.vflip and self.rot_k == 0


def sample_geometric_ops(
    config: DataConfig, generator: torch.Generator | None = None
) -> GeometricOps:
    """Draw an augmentation from the frozen probabilities.

    Args:
        config: Dataset configuration holding the probabilities.
        generator: Optional RNG for reproducibility.

    Returns:
        The sampled operations.
    """
    draws = torch.rand(3, generator=generator)
    rot_k = 0
    if draws[2].item() < config.rot90_prob:
        rot_k = int(torch.randint(1, 4, (1,), generator=generator).item())
    return GeometricOps(
        hflip=bool(draws[0].item() < config.hflip_prob),
        vflip=bool(draws[1].item() < config.vflip_prob),
        rot_k=rot_k,
    )


def apply_geometric(x: torch.Tensor, ops: GeometricOps) -> torch.Tensor:
    """Apply geometric operations to a single ``(C, H, W)`` tensor.

    Args:
        x: Tensor of shape ``(C, H, W)``.
        ops: Operations to apply.

    Returns:
        The transformed tensor, contiguous.

    Raises:
        ValueError: If ``x`` is not 3-D.
    """
    if x.dim() != 3:
        raise ValueError(f"apply_geometric expects a 3-D tensor, got {tuple(x.shape)}.")
    if ops.hflip:
        x = torch.flip(x, dims=(2,))
    if ops.vflip:
        x = torch.flip(x, dims=(1,))
    if ops.rot_k:
        x = torch.rot90(x, k=ops.rot_k, dims=(1, 2))
    return x.contiguous()


def apply_geometric_pair(
    lr: torch.Tensor, gt: torch.Tensor, ops: GeometricOps
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the *same* geometric operations to an LR/GT pair.

    Both tensors are square and the scale factor is 2, so every operation commutes
    with the resolution change and the pair stays aligned.

    Args:
        lr: Low-resolution tensor ``(C, h, w)``.
        gt: Ground-truth tensor ``(C, 2h, 2w)``.
        ops: Operations to apply to both.

    Returns:
        The transformed pair.
    """
    return apply_geometric(lr, ops), apply_geometric(gt, ops)
