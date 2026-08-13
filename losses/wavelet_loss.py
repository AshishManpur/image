"""Band-weighted Haar wavelet loss (Contract Part 6, weight 0.10).

The reconstruction head predicts the four Haar sub-bands of the output directly
(Contract Part 1, stage 25), so supervising in the same basis is not an analogy — it
is supervision in the network's own output coordinates.

Contract Part 6::

    2-level Haar, L1 per band, band weights LL=0.25, LH=HL=1.0, HH=1.5

The weighting is the point. An unweighted L1 in the wavelet basis would be dominated by
LL, which carries almost all the energy and is also the easiest band to fit; the
detail bands are where restoration actually succeeds or fails. LL is therefore
down-weighted to 0.25 and the diagonal band up-weighted to 1.5.

Decomposition reuses :func:`models.wavelet.haar.haar_dwt` — the transform is defined
once, in one place, and is already verified invertible to 1e-6 by ``tests/test_haar.py``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from models.wavelet.haar import haar_dwt

BAND_NAMES: tuple[str, ...] = ("LL", "LH", "HL", "HH")
"""Channel order produced by :func:`haar_dwt`."""


class WaveletLoss(nn.Module):
    """Multi-level band-weighted L1 in the Haar basis.

    Args:
        levels: Number of decomposition levels; the contract fixes this at 2. Each
            level re-decomposes the previous level's LL band.
        band_weights: Weights for ``(LL, LH, HL, HH)``. The contract fixes these at
            ``(0.25, 1.0, 1.0, 1.5)``.
        level_decay: Multiplier applied to each successive coarser level. Defaults to
            1.0, i.e. every level contributes equally, which is what the contract
            specifies by not distinguishing them.

    Raises:
        ValueError: If ``levels`` is not positive or ``band_weights`` is not length 4.
    """

    def __init__(
        self,
        levels: int = 2,
        band_weights: tuple[float, float, float, float] = (0.25, 1.0, 1.0, 1.5),
        level_decay: float = 1.0,
    ) -> None:
        super().__init__()
        if levels <= 0:
            raise ValueError(f"levels must be positive, got {levels}.")
        if len(band_weights) != 4:
            raise ValueError(
                f"band_weights must have 4 entries (LL, LH, HL, HH), "
                f"got {len(band_weights)}."
            )
        if any(w < 0 for w in band_weights):
            raise ValueError(f"band_weights must be non-negative, got {band_weights}.")

        self.levels = levels
        self.level_decay = level_decay
        self.register_buffer(
            "band_weights", torch.tensor(band_weights, dtype=torch.float32),
            persistent=False,
        )

    def band_losses(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Per-band, per-level L1 discrepancies.

        Exposed so that a diagnostic run can see which band is failing rather than only
        the aggregate.

        Args:
            pred: Predicted images ``(B, C, H, W)``.
            target: Ground truth of the same shape.

        Returns:
            Mapping from ``"L{level}_{band}"`` to a scalar tensor.
        """
        losses: dict[str, torch.Tensor] = {}
        pred_ll, target_ll = pred, target
        for level in range(self.levels):
            pred_bands = haar_dwt(pred_ll)
            target_bands = haar_dwt(target_ll)
            channels = pred_ll.shape[1]
            for index, name in enumerate(BAND_NAMES):
                start = index * channels
                stop = start + channels
                losses[f"L{level + 1}_{name}"] = F.l1_loss(
                    pred_bands[:, start:stop], target_bands[:, start:stop]
                )
            pred_ll = pred_bands[:, :channels]
            target_ll = target_bands[:, :channels]
        return losses

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the weighted multi-level wavelet L1.

        Args:
            pred: Predicted images ``(B, C, H, W)``.
            target: Ground truth of the same shape.

        Returns:
            Scalar loss.

        Raises:
            ValueError: If the shapes differ or the image cannot be decomposed
                ``levels`` times.
        """
        # String concatenation, not f-strings: TorchScript (Part 9) cannot size
        # `tuple(tensor.shape)`.
        if pred.shape != target.shape:
            raise ValueError(
                "Shape mismatch: " + str(list(pred.shape))
                + " vs " + str(list(target.shape)) + "."
            )
        divisor = int(2**self.levels)
        if pred.shape[-1] % divisor != 0 or pred.shape[-2] % divisor != 0:
            raise ValueError(
                str(self.levels) + "-level Haar needs spatial dimensions divisible by "
                + str(divisor) + ", got " + str(pred.shape[-2]) + "x"
                + str(pred.shape[-1]) + "."
            )

        pred = pred.float()
        target = target.float()
        weights = self.band_weights

        total = pred.new_zeros(())
        weight_sum = 0.0
        pred_ll, target_ll = pred, target
        for level in range(self.levels):
            pred_bands = haar_dwt(pred_ll)
            target_bands = haar_dwt(target_ll)
            channels = pred_ll.shape[1]
            scale = self.level_decay**level
            for index in range(4):
                start = index * channels
                stop = start + channels
                band_l1 = F.l1_loss(
                    pred_bands[:, start:stop], target_bands[:, start:stop]
                )
                total = total + scale * weights[index] * band_l1
                weight_sum += scale * float(weights[index])
            pred_ll = pred_bands[:, :channels]
            target_ll = target_bands[:, :channels]

        # Normalise by the total weight so the term's magnitude is comparable to a
        # plain L1; otherwise the fixed 0.10 contract weight would implicitly depend on
        # the number of levels.
        return total / max(weight_sum, 1e-12)

    def extra_repr(self) -> str:
        return (
            f"levels={self.levels}, "
            f"band_weights={tuple(self.band_weights.tolist())}"
        )
