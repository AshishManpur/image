"""Auxiliary noise-estimation loss (Contract Part 6, weight 0.02).

The noise head (Phase 4.9) predicts the two-parameter degradation statistics blindly
from the observed LR image. Nothing in the reconstruction objective supervises it
directly, so without this term it would only ever be trained through whatever gradient
the trunk happens to send back — a weak and indirect signal for a branch whose whole
purpose is to be *correct* about the noise level.

Contract Part 2.9 defines the target::

    Per-image closed-form least squares of r^2 = a + c*I_hat^2 on (D(GT), y);
    loss on log sigma_hat

and Part 6 fixes the comparison as an L1 on ``log sigma_hat`` against the analytic
sigma derived from the ground truth.

**Why log space.** Phase 1 measured sigma varying 8.5x across the dataset. A linear L1
would be dominated by the noisiest images and would effectively ignore the clean ones;
in log space the loss is on *relative* error, so a 10 % misestimate costs the same
whether the image is clean or noisy. It also matches how the head parameterises its
output — softplus of an unbounded pre-activation.

The target is computed under :func:`torch.no_grad` and detached: it is data, not a
differentiable function of the prediction.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from datasets.degradation import (
    analytic_sigma_map,
    fit_noise_parameters,
    forward_operator,
)


class NoiseAuxLoss(nn.Module):
    """L1 between predicted and analytic log-sigma maps.

    Args:
        blur_sigma: Blur applied by the forward operator when reconstructing
            ``D(GT)``. The contract samples the real blur from ``U(0.3, 0.5)``; the
            midpoint is used because the per-sample value is not recoverable at loss
            time.
        log_eps: Floor applied before taking logarithms, guarding ``log(0)``.
        sigma_min: Lower clamp on the analytic target, matching the head's own range.
        sigma_max: Upper clamp on the analytic target.

    Raises:
        ValueError: If ``log_eps`` is not positive or the sigma bounds are invalid.
    """

    def __init__(
        self,
        blur_sigma: float = 0.4,
        log_eps: float = 1e-6,
        sigma_min: float = 1e-4,
        sigma_max: float = 2.0,
    ) -> None:
        super().__init__()
        if log_eps <= 0.0:
            raise ValueError(f"log_eps must be positive, got {log_eps}.")
        if not 0.0 < sigma_min < sigma_max:
            raise ValueError(
                f"Require 0 < sigma_min < sigma_max, got {sigma_min}, {sigma_max}."
            )
        self.blur_sigma = blur_sigma
        self.log_eps = log_eps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    @torch.no_grad()
    def analytic_target(
        self, noisy_lr: torch.Tensor, gt: torch.Tensor
    ) -> torch.Tensor:
        """Build the supervision target from the ground truth.

        Args:
            noisy_lr: Observed LR image ``(B, 1, h, w)``.
            gt: Ground truth ``(B, 1, 2h, 2w)``.

        Returns:
            Analytic sigma map ``(B, 1, h, w)``, detached.

        Raises:
            ValueError: If the reconstructed ``D(GT)`` does not match ``noisy_lr``.
        """
        clean_lr = forward_operator(gt.float(), blur_sigma=self.blur_sigma)
        if clean_lr.shape != noisy_lr.shape:
            raise ValueError(
                f"D(GT) has shape {tuple(clean_lr.shape)} but the observed LR is "
                f"{tuple(noisy_lr.shape)}; they must match."
            )
        sigma_gauss, sigma_speckle = fit_noise_parameters(noisy_lr.float(), clean_lr)
        target = analytic_sigma_map(clean_lr, sigma_gauss, sigma_speckle)
        return target.clamp(self.sigma_min, self.sigma_max).detach()

    def forward(
        self,
        sigma_hat: torch.Tensor,
        noisy_lr: torch.Tensor,
        gt: torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the L1 log-sigma discrepancy.

        Args:
            sigma_hat: Predicted sigma map ``(B, 1, h, w)`` in image units.
            noisy_lr: Observed LR image ``(B, 1, h, w)``.
            gt: Ground truth ``(B, 1, 2h, 2w)``.
            target: Precomputed analytic target; derived from ``gt`` when ``None``.

        Returns:
            Scalar loss.

        Raises:
            ValueError: If the predicted map and the target disagree in shape.
        """
        if target is None:
            target = self.analytic_target(noisy_lr, gt)
        if sigma_hat.shape != target.shape:
            raise ValueError(
                f"Shape mismatch: prediction {tuple(sigma_hat.shape)} vs target "
                f"{tuple(target.shape)}."
            )
        log_pred = torch.log(sigma_hat.float().clamp_min(self.log_eps))
        log_target = torch.log(target.float().clamp_min(self.log_eps))
        return F.l1_loss(log_pred, log_target)

    def extra_repr(self) -> str:
        return f"blur_sigma={self.blur_sigma}, log_eps={self.log_eps}"
