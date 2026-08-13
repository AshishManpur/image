"""Channel-wise LayerNorm for NCHW tensors.

Phase 3 specifies LayerNorm (not BatchNorm) everywhere: BatchNorm is harmful in
restoration networks and is incompatible with the Phase 1 finding that per-image
means span 0.016-0.959, which makes batch statistics meaningless here.

.. note::
   **Moments are computed in float32 (Phase 4.10.1).** This layer is hand-rolled from
   ``mean``/``var``/``rsqrt`` rather than ``F.layer_norm``, which costs it autocast's
   protection: CUDA autocast promotes ``layer_norm`` to fp32, but ``mean`` and ``var``
   are on no policy list and simply follow their input dtype. Under fp16 autocast the
   moments were therefore being computed in half precision, and a variance is a sum of
   squares — it saturates fp16 once the channel standard deviation reaches ~256.

   The failure that produces is silent rather than loud: ``rsqrt(inf)`` is ``0``, so
   the layer returns zeros and deletes the signal without raising. Measured error
   against a float64 reference is 4.9e-07 with fp32 moments versus 3.5e-03 with fp16
   moments — a 7000x improvement in the regime where fp16 works at all, and the
   difference between working and returning zeros in the regime where it does not.

   The arithmetic is unchanged; only the dtype it runs in is. See
   ``scripts/layernorm_analysis.py`` and ``reports/PHASE4_10_1_NUMERICAL_STABILITY.md``.
"""

from __future__ import annotations

import torch
from torch import nn


class LayerNorm2d(nn.Module):
    """LayerNorm over the channel dimension of an NCHW tensor.

    Implemented with explicit moment arithmetic rather than ``F.layer_norm`` on a
    permuted tensor: this avoids two permutes per call and exports to a plain
    sequence of ONNX reduction ops.

    Args:
        num_channels: Number of channels ``C``.
        eps: Numerical stabiliser added to the variance.
        affine: Whether to learn per-channel scale and shift.
    """

    def __init__(self, num_channels: int, eps: float = 1e-6, affine: bool = True) -> None:
        super().__init__()
        if num_channels <= 0:
            raise ValueError(f"num_channels must be positive, got {num_channels}.")
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self._custom_init = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise ``x`` over its channel dimension.

        Args:
            x: Tensor of shape ``(B, C, H, W)``.

        Returns:
            Tensor of the same shape.

        Raises:
            ValueError: If ``x`` is not 4-D or has the wrong channel count.
        """
        if x.dim() != 4:
            raise ValueError(
                "LayerNorm2d expects a 4-D tensor, got " + str(x.dim()) + " dims."
            )
        if x.shape[1] != self.num_channels:
            raise ValueError(
                "LayerNorm2d configured for " + str(self.num_channels)
                + " channels, got " + str(x.shape[1]) + "."
            )
        # float32 moments regardless of the surrounding autocast dtype. Casting the
        # input rather than only the statistics matters: `x - mean` must also be done
        # in fp32, since an fp16 input that has already reached the ceiling gives
        # `inf - inf` and produces NaN before the variance is ever consulted.
        x32 = x.float()
        mean = x32.mean(dim=1, keepdim=True)
        var = x32.var(dim=1, keepdim=True, unbiased=False)
        normalized = (x32 - mean) * torch.rsqrt(var + self.eps)
        if self.affine:
            normalized = normalized * self.weight.float() + self.bias.float()
        # Returned in float32, which is both what CUDA autocast's own `F.layer_norm`
        # returns and what this layer already returned in practice: the affine
        # parameters are fp32, so `normalized * self.weight` promoted every affine site
        # to fp32 even when the moments had been computed in fp16. Keeping fp32 here
        # therefore changes no dtype the audit observed, and gives the residual stream
        # the headroom that the fp16 path did not have.
        return normalized

    def extra_repr(self) -> str:
        return f"num_channels={self.num_channels}, eps={self.eps}, affine={self.affine}"
