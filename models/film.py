"""Multi-level FiLM conditioning on the NoiseHead's noise parameters (Phase 5).

The trunk already receives noise information, but only once: ``sigma_map_normalized`` is
concatenated to the image as a second input channel and must then survive the DWT stem
and three levels of lossless resampling before it reaches the blocks that actually make
denoising decisions. FiLM (Perez et al., AAAI 2018) re-asserts it where those decisions
happen, as a per-channel affine modulation::

    gamma, beta = MLP(log sigma_g, log sigma_s)
    x <- x * (1 + gamma) + beta

Three properties are deliberate:

* **The sigmas are fed in log space.** ``sigma_g`` spans 1e-4 to 0.06 and ``sigma_s``
  0.14 to 0.19 across the measured per-image fits; a linear input would leave the
  additive term numerically invisible next to the multiplicative one.
* **Identity at initialisation.** Both head weights *and* biases are zeroed, so
  ``gamma = beta = 0`` and the modulation is exactly the identity on the first step.
  A FiLM layer that starts as a random affine map would perturb a backbone whose
  behaviour is already known, and would confound the ablation.
* **One shared trunk, one head per site.** The trunk is 2 -> 64 -> 32 with SimpleGate
  (the project's activation, Contract Part 5 fixes the activation set to "none
  (SimpleGate only)"), and each site gets its own ``Linear(32 -> 2C)``. That keeps the
  conditioner at ~47 k parameters rather than duplicating a trunk five times.

The modulation is applied once per trunk site, after that site's NAF stack, at
``L0, L1, L2, D1, D0``.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from models.blocks.simple_gate import SimpleGate

_SIGMA_FLOOR = 1e-6
"""Floor before the logarithm. The NoiseHead already clamps to ``sigma_min=1e-4``."""


class FiLMConditioner(nn.Module):
    """Maps ``(sigma_g, sigma_s)`` to per-site FiLM coefficients.

    Args:
        site_channels: Channel width of each conditioning site, in application order.
        hidden: Pre-SimpleGate width of the shared trunk; must be positive and even.

    Raises:
        ValueError: If ``site_channels`` is empty, any width is not a positive even
            number, or ``hidden`` is not a positive even number.
    """

    def __init__(self, site_channels: Sequence[int], hidden: int = 64) -> None:
        super().__init__()
        sites = tuple(int(c) for c in site_channels)
        if not sites:
            raise ValueError("site_channels must not be empty.")
        for channels in sites:
            if channels <= 0 or channels % 2 != 0:
                raise ValueError(
                    f"Each site width must be a positive even number, got {channels}."
                )
        if hidden <= 0 or hidden % 2 != 0:
            raise ValueError(f"hidden must be a positive even number, got {hidden}.")

        self.site_channels = sites
        self.hidden = hidden
        half = hidden // 2

        self.fc1 = nn.Linear(2, hidden)
        self.gate1 = SimpleGate()
        self.fc2 = nn.Linear(half, hidden)
        self.gate2 = SimpleGate()
        self.heads = nn.ModuleList(
            nn.Linear(half, 2 * channels) for channels in sites
        )
        for head in self.heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    @staticmethod
    def parameter_count(site_channels: Sequence[int], hidden: int = 64) -> int:
        """Analytic parameter count, used by the budget checks.

        Args:
            site_channels: Channel width of each conditioning site.
            hidden: Pre-SimpleGate trunk width.

        Returns:
            Number of parameters.
        """
        half = hidden // 2
        trunk = (2 * hidden + hidden) + (half * hidden + hidden)
        heads = sum(half * 2 * c + 2 * c for c in site_channels)
        return trunk + heads

    def forward(
        self, sigma_gauss: torch.Tensor, sigma_speckle: torch.Tensor
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Produce ``(gamma, beta)`` for every site.

        Args:
            sigma_gauss: Additive sigma ``(B, 1)``.
            sigma_speckle: Multiplicative sigma ``(B, 1)``.

        Returns:
            One ``(gamma, beta)`` pair per site, each shaped ``(B, C, 1, 1)``.
        """
        g = sigma_gauss.reshape(sigma_gauss.shape[0], 1)
        s = sigma_speckle.reshape(sigma_speckle.shape[0], 1)
        features = torch.log(
            torch.cat((g, s), dim=1).float().clamp_min(_SIGMA_FLOOR)
        )
        hidden = self.gate2(self.fc2(self.gate1(self.fc1(features))))

        out: list[tuple[torch.Tensor, torch.Tensor]] = []
        for head, channels in zip(self.heads, self.site_channels):
            coefficients = head(hidden)
            gamma, beta = coefficients[:, :channels], coefficients[:, channels:]
            out.append(
                (gamma.reshape(-1, channels, 1, 1), beta.reshape(-1, channels, 1, 1))
            )
        return out

    def extra_repr(self) -> str:
        return f"sites={self.site_channels}, hidden={self.hidden}"


def apply_film(
    x: torch.Tensor, coefficients: tuple[torch.Tensor, torch.Tensor] | None
) -> torch.Tensor:
    """Apply one FiLM modulation, or pass ``x`` through when unconditioned.

    Args:
        x: Feature map ``(B, C, H, W)``.
        coefficients: ``(gamma, beta)`` each ``(B, C, 1, 1)``, or ``None``.

    Returns:
        The modulated feature map.

    Raises:
        ValueError: If the coefficient channel count does not match ``x``.
    """
    if coefficients is None:
        return x
    gamma, beta = coefficients
    if gamma.shape[1] != x.shape[1]:
        raise ValueError(
            "FiLM coefficients carry " + str(gamma.shape[1])
            + " channels but the feature map has " + str(x.shape[1]) + "."
        )
    return x * (1.0 + gamma.to(x.dtype)) + beta.to(x.dtype)
