"""Weight initialisation utilities for SPARC-Net.

The network follows the NAFNet convention: every residual branch is scaled by a
learnable parameter initialised near zero, so the whole network is close to an
identity map at initialisation. Convolutions therefore only need a conservative
fan-in initialisation.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from utils.logging_utils import get_logger

_LOGGER = get_logger(__name__)


def trunc_normal_(tensor: torch.Tensor, std: float = 0.02) -> torch.Tensor:
    """In-place truncated normal initialisation clipped at +/- 2 std."""
    with torch.no_grad():
        tensor.normal_(0.0, std).clamp_(-2 * std, 2 * std)
    return tensor


def init_conv(module: nn.Conv2d, zero_bias: bool = True) -> None:
    """Initialise a convolution with Kaiming-uniform weights and zero bias."""
    nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5), nonlinearity="linear")
    if module.bias is not None and zero_bias:
        nn.init.zeros_(module.bias)


def default_init(module: nn.Module) -> None:
    """Apply the project-default initialisation to a module.

    Intended for use with :meth:`torch.nn.Module.apply`. Modules that manage their
    own initialisation must expose a ``_custom_init`` attribute set to ``True``.
    """
    if getattr(module, "_custom_init", False):
        return
    if isinstance(module, nn.Conv2d):
        init_conv(module)
    elif isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def icnr_(tensor: torch.Tensor, upscale_factor: int = 2) -> torch.Tensor:
    """ICNR initialisation for sub-pixel convolution weights.

    Initialises the weight so that the ``upscale_factor**2`` output channels that
    feed one spatial block start identical, which removes checkerboard artefacts.

    Args:
        tensor: Weight of shape ``(out_channels, in_channels, kH, kW)`` where
            ``out_channels`` is divisible by ``upscale_factor ** 2``.
        upscale_factor: Sub-pixel upscaling factor.

    Returns:
        The same tensor, modified in place.

    Raises:
        ValueError: If ``out_channels`` is not divisible by ``upscale_factor ** 2``.
    """
    out_channels = tensor.shape[0]
    block = upscale_factor**2
    if out_channels % block != 0:
        raise ValueError(
            f"ICNR needs out_channels divisible by {block}, got {out_channels}."
        )
    sub = torch.zeros(
        (out_channels // block, *tensor.shape[1:]),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    nn.init.kaiming_uniform_(sub, a=math.sqrt(5), nonlinearity="linear")
    with torch.no_grad():
        tensor.copy_(sub.repeat_interleave(block, dim=0))
    return tensor
