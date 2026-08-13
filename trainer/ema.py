"""Exponential moving average of model weights (Contract Part 5).

An EMA shadow copy is maintained alongside the live weights and updated every
optimisation step. Evaluation is reported for both: the EMA weights are typically
worth a small, free PSNR gain and are markedly less noisy epoch to epoch, which is
what the early-stopping rule keys on.

Buffers (none in SPARC-Net V1, but the class is written to be correct regardless) are
copied rather than averaged, since averaging running statistics is meaningless.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterator

import torch
from torch import nn

from utils.logging_utils import get_logger

_LOGGER = get_logger(__name__)


class ModelEma:
    """Maintains an exponentially-averaged copy of a model's parameters.

    Args:
        model: The live model to shadow.
        decay: EMA decay; the contract fixes this at 0.999.
        warmup_steps: Number of steps over which the effective decay ramps from 0 to
            ``decay``. This stops the average being dominated by the random
            initialisation during the first few hundred steps.

    Raises:
        ValueError: If ``decay`` is outside ``[0, 1)``.
    """

    def __init__(
        self, model: nn.Module, decay: float = 0.999, warmup_steps: int = 1000
    ) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"decay must be in [0, 1), got {decay}.")
        self.decay = decay
        self.warmup_steps = max(int(warmup_steps), 0)
        self.steps = 0
        self.module = deepcopy(model).eval()
        for param in self.module.parameters():
            param.requires_grad_(False)

    def effective_decay(self) -> float:
        """Decay for the current step, ramped in over ``warmup_steps``.

        The ramp is ``1 - 1/(1+step)`` clipped at ``decay``, and it is applied only
        while ``step < warmup_steps``. Early on this makes the shadow track the live
        weights almost exactly, so the average is not dominated by the random
        initialisation; it reaches the nominal decay at ``step = 1/(1-decay) - 1``
        (999 steps at decay 0.999), which is why ``warmup_steps`` defaults to 1000.

        Returns:
            The decay to use for this update.
        """
        if self.warmup_steps == 0 or self.steps >= self.warmup_steps:
            return self.decay
        ramp = 1.0 - 1.0 / (1.0 + self.steps)
        return min(self.decay, ramp)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Fold one optimisation step into the average.

        Args:
            model: The live model whose weights were just updated.
        """
        self.steps += 1
        decay = self.effective_decay()
        ema_params = dict(self.module.named_parameters())
        for name, param in model.named_parameters():
            shadow = ema_params[name]
            shadow.mul_(decay).add_(param.detach(), alpha=1.0 - decay)
        ema_buffers = dict(self.module.named_buffers())
        for name, buffer in model.named_buffers():
            ema_buffers[name].copy_(buffer)

    def state_dict(self) -> dict[str, Any]:
        """Serialise the shadow weights and the step counter."""
        return {
            "module": self.module.state_dict(),
            "steps": self.steps,
            "decay": self.decay,
            "warmup_steps": self.warmup_steps,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore from :meth:`state_dict`.

        Args:
            state: Previously serialised state.

        Raises:
            KeyError: If the payload is missing the shadow weights.
        """
        if "module" not in state:
            raise KeyError("EMA state is missing the 'module' entry.")
        self.module.load_state_dict(state["module"])
        self.steps = int(state.get("steps", 0))
        self.decay = float(state.get("decay", self.decay))
        self.warmup_steps = int(state.get("warmup_steps", self.warmup_steps))

    def parameters(self) -> Iterator[nn.Parameter]:
        """Iterate the shadow parameters."""
        return self.module.parameters()

    def to(self, device: torch.device | str) -> "ModelEma":
        """Move the shadow model to ``device``."""
        self.module.to(device)
        return self
