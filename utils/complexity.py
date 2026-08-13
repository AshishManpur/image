"""Parameter and FLOP accounting for SPARC-Net.

FLOPs are measured with :class:`torch.utils.flop_counter.FlopCounterMode`, which
instruments the real dispatched operators (convolution, matmul, scaled dot-product
attention). This is a measurement, not an analytic estimate, and is what Phase 4
compares against the Phase 3 analytic budget.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import nn

from utils.logging_utils import get_logger

_LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class ComplexityReport:
    """Measured complexity of a module for one forward pass."""

    params_total: int
    params_trainable: int
    flops: int
    macs: int
    per_module_flops: Mapping[str, int]

    @property
    def params_millions(self) -> float:
        return self.params_total / 1e6

    @property
    def gmacs(self) -> float:
        return self.macs / 1e9

    @property
    def gflops(self) -> float:
        return self.flops / 1e9

    def summary(self) -> str:
        return (
            f"params={self.params_millions:.3f}M "
            f"(trainable {self.params_trainable / 1e6:.3f}M) | "
            f"{self.gmacs:.3f} GMAC | {self.gflops:.3f} GFLOP"
        )


def count_parameters(module: nn.Module) -> tuple[int, int]:
    """Return ``(total, trainable)`` parameter counts."""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def parameter_table(module: nn.Module, depth: int = 1) -> dict[str, int]:
    """Parameter count per direct child (or deeper, controlled by ``depth``)."""
    table: dict[str, int] = {}
    for name, child in module.named_children():
        count = sum(p.numel() for p in child.parameters())
        table[name] = count
        if depth > 1:
            for sub_name, sub_count in parameter_table(child, depth - 1).items():
                table[f"{name}.{sub_name}"] = sub_count
    return table


def measure_complexity(
    module: nn.Module,
    inputs: Sequence[torch.Tensor] | torch.Tensor,
    forward_fn: Callable[..., Any] | None = None,
    module_depth: int = 2,
) -> ComplexityReport:
    """Measure parameters and FLOPs for a single forward pass.

    Args:
        module: Module to profile. Placed in ``eval`` mode for the measurement and
            restored to its previous mode afterwards.
        inputs: A tensor or sequence of tensors passed positionally to the module.
        forward_fn: Optional callable invoked instead of ``module(*inputs)``.
        module_depth: Depth of the per-module FLOP breakdown.

    Returns:
        A :class:`ComplexityReport`.

    Raises:
        RuntimeError: If the forward pass fails.
    """
    if isinstance(inputs, torch.Tensor):
        inputs = (inputs,)
    total, trainable = count_parameters(module)

    was_training = module.training
    module.eval()
    try:
        from torch.utils.flop_counter import FlopCounterMode

        counter = FlopCounterMode(display=False, depth=module_depth)
        ctx: Any = counter
    except ImportError:  # pragma: no cover - torch always ships this in 2.x
        _LOGGER.warning("FlopCounterMode unavailable; FLOPs will be reported as 0.")
        counter = None
        ctx = nullcontext()

    try:
        with torch.no_grad(), ctx:
            if forward_fn is not None:
                forward_fn(*inputs)
            else:
                module(*inputs)
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        raise RuntimeError(f"Forward pass failed during FLOP measurement: {exc}") from exc
    finally:
        module.train(was_training)

    if counter is None:
        flops = 0
        per_module: dict[str, int] = {}
    else:
        flops = counter.get_total_flops()
        per_module = {
            str(name): int(sum(values.values()))
            for name, values in counter.get_flop_counts().items()
        }

    return ComplexityReport(
        params_total=total,
        params_trainable=trainable,
        flops=int(flops),
        macs=int(flops) // 2,
        per_module_flops=per_module,
    )
