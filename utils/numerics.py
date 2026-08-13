"""Numerical-stability instrumentation (Phase 4.10.1).

The Phase 4.10 GPU shakedown diverged to a non-finite loss at step 423 and never
recovered. Nothing in the pipeline could say *which tensor* went bad first, so the
divergence could only be guessed at. This module supplies the missing evidence: cheap
tensor statistics, a forward-hook tracer that records the dtype and range of every
module output, and helpers that localise the first non-finite tensor in a forward pass.

Everything here is diagnostic. The tracer is off unless explicitly attached, and
:func:`tensor_stats` is only called from paths already gated behind a debug flag,
because ``.item()`` on a CUDA tensor forces a device synchronisation and would
otherwise dominate step time.

See ``reports/PHASE4_10_1_NUMERICAL_STABILITY.md`` for the findings this produced.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterator

import torch
from torch import nn

__all__ = [
    "FP16_MAX",
    "ModuleTracer",
    "TensorStats",
    "detect_anomaly",
    "first_nonfinite",
    "fp16_headroom",
    "fp32_island",
    "is_finite",
    "tensor_stats",
]

FP16_MAX: float = float(torch.finfo(torch.float16).max)
"""65504.0 — the largest finite fp16 value.

The binding constraint on this architecture under fp16 autocast: ``SimpleGate``
multiplies two channel halves and Simplified Channel Attention multiplies the result
again, so a NAF block's spatial branch grows magnitude to roughly the fourth power of
its convolution output before anything reduces it.
"""


@dataclass(frozen=True, slots=True)
class TensorStats:
    """Range and health summary of one tensor.

    Attributes:
        dtype: String form of the tensor dtype, e.g. ``"torch.float16"``.
        shape: Tensor shape as a tuple.
        min: Minimum value.
        max: Maximum value.
        absmax: Maximum absolute value — the quantity that overflows fp16.
        mean: Arithmetic mean over finite elements, computed in float32.
        std: Population standard deviation over finite elements, in float32.
        has_nan: Whether any element is NaN.
        has_inf: Whether any element is +/-Inf.
    """

    dtype: str
    shape: tuple[int, ...]
    min: float
    max: float
    absmax: float
    mean: float
    std: float
    has_nan: bool
    has_inf: bool

    @property
    def finite(self) -> bool:
        """Whether the tensor is entirely finite."""
        return not (self.has_nan or self.has_inf)

    @property
    def fp16_headroom(self) -> float:
        """Ratio of the fp16 ceiling to ``absmax``; below 1.0 means overflow."""
        return FP16_MAX / self.absmax if self.absmax > 0.0 else float("inf")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping."""
        return asdict(self)

    def __str__(self) -> str:
        flag = "" if self.finite else ("  <NaN>" if self.has_nan else "  <Inf>")
        return (
            f"{self.dtype:<15} absmax={self.absmax:>11.4g} "
            f"mean={self.mean:>10.3g} std={self.std:>10.3g}{flag}"
        )


def tensor_stats(x: torch.Tensor) -> TensorStats:
    """Summarise a tensor's range and finiteness.

    Statistics are computed in float32 so that summarising an fp16 tensor cannot
    itself overflow: ``x.std()`` on an fp16 tensor near the ceiling returns ``inf`` and
    would report a merely-large tensor as a broken one. Mean and standard deviation are
    taken over the finite subset, so a single Inf does not hide how far the rest of the
    tensor had already drifted.

    Args:
        x: Tensor to summarise. Detached internally; may require grad.

    Returns:
        The summary.
    """
    x32 = x.detach().float()
    has_nonfinite = bool((~torch.isfinite(x32)).any())
    finite_vals = x32[torch.isfinite(x32)] if has_nonfinite else x32.reshape(-1)
    if finite_vals.numel() == 0:
        mean = std = float("nan")
    else:
        mean = float(finite_vals.mean())
        std = float(finite_vals.std(unbiased=False)) if finite_vals.numel() > 1 else 0.0
    return TensorStats(
        dtype=str(x.dtype),
        shape=tuple(x.shape),
        min=float(x32.min()),
        max=float(x32.max()),
        absmax=float(x32.abs().max()),
        mean=mean,
        std=std,
        has_nan=bool(torch.isnan(x32).any()),
        has_inf=bool(torch.isinf(x32).any()),
    )


def is_finite(x: torch.Tensor) -> bool:
    """Whether every element of ``x`` is finite.

    Args:
        x: Tensor to test.

    Returns:
        ``True`` if no element is NaN or Inf.
    """
    return bool(torch.isfinite(x).all())


def fp16_headroom(x: torch.Tensor) -> float:
    """How many times ``x`` could grow before overflowing fp16.

    Args:
        x: Tensor to measure.

    Returns:
        ``65504 / absmax(x)``, or infinity for an all-zero tensor. A value below 1.0
        means the tensor already contains non-representable magnitudes.
    """
    absmax = float(x.detach().float().abs().max())
    return FP16_MAX / absmax if absmax > 0.0 else float("inf")


class ModuleTracer:
    """Records the output dtype and range of every module in a forward pass.

    Answers the two questions the Phase 4.10 logs could not: which modules still run in
    fp16 under autocast, and which tensor becomes non-finite first.

    Leaf modules are traced by default. Container modules are skipped because their
    output is by definition some leaf's output, and tracing both doubles the record
    count without adding information.

    Args:
        model: Model to instrument.
        leaves_only: Trace only modules with no children.
        include: Extra module type names to trace even when they have children.

    Example:
        >>> tracer = ModuleTracer(model)
        >>> with tracer:
        ...     model(x)
        >>> tracer.first_bad()
    """

    def __init__(
        self,
        model: nn.Module,
        leaves_only: bool = True,
        include: tuple[str, ...] = (),
    ) -> None:
        self.model = model
        self.leaves_only = leaves_only
        self.include = include
        self.records: list[dict[str, Any]] = []
        self._handles: list[Any] = []

    def _should_trace(self, module: nn.Module) -> bool:
        if type(module).__name__ in self.include:
            return True
        if not self.leaves_only:
            return True
        return next(module.children(), None) is None

    def _hook(self, name: str) -> Callable[..., None]:
        def hook(module: nn.Module, inputs: Any, output: Any) -> None:
            tensor = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(tensor):
                return
            in_dtype = (
                str(inputs[0].dtype)
                if inputs and torch.is_tensor(inputs[0])
                else "n/a"
            )
            self.records.append(
                {
                    "index": len(self.records),
                    "name": name,
                    "class": type(module).__name__,
                    "input_dtype": in_dtype,
                    **tensor_stats(tensor).to_dict(),
                }
            )

        return hook

    def attach(self) -> "ModuleTracer":
        """Register hooks on every traced module.

        Returns:
            ``self``, for chaining.
        """
        self.detach()
        for name, module in self.model.named_modules():
            if name and self._should_trace(module):
                self._handles.append(module.register_forward_hook(self._hook(name)))
        return self

    def detach(self) -> None:
        """Remove every registered hook."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def clear(self) -> None:
        """Drop recorded statistics, keeping the hooks attached."""
        self.records.clear()

    def __enter__(self) -> "ModuleTracer":
        return self.attach()

    def __exit__(self, *exc: Any) -> None:
        self.detach()

    # ----------------------------------------------------------------- queries
    def first_bad(self) -> dict[str, Any] | None:
        """The first traced module whose output was non-finite.

        Returns:
            The record, or ``None`` if every output was finite.
        """
        for record in self.records:
            if record["has_nan"] or record["has_inf"]:
                return record
        return None

    def bad(self) -> list[dict[str, Any]]:
        """Every record whose output was non-finite."""
        return [r for r in self.records if r["has_nan"] or r["has_inf"]]

    def by_dtype(self, dtype: str) -> list[dict[str, Any]]:
        """Records whose output had the given dtype string.

        Args:
            dtype: Dtype string to match, e.g. ``"torch.float16"``.

        Returns:
            Matching records, in execution order.
        """
        return [r for r in self.records if r["dtype"] == dtype]

    def by_input_dtype(self, dtype: str) -> list[dict[str, Any]]:
        """Records whose *input* had the given dtype string.

        This is the one that matters for hand-rolled normalisation layers: the output
        can be promoted to fp32 by a fp32 affine parameter while the moments that
        produced it were still computed in fp16.

        Args:
            dtype: Dtype string to match.

        Returns:
            Matching records, in execution order.
        """
        return [r for r in self.records if r["input_dtype"] == dtype]

    def worst_headroom(self, count: int = 10) -> list[dict[str, Any]]:
        """The records closest to the fp16 ceiling, largest ``absmax`` first.

        Args:
            count: How many records to return.

        Returns:
            Up to ``count`` records sorted by descending ``absmax``, non-finite first.
        """

        def key(record: dict[str, Any]) -> float:
            absmax = record["absmax"]
            return float("inf") if absmax != absmax else absmax  # NaN sorts first

        return sorted(self.records, key=key, reverse=True)[:count]


def first_nonfinite(
    model: nn.Module,
    *args: Any,
    leaves_only: bool = True,
    **kwargs: Any,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Run a forward pass and localise the first non-finite module output.

    Args:
        model: Model to run.
        *args: Positional arguments forwarded to ``model``.
        leaves_only: Passed to :class:`ModuleTracer`.
        **kwargs: Keyword arguments forwarded to ``model``.

    Returns:
        ``(first_bad_record_or_None, all_records)``.
    """
    tracer = ModuleTracer(model, leaves_only=leaves_only)
    with tracer, torch.no_grad():
        model(*args, **kwargs)
    return tracer.first_bad(), list(tracer.records)


@contextmanager
def fp32_island(device_type: str) -> Iterator[None]:
    """Suspend autocast for a block that must genuinely run in float32.

    An explicit ``.float()`` on a tensor does **not** keep it in float32 under
    autocast: ``conv2d``, ``linear``, ``matmul`` and ``einsum`` re-cast their arguments
    to the reduced dtype before the kernel runs, whatever the caller passed. Several
    modules in this codebase carried ``.float()`` calls and docstrings promising float32
    that were, measurably, not being honoured — see ``scripts/amp_audit.py``.

    Wrapping the sensitive region in this context manager is the only way to make that
    promise real. It disables autocast for one region rather than globally, so the
    trunk keeps its mixed-precision speedup.

    Callers must still cast their inputs: suspending autocast stops *new* demotions but
    does not upcast a tensor that is already fp16.

    Args:
        device_type: Device type string to suspend autocast for, e.g. ``"cuda"`` or
            ``"cpu"``. Autocast state is tracked per device type.

    Yields:
        Nothing.
    """
    with torch.amp.autocast(device_type, enabled=False):
        yield


@contextmanager
def detect_anomaly(enabled: bool = True) -> Iterator[None]:
    """Enable ``torch.autograd.detect_anomaly`` only when asked.

    A plain ``with torch.autograd.detect_anomaly():`` cannot be made conditional
    without duplicating the block it wraps; this keeps ``train_epoch`` single-pathed.

    Anomaly detection roughly doubles backward-pass cost and is never enabled by
    default — see the ``--detect-anomaly`` flag on ``train.py``.

    Args:
        enabled: When ``False`` this is a no-op context manager.

    Yields:
        Nothing.
    """
    if not enabled:
        yield
        return
    with torch.autograd.detect_anomaly(check_nan=True):
        yield
