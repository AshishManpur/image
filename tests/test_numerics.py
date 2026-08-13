"""Numerical-stability tests (Phase 4.10.1).

Covers the instrumentation in :mod:`utils.numerics`, the float32 moments in
:class:`models.blocks.layer_norm.LayerNorm2d`, and the float32 island in
:class:`losses.composite_loss.CompositeLoss`.

The regression these guard against is the Phase 4.10 shakedown: a fp16 activation
overflow produced a non-finite loss at step 423, and every subsequent batch failed
identically because the skip path never updates the weights.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from losses.composite_loss import CompositeLoss
from models.blocks.layer_norm import LayerNorm2d
from models.sparc_net import SPARCNet
from configs.sparc_config import build_sparc_config
from utils.numerics import (
    FP16_MAX,
    ModuleTracer,
    detect_anomaly,
    first_nonfinite,
    fp16_headroom,
    fp32_island,
    is_finite,
    tensor_stats,
)


# ------------------------------------------------------------------ tensor stats
def test_tensor_stats_reports_range_and_health() -> None:
    x = torch.tensor([[-2.0, 0.0, 3.0]])
    stats = tensor_stats(x)
    assert stats.min == -2.0
    assert stats.max == 3.0
    assert stats.absmax == 3.0
    assert stats.finite
    assert not stats.has_nan and not stats.has_inf


def test_tensor_stats_flags_nan_and_inf() -> None:
    assert tensor_stats(torch.tensor([1.0, float("nan")])).has_nan
    assert tensor_stats(torch.tensor([1.0, float("inf")])).has_inf
    assert not tensor_stats(torch.tensor([1.0, float("inf")])).finite


def test_tensor_stats_summarises_fp16_without_overflowing() -> None:
    """Statistics are computed in fp32 so a near-ceiling fp16 tensor reads as healthy.

    ``x.std()`` on an fp16 tensor at 6e4 overflows to inf and would report a merely
    large tensor as a broken one.
    """
    x = torch.full((256,), 60000.0, dtype=torch.float16)
    x[0] = -60000.0
    stats = tensor_stats(x)
    assert stats.finite
    assert math.isfinite(stats.std)
    assert stats.absmax == pytest.approx(60000.0)


def test_tensor_stats_mean_ignores_nonfinite_elements() -> None:
    """One Inf must not hide how far the rest of the tensor had drifted."""
    x = torch.tensor([1.0, 2.0, 3.0, float("inf")])
    stats = tensor_stats(x)
    assert stats.has_inf
    assert stats.mean == pytest.approx(2.0)


def test_fp16_headroom_matches_the_ceiling() -> None:
    assert fp16_headroom(torch.tensor([FP16_MAX])) == pytest.approx(1.0)
    assert fp16_headroom(torch.tensor([FP16_MAX / 4])) == pytest.approx(4.0)
    assert math.isinf(fp16_headroom(torch.zeros(4)))


def test_is_finite() -> None:
    assert is_finite(torch.ones(3))
    assert not is_finite(torch.tensor([1.0, float("nan")]))


# ---------------------------------------------------------------- module tracer
class _Nan(nn.Module):
    """A layer whose own output is NaN, so the tracer can attribute it."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * float("nan")


class _Breaks(nn.Sequential):
    """A healthy layer named ``good`` followed by a NaN-producing one named ``bad``."""

    def __init__(self) -> None:
        super().__init__()
        self.good = nn.Linear(4, 4)
        self.bad = _Nan()


def test_module_tracer_records_every_leaf() -> None:
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    tracer = ModuleTracer(model)
    with tracer:
        model(torch.randn(2, 4))
    assert len(tracer.records) == 3
    assert [r["class"] for r in tracer.records] == ["Linear", "ReLU", "Linear"]


def test_module_tracer_detaches_hooks_on_exit() -> None:
    model = nn.Sequential(nn.Linear(4, 4))
    tracer = ModuleTracer(model)
    with tracer:
        model(torch.randn(2, 4))
    before = len(tracer.records)
    model(torch.randn(2, 4))
    assert len(tracer.records) == before


def test_first_nonfinite_names_the_failing_module() -> None:
    model = _Breaks()
    first_bad, records = first_nonfinite(model, torch.randn(2, 4))
    assert first_bad is not None
    assert first_bad["name"] == "bad"
    assert first_bad["has_nan"]
    assert len(records) == 2


def test_first_nonfinite_returns_none_when_healthy() -> None:
    model = nn.Sequential(nn.Linear(4, 4))
    first_bad, _ = first_nonfinite(model, torch.randn(2, 4))
    assert first_bad is None


def test_tracer_worst_headroom_sorts_nonfinite_first() -> None:
    model = _Breaks()
    tracer = ModuleTracer(model)
    with tracer, torch.no_grad():
        model(torch.randn(2, 4))
    assert tracer.worst_headroom(1)[0]["name"] == "bad"


# ------------------------------------------------------------------ fp32 island
def test_fp32_island_defeats_autocast_demotion() -> None:
    """``.float()`` alone does not survive autocast; the island is what makes it hold.

    This is the measurement behind the Phase 4.10.1 loss fix: ``conv2d`` re-casts its
    arguments to the autocast dtype whatever the caller passed.
    """
    x = torch.randn(1, 4, 8, 8)
    w = torch.randn(4, 4, 3, 3)

    with torch.amp.autocast("cpu", dtype=torch.float16):
        assert F.conv2d(x.float(), w.float()).dtype is torch.float16
        with fp32_island("cpu"):
            assert F.conv2d(x.float(), w.float()).dtype is torch.float32


def test_detect_anomaly_is_a_no_op_when_disabled() -> None:
    with detect_anomaly(False):
        assert not torch.is_anomaly_enabled()
    with detect_anomaly(True):
        assert torch.is_anomaly_enabled()
    assert not torch.is_anomaly_enabled()


# -------------------------------------------------------------------- LayerNorm
def test_layer_norm_computes_moments_in_fp32_under_autocast() -> None:
    """The moments must not follow the autocast dtype.

    ``mean`` and ``var`` are on no autocast policy list, so before Phase 4.10.1 they
    ran in fp16 and the variance saturated at 65504.
    """
    layer = LayerNorm2d(48)
    x = torch.randn(2, 48, 8, 8, dtype=torch.float16)
    with torch.amp.autocast("cpu", dtype=torch.float16):
        assert layer(x).dtype is torch.float32


def test_layer_norm_survives_the_fp16_variance_overflow_regime() -> None:
    """Regression for the silent-zero failure.

    With fp16 moments, a channel standard deviation past ~256 overflows the variance,
    ``rsqrt(inf)`` is 0, and the layer returns zeros — deleting the signal without
    raising. With fp32 moments it normalises correctly.
    """
    layer = LayerNorm2d(48, affine=False)
    x = (torch.randn(2, 48, 8, 8) * 5000.0).half()

    out = layer(x)
    assert torch.isfinite(out).all()
    assert not bool((out == 0).all())
    # A correctly normalised tensor has unit variance regardless of input scale.
    assert out.float().std().item() == pytest.approx(1.0, abs=0.05)


def test_layer_norm_maths_is_unchanged() -> None:
    """The fix changes dtype, not definition: agreement with a float64 reference."""
    layer = LayerNorm2d(48, affine=False)
    for magnitude in (1.0, 10.0, 100.0):
        x = torch.randn(4, 48, 16, 16) * magnitude
        x64 = x.double()
        reference = (x64 - x64.mean(dim=1, keepdim=True)) * torch.rsqrt(
            x64.var(dim=1, keepdim=True, unbiased=False) + 1e-6
        )
        assert (layer(x).double() - reference).abs().max().item() < 1e-5


def test_layer_norm_is_still_scriptable() -> None:
    """Contract Part 9: the fix must not use autocast contexts, which do not script."""
    layer = LayerNorm2d(16).eval()
    x = torch.randn(2, 16, 8, 8)
    assert torch.allclose(torch.jit.script(layer)(x), layer(x), atol=1e-5)


# --------------------------------------------------------------- composite loss
def test_composite_loss_runs_in_fp32_under_autocast() -> None:
    """Every term must execute in fp32 even when the caller is inside autocast."""
    criterion = CompositeLoss(enabled={"noise": False})
    pred = torch.rand(1, 1, 256, 256)
    target = torch.rand(1, 1, 256, 256)

    with torch.amp.autocast("cpu", dtype=torch.float16):
        total, terms = criterion(pred, target)

    assert total.dtype is torch.float32
    assert all(math.isfinite(v) for v in terms.values())


def test_composite_loss_matches_between_autocast_and_fp32() -> None:
    """The island makes the objective independent of the surrounding autocast state."""
    torch.manual_seed(0)
    criterion = CompositeLoss(enabled={"noise": False})
    pred = torch.rand(1, 1, 256, 256)
    target = torch.rand(1, 1, 256, 256)

    plain, _ = criterion(pred, target)
    with torch.amp.autocast("cpu", dtype=torch.float16):
        autocast_value, _ = criterion(pred, target)

    assert plain.item() == pytest.approx(autocast_value.item(), rel=1e-6)


def test_composite_loss_gradients_are_finite_under_autocast() -> None:
    criterion = CompositeLoss(enabled={"noise": False})
    pred = torch.rand(1, 1, 256, 256, requires_grad=True)
    target = torch.rand(1, 1, 256, 256)

    with torch.amp.autocast("cpu", dtype=torch.float16):
        total, _ = criterion(pred, target)
    total.backward()

    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


# ------------------------------------------------------------- model-level trace
def test_model_forward_is_finite_at_initialisation() -> None:
    """The shipped initialisation must have real fp16 headroom, not marginal headroom."""
    model = SPARCNet(build_sparc_config("sparc-base", use_attention=False)).eval()
    x = torch.rand(1, 1, 128, 128) * 0.6

    tracer = ModuleTracer(model)
    with tracer, torch.no_grad(), torch.amp.autocast("cpu", dtype=torch.float16):
        output = model.forward_with_aux(x)

    assert torch.isfinite(output.image).all()
    assert tracer.first_bad() is None
    peak = max(r["absmax"] for r in tracer.records)
    assert peak < FP16_MAX / 10, f"only {FP16_MAX / peak:.1f}x fp16 headroom at init"
