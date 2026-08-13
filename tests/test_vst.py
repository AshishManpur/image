"""Numerical contract of the variance-stabilising transform (Phase 5)."""

from __future__ import annotations

import pytest
import torch

from models.vst import VarianceStabiliser

BATCH, SIZE = 4, 32


def _sigmas(batch: int = BATCH, gauss: float = 0.024, speckle: float = 0.165):
    """Per-image sigmas shaped as the NoiseHead emits them."""
    return (
        torch.full((batch, 1), gauss),
        torch.full((batch, 1), speckle),
    )


def test_forward_and_inverse_are_finite() -> None:
    vst = VarianceStabiliser()
    g, s = _sigmas()
    y = torch.rand(BATCH, 1, SIZE, SIZE) * 1.5 - 0.25
    z = vst(y, g, s)
    assert torch.isfinite(z).all()
    assert torch.isfinite(vst.inverse(z, g, s)).all()


def test_round_trip_reconstructs_the_input() -> None:
    """T^-1(T(y)) == y at initialisation, where the bias correction is exactly zero."""
    vst = VarianceStabiliser()
    g, s = _sigmas()
    y = torch.rand(BATCH, 1, SIZE, SIZE) * 1.5 - 0.25
    torch.testing.assert_close(vst.inverse(vst(y, g, s), g, s), y, atol=1e-5, rtol=1e-4)


def test_bias_correction_starts_negligible_but_not_stationary() -> None:
    """v must be ~0 at init, yet have a non-zero gradient so it can actually train.

    ``v = raw ** 2`` has gradient ``2 * raw``; seeding ``raw`` at exactly zero would pin
    the correction at a stationary point forever. This test is the regression guard for
    that bug.
    """
    vst = VarianceStabiliser(bias_correction=True)
    assert vst.residual_variance is not None
    variance = float(vst.residual_variance.detach().pow(2))
    assert 0.0 < variance < 1e-3, variance

    g, s = _sigmas()
    y = torch.rand(BATCH, 1, SIZE, SIZE)
    vst.inverse(vst(y, g, s), g, s).square().mean().backward()
    assert vst.residual_variance.grad is not None
    assert float(vst.residual_variance.grad.abs()) > 0.0, "correction cannot train"


def test_bias_correction_scales_the_inverse_once_trained() -> None:
    vst = VarianceStabiliser(bias_correction=True)
    g, s = _sigmas()
    y = torch.rand(BATCH, 1, SIZE, SIZE)
    z = vst(y, g, s)
    naive = vst.inverse(z, g, s)
    with torch.no_grad():
        vst.residual_variance.fill_(0.5)
    corrected = vst.inverse(z, g, s)
    expected = naive * (1.0 + 0.5 * (0.165**2) * 0.25)
    torch.testing.assert_close(corrected, expected, atol=1e-5, rtol=1e-4)


def test_negative_and_out_of_range_inputs_survive() -> None:
    """Phase 1 measured 0.30 % of LR pixels below 0 and 3.36 % above 1; none are clipped."""
    vst = VarianceStabiliser()
    g, s = _sigmas(batch=1)
    y = torch.tensor([[-0.28, -0.01, 0.0, 0.5, 1.0, 2.16]]).reshape(1, 1, 1, 6)
    z = vst(y, g, s)
    assert torch.isfinite(z).all()
    torch.testing.assert_close(vst.inverse(z, g, s), y, atol=1e-5, rtol=1e-4)
    # Sign is preserved: asinh is odd, so negatives stay negative.
    assert bool((torch.sign(z) == torch.sign(y)).all())


def test_zero_maps_to_zero() -> None:
    vst = VarianceStabiliser()
    g, s = _sigmas(batch=1)
    z = vst(torch.zeros(1, 1, 4, 4), g, s)
    torch.testing.assert_close(z, torch.zeros(1, 1, 4, 4))


@pytest.mark.parametrize("gauss,speckle", [(1e-4, 2.0), (2.0, 1e-4), (1e-4, 1e-4), (2.0, 2.0)])
def test_extreme_sigmas_stay_finite(gauss: float, speckle: float) -> None:
    """The NoiseHead clamps to [1e-4, 2.0]; every corner of that box must hold."""
    vst = VarianceStabiliser()
    g, s = _sigmas(gauss=gauss, speckle=speckle)
    y = torch.rand(BATCH, 1, SIZE, SIZE) * 2.0 - 0.5
    z = vst(y, g, s)
    assert torch.isfinite(z).all()
    assert torch.isfinite(vst.inverse(z, g, s)).all()


def test_gradients_are_finite_through_both_directions() -> None:
    vst = VarianceStabiliser()
    g, s = _sigmas()
    y = (torch.rand(BATCH, 1, SIZE, SIZE) * 1.5 - 0.25).requires_grad_(True)
    vst.inverse(vst(y, g, s), g, s).square().mean().backward()
    assert y.grad is not None and torch.isfinite(y.grad).all()
    assert vst.residual_variance.grad is not None
    assert torch.isfinite(vst.residual_variance.grad).all()


def test_stabilisation_flattens_the_intensity_dependence() -> None:
    """The measured claim: Var(T(y)) is ~1 regardless of I, where Var(y|I) is not.

    Synthesises the Phase 1 model Var(y|I) = sg^2 + ss^2 I^2 at two intensities two
    decades apart and checks the linear-domain spread collapses after stabilisation.
    """
    torch.manual_seed(0)
    sg, ss, n = 0.024, 0.165, 200_000
    vst = VarianceStabiliser()
    spread = []
    for intensity in (0.05, 0.95):
        clean = torch.full((1, 1, 1, n), intensity)
        noise = torch.randn(1, 1, 1, n) * (sg**2 + ss**2 * intensity**2) ** 0.5
        g, s = _sigmas(batch=1)
        z = vst(clean + noise, g, s) - vst(clean, g, s)
        spread.append((float(noise.std()), float(z.std())))
    linear_ratio = spread[1][0] / spread[0][0]
    stabilised_ratio = spread[1][1] / spread[0][1]
    assert linear_ratio > 5.0, f"expected strong heteroscedasticity, got {linear_ratio}"
    assert stabilised_ratio < 1.5, f"stabilisation failed, ratio {stabilised_ratio}"


def test_disabled_stabiliser_is_the_identity_and_weightless() -> None:
    vst = VarianceStabiliser(enabled=False)
    assert vst.residual_variance is None
    assert sum(p.numel() for p in vst.parameters()) == 0
    g, s = _sigmas()
    y = torch.rand(BATCH, 1, SIZE, SIZE)
    torch.testing.assert_close(vst(y, g, s), y)
    torch.testing.assert_close(vst.inverse(y, g, s), y)


def test_batch_mismatch_is_rejected() -> None:
    vst = VarianceStabiliser()
    g, s = _sigmas(batch=2)
    with pytest.raises(ValueError, match="expected sigmas for 4"):
        vst(torch.rand(4, 1, SIZE, SIZE), g, s)


def test_non_positive_eps_is_rejected() -> None:
    with pytest.raises(ValueError, match="eps must be positive"):
        VarianceStabiliser(eps=0.0)
