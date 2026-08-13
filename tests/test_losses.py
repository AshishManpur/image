"""Loss-system tests (Contract Part 6, Part 9).

Part 9 requires every loss to pass shape, gradient-propagation and numerical-stability
checks **individually, before combination**. The file is therefore organised per term,
with the composite tested only after each component is established.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from configs.sparc_config import LossConfig, build_sparc_config
from losses import (
    TERM_NAMES,
    CharbonnierLoss,
    CompositeLoss,
    FFTLoss,
    GradientLoss,
    MSSSIMLoss,
    NoiseAuxLoss,
    WaveletLoss,
)
from models.sparc_net import SPARCNet

SIZE = 256
"""MS-SSIM at 5 scales needs >= 161 px; the real training size is 256."""


def _pair(batch: int = 2, size: int = SIZE, noise: float = 0.05):
    generator = torch.Generator().manual_seed(1337)
    target = torch.rand(batch, 1, size, size, generator=generator)
    pred = (target + noise * torch.randn(target.shape, generator=generator)).clamp(0, 1)
    return pred.requires_grad_(True), target


def _reconstruction_losses() -> list[tuple[str, nn.Module]]:
    """Every term with a plain ``(pred, target)`` signature."""
    return [
        ("charbonnier", CharbonnierLoss()),
        ("ms_ssim", MSSSIMLoss()),
        ("wavelet", WaveletLoss()),
        ("fft", FFTLoss()),
        ("gradient", GradientLoss()),
    ]


# =============================================================== shared invariants
@pytest.mark.parametrize("name,loss", _reconstruction_losses())
@pytest.mark.parametrize("batch", [1, 2, 8])
def test_loss_returns_a_scalar(name: str, loss: nn.Module, batch: int) -> None:
    pred, target = _pair(batch)
    value = loss(pred, target)
    assert value.ndim == 0
    assert torch.isfinite(value)


@pytest.mark.parametrize("name,loss", _reconstruction_losses())
def test_identical_inputs_give_zero_loss(name: str, loss: nn.Module) -> None:
    """Every reconstruction term must vanish when the prediction is exact.

    Charbonnier is the deliberate exception: ``sqrt(0 + eps)`` is ``1e-3``, by design —
    the epsilon is what makes it differentiable at zero.
    """
    _, target = _pair()
    value = loss(target, target).item()
    if name == "charbonnier":
        assert value == pytest.approx(math.sqrt(1e-6), abs=1e-9)
    else:
        assert value == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("name,loss", _reconstruction_losses())
def test_increasing_error_increases_loss(name: str, loss: nn.Module) -> None:
    _, target = _pair()
    generator = torch.Generator().manual_seed(4)
    perturbation = torch.randn(target.shape, generator=generator)
    values = [
        loss((target + level * perturbation).clamp(0, 1), target).item()
        for level in (0.01, 0.05, 0.2)
    ]
    assert values[0] < values[1] < values[2], f"{name} not monotone: {values}"


@pytest.mark.parametrize("name,loss", _reconstruction_losses())
def test_gradients_propagate(name: str, loss: nn.Module) -> None:
    pred, target = _pair()
    loss(pred, target).backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert torch.count_nonzero(pred.grad) > 0


@pytest.mark.parametrize("name,loss", _reconstruction_losses())
@pytest.mark.parametrize("magnitude", [1e-3, 1e3])
def test_numerical_stability_at_extreme_magnitudes(
    name: str, loss: nn.Module, magnitude: float
) -> None:
    generator = torch.Generator().manual_seed(11)
    target = torch.rand(2, 1, SIZE, SIZE, generator=generator) * magnitude
    pred = (target + magnitude * 0.01 * torch.randn(target.shape, generator=generator))
    pred.requires_grad_(True)
    value = loss(pred, target)
    assert torch.isfinite(value), f"{name} produced {value}"
    value.backward()
    assert torch.isfinite(pred.grad).all()


@pytest.mark.parametrize("name,loss", _reconstruction_losses())
def test_autocast_produces_finite_loss(name: str, loss: nn.Module) -> None:
    """AMP readiness: FFT and MS-SSIM internally force float32 for this reason."""
    pred, target = _pair()
    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        value = loss(pred, target)
    assert torch.isfinite(value)


@pytest.mark.parametrize("name,loss", _reconstruction_losses())
def test_channels_last_matches_contiguous(name: str, loss: nn.Module) -> None:
    pred, target = _pair()
    reference = loss(pred, target)
    produced = loss(
        pred.to(memory_format=torch.channels_last),
        target.to(memory_format=torch.channels_last),
    )
    assert produced.item() == pytest.approx(reference.item(), rel=1e-5)


@pytest.mark.parametrize("name,loss", _reconstruction_losses())
def test_rejects_shape_mismatch(name: str, loss: nn.Module) -> None:
    with pytest.raises(ValueError):
        loss(torch.rand(2, 1, SIZE, SIZE), torch.rand(2, 1, SIZE, SIZE // 2))


@pytest.mark.parametrize("name,loss", _reconstruction_losses())
def test_losses_contribute_no_parameters(name: str, loss: nn.Module) -> None:
    """Kernels must be buffers. A stray Parameter would corrupt the Part 3 budget."""
    assert sum(p.numel() for p in loss.parameters()) == 0


@pytest.mark.parametrize("name,loss", _reconstruction_losses())
def test_torchscript_matches_eager(name: str, loss: nn.Module) -> None:
    """Part 9 requires TorchScript for every loss term.

    This is why the shape checks use string concatenation rather than f-strings:
    TorchScript cannot size ``tuple(tensor.shape)``.
    """
    pred, target = _pair()
    scripted = torch.jit.script(loss)
    with torch.no_grad():
        assert torch.allclose(scripted(pred, target), loss(pred, target), atol=1e-5)


@pytest.mark.parametrize("name,loss", _reconstruction_losses())
def test_torch_compile_matches_eager(name: str, loss: nn.Module) -> None:
    """Dynamo must trace each term in a single graph.

    ``backend="eager"`` deliberately: it exercises tracing without invoking Inductor's
    C++ codegen, which needs an MSVC toolchain this development host lacks.
    """
    pred, target = _pair()
    compiled = torch.compile(loss, backend="eager", fullgraph=True)
    with torch.no_grad():
        assert torch.allclose(compiled(pred, target), loss(pred, target), atol=1e-5)


# ========================================================================= MS-SSIM
def test_ms_ssim_rejects_images_smaller_than_the_pyramid() -> None:
    loss = MSSSIMLoss()
    assert loss.minimum_size == 161
    with pytest.raises(ValueError, match="at least"):
        loss(torch.rand(1, 1, 64, 64), torch.rand(1, 1, 64, 64))


def test_ms_ssim_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError):
        MSSSIMLoss(scales=6)
    with pytest.raises(ValueError):
        MSSSIMLoss(window_size=10)
    with pytest.raises(ValueError):
        MSSSIMLoss(sigma=0.0)


def test_ms_ssim_is_bounded_in_unit_interval() -> None:
    loss = MSSSIMLoss()
    _, target = _pair()
    generator = torch.Generator().manual_seed(5)
    for pred in (target, torch.rand(target.shape, generator=generator), 1.0 - target):
        value = loss(pred, target).item()
        assert -1e-5 <= value <= 2.0


# ======================================================================== gradient
def test_gradient_kernels_are_buffers_not_parameters() -> None:
    loss = GradientLoss()
    names = dict(loss.named_buffers())
    assert "kernel_x" in names and "kernel_y" in names
    assert not list(loss.parameters())
    assert not any(isinstance(m, nn.Conv2d) for m in loss.modules())


def test_sobel_detects_a_vertical_edge() -> None:
    """A vertical step must produce a horizontal response and no vertical one.

    With ``normalize=True`` the kernel is divided by 8, so a clean 0->1 step gives a
    response of exactly ``(1 + 2 + 1) / 8 = 0.5``. Asserting the exact value pins the
    normalisation, which the fixed contract weight of 0.05 depends on.
    """
    loss = GradientLoss()
    image = torch.zeros(1, 1, 32, 32)
    image[..., 16:] = 1.0
    grad_x, grad_y = loss.gradients(image)
    assert grad_x.abs().max().item() == pytest.approx(0.5, abs=1e-6)
    assert grad_y.abs().max().item() < 1e-6


def test_gradient_loss_is_blind_to_a_constant_offset() -> None:
    """A DC shift changes no derivative, so the gradient term must not react."""
    loss = GradientLoss()
    _, target = _pair()
    assert loss(target + 0.3, target).item() == pytest.approx(0.0, abs=1e-6)


def test_gradient_magnitude_is_available_for_diagnostics() -> None:
    loss = GradientLoss()
    image = torch.zeros(1, 1, 16, 16)
    image[..., 8:] = 1.0
    # Pure vertical edge: magnitude collapses to |grad_x| = 0.5.
    assert loss.magnitude(image).max().item() == pytest.approx(0.5, abs=1e-5)


# ============================================================================= FFT
def test_fft_is_blind_to_a_global_shift_of_phase_only() -> None:
    """Amplitude-only: a circular shift preserves the magnitude spectrum exactly."""
    loss = FFTLoss()
    _, target = _pair(batch=1)
    shifted = torch.roll(target, shifts=(7, 11), dims=(-2, -1))
    assert loss(shifted, target).item() == pytest.approx(0.0, abs=1e-4)


def test_fft_detects_a_blur() -> None:
    """Blurring removes high-frequency energy, which is what this term exists to see."""
    loss = FFTLoss()
    _, target = _pair(batch=1)
    blurred = torch.nn.functional.avg_pool2d(target, 4)
    blurred = torch.nn.functional.interpolate(blurred, scale_factor=4, mode="nearest")
    assert loss(blurred, target).item() > loss(target, target).item() + 1e-4


def test_fft_rejects_unknown_norm() -> None:
    with pytest.raises(ValueError):
        FFTLoss(norm="nonsense")


# ========================================================================= wavelet
def test_wavelet_reports_every_band_and_level() -> None:
    loss = WaveletLoss(levels=2)
    pred, target = _pair()
    bands = loss.band_losses(pred, target)
    assert set(bands) == {
        f"L{level}_{band}"
        for level in (1, 2)
        for band in ("LL", "LH", "HL", "HH")
    }
    assert all(torch.isfinite(v) for v in bands.values())


def test_wavelet_band_weighting_favours_detail_bands() -> None:
    """A pure-HH error must cost more than an equal-size pure-LL error."""
    loss = WaveletLoss(levels=1)
    base = torch.zeros(1, 1, 32, 32)

    ll_error = base.clone() + 0.1  # constant -> LL only
    checker = torch.tensor([[1.0, -1.0], [-1.0, 1.0]]).repeat(16, 16)
    hh_error = base + 0.1 * checker.reshape(1, 1, 32, 32)  # alternating -> HH only

    assert loss(hh_error, base).item() > loss(ll_error, base).item()


def test_wavelet_rejects_indivisible_sizes_and_bad_config() -> None:
    loss = WaveletLoss(levels=2)
    with pytest.raises(ValueError, match="divisible"):
        loss(torch.rand(1, 1, 30, 30), torch.rand(1, 1, 30, 30))
    with pytest.raises(ValueError):
        WaveletLoss(levels=0)
    with pytest.raises(ValueError):
        WaveletLoss(band_weights=(1.0, 1.0, 1.0))
    with pytest.raises(ValueError):
        WaveletLoss(band_weights=(-1.0, 1.0, 1.0, 1.0))


# =========================================================================== noise
def _noise_batch(batch: int = 2):
    generator = torch.Generator().manual_seed(21)
    gt = torch.rand(batch, 1, 64, 64, generator=generator)
    lr = torch.rand(batch, 1, 32, 32, generator=generator)
    sigma = torch.full((batch, 1, 32, 32), 0.1, requires_grad=True)
    return sigma, lr, gt


def test_noise_loss_shape_and_gradient() -> None:
    loss = NoiseAuxLoss()
    sigma, lr, gt = _noise_batch()
    value = loss(sigma, lr, gt)
    assert value.ndim == 0 and torch.isfinite(value)
    value.backward()
    assert sigma.grad is not None and torch.isfinite(sigma.grad).all()


def test_noise_loss_is_zero_against_its_own_target() -> None:
    loss = NoiseAuxLoss()
    _, lr, gt = _noise_batch()
    target = loss.analytic_target(lr, gt)
    assert loss(target, lr, gt, target=target).item() == pytest.approx(0.0, abs=1e-6)


def test_noise_loss_grows_with_relative_error() -> None:
    """Log space: the loss must respond to the ratio, not the absolute difference."""
    loss = NoiseAuxLoss()
    _, lr, gt = _noise_batch()
    target = loss.analytic_target(lr, gt)
    close = loss(target * 1.1, lr, gt, target=target).item()
    far = loss(target * 2.0, lr, gt, target=target).item()
    assert 0.0 < close < far
    assert far == pytest.approx(math.log(2.0), abs=1e-4)


def test_noise_target_is_detached() -> None:
    loss = NoiseAuxLoss()
    _, lr, gt = _noise_batch()
    assert not loss.analytic_target(lr, gt).requires_grad


def test_noise_loss_rejects_bad_shapes_and_config() -> None:
    loss = NoiseAuxLoss()
    _, lr, gt = _noise_batch()
    with pytest.raises(ValueError):
        loss(torch.rand(2, 1, 16, 16), lr, gt)
    with pytest.raises(ValueError):
        NoiseAuxLoss(log_eps=0.0)
    with pytest.raises(ValueError):
        NoiseAuxLoss(sigma_min=2.0, sigma_max=1.0)


# ======================================================================= composite
def test_composite_weights_come_from_the_config() -> None:
    config = LossConfig()
    loss = CompositeLoss(config)
    assert loss.weights == {
        "charbonnier": config.charbonnier,
        "ms_ssim": config.ms_ssim,
        "wavelet": config.wavelet,
        "fft": config.fft,
        "gradient": config.gradient,
        "noise": config.noise_aux,
        "clean_lr": config.clean_lr,
    }
    assert loss.wants_aux is True


def test_composite_reports_every_term() -> None:
    loss = CompositeLoss()
    pred, target = _pair()
    total, terms = loss(pred, target)

    for name in TERM_NAMES:
        if name in ("noise", "clean_lr"):
            # Neither is available on the plain-tensor path: `noise` needs a sigma map
            # and `clean_lr` needs the model's clean-LR estimate, both of which only
            # exist on a `SparcOutput`.
            continue
        assert name in terms and f"raw_{name}" in terms
    assert "total" in terms
    assert torch.isfinite(total)


def test_composite_total_equals_the_manual_weighted_sum() -> None:
    loss = CompositeLoss()
    pred, target = _pair()
    total, terms = loss(pred, target)
    expected = sum(
        loss.weights[name] * terms[f"raw_{name}"]
        for name in TERM_NAMES
        if f"raw_{name}" in terms
    )
    assert total.item() == pytest.approx(expected, rel=1e-5)
    assert terms["total"] == pytest.approx(total.item(), rel=1e-6)


def test_composite_terms_can_be_disabled_individually() -> None:
    loss = CompositeLoss(enabled={"ms_ssim": False, "fft": False})
    pred, target = _pair()
    _, terms = loss(pred, target)
    assert "ms_ssim" not in terms and "fft" not in terms
    assert "charbonnier" in terms
    assert loss.ms_ssim is None and loss.fft is None


def test_composite_rejects_unknown_term_names() -> None:
    with pytest.raises(ValueError, match="Unknown loss term"):
        CompositeLoss(enabled={"perceptual": True})


def test_composite_gradients_propagate() -> None:
    loss = CompositeLoss()
    pred, target = _pair()
    total, _ = loss(pred, target)
    total.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
    assert torch.count_nonzero(pred.grad) > 0


def test_composite_contributes_no_parameters() -> None:
    assert sum(p.numel() for p in CompositeLoss().parameters()) == 0


def test_composite_consumes_the_model_aux_output() -> None:
    """End-to-end: SparcOutput + batch dict, with the noise term active."""
    model = SPARCNet(
        build_sparc_config("sparc-base", use_attention=False, use_gated_fusion=False)
    )
    loss = CompositeLoss()
    generator = torch.Generator().manual_seed(3)
    batch = {
        "lr": torch.rand(1, 1, 128, 128, generator=generator),
        "gt": torch.rand(1, 1, 256, 256, generator=generator),
    }
    output = model.forward_with_aux(batch["lr"])
    total, terms = loss(output, batch)

    assert "noise" in terms, "noise term must activate when sigma and lr are present"
    assert torch.isfinite(total)
    total.backward()
    grads = [p.grad for p in model.noise_head.parameters() if p.grad is not None]
    assert grads, "the auxiliary loss must reach the noise head"
    assert any(torch.count_nonzero(g) > 0 for g in grads)


def test_composite_skips_noise_when_the_head_is_absent() -> None:
    """An ablation without the noise head must not crash the objective."""
    model = SPARCNet(build_sparc_config("sparc-tiny"))
    loss = CompositeLoss()
    batch = {"lr": torch.rand(1, 1, 128, 128), "gt": torch.rand(1, 1, 256, 256)}
    total, terms = loss(model.forward_with_aux(batch["lr"]), batch)
    assert "noise" not in terms
    assert torch.isfinite(total)


def test_composite_rejects_a_batch_without_ground_truth() -> None:
    loss = CompositeLoss()
    with pytest.raises(ValueError, match="'gt'"):
        loss(torch.rand(1, 1, SIZE, SIZE), {"lr": torch.rand(1, 1, 128, 128)})
