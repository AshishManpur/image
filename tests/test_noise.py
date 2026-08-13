"""Noise-head tests (Contract Part 2.9, Part 3 stage 1, Part 9).

The head is the one module in SPARC-Net whose *initial output value* is fixed by the
contract, so the initialisation tests here are acceptance criteria rather than
sanity checks.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from configs.sparc_config import NoiseHeadConfig, build_sparc_config
from datasets.degradation import analytic_sigma_map
from models.noise import NoiseHead, NoiseHeadOutput
from models.noise.noise_map import (
    assemble_sigma_map,
    build_smoothing_kernel,
    estimate_local_intensity,
)
from models.sparc_net import SPARCNet
from utils.complexity import measure_complexity

CONTRACT_PARAMETERS = 42_050
"""Contract Part 3, stage 1. Part 9 requires an exact match, not a tolerance."""

CONTRACT_SIGMA_GAUSS = 0.024
CONTRACT_SIGMA_SPECKLE = 0.165


@pytest.fixture
def head() -> NoiseHead:
    return NoiseHead()


def _observation(batch: int = 2, size: int = 128) -> torch.Tensor:
    generator = torch.Generator().manual_seed(1337)
    return torch.rand(batch, 1, size, size, generator=generator)


# --------------------------------------------------------------------------- shapes
@pytest.mark.parametrize("batch", [1, 2, 8])
def test_output_shapes(head: NoiseHead, batch: int) -> None:
    """Part 2.9: (B,1,128,128) -> (B,2) -> sigma map (B,1,128,128)."""
    out = head(_observation(batch))
    assert isinstance(out, NoiseHeadOutput)
    assert out.sigma_gauss.shape == (batch, 1)
    assert out.sigma_speckle.shape == (batch, 1)
    assert out.sigma_map.shape == (batch, 1, 128, 128)
    assert out.sigma_map_normalized.shape == (batch, 1, 128, 128)


def test_trunk_resolutions_halve_four_times(head: NoiseHead) -> None:
    """The four strided stages must take 128 -> 64 -> 32 -> 16 -> 8."""
    x = _observation(1)
    expected = [(16, 64), (24, 32), (32, 16), (32, 8)]
    for stage, (channels, size) in zip(head.stages, expected):
        x = stage(x)
        assert x.shape == (1, channels, size, size)


def test_rejects_bad_inputs(head: NoiseHead) -> None:
    with pytest.raises(ValueError):
        head(torch.rand(2, 1, 128))
    with pytest.raises(ValueError):
        head(torch.rand(2, 3, 128, 128))


# --------------------------------------------------------------------- parameters
def test_parameter_count_is_exactly_the_contract_value(head: NoiseHead) -> None:
    assert sum(p.numel() for p in head.parameters()) == CONTRACT_PARAMETERS


def test_analytic_parameter_count_matches_the_module(head: NoiseHead) -> None:
    assert NoiseHead.parameter_count() == CONTRACT_PARAMETERS
    assert NoiseHead.parameter_count() == sum(p.numel() for p in head.parameters())


def test_smoothing_kernel_contributes_no_parameters(head: NoiseHead) -> None:
    """The kernel is a buffer; if it became a Parameter the budget would break."""
    assert "smoothing_kernel" in dict(head.named_buffers())
    assert "smoothing_kernel" not in dict(head.named_parameters())


# ------------------------------------------------------------------ initialisation
def test_softplus_at_init_gives_the_contract_sigmas(head: NoiseHead) -> None:
    """Part 2.9: weight = 0, bias = (-3.718, -1.718) -> softplus = (0.024, 0.165)."""
    out = head(_observation(4))
    assert torch.allclose(
        out.sigma_gauss, torch.full_like(out.sigma_gauss, CONTRACT_SIGMA_GAUSS), atol=1e-4
    )
    assert torch.allclose(
        out.sigma_speckle,
        torch.full_like(out.sigma_speckle, CONTRACT_SIGMA_SPECKLE),
        atol=1e-4,
    )


def test_initial_prediction_is_input_independent(head: NoiseHead) -> None:
    """A zero final weight makes the initial output constant regardless of input."""
    low = head(torch.zeros(1, 1, 128, 128))
    high = head(torch.full((1, 1, 128, 128), 5.0))
    assert torch.allclose(low.sigma_gauss, high.sigma_gauss, atol=1e-6)
    assert torch.allclose(low.sigma_speckle, high.sigma_speckle, atol=1e-6)


def test_final_layer_survives_the_model_wide_init_sweep() -> None:
    """`SPARCNet.apply(default_init)` must not overwrite the contract initialisation.

    `default_init` re-initialises every `nn.Linear` with `trunc_normal_`. Without the
    `_custom_init` marker the zero weight and the two calibrated biases would be
    silently destroyed during model construction, and the head would start from a
    random sigma instead of the Phase 1 measured one.
    """
    model = SPARCNet(build_sparc_config("sparc-base", use_attention=False,
                                        use_gated_fusion=False))
    fc2 = model.noise_head.fc2
    assert torch.count_nonzero(fc2.weight) == 0
    assert fc2.bias[0].item() == pytest.approx(-3.718, abs=1e-6)
    assert fc2.bias[1].item() == pytest.approx(-1.718, abs=1e-6)


# ------------------------------------------------------------------------- outputs
def test_sigmas_are_strictly_positive(head: NoiseHead) -> None:
    out = head(_observation(8) * 4.0 - 2.0)
    assert (out.sigma_gauss > 0).all()
    assert (out.sigma_speckle > 0).all()
    assert (out.sigma_map > 0).all()


@pytest.mark.parametrize("magnitude", [1e-3, 1.0, 1e3])
def test_sigma_stays_in_the_contract_range(head: NoiseHead, magnitude: float) -> None:
    """Part 2.9: sigma_hat in [1e-4, 2.0] after clamp, for any input magnitude."""
    config = NoiseHeadConfig()
    out = head(_observation(4) * magnitude)
    assert out.sigma_map.min().item() >= config.sigma_min
    assert out.sigma_map.max().item() <= config.sigma_max
    assert torch.isfinite(out.sigma_map).all()


def test_sigma_map_follows_the_phase1_variance_model() -> None:
    """The map must equal sqrt(sigma_g^2 + sigma_s^2 I^2) on a constant image.

    On a constant input the smoother is an identity, so the closed form is exact and
    the assembly can be checked against the analytic target used for supervision.
    """
    kernel = build_smoothing_kernel(5)
    intensity = 0.4
    y = torch.full((3, 1, 32, 32), intensity)
    g = torch.full((3, 1), 0.024)
    s = torch.full((3, 1), 0.165)

    produced = assemble_sigma_map(y, g, s, kernel)
    expected = math.sqrt(0.024**2 + (0.165 * intensity) ** 2)
    assert produced.mean().item() == pytest.approx(expected, abs=1e-5)

    reference = analytic_sigma_map(y, g.flatten(), s.flatten())
    assert torch.allclose(produced, reference, atol=1e-5)


def test_sigma_map_increases_with_intensity() -> None:
    """Speckle is intensity-dependent: brighter regions must get a larger sigma."""
    kernel = build_smoothing_kernel(5)
    y = torch.cat([torch.full((1, 1, 16, 16), 0.1), torch.full((1, 1, 16, 16), 0.9)], dim=0)
    out = assemble_sigma_map(y, torch.full((2, 1), 0.024), torch.full((2, 1), 0.165), kernel)
    assert out[1].mean().item() > out[0].mean().item()


def test_local_intensity_is_non_negative_and_smoothing_preserves_constants() -> None:
    kernel = build_smoothing_kernel(5)
    assert torch.allclose(
        estimate_local_intensity(torch.full((1, 1, 16, 16), 0.7), kernel),
        torch.full((1, 1, 16, 16), 0.7),
        atol=1e-6,
    )
    assert (estimate_local_intensity(torch.full((1, 1, 8, 8), -3.0), kernel) >= 0).all()


def test_normalised_map_is_the_physical_map_over_the_scale(head: NoiseHead) -> None:
    y = _observation(3)
    scale = torch.tensor([0.5, 1.0, 2.0]).reshape(3, 1, 1, 1)
    out = head(y, scale)
    assert torch.allclose(out.sigma_map_normalized, out.sigma_map / scale, atol=1e-5)
    assert torch.allclose(head(y).sigma_map_normalized, head(y).sigma_map, atol=1e-6)


def test_build_smoothing_kernel_rejects_even_sizes() -> None:
    for bad in (0, -1, 4):
        with pytest.raises(ValueError):
            build_smoothing_kernel(bad)


def test_assemble_rejects_shape_mismatch() -> None:
    kernel = build_smoothing_kernel(5)
    with pytest.raises(ValueError):
        assemble_sigma_map(torch.rand(2, 1, 8, 8), torch.rand(3, 1), torch.rand(2, 1), kernel)
    with pytest.raises(ValueError):
        assemble_sigma_map(torch.rand(2, 1, 8), torch.rand(2, 1), torch.rand(2, 1), kernel)


def test_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError):
        NoiseHead(NoiseHeadConfig(trunk_channels=()))
    with pytest.raises(ValueError):
        NoiseHead(NoiseHeadConfig(sigma_min=1.0, sigma_max=0.5))


# ----------------------------------------------------------------------- gradients
def test_trunk_is_gradient_dormant_at_initialisation(head: NoiseHead) -> None:
    """At exact initialisation the trunk receives **zero** gradient, by construction.

    This is a consequence of the contract, not a defect. Part 2.9 fixes the final
    layer's weight at zero, so at step 0 the prediction is a constant and
    ``d sigma / d trunk`` is identically zero. The trunk is therefore dormant for
    exactly one optimiser step: ``fc2.weight`` itself does receive a non-zero
    gradient, so after the first update the weight is non-zero and gradient reaches
    the whole module (see the next test).

    The test exists to pin this down deliberately. Without it, a future change that
    accidentally detached the trunk would be indistinguishable from the intended
    zero-init behaviour.
    """
    out = head(_observation(2))
    (out.sigma_map.mean() + out.sigma_gauss.sum() + out.sigma_speckle.sum()).backward()

    assert torch.count_nonzero(head.fc2.weight.grad) > 0, "fc2.weight must get gradient"
    assert torch.count_nonzero(head.fc2.bias.grad) > 0, "fc2.bias must get gradient"
    assert torch.count_nonzero(head.stages[0].conv.weight.grad) == 0


def test_gradients_reach_every_parameter_once_training_has_started(
    head: NoiseHead,
) -> None:
    """After the final weight leaves zero, every parameter must get a finite gradient.

    `fc2.weight` is zero *in value* by contract, so the assertion throughout is that
    each parameter **receives** a non-zero gradient — never that its value is non-zero.
    """
    # Simulate the state after one optimiser step.
    with torch.no_grad():
        head.fc2.weight.normal_(0.0, 0.02)

    out = head(_observation(2))
    (out.sigma_map.mean() + out.sigma_gauss.sum() + out.sigma_speckle.sum()).backward()

    for name, param in head.named_parameters():
        assert param.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(param.grad).all(), f"{name} has non-finite gradient"
        assert torch.count_nonzero(param.grad) > 0, f"{name} has an all-zero gradient"


# -------------------------------------------------------------------- stability
def test_no_nan_over_many_batches(head: NoiseHead) -> None:
    """Part 9: no NaN/Inf over 100 random batches, including extreme magnitudes."""
    generator = torch.Generator().manual_seed(7)
    for index in range(100):
        magnitude = {0: 1e-3, 1: 1e3}.get(index % 3, 1.0)
        y = torch.randn(2, 1, 128, 128, generator=generator) * magnitude
        out = head(y)
        assert torch.isfinite(out.sigma_map).all()
        assert torch.isfinite(out.sigma_gauss).all()
        assert torch.isfinite(out.sigma_speckle).all()


def test_contains_no_forbidden_layers(head: NoiseHead) -> None:
    """Part 7 forbids BatchNorm anywhere in the network."""
    for module in head.modules():
        assert not isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))


# --------------------------------------------------------- GPU-readiness (CPU-safe)
def test_autocast_fp16_produces_finite_output(head: NoiseHead) -> None:
    """AMP readiness. CPU autocast uses bfloat16; CUDA uses fp16. Both must be finite.

    The sigma arithmetic is deliberately performed in float32 inside the module: the
    squared sigmas are ~5.8e-4 at initialisation, close enough to the fp16 subnormal
    range that half-precision accumulation would lose accuracy.
    """
    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        out = head(_observation(2))
    assert torch.isfinite(out.sigma_map).all()
    assert out.sigma_map.dtype == torch.float32


def test_channels_last_is_supported(head: NoiseHead) -> None:
    """Contract Part 5 runs the model with `memory_format=torch.channels_last`."""
    y = _observation(2).to(memory_format=torch.channels_last)
    reference = head(_observation(2))
    out = head(y)
    assert torch.allclose(out.sigma_map, reference.sigma_map, atol=1e-5)


def test_torchscript_matches_eager(head: NoiseHead) -> None:
    """Part 9: `torch.jit.script` succeeds and matches eager to 1e-5.

    This is why `NoiseHeadOutput` is a `NamedTuple` and why `sigma_min`/`sigma_max` are
    plain float attributes rather than reads through `self.config`: TorchScript cannot
    represent a dataclass return type or hold an arbitrary Python object as a module
    attribute.
    """
    head.eval()
    scripted = torch.jit.script(head)
    y = _observation(2)
    with torch.no_grad():
        expected = head(y)
        produced = scripted(y)
    assert torch.allclose(expected.sigma_map, produced.sigma_map, atol=1e-5)
    assert torch.allclose(expected.sigma_gauss, produced.sigma_gauss, atol=1e-5)
    assert torch.allclose(expected.sigma_speckle, produced.sigma_speckle, atol=1e-5)


def test_torchscript_handles_the_optional_scale_argument(head: NoiseHead) -> None:
    """The `Optional[Tensor]` scale must survive scripting in both states."""
    head.eval()
    scripted = torch.jit.script(head)
    y = _observation(2)
    scale = torch.tensor([0.5, 2.0]).reshape(2, 1, 1, 1)
    with torch.no_grad():
        assert torch.allclose(
            scripted(y, scale).sigma_map_normalized,
            head(y, scale).sigma_map_normalized,
            atol=1e-5,
        )


def test_torch_compile_traces_without_graph_breaks(head: NoiseHead) -> None:
    """Contract Part 5 keeps `torch.compile` off in V1 but requires compatibility.

    The `eager` backend is used deliberately: it exercises the part that can actually
    be wrong in our code — whether Dynamo can trace the module in a single graph —
    without invoking the Inductor C++ codegen, which needs an MSVC toolchain that this
    development host lacks. Inductor codegen must still be confirmed on the RTX A400.
    """
    compiled = torch.compile(head, backend="eager", fullgraph=True)
    y = _observation(1)
    with torch.no_grad():
        expected = head(y).sigma_map
        produced = compiled(y).sigma_map
    assert torch.allclose(expected, produced, atol=1e-4)


def test_onnx_export_matches_eager(head: NoiseHead, tmp_path) -> None:
    """Part 9: ONNX export succeeds and onnxruntime matches eager to 1e-3.

    ``opset_version=18`` rather than the contract's 17: the torch 2.10 exporter has no
    implementation below 18 and silently upgrades a 17 request, so pinning 17 here
    would test something that does not happen. Flagged for Phase 4.15.
    """
    onnxruntime = pytest.importorskip("onnxruntime")
    head.eval()
    path = tmp_path / "noise_head.onnx"
    y = _observation(1)

    torch.onnx.export(
        head, (y,), str(path), opset_version=18, input_names=["y"],
        output_names=["sigma_gauss", "sigma_speckle", "sigma_map", "sigma_map_norm"],
    )
    session = onnxruntime.InferenceSession(
        str(path), providers=["CPUExecutionProvider"]
    )
    produced = session.run(None, {"y": y.numpy()})
    with torch.no_grad():
        expected = head(y)

    for index, reference in enumerate(expected):
        assert abs(produced[index] - reference.numpy()).max() < 1e-3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_execution_matches_cpu(head: NoiseHead) -> None:
    y = _observation(2)
    expected = head(y).sigma_map
    produced = head.cuda()(y.cuda()).sigma_map.cpu()
    assert torch.allclose(expected, produced, atol=1e-4)


# ------------------------------------------------------------------------- MACs
def test_measured_macs_match_the_corrected_accounting(head: NoiseHead) -> None:
    """Part 3 records 52.32 MMAC; the true measured value is ~13.4 MMAC.

    The table counts each strided convolution at its *input* resolution (4x
    over-count): 51.90 M that way plus 0.41 M for the smoother gives the tabulated
    52.31 M. Counted at output resolution the convolutions cost 12.98 M. The module is
    deliberately not inflated to match the erratum — see finding V-3 in
    `reports/PHASE4_7_AUDIT.md`.
    """
    report = measure_complexity(head, torch.rand(1, 1, 128, 128))
    measured_mmac = report.macs / 1e6
    assert 12.0 <= measured_mmac <= 15.0, f"measured {measured_mmac:.2f} MMAC"
    assert measured_mmac < 52.32 * 0.95, "must not silently match the erratum"


# ------------------------------------------------------------------ integration
def test_sparcnet_exposes_the_noise_prediction() -> None:
    """Training path returns image + noise; inference path returns the image only."""
    model = SPARCNet(
        build_sparc_config("sparc-base", use_attention=False, use_gated_fusion=False)
    ).eval()
    y = _observation(1)

    with torch.no_grad():
        image = model(y)
        aux = model.forward_with_aux(y)

    assert image.shape == (1, 1, 256, 256)
    assert torch.equal(image, aux.image), "aux path must not change the restored image"
    assert aux.sigma is not None and aux.sigma.shape == (1, 1, 128, 128)
    assert aux.noise is not None
    assert torch.equal(aux.sigma, aux.noise.sigma_map)
    assert aux.noise.sigma_gauss.shape == (1, 1)


def test_noise_head_feeds_two_channels_into_the_stem() -> None:
    """With the head enabled the stem must receive concat[y_hat, sigma_hat]."""
    model = SPARCNet(
        build_sparc_config("sparc-base", use_attention=False, use_gated_fusion=False)
    )
    assert model.encoder.stem.in_channels == 2

    without = SPARCNet(build_sparc_config("sparc-tiny"))
    assert without.encoder.stem.in_channels == 1
    assert without.forward_with_aux(_observation(1)).noise is None
