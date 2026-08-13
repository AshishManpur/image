"""Skip-fusion tests (Contract Part 2.7, Part 3 stages 15/20, Part 9).

Covers both fusion implementations: ``GatedFuse`` (the V1 module) and ``ConcatFusion``
(the step-6 placeholder retained as the ablation A3 control arm). The regression tests
at the end compare them directly, which is what ablation A3 measures.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from configs.sparc_config import build_sparc_config
from models.fusion import ConcatFusion, GatedFuse, build_fusion
from models.sparc_net import SPARCNet
from utils.complexity import measure_complexity

CONTRACT_PARAMETERS = {96: 23_256, 48: 5_868}
"""Contract Part 3 stages 15 and 20. Part 9 requires an exact match."""


def _inputs(batch: int = 2, channels: int = 96, size: int = 32):
    generator = torch.Generator().manual_seed(1337)
    skip = torch.rand(batch, channels, size, size, generator=generator)
    decoded = torch.rand(batch, channels, size, size, generator=generator)
    return skip, decoded


# --------------------------------------------------------------------- parameters
@pytest.mark.parametrize("channels,expected", sorted(CONTRACT_PARAMETERS.items()))
def test_parameter_count_is_exactly_the_contract_value(
    channels: int, expected: int
) -> None:
    module = GatedFuse(channels)
    assert sum(p.numel() for p in module.parameters()) == expected


@pytest.mark.parametrize("channels,expected", sorted(CONTRACT_PARAMETERS.items()))
def test_analytic_parameter_count_matches_the_module(
    channels: int, expected: int
) -> None:
    assert GatedFuse.parameter_count(channels) == expected
    assert GatedFuse.parameter_count(channels) == sum(
        p.numel() for p in GatedFuse(channels).parameters()
    )


def test_rejects_invalid_configuration() -> None:
    for channels, reduction in ((0, 4), (-8, 4), (96, 0), (96, -1), (2, 4)):
        with pytest.raises(ValueError):
            GatedFuse(channels, reduction)


# ------------------------------------------------------------------------- shapes
@pytest.mark.parametrize("batch", [1, 2, 8])
def test_output_shape_is_preserved(batch: int) -> None:
    module = GatedFuse(96)
    skip, decoded = _inputs(batch)
    assert module(skip, decoded).shape == skip.shape


def test_gate_is_per_channel_not_per_pixel() -> None:
    """Contract Part 2.7 pools globally: the gate is (B, C, 1, 1)."""
    module = GatedFuse(96)
    skip, decoded = _inputs()
    assert module.gate(skip, decoded).shape == (2, 96, 1, 1)


def test_rejects_mismatched_or_wrong_channel_inputs() -> None:
    module = GatedFuse(96)
    with pytest.raises(ValueError, match="must match"):
        module(torch.rand(2, 96, 32, 32), torch.rand(2, 96, 16, 16))
    with pytest.raises(ValueError, match="configured for"):
        module(torch.rand(2, 48, 32, 32), torch.rand(2, 48, 32, 32))


# ------------------------------------------------------------------------ gating
@pytest.mark.parametrize("magnitude", [1e-3, 1.0])
def test_gate_is_strictly_inside_the_unit_interval(magnitude: float) -> None:
    """At realistic feature magnitudes the gate stays strictly between 0 and 1."""
    module = GatedFuse(96)
    gate = module.gate(*(t * magnitude for t in _inputs()))
    assert (gate > 0).all() and (gate < 1).all()


def test_gate_saturates_safely_at_extreme_magnitudes() -> None:
    """At x1e3 the sigmoid saturates to exactly 0.0 / 1.0 in float32.

    That is arithmetic, not a defect: the gate remains in ``[0, 1]``, stays finite, and
    the convex-combination guarantee is unaffected — a saturated gate simply selects one
    input outright. The consequence worth knowing is that the *gate's own* gradient
    vanishes there, so a run whose features reach this magnitude would stop adapting the
    mix. The trunk operates on normalised features, so this is a stress test rather than
    an operating point.
    """
    module = GatedFuse(96)
    skip, decoded = (t * 1e3 for t in _inputs())
    gate = module.gate(skip, decoded)

    assert torch.isfinite(gate).all()
    assert (gate >= 0).all() and (gate <= 1).all()

    out = module(skip, decoded)
    assert torch.isfinite(out).all()
    assert (out >= torch.minimum(skip, decoded) - 1e-3).all()
    assert (out <= torch.maximum(skip, decoded) + 1e-3).all()


def test_output_is_a_convex_combination_of_the_inputs() -> None:
    """The coefficients sum to 1, so the fusion can never amplify.

    This is the property that keeps the residual path stable: the output is bounded
    elementwise by the two inputs, whatever the gate learns.
    """
    module = GatedFuse(96)
    skip, decoded = _inputs()
    out = module(skip, decoded)
    lower = torch.minimum(skip, decoded)
    upper = torch.maximum(skip, decoded)
    assert (out >= lower - 1e-6).all()
    assert (out <= upper + 1e-6).all()


def test_saturated_gate_selects_one_input_exactly() -> None:
    """Forcing g -> 1 must return the skip, and g -> 0 the decoder.

    Guards the operand order. Swapping ``skip`` and ``decoded`` still trains and still
    produces plausible numbers, so nothing else in the suite would catch it — but it
    inverts the module's meaning.
    """
    module = GatedFuse(96)
    skip, decoded = _inputs()

    with torch.no_grad():
        module.excite.bias.fill_(50.0)  # sigmoid saturates high
        module.excite.weight.zero_()
    assert torch.allclose(module(skip, decoded), skip, atol=1e-5)

    with torch.no_grad():
        module.excite.bias.fill_(-50.0)  # sigmoid saturates low
    assert torch.allclose(module(skip, decoded), decoded, atol=1e-5)


def test_identical_inputs_are_returned_unchanged() -> None:
    """When both inputs agree, any convex combination is that value."""
    module = GatedFuse(96)
    skip, _ = _inputs()
    assert torch.allclose(module(skip, skip), skip, atol=1e-6)


# --------------------------------------------------------------------- gradients
def test_gradients_reach_every_parameter_and_both_inputs() -> None:
    module = GatedFuse(96)
    skip, decoded = _inputs()
    skip.requires_grad_(True)
    decoded.requires_grad_(True)

    module(skip, decoded).pow(2).mean().backward()

    for name, param in module.named_parameters():
        assert param.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(param.grad).all(), f"{name} has non-finite gradient"
        assert torch.count_nonzero(param.grad) > 0, f"{name} has an all-zero gradient"
    for name, tensor in (("skip", skip), ("decoded", decoded)):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
        assert torch.count_nonzero(tensor.grad) > 0, f"{name} got no gradient"


# -------------------------------------------------------------------- stability
def test_no_nan_over_many_batches() -> None:
    """Part 9: no NaN/Inf over 100 random batches, including extreme magnitudes."""
    module = GatedFuse(96)
    generator = torch.Generator().manual_seed(7)
    for index in range(100):
        magnitude = {0: 1e-3, 1: 1e3}.get(index % 3, 1.0)
        skip = torch.randn(2, 96, 16, 16, generator=generator) * magnitude
        decoded = torch.randn(2, 96, 16, 16, generator=generator) * magnitude
        out = module(skip, decoded)
        assert torch.isfinite(out).all()


def test_contains_no_forbidden_layers() -> None:
    """Part 7 forbids BatchNorm anywhere in the network."""
    for module in GatedFuse(96).modules():
        assert not isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))


# ------------------------------------------------------- GPU / export readiness
def test_autocast_produces_finite_output() -> None:
    module = GatedFuse(96)
    skip, decoded = _inputs()
    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        out = module(skip, decoded)
    assert torch.isfinite(out.float()).all()


def test_channels_last_matches_contiguous() -> None:
    module = GatedFuse(96)
    skip, decoded = _inputs()
    reference = module(skip, decoded)
    produced = module(
        skip.to(memory_format=torch.channels_last),
        decoded.to(memory_format=torch.channels_last),
    )
    assert torch.allclose(produced, reference, atol=1e-5)


def test_torchscript_matches_eager() -> None:
    module = GatedFuse(96).eval()
    skip, decoded = _inputs()
    scripted = torch.jit.script(module)
    with torch.no_grad():
        assert torch.allclose(scripted(skip, decoded), module(skip, decoded), atol=1e-5)


def test_torch_compile_matches_eager() -> None:
    """``backend="eager"``: exercises Dynamo tracing without Inductor's C++ codegen,
    which needs an MSVC toolchain this development host lacks."""
    module = GatedFuse(96).eval()
    skip, decoded = _inputs()
    compiled = torch.compile(module, backend="eager", fullgraph=True)
    with torch.no_grad():
        assert torch.allclose(compiled(skip, decoded), module(skip, decoded), atol=1e-5)


def test_onnx_export_matches_eager(tmp_path) -> None:
    onnxruntime = pytest.importorskip("onnxruntime")
    module = GatedFuse(96).eval()
    skip, decoded = _inputs(batch=1)
    path = tmp_path / "gated_fuse.onnx"

    torch.onnx.export(
        module, (skip, decoded), str(path), opset_version=18,
        input_names=["skip", "decoded"], output_names=["fused"],
    )
    session = onnxruntime.InferenceSession(
        str(path), providers=["CPUExecutionProvider"]
    )
    produced = session.run(None, {"skip": skip.numpy(), "decoded": decoded.numpy()})[0]
    with torch.no_grad():
        expected = module(skip, decoded)
    assert abs(produced - expected.numpy()).max() < 1e-3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_matches_cpu() -> None:
    module = GatedFuse(96)
    skip, decoded = _inputs()
    expected = module(skip, decoded)
    produced = module.cuda()(skip.cuda(), decoded.cuda()).cpu()
    assert torch.allclose(expected, produced, atol=1e-4)


# ------------------------------------------------------------------- integration
def test_build_fusion_selects_the_gated_module() -> None:
    """The existing factory must switch implementations; no interface change."""
    gated = build_sparc_config("sparc-base", use_gated_fusion=True)
    concat = build_sparc_config("sparc-base", use_gated_fusion=False)
    assert isinstance(build_fusion(gated, 96), GatedFuse)
    assert isinstance(build_fusion(concat, 96), ConcatFusion)
    assert build_fusion(gated, 96).reduction == gated.fusion_reduction


def test_sparcnet_uses_gated_fusion_at_both_decoder_stages() -> None:
    model = SPARCNet(build_sparc_config("sparc-base", use_attention=False))
    fusions = [stage.fusion for stage in model.decoder.stages]
    assert all(isinstance(f, GatedFuse) for f in fusions)
    assert [f.channels for f in fusions] == [96, 48]
    assert [sum(p.numel() for p in f.parameters()) for f in fusions] == [23_256, 5_868]


def test_sparcnet_forward_is_unchanged_in_shape_and_range() -> None:
    model = SPARCNet(build_sparc_config("sparc-base", use_attention=False)).eval()
    y = torch.rand(2, 1, 128, 128)
    with torch.no_grad():
        out = model(y)
    assert out.shape == (2, 1, 256, 256)
    assert out.min() >= 0.0 and out.max() <= 1.0


# --------------------------------------------------- regression vs concat fusion
def test_gated_fusion_costs_the_expected_extra_parameters() -> None:
    """GatedFuse vs ConcatFusion, the ablation A3 arms."""
    for channels in (96, 48):
        gated = sum(p.numel() for p in GatedFuse(channels).parameters())
        concat = sum(p.numel() for p in ConcatFusion(channels).parameters())
        # Concat is just the 2C -> C projection; the gate MLP is the difference.
        assert concat == 2 * channels * channels + channels
        assert gated - concat == 2 * (channels * (channels // 4)) + channels // 4 + channels


def test_model_level_parameter_delta_matches_the_contract() -> None:
    """Swapping the fusion must move the model total by exactly 23,256 + 5,868 - concat."""
    gated_model = SPARCNet(build_sparc_config("sparc-base", use_attention=False))
    concat_model = SPARCNet(
        build_sparc_config("sparc-base", use_attention=False, use_gated_fusion=False)
    )
    gated = sum(p.numel() for p in gated_model.parameters())
    concat = sum(p.numel() for p in concat_model.parameters())

    expected = (23_256 + 5_868) - (
        sum(p.numel() for p in ConcatFusion(96).parameters())
        + sum(p.numel() for p in ConcatFusion(48).parameters())
    )
    assert gated - concat == expected


def test_gated_fusion_flop_overhead_is_negligible() -> None:
    """The gate adds two 1x1 convs on a pooled (B,C,1,1) tensor, so almost no MACs."""
    sample = torch.rand(1, 1, 128, 128)
    gated = measure_complexity(
        SPARCNet(build_sparc_config("sparc-base", use_attention=False)), sample
    )
    concat = measure_complexity(
        SPARCNet(
            build_sparc_config("sparc-base", use_attention=False, use_gated_fusion=False)
        ),
        sample,
    )
    overhead = (gated.macs - concat.macs) / concat.macs
    assert overhead < 0.01, f"gate added {overhead:.3%} MACs"


# -------------------------------------------------------------- MACs and memory
CONTRACT_MMAC = {96: 18.88, 48: 18.88}
"""Contract Part 3 stages 15 and 20, at (C=96, 32x32) and (C=48, 64x64) respectively.

Both land on the same figure because ``2C^2*H*W`` is invariant when C halves and H*W
quadruples — the two fusion sites cost the same despite their different shapes.
"""


@pytest.mark.parametrize("channels,size", [(96, 32), (48, 64)])
def test_macs_are_within_five_percent_of_the_contract(channels: int, size: int) -> None:
    """Part 9: MACs within +/-5 % of Part 3, measured with ``FlopCounterMode``."""
    report = measure_complexity(GatedFuse(channels), _inputs(1, channels, size))
    measured = report.macs / 1e6
    expected = CONTRACT_MMAC[channels]
    assert abs(measured - expected) / expected < 0.05, (
        f"C={channels}: measured {measured:.2f} MMAC vs contract {expected} MMAC"
    )


@pytest.mark.parametrize("channels,size,expected_mb", [(96, 32, 0.590), (48, 64, 1.180)])
def test_activation_memory_is_within_fifteen_percent_of_the_contract(
    channels: int, size: int, expected_mb: float
) -> None:
    """Part 9: measured activation bytes within +/-15 % of Part 3.

    Part 2.7 budgets ``3*C*H*W`` elements, which is the concatenated tensor
    (``2*C*H*W``) plus the output (``C*H*W``). Those are exactly the tensors this
    module *allocates*; ``skip`` and ``decoded`` belong to the stages that produced
    them and the two ``(B,C,1,1)`` gate tensors are negligible.

    So the measurement is: tensors autograd retains for backward, minus the module's
    own inputs and parameters, plus the output. Part 3 quotes fp16 decimal megabytes.
    """
    module = GatedFuse(channels)
    skip, decoded = _inputs(1, channels, size)
    skip.requires_grad_(True)
    decoded.requires_grad_(True)

    own_tensors = {id(skip), id(decoded), *(id(p) for p in module.parameters())}
    saved: dict[int, int] = {}

    def pack(tensor: torch.Tensor) -> torch.Tensor:
        if id(tensor) not in own_tensors:
            saved[id(tensor)] = tensor.numel()
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = module(skip, decoded)
        out.sum().backward()

    elements = sum(saved.values()) + out.numel()
    measured_mb = elements * 2 / 1e6  # fp16, decimal MB as in Part 3
    assert abs(measured_mb - expected_mb) / expected_mb < 0.15, (
        f"C={channels}: measured {measured_mb:.3f} MB vs contract {expected_mb} MB"
    )


# ------------------------------------------- full-model integration and recovery
def test_full_model_gradients_reach_both_fusion_modules() -> None:
    """The gate must train end-to-end, not just in isolation.

    A gate that receives no gradient inside the assembled network would still pass
    every isolated test while silently degenerating to a constant mix.
    """
    model = SPARCNet(build_sparc_config("sparc-base", use_attention=False))
    model(torch.rand(2, 1, 128, 128)).pow(2).mean().backward()

    for stage_index, stage in enumerate(model.decoder.stages):
        for name, param in stage.fusion.named_parameters():
            assert param.grad is not None, f"stage {stage_index} {name}: no gradient"
            assert torch.isfinite(param.grad).all(), f"stage {stage_index} {name}: non-finite"
            assert torch.count_nonzero(param.grad) > 0, f"stage {stage_index} {name}: all zero"


def test_gate_adapts_to_the_input() -> None:
    """Contract Part 2.7 conditions the gate on the joint statistics of both inputs.

    A gate that ignored its inputs — a bug that leaves every other test green, since a
    constant gate is still a valid convex combination — would produce the same value
    for every image. This asserts the gate genuinely varies across differing inputs and
    across channels.
    """
    module = GatedFuse(96)
    generator = torch.Generator().manual_seed(11)
    skip = torch.randn(4, 96, 32, 32, generator=generator)
    decoded = torch.randn(4, 96, 32, 32, generator=generator)
    # Give each batch element a distinct scale, the way differing noise levels would.
    skip = skip * torch.tensor([0.1, 0.5, 1.0, 2.0]).view(4, 1, 1, 1)

    gate = module.gate(skip, decoded)
    assert gate.std(dim=0).max() > 1e-4, "gate does not vary across images"
    assert gate.std(dim=1).max() > 1e-4, "gate does not vary across channels"


def test_state_dict_round_trips_and_concat_checkpoints_are_detected() -> None:
    """Checkpoint compatibility, both directions.

    A GatedFuse checkpoint must reload exactly. A *concat* checkpoint must not load
    silently into a gated model — the gate weights would stay at their random
    initialisation while the run reported a successful resume.
    """
    model = SPARCNet(build_sparc_config("sparc-base", use_attention=False))
    reloaded = SPARCNet(build_sparc_config("sparc-base", use_attention=False))
    reloaded.load_state_dict(model.state_dict())

    sample = torch.rand(1, 1, 128, 128)
    model.eval()
    reloaded.eval()
    with torch.no_grad():
        assert torch.allclose(model(sample), reloaded(sample), atol=1e-6)

    concat_model = SPARCNet(
        build_sparc_config("sparc-base", use_attention=False, use_gated_fusion=False)
    )
    with pytest.raises(RuntimeError, match="Missing key"):
        model.load_state_dict(concat_model.state_dict())


def test_composite_loss_and_noise_head_still_work_with_gated_fusion() -> None:
    """Phase 4.9 and 4.10 must be unaffected by the fusion swap."""
    from losses import CompositeLoss

    model = SPARCNet(build_sparc_config("sparc-base", use_attention=False))
    criterion = CompositeLoss()  # all six terms; 256 px clears the MS-SSIM 161 px floor
    batch = {"lr": torch.rand(2, 1, 128, 128), "gt": torch.rand(2, 1, 256, 256)}

    output = model.forward_with_aux(batch["lr"])
    total, terms = criterion(output, batch)

    assert output.sigma.shape == (2, 1, 128, 128)
    assert torch.isfinite(total) and total > 0
    assert "noise" in terms and torch.isfinite(torch.tensor(terms["noise"]))

    total.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for stage in model.decoder.stages
        for p in stage.fusion.parameters()
    )
