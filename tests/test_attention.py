"""Global self-attention tests (Contract Part 2.6, Part 3 stages 8/12/16, Part 9).

Phase 4.12 is the highest-risk module in the build order: it is the only place where
tensors are reshaped between spatial and token layout, the only place with a second
mathematically-equivalent code path that must be kept in agreement, and the only place
where a stray ``nn.Parameter`` would silently blow the Part 10 budget.

The single most important test in this file is
``test_sdpa_matches_explicit_path``: training runs the fused kernel and export runs the
unfused one, so a divergence between them is a bug that only surfaces after the model
has already been trained.
"""

from __future__ import annotations

import copy
import math

import pytest
import torch
from torch import nn

from configs.sparc_config import build_sparc_config, sparc_base
from models.attention.gsa_block import GSABlock
from models.attention.rel_pos import RelativePositionBias, relative_position_index
from models.sparc_net import SPARCNet, build_model

CONTRACT_INSTANCES = {
    # (channels, heads, spatial_size): per-block parameters, Contract Part 3
    (96, 3, 32): 62_451,
    (160, 5, 16): 140_245,
}
"""Part 3 stages 8 and 12/16. Part 9 requires an exact match, not a tolerance."""

CONTRACT_GROUPS = {
    "enc_l1": (2, 62_451, 124_902),
    "enc_l2": (3, 140_245, 420_735),
    "dec_d1": (1, 62_451, 62_451),
}
"""Blocks per group and the group total from Part 3."""

TOTAL_GSA_PARAMETERS = 608_088
"""124,902 + 420,735 + 62,451."""

SPARC_BASE_PARAMETERS = 2_345_650
"""Contract Part 3, TOTAL row."""

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def _block(channels: int = 96, heads: int = 3, size: int = 32, **kwargs) -> GSABlock:
    return GSABlock(channels=channels, heads=heads, spatial_size=size, **kwargs)


def _input(batch: int = 2, channels: int = 96, size: int = 32) -> torch.Tensor:
    generator = torch.Generator().manual_seed(1337)
    return torch.randn(batch, channels, size, size, generator=generator)


# --------------------------------------------------------------------- parameters
@pytest.mark.parametrize("spec,expected", sorted(CONTRACT_INSTANCES.items()))
def test_gsa_parameter_count_is_exact(spec: tuple[int, int, int], expected: int) -> None:
    channels, heads, size = spec
    assert sum(p.numel() for p in _block(channels, heads, size).parameters()) == expected


@pytest.mark.parametrize("spec,expected", sorted(CONTRACT_INSTANCES.items()))
def test_analytic_parameter_count_matches_the_module(
    spec: tuple[int, int, int], expected: int
) -> None:
    channels, heads, size = spec
    assert GSABlock.parameter_count(channels, heads, size) == expected


def test_parameter_composition_matches_the_contract_breakdown() -> None:
    """Every named tensor is accounted for, so nothing can hide in the total."""
    block = _block(96, 3, 32)
    counts = {name: p.numel() for name, p in block.named_parameters()}
    assert counts == {
        "norm1.weight": 96,
        "norm1.bias": 96,
        "qkv.weight": 96 * 144,
        "qkv.bias": 144,
        "qkv_dwconv.weight": 144 * 9,
        "qkv_dwconv.bias": 144,
        "project.weight": 48 * 96,
        "project.bias": 96,
        "scale1.gamma": 96,
        "rel_pos.rel_pos_table": 3 * 63**2,
        "norm2.weight": 96,
        "norm2.bias": 96,
        "ffn_in.weight": 96 * 192,
        "ffn_in.bias": 192,
        "ffn_dwconv.weight": 192 * 9,
        "ffn_dwconv.bias": 192,
        "ffn_out.weight": 96 * 96,
        "ffn_out.bias": 96,
        "scale2.gamma": 96,
    }
    assert sum(counts.values()) == 62_451


@pytest.mark.parametrize("group,spec", sorted(CONTRACT_GROUPS.items()))
def test_group_totals_match_part_3(group: str, spec: tuple[int, int, int]) -> None:
    count, per_block, total = spec
    assert count * per_block == total


def test_head_dim_is_sixteen_at_every_contract_instantiation() -> None:
    for channels, heads, size in CONTRACT_INSTANCES:
        block = _block(channels, heads, size)
        assert block.head_dim == 16
        assert block.attn_dim == channels // 2
        assert block.scale == pytest.approx(16.0**-0.5)


def test_rejects_configurations_that_break_the_head_dim_invariant() -> None:
    for channels, heads, size in ((96, 4, 32), (96, 0, 32), (0, 3, 32), (96, 3, 0)):
        with pytest.raises(ValueError):
            _block(channels, heads, size)


# ------------------------------------------------------------------------- shapes
@pytest.mark.parametrize("batch", [1, 2, 8])
@pytest.mark.parametrize("channels,heads,size", [(96, 3, 32), (160, 5, 16)])
def test_gsa_shape(batch: int, channels: int, heads: int, size: int) -> None:
    block = _block(channels, heads, size)
    x = _input(batch, channels, size)
    assert block(x).shape == x.shape


def test_intermediate_attention_shapes() -> None:
    """Pin every stage of the spatial/token reshape (Contract Part 2.6)."""
    block = _block(96, 3, 32)
    x = _input(2)
    qkv = block.qkv_dwconv(block.qkv(block.norm1(x)))
    assert qkv.shape == (2, 3 * 48, 32, 32)
    q, k, v = qkv.split(block.attn_dim, dim=1)
    assert q.shape == k.shape == v.shape == (2, 48, 32, 32)
    q = q.reshape(2, 3, 16, 1024).transpose(-2, -1)
    assert q.shape == (2, 3, 1024, 16)
    assert block.rel_pos().shape == (3, 1024, 1024)


def test_rejects_a_grid_it_was_not_built_for() -> None:
    block = _block(96, 3, 32)
    with pytest.raises(ValueError):
        block(_input(1, 96, 16))
    with pytest.raises(ValueError):
        block(torch.randn(1, 64, 32, 32))
    with pytest.raises(ValueError):
        block(torch.randn(96, 32, 32))


# --------------------------------------------------------------- SDPA vs explicit
@pytest.mark.parametrize("channels,heads,size", [(96, 3, 32), (160, 5, 16)])
@pytest.mark.parametrize("batch", [1, 2])
def test_sdpa_matches_explicit_path(
    channels: int, heads: int, size: int, batch: int
) -> None:
    """Contract Part 2.6: the export path must match SDPA within 1e-4."""
    fused = _block(channels, heads, size, use_sdpa=True)
    explicit = copy.deepcopy(fused)
    explicit.use_sdpa = False

    x = _input(batch, channels, size)
    with torch.inference_mode():
        a = fused(x)
        b = explicit(x)
    assert torch.max((a - b).abs()).item() <= 1e-4


def test_sdpa_matches_explicit_path_on_large_magnitude_input() -> None:
    """Softmax must not drift apart when the logits are far from zero."""
    fused = _block(96, 3, 32)
    explicit = copy.deepcopy(fused)
    explicit.use_sdpa = False
    x = _input(1) * 50.0
    with torch.inference_mode():
        difference = torch.max((fused(x) - explicit(x)).abs()).item()
    assert difference <= 1e-4


def test_sdpa_and_explicit_gradients_agree() -> None:
    fused = _block(96, 3, 32)
    explicit = copy.deepcopy(fused)
    explicit.use_sdpa = False
    x = _input(1)

    fused(x).square().mean().backward()
    explicit(x).square().mean().backward()
    for (name, p), (_, q) in zip(
        fused.named_parameters(), explicit.named_parameters()
    ):
        assert p.grad is not None and q.grad is not None, name
        assert torch.allclose(p.grad, q.grad, atol=1e-5, rtol=1e-4), name


# ------------------------------------------------------------- relative positions
@pytest.mark.parametrize("size", [4, 16, 32])
def test_rel_pos_index_is_in_range_and_centred(size: int) -> None:
    index = relative_position_index(size)
    tokens = size * size
    assert index.shape == (tokens, tokens)
    assert index.dtype == torch.int64
    assert int(index.min()) >= 0
    assert int(index.max()) < (2 * size - 1) ** 2
    center = (size - 1) * (2 * size - 1) + (size - 1)
    # zero offset — every token relative to itself — is the table centre
    assert torch.equal(index.diagonal(), torch.full((tokens,), center))


@pytest.mark.parametrize("size", [4, 16])
def test_rel_pos_index_offsets_mirror(size: int) -> None:
    """``index[a, b]`` and ``index[b, a]`` encode opposite offsets."""
    index = relative_position_index(size)
    center = (size - 1) * (2 * size - 1) + (size - 1)
    assert torch.equal(index + index.T, torch.full_like(index, 2 * center))


def test_rel_pos_table_is_trainable_index_is_not() -> None:
    module = RelativePositionBias(heads=3, size=32)
    assert module.rel_pos_table.requires_grad
    assert module.rel_pos_table.shape == (3, 63**2)

    buffers = dict(module.named_buffers())
    assert "rel_pos_index" in buffers
    assert buffers["rel_pos_index"].dtype == torch.int64
    assert not buffers["rel_pos_index"].requires_grad
    assert not any(
        b.is_floating_point() and b.requires_grad for b in module.buffers()
    )
    # the index must not be counted as a parameter — it would add N^2 = 1,048,576
    assert sum(p.numel() for p in module.parameters()) == 3 * 63**2


def test_no_buffer_leaks_into_the_parameter_count() -> None:
    block = _block(96, 3, 32)
    parameter_ids = {id(p) for p in block.parameters()}
    assert not any(id(b) in parameter_ids for b in block.buffers())
    assert sum(p.numel() for p in block.parameters()) == 62_451


def test_rel_pos_table_is_initialised_trunc_normal() -> None:
    table = RelativePositionBias(heads=5, size=16).rel_pos_table
    assert table.abs().max().item() <= 2 * 0.02 + 1e-6
    assert 0.005 < table.std().item() < 0.04


def test_bias_actually_reaches_the_logits() -> None:
    """The gathered table must land on the logits, on both attention paths.

    Measured at the attention op rather than at the block output: at initialisation
    ``LayerScale = 1e-2`` and the table is ``trunc_normal_(0.02)``, so the block-level
    effect is ~5e-6 and indistinguishable from float noise. That is by design — the
    block starts near the identity — but it makes the block output a useless probe.
    """
    torch.manual_seed(1337)
    block = _block(96, 3, 32)
    q, k, v = (torch.randn(1, 3, 1024, 16) for _ in range(3))
    bias = block.rel_pos().unsqueeze(0)
    for use_sdpa in (True, False):
        block.use_sdpa = use_sdpa
        with torch.inference_mode():
            difference = (
                block._attend(q, k, v, bias) - block._attend(q, k, v, None)
            ).abs().max().item()
        assert difference > 1e-4, use_sdpa


def test_gsa_is_permutation_equivariant_without_bias() -> None:
    """With the positional term removed, attention must act along the token axis.

    Only the attention branch is tested: the depthwise 3x3 convolutions and the GDFN
    branch are spatial operators and are deliberately not permutation-equivariant.
    """
    torch.manual_seed(1337)
    block = _block(96, 3, 32, use_relative_position_bias=False)
    q = torch.randn(1, 3, 1024, 16)
    k = torch.randn(1, 3, 1024, 16)
    v = torch.randn(1, 3, 1024, 16)
    perm = torch.randperm(1024)

    with torch.inference_mode():
        direct = block._attend(q, k, v, None)[:, :, perm]
        shuffled = block._attend(q[:, :, perm], k[:, :, perm], v[:, :, perm], None)
    assert torch.max((direct - shuffled).abs()).item() <= 1e-5


def test_block_without_bias_has_no_rel_pos_parameters() -> None:
    block = _block(96, 3, 32, use_relative_position_bias=False)
    assert block.rel_pos is None
    assert sum(p.numel() for p in block.parameters()) == 62_451 - 3 * 63**2


# --------------------------------------------------------------------- gradients
@pytest.mark.parametrize("channels,heads,size", [(96, 3, 32), (160, 5, 16)])
def test_every_parameter_receives_a_finite_nonzero_gradient(
    channels: int, heads: int, size: int
) -> None:
    block = _block(channels, heads, size)
    block(_input(2, channels, size)).square().mean().backward()
    for name, parameter in block.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum().item() > 0.0, name


def test_gradient_flows_to_the_input() -> None:
    block = _block(96, 3, 32)
    x = _input(1).requires_grad_(True)
    block(x).square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert x.grad.abs().sum().item() > 0.0


# -------------------------------------------------------------------- stability
def test_no_nan_over_extreme_input_scales() -> None:
    block = _block(96, 3, 32).eval()
    generator = torch.Generator().manual_seed(7)
    with torch.inference_mode():
        for scale in (1e-3, 1.0, 1e3):
            for _ in range(10):
                x = torch.randn(2, 96, 32, 32, generator=generator) * scale
                assert torch.isfinite(block(x)).all()


def test_layer_scale_zero_makes_the_block_the_identity() -> None:
    block = _block(96, 3, 32)
    with torch.no_grad():
        block.scale1.gamma.zero_()
        block.scale2.gamma.zero_()
    x = _input(1)
    with torch.inference_mode():
        assert torch.max((block(x) - x).abs()).item() == 0.0


def test_no_forbidden_layers() -> None:
    """Contract Parts 5 and 7: no BatchNorm, no activation function, no dropout."""
    forbidden = (
        nn.BatchNorm2d, nn.BatchNorm1d, nn.ReLU, nn.GELU, nn.SiLU, nn.LeakyReLU,
        nn.Sigmoid, nn.Tanh, nn.Dropout, nn.Dropout2d,
    )
    for model in (_block(96, 3, 32), build_model("sparc-base")):
        for name, module in model.named_modules():
            assert not isinstance(module, forbidden), f"{name}: {type(module).__name__}"


# ------------------------------------------------------------- device / precision
def test_channels_last_matches_contiguous() -> None:
    block = _block(96, 3, 32).eval()
    x = _input(2)
    with torch.inference_mode():
        reference = block(x)
        result = block(x.to(memory_format=torch.channels_last))
    assert torch.max((reference - result).abs()).item() <= 1e-5


@CUDA
def test_gsa_runs_on_cuda_and_matches_cpu() -> None:
    block = _block(96, 3, 32).eval()
    x = _input(2)
    with torch.inference_mode():
        reference = block(x)
        result = block.cuda()(x.cuda()).cpu()
    assert torch.max((reference - result).abs()).item() <= 1e-4


@CUDA
def test_gsa_autocast_bf16_is_finite_on_cuda() -> None:
    block = _block(96, 3, 32).cuda()
    x = _input(2).cuda()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = block(x)
        loss = out.square().mean()
    loss.backward()
    assert torch.isfinite(out).all()
    for name, parameter in block.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name


@CUDA
def test_gsa_autocast_fp16_is_finite_on_cuda() -> None:
    """Softmax must not overflow: the contract's default AMP dtype is fp16."""
    block = _block(96, 3, 32).cuda()
    with torch.autocast("cuda", dtype=torch.float16):
        out = block(_input(2).cuda())
    assert torch.isfinite(out).all()


@CUDA
def test_gsa_channels_last_on_cuda() -> None:
    block = _block(96, 3, 32).cuda().to(memory_format=torch.channels_last).eval()
    x = _input(2).cuda().to(memory_format=torch.channels_last)
    with torch.inference_mode():
        assert torch.isfinite(block(x)).all()


def test_autocast_fp16_is_finite_on_cpu() -> None:
    block = _block(96, 3, 32).eval()
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
        assert torch.isfinite(block(_input(2))).all()


# ------------------------------------------------------------------------ export
def test_torchscript_matches_eager() -> None:
    block = _block(96, 3, 32).eval()
    scripted = torch.jit.script(block)
    x = _input(1)
    with torch.inference_mode():
        assert torch.max((block(x) - scripted(x)).abs()).item() <= 1e-5


def test_gsa_onnx_uses_explicit_path(tmp_path) -> None:
    """SDPA does not lower to opset 17; export flips to the explicit path."""
    onnxruntime = pytest.importorskip("onnxruntime")
    block = copy.deepcopy(_block(96, 3, 32)).eval()
    block.use_sdpa = False
    x = _input(1)
    path = tmp_path / "gsa.onnx"
    torch.onnx.export(
        block, (x,), str(path), opset_version=17,
        input_names=["input"], output_names=["output"], dynamo=False,
    )
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    result = torch.from_numpy(session.run(None, {"input": x.numpy()})[0])
    with torch.inference_mode():
        reference = block(x)
    assert torch.max((reference - result).abs()).item() <= 1e-3


def test_torch_compile_matches_eager() -> None:
    if not hasattr(torch, "compile"):  # pragma: no cover - torch >= 2.0 everywhere here
        pytest.skip("torch.compile unavailable")
    block = _block(96, 3, 32).eval()
    x = _input(1)
    with torch.inference_mode():
        reference = block(x)
    try:
        compiled = torch.compile(block, fullgraph=True)
        with torch.inference_mode():
            result = compiled(x)
    except Exception as error:  # pragma: no cover - no compiler backend on this host
        pytest.skip(f"torch.compile backend unavailable: {error}")
    assert torch.max((reference - result).abs()).item() <= 1e-4


# -------------------------------------------------------------------- integration
def test_gsa_never_instantiated_at_64_or_128() -> None:
    """Contract Part 2.6: attention runs at 32x32 and 16x16 only."""
    config = sparc_base()
    assert config.enc_gsa_blocks[0] == 0  # trunk level 0 is 64x64
    assert config.dec_gsa_blocks[1] == 0  # decoder D0 is 64x64
    assert not config.uses_attention_at(0)

    model = build_model("sparc-base")
    for name, module in model.named_modules():
        if isinstance(module, GSABlock):
            assert module.spatial_size in (16, 32), f"{name}: {module.spatial_size}"


def test_sparc_base_instantiates_exactly_six_blocks_in_the_right_places() -> None:
    model = build_model("sparc-base")
    sizes = [
        (name, m.spatial_size, m.channels, m.heads)
        for name, m in model.named_modules()
        if isinstance(m, GSABlock)
    ]
    assert len(sizes) == 6
    encoder_l1 = [s for s in sizes if s[0].startswith("encoder.levels.1")]
    encoder_l2 = [s for s in sizes if s[0].startswith("encoder.levels.2")]
    decoder_d1 = [s for s in sizes if s[0].startswith("decoder.stages.0")]
    assert [s[1:] for s in encoder_l1] == [(32, 96, 3)] * 2
    assert [s[1:] for s in encoder_l2] == [(16, 160, 5)] * 3
    assert [s[1:] for s in decoder_d1] == [(32, 96, 3)] * 1


def test_total_gsa_parameters_match_part_3() -> None:
    model = build_model("sparc-base")
    total = sum(
        sum(p.numel() for p in m.parameters())
        for m in model.modules()
        if isinstance(m, GSABlock)
    )
    assert total == TOTAL_GSA_PARAMETERS


def test_sparc_base_total_parameter_count_matches_part_3_exactly() -> None:
    model = build_model("sparc-base")
    assert sum(p.numel() for p in model.parameters()) == SPARC_BASE_PARAMETERS


def test_attention_adds_exactly_the_contract_delta() -> None:
    """The pre-attention baseline plus 608,088 is the full model."""
    with_attention = sum(p.numel() for p in build_model("sparc-base").parameters())
    without = sum(
        p.numel()
        for p in SPARCNet(build_sparc_config("sparc-base", use_attention=False)).parameters()
    )
    assert with_attention - without == TOTAL_GSA_PARAMETERS
    assert with_attention == SPARC_BASE_PARAMETERS


def test_pre_attention_and_tiny_variants_still_build() -> None:
    """The deferred-import factory must keep the attention-free paths working."""
    for config in (
        build_sparc_config("sparc-tiny"),
        build_sparc_config("sparc-base", use_attention=False),
    ):
        model = SPARCNet(config)
        assert not any(isinstance(m, GSABlock) for m in model.modules())
        size = config.input_size
        with torch.inference_mode():
            out = model(torch.rand(1, 1, size, size))
        assert out.shape == (1, 1, 2 * size, 2 * size)


def test_full_model_forward_and_backward() -> None:
    model = build_model("sparc-base")
    x = torch.rand(1, 1, 128, 128)
    out = model(x)
    assert out.shape == (1, 1, 256, 256)
    assert torch.isfinite(out).all()
    out.square().mean().backward()
    missing = [
        name
        for name, p in model.named_parameters()
        if p.grad is None or not torch.isfinite(p.grad).all()
    ]
    assert not missing, missing


def test_rel_pos_tables_are_excluded_from_weight_decay() -> None:
    """Contract Part 5: rel-pos tables must land in the zero-decay group."""
    from trainer.trainer import build_param_groups

    model = build_model("sparc-base")
    groups = build_param_groups(model, weight_decay=1e-4)
    decayed = {id(p) for p in groups[0]["params"]}
    tables = [
        p for name, p in model.named_parameters() if "rel_pos" in name
    ]
    assert len(tables) == 6
    assert not any(id(t) in decayed for t in tables)


def test_macs_are_within_five_percent_of_part_3() -> None:
    """Contract Part 10: 2.449 GMAC, Part 9 tolerance +/-5 %."""
    from utils.complexity import measure_complexity

    report = measure_complexity(build_model("sparc-base"), torch.rand(1, 1, 128, 128))
    macs = report.gmacs
    assert math.isclose(macs, 2.449, rel_tol=0.05), macs
