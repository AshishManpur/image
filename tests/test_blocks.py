"""Core building block tests (Contract Part 8, step 5; Part 9)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from models.blocks.layer_norm import LayerNorm2d
from models.blocks.naf_block import NAFBlock
from models.blocks.simple_gate import LayerScale, SimpleGate
from utils.complexity import count_parameters, measure_complexity

# Contract Part 3: NAF block parameter counts at each width used by SPARC-Base.
CONTRACT_NAF_PARAMS = {32: 8_224, 48: 17_712, 96: 67_680, 160: 184_480}


# ------------------------------------------------------------------ LayerNorm2d
def test_layernorm_normalises_channels() -> None:
    norm = LayerNorm2d(16, affine=False)
    out = norm(torch.randn(4, 16, 8, 8) * 5.0 + 3.0)
    assert torch.allclose(out.mean(dim=1), torch.zeros(4, 8, 8), atol=1e-4)
    assert torch.allclose(out.std(dim=1, unbiased=False), torch.ones(4, 8, 8), atol=1e-3)


def test_layernorm_is_identity_at_init() -> None:
    norm = LayerNorm2d(8)
    x = torch.randn(2, 8, 4, 4)
    assert torch.allclose(norm(x), LayerNorm2d(8, affine=False)(x), atol=1e-6)


def test_layernorm_parameter_count() -> None:
    assert count_parameters(LayerNorm2d(48))[0] == 2 * 48
    assert count_parameters(LayerNorm2d(48, affine=False))[0] == 0


def test_layernorm_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        LayerNorm2d(0)
    with pytest.raises(ValueError):
        LayerNorm2d(8)(torch.randn(2, 8, 4))
    with pytest.raises(ValueError):
        LayerNorm2d(8)(torch.randn(2, 4, 4, 4))


def test_layernorm_is_stable_across_magnitudes() -> None:
    norm = LayerNorm2d(8)
    for scale in (1e-3, 1.0, 1e3):
        out = norm(torch.randn(2, 8, 8, 8) * scale)
        assert torch.isfinite(out).all()


# ------------------------------------------------------------------- SimpleGate
def test_simple_gate_halves_channels_and_multiplies() -> None:
    x = torch.arange(16, dtype=torch.float32).reshape(1, 4, 2, 2)
    out = SimpleGate()(x)
    assert out.shape == (1, 2, 2, 2)
    assert torch.allclose(out, x[:, :2] * x[:, 2:])


def test_simple_gate_rejects_odd_channels() -> None:
    with pytest.raises(ValueError):
        SimpleGate()(torch.randn(1, 3, 4, 4))


def test_simple_gate_is_parameter_free() -> None:
    assert count_parameters(SimpleGate())[0] == 0


# ------------------------------------------------------------------- LayerScale
def test_layer_scale_initialises_to_contract_value() -> None:
    scale = LayerScale(16, 1e-2)
    assert torch.allclose(scale.gamma, torch.full((1, 16, 1, 1), 1e-2))
    assert count_parameters(scale)[0] == 16


def test_layer_scale_multiplies_channelwise() -> None:
    scale = LayerScale(4, 0.5)
    x = torch.ones(2, 4, 3, 3)
    assert torch.allclose(scale(x), x * 0.5)


def test_layer_scale_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        LayerScale(0)
    with pytest.raises(ValueError):
        LayerScale(8)(torch.randn(1, 4, 4, 4))


# --------------------------------------------------------------------- NAFBlock
@pytest.mark.parametrize("channels", sorted(CONTRACT_NAF_PARAMS))
def test_naf_parameter_count_matches_contract_exactly(channels: int) -> None:
    """Contract Part 9: parameter counts must match exactly, not within a tolerance."""
    block = NAFBlock(channels)
    assert count_parameters(block)[0] == CONTRACT_NAF_PARAMS[channels]
    assert NAFBlock.parameter_count(channels) == CONTRACT_NAF_PARAMS[channels]


@pytest.mark.parametrize("batch", [1, 2, 8])
def test_naf_preserves_shape(batch: int) -> None:
    block = NAFBlock(48)
    x = torch.randn(batch, 48, 16, 16)
    assert block(x).shape == x.shape


def test_naf_is_identity_when_layer_scales_are_zero() -> None:
    """Contract Part 9 invariant."""
    block = NAFBlock(16)
    with torch.no_grad():
        block.scale1.gamma.zero_()
        block.scale2.gamma.zero_()
    x = torch.randn(2, 16, 8, 8)
    assert torch.allclose(block(x), x, atol=1e-6)


def test_naf_is_near_identity_at_initialisation() -> None:
    """LayerScale init of 1e-2 keeps the block close to identity before training."""
    block = NAFBlock(32)
    x = torch.randn(2, 32, 16, 16)
    relative = (block(x) - x).norm() / x.norm()
    assert relative < 0.5


def test_naf_gradients_reach_every_parameter() -> None:
    block = NAFBlock(32)
    block(torch.randn(2, 32, 16, 16)).pow(2).mean().backward()
    for name, param in block.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name
        assert param.grad.abs().sum().item() > 0.0, name


def test_naf_is_numerically_stable_across_magnitudes() -> None:
    block = NAFBlock(16)
    for scale in (1e-3, 1.0, 1e3):
        out = block(torch.randn(2, 16, 16, 16) * scale)
        assert torch.isfinite(out).all()


def test_naf_no_nan_over_many_batches() -> None:
    block = NAFBlock(16)
    for _ in range(100):
        out = block(torch.randn(2, 16, 8, 8))
        assert torch.isfinite(out).all()


def test_naf_macs_match_contract() -> None:
    """Contract Part 3: NAF at C=48, 64x64 is 60.16 MMAC."""
    report = measure_complexity(NAFBlock(48), torch.randn(1, 48, 64, 64))
    expected = 60.16e6
    assert abs(report.macs - expected) / expected < 0.05


def test_naf_rejects_bad_configuration() -> None:
    with pytest.raises(ValueError):
        NAFBlock(0)
    with pytest.raises(ValueError):
        NAFBlock(15)  # odd channels break SimpleGate
    with pytest.raises(ValueError):
        NAFBlock(16, expansion=0)
    with pytest.raises(ValueError):
        NAFBlock(16)(torch.randn(1, 8, 4, 4))


def test_naf_contains_no_forbidden_layers() -> None:
    """Contract Part 7: BatchNorm and activation functions are forbidden."""
    forbidden = (nn.BatchNorm2d, nn.ReLU, nn.GELU, nn.SiLU, nn.LeakyReLU, nn.Dropout)
    assert not any(isinstance(m, forbidden) for m in NAFBlock(32).modules())


def test_naf_torchscript_matches_eager() -> None:
    block = NAFBlock(16).eval()
    x = torch.randn(2, 16, 8, 8)
    with torch.no_grad():
        assert torch.allclose(torch.jit.script(block)(x), block(x), atol=1e-5)


def test_naf_onnx_export(tmp_path) -> None:
    onnx = pytest.importorskip("onnx")
    path = tmp_path / "naf.onnx"
    torch.onnx.export(
        NAFBlock(16).eval(), torch.randn(1, 16, 16, 16), str(path),
        opset_version=17, input_names=["x"], output_names=["y"],
    )
    onnx.checker.check_model(onnx.load(str(path)))
