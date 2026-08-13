"""Haar transform tests (Contract Part 8, step 4; Part 9)."""

from __future__ import annotations

import pytest
import torch

from models.wavelet.haar import HaarDWT, HaarIDWT, haar_dwt, haar_idwt


@pytest.mark.parametrize("shape", [(1, 1, 8, 8), (2, 3, 32, 32), (8, 48, 64, 64)])
def test_dwt_shapes(shape: tuple[int, ...]) -> None:
    out = haar_dwt(torch.randn(*shape))
    b, c, h, w = shape
    assert out.shape == (b, 4 * c, h // 2, w // 2)


@pytest.mark.parametrize("shape", [(1, 4, 4, 4), (2, 12, 16, 16), (8, 128, 64, 64)])
def test_idwt_shapes(shape: tuple[int, ...]) -> None:
    out = haar_idwt(torch.randn(*shape))
    b, c, h, w = shape
    assert out.shape == (b, c // 4, h * 2, w * 2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_perfect_reconstruction(dtype: torch.dtype) -> None:
    """Contract acceptance: IDWT(DWT(x)) == x to better than 1e-6."""
    x = torch.randn(4, 8, 64, 64, dtype=dtype)
    error = (haar_idwt(haar_dwt(x)) - x).abs().max().item()
    assert error < 1e-6, f"max reconstruction error {error:.3e}"


def test_perfect_reconstruction_in_the_other_order() -> None:
    x = torch.randn(2, 16, 32, 32)
    assert (haar_dwt(haar_idwt(x)) - x).abs().max().item() < 1e-6


def test_transform_is_orthonormal_energy_preserving() -> None:
    """Parseval: an orthonormal transform preserves the L2 norm exactly."""
    x = torch.randn(4, 6, 32, 32, dtype=torch.float64)
    assert pytest.approx(haar_dwt(x).pow(2).sum().item(), rel=1e-10) == x.pow(2).sum().item()


def test_basis_matrix_is_orthonormal() -> None:
    """Verify the 4x4 analysis matrix satisfies M M^T = I."""
    basis = []
    for index in range(4):
        block = torch.zeros(1, 1, 2, 2)
        block.view(-1)[index] = 1.0
        basis.append(haar_dwt(block).view(-1))
    matrix = torch.stack(basis, dim=1)
    assert torch.allclose(matrix @ matrix.T, torch.eye(4), atol=1e-6)


def test_ll_band_is_the_block_mean_scaled() -> None:
    x = torch.randn(2, 3, 16, 16)
    ll = haar_dwt(x)[:, :3]
    pooled = torch.nn.functional.avg_pool2d(x, 2) * 2.0
    assert torch.allclose(ll, pooled, atol=1e-6)


def test_constant_input_has_only_an_ll_band() -> None:
    """A flat image must produce zero detail coefficients."""
    out = haar_dwt(torch.full((2, 4, 16, 16), 0.37))
    assert out[:, 4:].abs().max().item() < 1e-6
    assert out[:, :4].abs().min().item() > 0.0


def test_idwt_of_ll_only_is_flat() -> None:
    """No checkerboard: LL-only sub-bands reconstruct to a constant image."""
    bands = torch.zeros(1, 4, 8, 8)
    bands[:, 0] = 1.0
    out = haar_idwt(bands)
    assert out.std().item() < 1e-6


def test_gradients_flow_through_both_transforms() -> None:
    x = torch.randn(2, 4, 16, 16, requires_grad=True)
    haar_idwt(haar_dwt(x)).pow(2).sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum().item() > 0.0


def test_transforms_are_parameter_free() -> None:
    assert sum(p.numel() for p in HaarDWT().parameters()) == 0
    assert sum(p.numel() for p in HaarIDWT().parameters()) == 0


def test_numerical_stability_across_magnitudes() -> None:
    for scale in (1e-3, 1.0, 1e3):
        x = torch.randn(2, 4, 32, 32) * scale
        out = haar_idwt(haar_dwt(x))
        assert torch.isfinite(out).all()
        assert (out - x).abs().max().item() < 1e-6 * max(scale, 1.0) * 10


def test_invalid_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        haar_dwt(torch.randn(1, 4, 8))
    with pytest.raises(ValueError):
        haar_dwt(torch.randn(1, 4, 7, 8))
    with pytest.raises(ValueError):
        haar_idwt(torch.randn(1, 6, 8, 8))
    with pytest.raises(ValueError):
        haar_idwt(torch.randn(1, 8, 8))


def test_torchscript_compatible() -> None:
    scripted_dwt = torch.jit.script(HaarDWT())
    scripted_idwt = torch.jit.script(HaarIDWT())
    x = torch.randn(2, 4, 16, 16)
    assert torch.allclose(scripted_dwt(x), haar_dwt(x), atol=1e-6)
    assert torch.allclose(scripted_idwt(scripted_dwt(x)), x, atol=1e-6)


def test_onnx_export(tmp_path) -> None:
    onnx = pytest.importorskip("onnx")

    class RoundTrip(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return haar_idwt(haar_dwt(x))

    path = tmp_path / "haar.onnx"
    torch.onnx.export(
        RoundTrip(), torch.randn(1, 4, 16, 16), str(path), opset_version=17,
        input_names=["x"], output_names=["y"],
    )
    onnx.checker.check_model(onnx.load(str(path)))
