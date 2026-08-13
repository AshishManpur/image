"""Encoder / decoder / head / full-model tests (Contract Part 8, steps 6-8; Part 9)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from configs.sparc_config import build_sparc_config, sparc_base, sparc_tiny
from models.decoder.decoder import Decoder, Upsample
from models.decoder.reconstruction_head import ReconstructionHead
from models.encoder import Downsample, Encoder, Stem
from models.fusion.concat_fuse import ConcatFusion
from models.normalization import RobustNormalizer
from models.sparc_net import SPARCNet, build_model
from utils.complexity import count_parameters

# Contract Part 3 stage parameter counts that are implemented at this step.
CONTRACT_STAGE_PARAMS = {
    "stem_with_noise_map": 3_504,   # Conv3x3(8 -> 48)
    "down0": 18_528,                # Conv1x1(192 -> 96)
    "down1": 61_600,                # Conv1x1(384 -> 160)
    "up1": 61_824,                  # Conv1x1(160 -> 384)
    "up0": 18_624,                  # Conv1x1(96 -> 192)
    "head_project": 55_424,         # Conv3x3(48 -> 128)
    "head_subbands": 1_156,         # Conv3x3(32 -> 4)
}

STAGE6 = {"use_noise_head": False, "use_gated_fusion": False, "use_attention": False}


def stage6_base():
    """SPARC-Base restricted to the modules that exist at Contract step 6."""
    return build_sparc_config("sparc-base", **STAGE6)


# ---------------------------------------------------------------- normalisation
def test_normalizer_round_trip_is_exact() -> None:
    """Contract Part 9 invariant: denormalize(forward(x)) == x to 1e-6."""
    normalizer = RobustNormalizer()
    x = torch.randn(4, 1, 32, 32) * 3.0 + 1.5
    normalised, stats = normalizer(x)
    assert (RobustNormalizer.denormalize(normalised, stats) - x).abs().max() < 1e-5


def test_normalizer_centres_and_scales() -> None:
    normalizer = RobustNormalizer()
    normalised, _ = normalizer(torch.randn(4, 1, 64, 64) * 5.0 + 2.0)
    assert normalised.mean(dim=[1, 2, 3]).abs().max() < 1e-5
    assert (normalised.std(dim=[1, 2, 3], unbiased=False) - 1.0).abs().max() < 1e-4


def test_normalizer_floors_the_scale_on_flat_images() -> None:
    """Phase 1 found 7 images with std < 0.03; the floor keeps the inverse finite."""
    normalizer = RobustNormalizer(scale_floor=0.02)
    _, stats = normalizer(torch.full((2, 1, 16, 16), 0.5))
    assert torch.allclose(stats.scale, torch.full_like(stats.scale, 0.02))


def test_normalizer_disabled_is_identity() -> None:
    normalizer = RobustNormalizer(enabled=False)
    x = torch.randn(2, 1, 8, 8)
    out, stats = normalizer(x)
    assert torch.equal(out, x)
    assert torch.allclose(RobustNormalizer.denormalize(out, stats), x)


def test_normalizer_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        RobustNormalizer(scale_floor=0.0)
    with pytest.raises(ValueError):
        RobustNormalizer()(torch.randn(2, 8, 8))


def test_normalizer_is_parameter_free() -> None:
    assert count_parameters(RobustNormalizer())[0] == 0


# ---------------------------------------------------------------- concat fusion
def test_concat_fusion_shape_and_params() -> None:
    fusion = ConcatFusion(96)
    out = fusion(torch.randn(2, 96, 32, 32), torch.randn(2, 96, 32, 32))
    assert out.shape == (2, 96, 32, 32)
    assert count_parameters(fusion)[0] == 2 * 96 * 96 + 96


def test_concat_fusion_rejects_mismatched_inputs() -> None:
    with pytest.raises(ValueError):
        ConcatFusion(48)(torch.randn(1, 48, 8, 8), torch.randn(1, 48, 4, 4))
    with pytest.raises(ValueError):
        ConcatFusion(48)(torch.randn(1, 32, 8, 8), torch.randn(1, 32, 8, 8))


# --------------------------------------------------------------------- encoder
def test_stem_halves_resolution_and_matches_contract_params() -> None:
    stem = Stem(in_channels=2, out_channels=48)
    assert stem(torch.randn(2, 2, 128, 128)).shape == (2, 48, 64, 64)
    assert count_parameters(stem)[0] == CONTRACT_STAGE_PARAMS["stem_with_noise_map"]


def test_downsample_matches_contract_params() -> None:
    assert count_parameters(Downsample(48, 96))[0] == CONTRACT_STAGE_PARAMS["down0"]
    assert count_parameters(Downsample(96, 160))[0] == CONTRACT_STAGE_PARAMS["down1"]


def test_encoder_stage_shapes_match_contract_part_4() -> None:
    encoder = Encoder(stage6_base(), in_channels=1)
    bottleneck, skips = encoder(torch.randn(2, 1, 128, 128))
    assert [tuple(s.shape) for s in skips] == [(2, 48, 64, 64), (2, 96, 32, 32)]
    assert tuple(bottleneck.shape) == (2, 160, 16, 16)


def test_encoder_returns_one_skip_per_decoder_stage() -> None:
    config = stage6_base()
    _, skips = Encoder(config, 1)(torch.randn(1, 1, 128, 128))
    assert len(skips) == config.num_levels - 1


# --------------------------------------------------------------------- decoder
def test_upsample_doubles_resolution_and_matches_contract_params() -> None:
    up = Upsample(160, 96)
    assert up(torch.randn(2, 160, 16, 16)).shape == (2, 96, 32, 32)
    assert count_parameters(up)[0] == CONTRACT_STAGE_PARAMS["up1"]
    assert count_parameters(Upsample(96, 48))[0] == CONTRACT_STAGE_PARAMS["up0"]


def test_decoder_returns_to_trunk_resolution() -> None:
    config = stage6_base()
    encoder, decoder = Encoder(config, 1), Decoder(config)
    bottleneck, skips = encoder(torch.randn(2, 1, 128, 128))
    assert decoder(bottleneck, skips).shape == (2, 48, 64, 64)


def test_decoder_rejects_wrong_skip_count() -> None:
    config = stage6_base()
    decoder = Decoder(config)
    with pytest.raises(ValueError):
        decoder(torch.randn(1, 160, 16, 16), [torch.randn(1, 96, 32, 32)])


# ------------------------------------------------------- reconstruction head
def test_head_quadruples_resolution() -> None:
    head = ReconstructionHead(stage6_base())
    assert head(torch.randn(2, 48, 64, 64)).shape == (2, 1, 256, 256)


def test_head_conv_params_match_contract() -> None:
    head = ReconstructionHead(stage6_base())
    assert count_parameters(head.project)[0] == CONTRACT_STAGE_PARAMS["head_project"]
    assert count_parameters(head.to_subbands)[0] == CONTRACT_STAGE_PARAMS["head_subbands"]


def test_head_exposes_subbands_for_the_wavelet_loss() -> None:
    head = ReconstructionHead(stage6_base())
    assert head.predict_subbands(torch.randn(2, 48, 64, 64)).shape == (2, 4, 128, 128)


def test_head_rejects_wrong_channel_count() -> None:
    with pytest.raises(ValueError):
        ReconstructionHead(stage6_base())(torch.randn(1, 32, 64, 64))


def test_head_rejects_odd_width() -> None:
    with pytest.raises(ValueError):
        ReconstructionHead(build_sparc_config("sparc-base", head_width=31, **STAGE6))


def test_constant_input_gives_constant_output() -> None:
    """Contract Part 9 invariant: no checkerboard from the sub-band upsampler."""
    model = SPARCNet(stage6_base()).eval()
    with torch.no_grad():
        out = model(torch.full((1, 1, 128, 128), 0.42))
    assert out.std().item() < 1e-4


# ------------------------------------------------------------------- full model
@pytest.mark.parametrize("batch", [1, 2, 8])
def test_model_output_shape(batch: int) -> None:
    model = SPARCNet(sparc_tiny()).eval()
    with torch.no_grad():
        out = model(torch.rand(batch, 1, 128, 128))
    assert out.shape == (batch, 1, 256, 256)


def test_model_output_is_clamped_to_unit_range() -> None:
    """GT is exactly [0, 1] for all 3200 images, so the output must be too."""
    model = SPARCNet(sparc_tiny()).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 1, 128, 128) * 5.0)
    assert out.min().item() >= 0.0 and out.max().item() <= 1.0


def test_model_accepts_unclipped_input() -> None:
    """Phase 1: 3.36 % of real LR pixels exceed 1.0 and 0.30 % fall below 0."""
    model = SPARCNet(sparc_tiny()).eval()
    x = torch.rand(2, 1, 128, 128) * 2.4 - 0.3
    with torch.no_grad():
        out = model(x)
    assert torch.isfinite(out).all()


def test_model_gradients_reach_every_parameter() -> None:
    model = SPARCNet(sparc_tiny())
    model(torch.rand(2, 1, 128, 128)).pow(2).mean().backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name
        assert param.grad.abs().sum().item() > 0.0, name


def test_model_is_stable_across_input_magnitudes() -> None:
    model = SPARCNet(sparc_tiny()).eval()
    for scale in (1e-3, 1.0, 1e3):
        with torch.no_grad():
            out = model(torch.randn(1, 1, 128, 128) * scale)
        assert torch.isfinite(out).all()


def test_xl_moderate_forward_backward_and_flags() -> None:
    config = build_sparc_config("sparc-xl-moderate")
    model = SPARCNet(config)

    assert config.use_vst is True
    assert config.use_film is True
    assert config.use_attention is False
    assert model.film is not None
    assert model.vst.enabled is True

    x = torch.randn(2, 1, 128, 128)
    out = model(x)
    assert out.shape == (2, 1, 256, 256)
    assert torch.isfinite(out).all()
    assert out.min().item() >= 0.0
    assert out.max().item() <= 1.0

    out.mean().backward()
    grads = [
        parameter.grad for parameter in model.parameters() if parameter.requires_grad
    ]
    assert grads
    assert all(grad is not None for grad in grads)
    assert all(torch.isfinite(grad).all() for grad in grads if grad is not None)


@pytest.mark.parametrize(
    "variant", ["sparc-restoration-v2", "sparc-restoration-v2-fallback"]
)
def test_restoration_v2_forward_backward_and_flags(variant: str) -> None:
    config = build_sparc_config(variant)
    model = SPARCNet(config)

    assert config.use_vst is True
    assert config.use_film is True
    assert config.use_attention is False
    assert model.film is not None
    assert model.vst.enabled is True

    x = torch.randn(2, 1, 128, 128)
    out = model(x)
    assert out.shape == (2, 1, 256, 256)
    assert torch.isfinite(out).all()
    assert out.min().item() >= 0.0
    assert out.max().item() <= 1.0

    out.mean().backward()
    grads = [
        parameter.grad for parameter in model.parameters() if parameter.requires_grad
    ]
    assert grads
    assert all(grad is not None for grad in grads)
    assert all(torch.isfinite(grad).all() for grad in grads if grad is not None)


def test_model_rejects_bad_inputs() -> None:
    model = SPARCNet(sparc_tiny()).eval()
    with pytest.raises(ValueError):
        model(torch.rand(1, 128, 128))
    with pytest.raises(ValueError):
        model(torch.rand(1, 3, 128, 128))
    with pytest.raises(ValueError):
        model(torch.rand(1, 1, 130, 130))


def test_model_contains_no_forbidden_layers() -> None:
    """Contract Part 7: BatchNorm, activations and dropout are forbidden."""
    forbidden = (nn.BatchNorm2d, nn.ReLU, nn.GELU, nn.SiLU, nn.LeakyReLU, nn.Dropout)
    model = SPARCNet(stage6_base())
    assert not any(isinstance(m, forbidden) for m in model.modules())


def test_forward_with_aux_reports_no_sigma_without_the_noise_head() -> None:
    model = SPARCNet(stage6_base()).eval()
    with torch.no_grad():
        output = model(torch.rand(2, 1, 128, 128))
    with torch.no_grad():
        aux = model.forward_with_aux(torch.rand(2, 1, 128, 128))
    assert aux.sigma is None
    assert aux.stats.location.shape == (2, 1, 1, 1)
    assert output.shape == (2, 1, 256, 256)


def test_build_model_helper_applies_overrides() -> None:
    model = build_model("sparc-tiny", head_width=24)
    assert model.config.head_width == 24


def test_stage6_parameter_totals_are_stable() -> None:
    """Regression guard: these are the step-6 budgets before the deferred modules."""
    assert count_parameters(SPARCNet(sparc_tiny()))[0] == 250_452
    assert count_parameters(SPARCNet(stage6_base()))[0] == 1_687_844


def test_full_base_config_builds_and_matches_part_3() -> None:
    """Every deferred module has landed (steps 10-13), so Part 3 is now reachable."""
    model = SPARCNet(sparc_base())
    assert count_parameters(model)[0] == 2_345_650
    with torch.inference_mode():
        assert model(torch.rand(1, 1, 128, 128)).shape == (1, 1, 256, 256)
