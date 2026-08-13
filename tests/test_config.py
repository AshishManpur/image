"""Configuration validation tests (Contract Part 8, step 1)."""

from __future__ import annotations

import pytest

from configs.sparc_config import (
    ATTENTION_HEAD_DIM,
    LossConfig,
    SparcConfig,
    TrainingConfig,
    build_sparc_config,
    sparc_base,
    sparc_tiny,
)


def test_base_config_matches_contract() -> None:
    cfg = sparc_base()
    assert cfg.widths == (48, 96, 160)
    assert cfg.enc_naf_blocks == (4, 4, 4)
    assert cfg.enc_gsa_blocks == (0, 2, 3)
    assert cfg.dec_naf_blocks == (4, 4)
    assert cfg.dec_gsa_blocks == (1, 0)
    assert cfg.num_heads == (0, 3, 5)
    assert cfg.head_width == 32
    assert cfg.head_naf_blocks == 3
    assert cfg.attn_dim_ratio == 0.5
    assert all(
        (cfg.use_normalization, cfg.use_noise_head, cfg.use_gated_fusion,
         cfg.use_attention, cfg.use_global_residual, cfg.clamp_output, cfg.use_sdpa)
    )


def test_level_sizes_follow_dwt_stem() -> None:
    cfg = sparc_base()
    assert cfg.trunk_size == 64
    assert [cfg.level_size(i) for i in range(3)] == [64, 32, 16]


def test_attention_head_dim_is_sixteen_everywhere() -> None:
    cfg = sparc_base()
    for level in range(cfg.num_levels):
        if cfg.uses_attention_at(level):
            assert cfg.attention_dim(level) // cfg.num_heads[level] == ATTENTION_HEAD_DIM


def test_decoder_attention_is_at_level_one_only() -> None:
    cfg = sparc_base()
    assert cfg.uses_attention_at(0) is False
    assert cfg.uses_attention_at(1) is True
    assert cfg.uses_attention_at(2) is True


def test_tiny_config_disables_optional_stages() -> None:
    cfg = sparc_tiny()
    assert cfg.use_noise_head is False
    assert cfg.use_gated_fusion is False
    assert cfg.use_attention is False
    assert sum(cfg.enc_gsa_blocks) == 0
    assert sum(cfg.dec_gsa_blocks) == 0


def test_xl_moderate_config_matches_the_approved_reduction() -> None:
    cfg = build_sparc_config("sparc-xl-moderate")
    assert cfg.widths == (80, 136, 176)
    assert cfg.enc_naf_blocks == (8, 6, 4)
    assert cfg.dec_naf_blocks == (4, 6)
    assert cfg.head_width == 48
    assert cfg.head_naf_blocks == 4
    assert cfg.use_attention is False
    assert cfg.use_vst is True
    assert cfg.use_film is True
    assert cfg.head_project_kernel_size == 3  # unchanged default: v1-lineage variants


def test_restoration_v2_config_matches_the_coarse_to_fine_reallocation() -> None:
    cfg = build_sparc_config("sparc-restoration-v2")
    assert cfg.widths == (64, 160, 224)
    assert cfg.enc_naf_blocks == (3, 8, 8)
    assert cfg.dec_naf_blocks == (6, 3)
    assert cfg.head_width == 48
    assert cfg.head_naf_blocks == 4
    assert cfg.head_project_kernel_size == 1
    assert cfg.use_attention is False
    assert cfg.use_vst is True
    assert cfg.use_film is True


def test_restoration_v2_fallback_config_is_a_smaller_step() -> None:
    cfg = build_sparc_config("sparc-restoration-v2-fallback")
    assert cfg.widths == (72, 144, 192)
    assert cfg.enc_naf_blocks == (4, 7, 6)
    assert cfg.dec_naf_blocks == (5, 4)
    assert cfg.head_project_kernel_size == 1
    assert cfg.use_attention is False


def test_head_project_kernel_size_rejects_even_values() -> None:
    with pytest.raises(ValueError):
        SparcConfig(head_project_kernel_size=2)


@pytest.mark.parametrize(
    "overrides",
    [
        {"num_heads": (0, 4, 5)},          # head_dim 12 at level 1
        {"widths": (48, 96, 161)},          # odd width
        {"enc_naf_blocks": (4, 4)},         # wrong length
        {"dec_gsa_blocks": (1, 0, 0)},      # wrong length
        {"scale": 4},
        {"input_size": 100},                # not divisible by 8
        {"attn_dim_ratio": 0.0},
    ],
)
def test_invalid_configs_are_rejected(overrides: dict) -> None:
    with pytest.raises(ValueError):
        SparcConfig(**overrides)


def test_build_sparc_config_rejects_unknown_variant() -> None:
    with pytest.raises(KeyError):
        build_sparc_config("sparc-enormous")


def test_config_is_serialisable_and_immutable() -> None:
    cfg = sparc_base()
    payload = cfg.to_dict()
    assert payload["widths"] == (48, 96, 160)
    with pytest.raises(Exception):
        cfg.widths = (1, 2, 3)  # type: ignore[misc]


def test_training_config_enforces_sanctioned_override() -> None:
    TrainingConfig().validate()
    TrainingConfig(batch_size=16, learning_rate=4.2e-4).validate()
    with pytest.raises(ValueError):
        TrainingConfig(batch_size=12).validate()
    with pytest.raises(ValueError):
        TrainingConfig(batch_size=16).validate()


def test_loss_weights_match_contract() -> None:
    loss = LossConfig()
    assert (loss.charbonnier, loss.ms_ssim, loss.wavelet) == (1.00, 0.15, 0.10)
    assert (loss.fft, loss.gradient, loss.noise_aux) == (0.05, 0.05, 0.02)
    assert loss.wavelet_band_weights == (0.25, 1.0, 1.0, 1.5)
