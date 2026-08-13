"""Dataset pipeline tests (Contract Part 8, step 2; Part 9)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from configs.sparc_config import DataConfig
from datasets.degradation import (
    DegradationParams,
    analytic_sigma_map,
    bicubic_downsample2,
    fit_noise_parameters,
    forward_operator,
    gaussian_blur,
    gaussian_kernel1d,
    sample_degradation_params,
    synthesize_lr,
)
from datasets.packed_dataset import PackedRestorationDataset, build_datasets, load_manifest
from datasets.splits import block_ids, group_aware_split, verify_no_group_overlap
from datasets.transforms import (
    GeometricOps,
    apply_geometric,
    apply_geometric_pair,
    sample_geometric_ops,
)

CONFIG = DataConfig()
PACKED_AVAILABLE = (Path(CONFIG.packed_root) / "manifest.json").exists()
requires_pack = pytest.mark.skipif(
    not PACKED_AVAILABLE, reason="Run `python scripts/pack_dataset.py` first."
)


# ----------------------------------------------------------------------- splits
def test_group_aware_split_sizes_match_contract() -> None:
    split = group_aware_split(3200, block_size=32, every_n=10)
    assert split.n_train == 2880
    assert split.n_val == 320
    assert split.n_train + split.n_val == 3200


def test_group_aware_split_has_no_index_or_block_overlap() -> None:
    split = group_aware_split(3200, block_size=32, every_n=10)
    assert np.intersect1d(split.train, split.val).size == 0
    verify_no_group_overlap(split, block_size=32, num_samples=3200)


def test_adjacent_ids_stay_in_the_same_partition() -> None:
    """Phase 1: 98.9 % of near-duplicate twins are within +/-2 IDs.

    The invariant is that every train/val boundary lands exactly on a block edge, so
    no pair of adjacent IDs is split unless a block boundary separates them.
    """
    block_size = 32
    split = group_aware_split(3200, block_size=block_size, every_n=10)
    is_val = np.zeros(3200, dtype=bool)
    is_val[split.val] = True
    boundaries = np.flatnonzero(is_val[:-1] != is_val[1:]) + 1
    assert boundaries.size > 0
    assert np.all(boundaries % block_size == 0)


def test_block_ids_are_contiguous() -> None:
    ids = block_ids(100, 32)
    assert ids[0] == 0 and ids[31] == 0 and ids[32] == 1


def test_split_rejects_degenerate_every_n() -> None:
    with pytest.raises(ValueError):
        group_aware_split(3200, every_n=1)


# ------------------------------------------------------------------ degradation
def test_gaussian_kernel_is_normalised_and_symmetric() -> None:
    kernel = gaussian_kernel1d(0.4, 5)
    assert kernel.shape == (5,)
    assert pytest.approx(kernel.sum().item(), abs=1e-6) == 1.0
    assert torch.allclose(kernel, kernel.flip(0), atol=1e-7)


def test_gaussian_kernel_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError):
        gaussian_kernel1d(0.0)
    with pytest.raises(ValueError):
        gaussian_kernel1d(0.4, kernel_size=4)


def test_gaussian_blur_preserves_constants() -> None:
    x = torch.full((2, 1, 32, 32), 0.37)
    assert torch.allclose(gaussian_blur(x, 0.4), x, atol=1e-6)


def test_forward_operator_halves_resolution() -> None:
    out = forward_operator(torch.rand(2, 1, 256, 256))
    assert out.shape == (2, 1, 128, 128)


def test_synthesize_lr_shape_and_no_clipping() -> None:
    gt = torch.rand(2, 1, 256, 256)
    params = DegradationParams(blur_sigma=0.4, looks=35.0, gauss_sigma=0.03)
    out = synthesize_lr(gt, params, torch.Generator().manual_seed(0))
    assert out.shape == (2, 1, 128, 128)
    assert out.min() < 0.0 or out.max() > 1.0  # unclipped noise escapes [0, 1]


def test_synthesize_lr_rejects_non_4d() -> None:
    with pytest.raises(ValueError):
        synthesize_lr(torch.rand(1, 256, 256), DegradationParams(0.4, 35.0, 0.0))


def test_speckle_has_unit_mean_and_expected_variance() -> None:
    """Gamma(L, L) has mean 1 and variance 1/L, so sigma_s = L**-0.5."""
    gt = torch.full((1, 1, 512, 512), 0.5)
    params = DegradationParams(blur_sigma=0.4, looks=36.0, gauss_sigma=0.0)
    out = synthesize_lr(gt, params, torch.Generator().manual_seed(0), noise_at_lr=True)
    assert pytest.approx(out.mean().item(), abs=0.005) == 0.5
    assert pytest.approx((out.std() / 0.5).item(), rel=0.10) == params.speckle_sigma


def test_noise_ordering_flag_changes_noise_level() -> None:
    """Amendment A-001: HR injection loses roughly half the speckle variance."""
    gt = torch.rand(1, 1, 256, 256)
    params = DegradationParams(blur_sigma=0.4, looks=35.0, gauss_sigma=0.0)
    clean = forward_operator(gt, 0.4)
    at_lr = synthesize_lr(gt, params, torch.Generator().manual_seed(0), noise_at_lr=True)
    at_hr = synthesize_lr(gt, params, torch.Generator().manual_seed(0), noise_at_lr=False)
    var_lr = (at_lr - clean).var().item()
    var_hr = (at_hr - clean).var().item()
    assert var_hr < 0.75 * var_lr


def test_sampled_params_lie_in_frozen_ranges() -> None:
    generator = torch.Generator().manual_seed(0)
    for _ in range(64):
        params = sample_degradation_params(CONFIG, generator)
        assert CONFIG.blur_sigma_range[0] <= params.blur_sigma <= CONFIG.blur_sigma_range[1]
        assert CONFIG.speckle_looks_range[0] <= params.looks <= CONFIG.speckle_looks_range[1]
        assert CONFIG.gauss_sigma_range[0] <= params.gauss_sigma <= CONFIG.gauss_sigma_range[1]


def test_fit_noise_parameters_recovers_known_sigmas() -> None:
    torch.manual_seed(0)
    clean = torch.rand(4, 1, 128, 128)
    sigma_g, sigma_s = 0.03, 0.16
    noisy = clean + torch.randn_like(clean) * sigma_g + clean * torch.randn_like(clean) * sigma_s
    fitted_g, fitted_s = fit_noise_parameters(noisy, clean)
    assert torch.allclose(fitted_s, torch.full((4,), sigma_s), atol=0.02)
    assert torch.allclose(fitted_g, torch.full((4,), sigma_g), atol=0.02)


def test_fit_noise_parameters_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        fit_noise_parameters(torch.rand(1, 1, 8, 8), torch.rand(1, 1, 4, 4))


def test_analytic_sigma_map_matches_formula() -> None:
    clean = torch.rand(2, 1, 8, 8)
    g = torch.tensor([0.02, 0.05])
    s = torch.tensor([0.16, 0.18])
    expected = torch.sqrt(
        g.view(-1, 1, 1, 1) ** 2 + s.view(-1, 1, 1, 1) ** 2 * clean**2 + 1e-12
    )
    assert torch.allclose(analytic_sigma_map(clean, g, s), expected, atol=1e-6)


# ------------------------------------------------------------------- transforms
def test_geometric_pair_stays_aligned() -> None:
    lr = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4)
    gt = torch.nn.functional.interpolate(
        lr.unsqueeze(0), scale_factor=2, mode="nearest"
    ).squeeze(0)
    ops = GeometricOps(hflip=True, vflip=False, rot_k=1)
    lr_aug, gt_aug = apply_geometric_pair(lr, gt, ops)
    expected = torch.nn.functional.interpolate(
        lr_aug.unsqueeze(0), scale_factor=2, mode="nearest"
    ).squeeze(0)
    assert torch.allclose(gt_aug, expected)


def test_identity_ops_are_a_no_op() -> None:
    x = torch.rand(1, 8, 8)
    ops = GeometricOps(hflip=False, vflip=False, rot_k=0)
    assert ops.is_identity
    assert torch.equal(apply_geometric(x, ops), x)


def test_geometric_transforms_are_measure_preserving() -> None:
    x = torch.rand(1, 8, 8)
    for ops in (
        GeometricOps(True, False, 0),
        GeometricOps(False, True, 0),
        GeometricOps(False, False, 2),
        GeometricOps(True, True, 3),
    ):
        out = apply_geometric(x, ops)
        assert out.shape == x.shape
        assert pytest.approx(out.sum().item(), abs=1e-5) == x.sum().item()


def test_apply_geometric_rejects_wrong_rank() -> None:
    with pytest.raises(ValueError):
        apply_geometric(torch.rand(1, 1, 8, 8), GeometricOps(False, False, 0))


def test_sample_geometric_ops_is_deterministic_under_seed() -> None:
    a = sample_geometric_ops(CONFIG, torch.Generator().manual_seed(7))
    b = sample_geometric_ops(CONFIG, torch.Generator().manual_seed(7))
    assert a == b


# --------------------------------------------------------------- packed dataset
@requires_pack
def test_manifest_records_expected_counts() -> None:
    manifest = load_manifest(Path(CONFIG.packed_root))
    assert manifest["train_lr"]["count"] == 3200
    assert manifest["train_gt"]["count"] == 3200
    assert manifest["test_lr"]["count"] == 400
    assert manifest["train_lr"]["size"] == 128
    assert manifest["train_gt"]["size"] == 256


@requires_pack
def test_packed_dataset_returns_contract_shapes() -> None:
    split = group_aware_split(3200, 32, 10)
    train, val = build_datasets(CONFIG, split.train, split.val)
    assert len(train) == 2880 and len(val) == 320
    sample = train[0]
    assert sample["lr"].shape == (1, 128, 128)
    assert sample["gt"].shape == (1, 256, 256)
    assert sample["lr"].dtype == torch.float32
    assert torch.isfinite(sample["lr"]).all() and torch.isfinite(sample["gt"]).all()


@requires_pack
def test_validation_split_is_not_augmented() -> None:
    split = group_aware_split(3200, 32, 10)
    _, val = build_datasets(CONFIG, split.train, split.val)
    assert torch.equal(val[5]["lr"], val[5]["lr"])
    assert val[5]["resynth"].item() == 0


@requires_pack
def test_training_sample_is_deterministic_for_a_fixed_seed() -> None:
    split = group_aware_split(3200, 32, 10)
    train_a, _ = build_datasets(CONFIG, split.train, split.val, seed=1337)
    train_b, _ = build_datasets(CONFIG, split.train, split.val, seed=1337)
    assert torch.equal(train_a[11]["lr"], train_b[11]["lr"])
    assert torch.equal(train_a[11]["gt"], train_b[11]["gt"])


@requires_pack
def test_gt_is_exactly_unit_range() -> None:
    """Phase 1: every GT image is min-max normalised to exactly [0, 1]."""
    split = group_aware_split(3200, 32, 10)
    _, val = build_datasets(CONFIG, split.train, split.val)
    for position in (0, 17, 100):
        gt = val[position]["gt"]
        assert pytest.approx(gt.min().item(), abs=1e-3) == 0.0
        assert pytest.approx(gt.max().item(), abs=1e-3) == 1.0


@requires_pack
def test_test_dataset_has_no_ground_truth() -> None:
    dataset = PackedRestorationDataset(
        Path(CONFIG.packed_root) / "test_lr.npy", None, None, CONFIG
    )
    assert len(dataset) == 400
    assert "gt" not in dataset[0]


@requires_pack
def test_dataset_rejects_out_of_range_indices() -> None:
    with pytest.raises(ValueError):
        PackedRestorationDataset(
            Path(CONFIG.packed_root) / "test_lr.npy", None, np.array([400]), CONFIG
        )


@requires_pack
def test_resynthesised_noise_matches_real_statistics() -> None:
    """Amendment A-001 acceptance criteria.

    Aggregate residual std within 10 %, sigma_s median within 10 %, and per-bin
    variance within 30 % for I > 0.10. The band tolerance cannot be tightened
    further with the frozen two-component noise model: the real noise contains a
    Poisson term that the contract deliberately omits (see AMENDMENTS.md, A-001).
    """
    lr_all = np.load(Path(CONFIG.packed_root) / "train_lr.npy", mmap_mode="r")
    gt_all = np.load(Path(CONFIG.packed_root) / "train_gt.npy", mmap_mode="r")
    index = np.sort(np.random.default_rng(0).choice(3200, 96, replace=False))
    gt = torch.from_numpy(np.asarray(gt_all[index], dtype=np.float32)).unsqueeze(1)
    real = torch.from_numpy(np.asarray(lr_all[index], dtype=np.float32)).unsqueeze(1)
    clean = forward_operator(gt, 0.4)

    generator = torch.Generator().manual_seed(0)
    synth = torch.cat(
        [
            synthesize_lr(
                gt[i : i + 1],
                sample_degradation_params(CONFIG, generator),
                generator,
                CONFIG.noise_at_lr,
            )
            for i in range(gt.shape[0])
        ]
    )

    real_std = (real - clean).std().item()
    synth_std = (synth - clean).std().item()
    assert abs(synth_std - real_std) / real_std < 0.10

    _, real_sigma_s = fit_noise_parameters(real, clean)
    _, synth_sigma_s = fit_noise_parameters(synth, clean)
    real_med = real_sigma_s.median().item()
    assert abs(synth_sigma_s.median().item() - real_med) / real_med < 0.10

    intensity = clean.flatten().double()
    for low, high in ((0.10, 0.30), (0.30, 0.60), (0.60, 1.10)):
        mask = (intensity >= low) & (intensity < high)
        var_real = (real - clean).flatten().double()[mask].var().item()
        var_synth = (synth - clean).flatten().double()[mask].var().item()
        assert abs(var_synth - var_real) / var_real < 0.30, (low, high)
