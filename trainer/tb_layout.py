"""TensorBoard scalar organisation for SPARC-Net (Contract Part 6 telemetry).

Contract Part 6 requires that **every loss term is logged separately every step**.
Without a layout that is 15+ ungrouped scalar plots in alphabetical order, which is
unreadable exactly when it matters — during a loss-balance investigation.

This module defines the tag vocabulary and the custom-scalars layout. It is pure
data plus two small helpers; it is **not** wired into :class:`trainer.trainer.Trainer`
yet. Wiring happens in Phase 4.8.

**Tag convention:** ``<group>/<name>``, where the group is one of :data:`GROUPS`.
Per-step scalars use the ``step_`` prefixed groups and are written at
``global_step``; per-epoch scalars use the plain groups and are written at ``epoch``.
Mixing the two step bases on one plot produces a misleading x-axis, so they are kept
in separate groups by construction.

Guarded rules:

* Noise-head and attention groups are only populated when those modules are enabled,
  so an ablation run does not emit empty plots.
* GPU-memory scalars are skipped entirely on a non-CUDA host rather than logged as
  zero — a flat zero line reads as "no memory used" rather than "not measured".
"""

from __future__ import annotations

from typing import Any, Mapping

__all__ = ["GROUPS", "LOSS_TERMS", "build_layout", "should_log_group"]

GROUPS: tuple[str, ...] = (
    "train",
    "val",
    "ema",
    "loss",
    "step_loss",
    "optim",
    "grad",
    "noise",
    "attention",
    "memory",
    "throughput",
)
"""Every permitted tag prefix. Anything else is a typo, not a new group."""

LOSS_TERMS: tuple[str, ...] = (
    "total",
    "charbonnier",
    "ms_ssim",
    "wavelet",
    "fft",
    "gradient",
    "noise_aux",
)
"""Contract Part 6 loss decomposition, in contract order."""


def build_layout() -> dict[str, dict[str, Any]]:
    """Return the TensorBoard custom-scalars layout.

    Pass the result to ``SummaryWriter.add_custom_scalars`` once at run start.

    Returns:
        A layout mapping in the ``{category: {chart: [type, [tags]]}}`` form that
        ``add_custom_scalars`` expects.
    """
    return {
        "Overview": {
            "PSNR": ["Multiline", ["val/psnr_mean", "ema/psnr_mean", "val/psnr_median"]],
            "SSIM": ["Multiline", ["val/ssim_mean", "ema/ssim_mean"]],
            "Loss": ["Multiline", ["train/total", "val/total"]],
        },
        "Loss terms (per epoch)": {
            "Weighted contributions": [
                "Multiline", [f"loss/{term}" for term in LOSS_TERMS]
            ],
            "Unweighted": [
                "Multiline", [f"loss/raw_{term}" for term in LOSS_TERMS if term != "total"]
            ],
        },
        "Loss terms (per step)": {
            "Every term, every step": [
                "Multiline", [f"step_loss/{term}" for term in LOSS_TERMS]
            ],
        },
        "Optimisation": {
            "Learning rate": ["Multiline", ["optim/lr"]],
            "Gradient norm": ["Multiline", ["grad/global_norm", "grad/clipped_fraction"]],
            "AMP scale": ["Multiline", ["optim/amp_scale"]],
        },
        "Noise head": {
            "Predicted sigma": [
                "Multiline", ["noise/sigma_mean", "noise/sigma_min", "noise/sigma_max"]
            ],
            "Correlation with analytic sigma": ["Multiline", ["noise/sigma_correlation"]],
        },
        "Attention": {
            "LayerScale magnitude": ["Multiline", ["attention/layer_scale_mean"]],
            "Relative position bias": ["Multiline", ["attention/rel_pos_std"]],
        },
        "System": {
            "Peak GPU memory (GB)": [
                "Multiline", ["memory/peak_allocated_gb", "memory/peak_reserved_gb"]
            ],
            "Throughput": [
                "Multiline", ["throughput/images_per_second", "throughput/seconds_per_epoch"]
            ],
        },
    }


def should_log_group(group: str, *, cuda: bool, flags: Mapping[str, bool]) -> bool:
    """Whether a tag group should be written for the current run.

    Args:
        group: One of :data:`GROUPS`.
        cuda: Whether the run is on a CUDA device.
        flags: Model feature flags; ``use_noise_head`` and ``use_attention`` are read.

    Returns:
        ``True`` when the group is meaningful for this run.

    Raises:
        ValueError: If ``group`` is not a known group.
    """
    if group not in GROUPS:
        raise ValueError(f"Unknown TensorBoard group '{group}'. Known: {GROUPS}.")
    if group == "memory":
        return cuda
    if group == "noise":
        return bool(flags.get("use_noise_head", False))
    if group == "attention":
        return bool(flags.get("use_attention", False))
    return True
