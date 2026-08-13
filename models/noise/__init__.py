"""Blind noise estimation (Contract Part 2.9, Part 14).

* ``noise_head.py`` — :class:`NoiseHead`, the two-parameter blind estimator.
* ``noise_map.py``  — sigma-map assembly and the analytic auxiliary target.
"""

from __future__ import annotations

from models.noise.noise_head import NoiseHead, NoiseHeadOutput, NoiseStage
from models.noise.noise_map import (
    analytic_sigma_map,
    assemble_sigma_map,
    build_smoothing_kernel,
    fit_noise_parameters,
)

__all__ = [
    "NoiseHead",
    "NoiseHeadOutput",
    "NoiseStage",
    "analytic_sigma_map",
    "assemble_sigma_map",
    "build_smoothing_kernel",
    "fit_noise_parameters",
]
