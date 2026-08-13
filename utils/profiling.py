"""Latency, throughput and memory profiling utilities."""

from __future__ import annotations

import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch
from torch import nn

from utils.logging_utils import get_logger

_LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class LatencyReport:
    """Measured latency and throughput for a fixed batch size."""

    batch_size: int
    mean_ms: float
    median_ms: float
    p90_ms: float
    std_ms: float
    images_per_second: float
    peak_memory_mb: float

    def summary(self) -> str:
        return (
            f"B={self.batch_size:3d} | {self.mean_ms:8.3f} ms mean | "
            f"{self.median_ms:8.3f} ms median | {self.images_per_second:9.1f} img/s | "
            f"peak {self.peak_memory_mb:8.1f} MB"
        )


@contextmanager
def timed(label: str) -> Iterator[None]:
    """Context manager logging the wall-clock duration of a block."""
    start = time.perf_counter()
    try:
        yield
    finally:
        _LOGGER.info("%s took %.3f s", label, time.perf_counter() - start)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_latency(
    module: nn.Module,
    input_shape: tuple[int, ...],
    batch_size: int = 1,
    warmup: int = 10,
    iterations: int = 50,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> LatencyReport:
    """Benchmark forward-pass latency.

    Args:
        module: Module to benchmark, moved to ``device`` and set to ``eval``.
        input_shape: Shape of a single sample, excluding the batch dimension.
        batch_size: Batch size to benchmark.
        warmup: Number of untimed warm-up iterations.
        iterations: Number of timed iterations.
        device: Device to run on.
        dtype: Input/parameter dtype (use ``torch.float16`` for FP16 timing).

    Returns:
        A :class:`LatencyReport`.

    Raises:
        ValueError: If ``iterations`` is not positive.
    """
    if iterations <= 0:
        raise ValueError("iterations must be positive.")

    device = torch.device(device)
    module = module.to(device=device, dtype=dtype).eval()
    sample = torch.randn((batch_size, *input_shape), device=device, dtype=dtype)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        for _ in range(warmup):
            module(sample)
        _synchronize(device)

        timings: list[float] = []
        for _ in range(iterations):
            start = time.perf_counter()
            module(sample)
            _synchronize(device)
            timings.append((time.perf_counter() - start) * 1000.0)

    peak_mb = (
        torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else 0.0
    )
    mean_ms = statistics.fmean(timings)
    return LatencyReport(
        batch_size=batch_size,
        mean_ms=mean_ms,
        median_ms=statistics.median(timings),
        p90_ms=sorted(timings)[int(0.9 * (len(timings) - 1))],
        std_ms=statistics.pstdev(timings) if len(timings) > 1 else 0.0,
        images_per_second=batch_size * 1000.0 / mean_ms if mean_ms > 0 else float("inf"),
        peak_memory_mb=peak_mb,
    )
