"""Utility infrastructure tests (Contract Part 8, step 1)."""

from __future__ import annotations

import json

import torch
from torch import nn

from utils.complexity import count_parameters, measure_complexity, parameter_table
from utils.init import icnr_, trunc_normal_
from utils.logging_utils import CsvLogger, JsonlLogger, get_logger
from utils.profiling import benchmark_latency
from utils.seed import set_seed


def _tiny_net() -> nn.Module:
    return nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.Conv2d(8, 1, 3, padding=1))


def test_seed_gives_bitwise_reproducibility() -> None:
    set_seed(1337)
    a = torch.randn(64)
    net_a = _tiny_net()
    set_seed(1337)
    b = torch.randn(64)
    net_b = _tiny_net()
    assert torch.equal(a, b)
    for pa, pb in zip(net_a.parameters(), net_b.parameters()):
        assert torch.equal(pa, pb)


def test_count_parameters_matches_manual() -> None:
    net = _tiny_net()
    total, trainable = count_parameters(net)
    expected = (9 * 1 * 8 + 8) + (9 * 8 * 1 + 1)
    assert total == expected == trainable


def test_parameter_table_covers_children() -> None:
    net = _tiny_net()
    table = parameter_table(net)
    assert set(table) == {"0", "1"}
    assert sum(table.values()) == count_parameters(net)[0]


def test_measure_complexity_matches_analytic_macs() -> None:
    net = _tiny_net()
    report = measure_complexity(net, torch.randn(1, 1, 32, 32))
    analytic = (9 * 1 * 8 + 9 * 8 * 1) * 32 * 32
    assert abs(report.macs - analytic) / analytic < 0.05
    assert report.params_total == count_parameters(net)[0]
    assert "params=" in report.summary()


def test_measure_complexity_restores_training_mode() -> None:
    net = _tiny_net().train()
    measure_complexity(net, torch.randn(1, 1, 16, 16))
    assert net.training is True


def test_trunc_normal_is_bounded() -> None:
    tensor = torch.empty(4096)
    trunc_normal_(tensor, std=0.02)
    assert tensor.abs().max().item() <= 2 * 0.02 + 1e-6


def test_icnr_repeats_blocks() -> None:
    weight = torch.empty(8, 4, 3, 3)
    icnr_(weight, upscale_factor=2)
    for group in range(weight.shape[0] // 4):
        block = weight[group * 4 : (group + 1) * 4]
        assert torch.allclose(block[0], block[1])
        assert torch.allclose(block[0], block[3])


def test_loggers_write_records(tmp_path) -> None:
    jsonl = JsonlLogger(tmp_path / "m.jsonl")
    jsonl.log({"epoch": 1, "psnr": 21.67})
    payload = json.loads((tmp_path / "m.jsonl").read_text().strip())
    assert payload["psnr"] == 21.67

    csv = CsvLogger(tmp_path / "m.csv", ["epoch", "psnr"])
    csv.log({"epoch": 1, "psnr": 21.67})
    lines = (tmp_path / "m.csv").read_text().strip().splitlines()
    assert lines[0] == "epoch,psnr" and lines[1] == "1,21.67"


def test_get_logger_returns_named_logger() -> None:
    assert get_logger("sparc.test").name == "sparc.test"


def test_benchmark_latency_reports_positive_throughput() -> None:
    report = benchmark_latency(
        _tiny_net(), (1, 32, 32), batch_size=2, warmup=1, iterations=3
    )
    assert report.mean_ms > 0.0
    assert report.images_per_second > 0.0
    assert report.batch_size == 2
