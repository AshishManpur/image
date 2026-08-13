"""Phase 6 smoke test: supervised clean-LR base.

Runs the seven required checks plus the two equivalence assertions that make the
warm-started experiment safe to launch:

1. model construction        A. zero-init equivalence vs sparc-base
2. forward                   B. residual_source="noisy" equivalence
3. backward
4. loss
5. strict partial checkpoint load
6. one training epoch  (subset)
7. one validation epoch (subset)

Nothing here writes into ``checkpoints/`` or ``outputs/``: the trainer is pointed at a
scratch ``run_name`` so no existing artefact is touched.

Usage::

    python scripts/smoke_clean_lr.py [--device cuda] [--train-batches 12] [--val-batches 8]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.sparc_config import (  # noqa: E402
    DataConfig,
    LossConfig,
    TrainingConfig,
    sparc_base,
    sparc_clean_lr,
)
from datasets.packed_dataset import build_datasets  # noqa: E402
from datasets.splits import group_aware_split, verify_no_group_overlap  # noqa: E402
from losses.composite_loss import CompositeLoss  # noqa: E402
from models.sparc_net import SPARCNet  # noqa: E402
from trainer.trainer import Trainer  # noqa: E402
from utils.checkpoint import load_backbone_weights  # noqa: E402
from utils.complexity import measure_complexity  # noqa: E402
from utils.logging_utils import get_logger  # noqa: E402

_LOGGER = get_logger(__name__)
CHECKPOINT = PROJECT_ROOT / "checkpoints" / "sparc_base_50" / "best_psnr.pt"

_results: dict[str, str] = {}


def _check(name: str, ok: bool, detail: str = "") -> None:
    _results[name] = ("PASS" if ok else "FAIL") + (f"  {detail}" if detail else "")
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not ok:
        raise SystemExit(f"Smoke test failed at: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--train-batches", type=int, default=12)
    parser.add_argument("--val-batches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(0)

    print("=" * 78)
    print("PHASE 6 SMOKE TEST - supervised clean-LR base")
    print("=" * 78)

    # ------------------------------------------------------------ 1. construction
    base_cfg = sparc_base()
    clean_cfg = sparc_clean_lr()
    old = SPARCNet(base_cfg).eval()
    new = SPARCNet(clean_cfg).eval()
    p_old = sum(p.numel() for p in old.parameters())
    p_new = sum(p.numel() for p in new.parameters())
    _check("1. model construction", p_new > p_old, f"{p_old:,} -> {p_new:,}")

    # ------------------------------------------------------- 5. partial load first
    # Done before the equivalence checks because they must compare *trained* weights,
    # not two random initialisations.
    if not CHECKPOINT.exists():
        raise SystemExit(f"Checkpoint not found: {CHECKPOINT}")
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    old.load_state_dict(payload["model"])
    missing, unexpected = load_backbone_weights(new, CHECKPOINT)
    _check(
        "5. partial checkpoint load",
        not unexpected and all(k.startswith("clean_branch.") for k in missing),
        f"missing={len(missing)} (all clean_branch.*), unexpected={len(unexpected)}",
    )

    # --------------------------------------------------------- A/B. equivalences
    x = torch.randn(2, 1, 128, 128)
    with torch.no_grad():
        out_old = old(x)
        out_new = new(x)
    delta = (out_old - out_new).abs().max().item()
    _check("A. zero-init equivalence", delta == 0.0, f"max|new-old| = {delta:.3e}")

    noisy_cfg = sparc_clean_lr().with_overrides(residual_source="noisy")
    noisy = SPARCNet(noisy_cfg).eval()
    load_backbone_weights(noisy, CHECKPOINT)
    with torch.no_grad():
        out_noisy = noisy(x)
    delta_b = (out_old - out_noisy).abs().max().item()
    _check(
        "B. residual_source='noisy' equivalence", delta_b == 0.0,
        f"max|noisy-old| = {delta_b:.3e}",
    )

    # -------------------------------------------------------------- 2/3. fwd/bwd
    new_train = new.train()
    aux = new_train.forward_with_aux(x)
    _check(
        "2. forward",
        aux.image.shape == (2, 1, 256, 256) and aux.clean_lr.shape == (2, 1, 128, 128),
        f"image {tuple(aux.image.shape)}, clean_lr {tuple(aux.clean_lr.shape)}",
    )
    (aux.image.mean() + aux.clean_lr.mean()).backward()
    grads = [p.grad for p in new_train.parameters() if p.grad is not None]
    finite = all(torch.isfinite(g).all().item() for g in grads)
    branch_grad = sum(
        1
        for n, p in new_train.named_parameters()
        if n.startswith("clean_branch.") and p.grad is not None and p.grad.abs().sum() > 0
    )
    _check(
        "3. backward", finite and branch_grad > 0,
        f"{len(grads)} params w/ grad, all finite={finite}, clean_branch nonzero={branch_grad}",
    )
    new_train.zero_grad(set_to_none=True)

    # -------------------------------------------------------------------- 4. loss
    loss_cfg = LossConfig()
    criterion = CompositeLoss(loss_cfg)
    gt = torch.rand(2, 1, 256, 256)
    aux = new_train.forward_with_aux(x)
    total, terms = criterion(aux, {"gt": gt, "lr": x})
    _check(
        "4. loss",
        "clean_lr" in terms and torch.isfinite(total).item(),
        f"total={float(total.detach()):.5f}, clean_lr(raw)={terms.get('raw_clean_lr', float('nan')):.5f}",
    )

    # ------------------------------------------------------- 6/7. train + val epoch
    data_cfg = DataConfig()
    train_cfg = TrainingConfig(batch_size=args.batch_size)
    split = group_aware_split(
        data_cfg.expected_train,
        block_size=train_cfg.val_block_size,
        every_n=train_cfg.val_every_n_blocks,
    )
    verify_no_group_overlap(split, train_cfg.val_block_size, data_cfg.expected_train)
    train_ds, val_ds = build_datasets(
        data_cfg, split.train, split.val, seed=train_cfg.seed
    )
    train_sub = Subset(train_ds, range(args.train_batches * args.batch_size))
    val_sub = Subset(val_ds, range(args.val_batches * args.batch_size))
    train_loader = DataLoader(train_sub, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_sub, batch_size=args.batch_size)

    model = SPARCNet(clean_cfg)
    load_backbone_weights(model, CHECKPOINT)
    trainer = Trainer(
        model=model,
        criterion=CompositeLoss(loss_cfg),
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_cfg,
        device=device,
        run_name="smoke_clean_lr",          # scratch; never an existing run
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    train_metrics = trainer.train_epoch()
    train_seconds = time.perf_counter() - t0
    _check(
        "6. one training epoch",
        torch.isfinite(torch.tensor(train_metrics["total"])).item(),
        f"loss={train_metrics['total']:.5f}, clean_lr_w(epoch0)={trainer.criterion.weights['clean_lr']:.4f}",
    )

    t0 = time.perf_counter()
    val_metrics = trainer.evaluate(trainer.model, desc="smoke-val")
    val_seconds = time.perf_counter() - t0
    _check(
        "7. one validation epoch",
        "clean_lr_psnr" in val_metrics and "rho_low" in val_metrics,
        f"psnr_mean={val_metrics['psnr_mean']:.3f}",
    )

    # ------------------------------------------------------------------- complexity
    c_old = measure_complexity(old.eval(), torch.randn(1, 1, 128, 128))
    c_new = measure_complexity(new.eval(), torch.randn(1, 1, 128, 128))
    peak = (
        torch.cuda.max_memory_allocated(device) / 1e9
        if device.type == "cuda"
        else float("nan")
    )

    per_epoch = train_seconds * (len(train_ds) / max(1, len(train_sub)))
    print("\n" + "=" * 78)
    print("REPORT")
    print("=" * 78)
    print(f"PARAMETERS   old={p_old:,}  new={p_new:,}  delta=+{p_new - p_old:,} "
          f"(+{100 * (p_new - p_old) / p_old:.2f}%)")
    print(f"GMAC         old={c_old.macs / 1e9:.4f}  new={c_new.macs / 1e9:.4f}  "
          f"(+{100 * (c_new.macs - c_old.macs) / c_old.macs:.1f}%)")
    print(f"VRAM peak    {peak:.4f} GB (device={device})")
    print(f"epoch time   {train_seconds:.1f} s for {len(train_sub)} samples "
          f"-> ~{per_epoch:.0f} s extrapolated full epoch ({len(train_ds)} samples)")
    print(f"val time     {val_seconds:.1f} s for {len(val_sub)} samples")
    print(f"PSNR mean    {val_metrics['psnr_mean']:.4f}")
    print(f"PSNR median  {val_metrics['psnr_median']:.4f}")
    print(f"PSNR pooled  {val_metrics['psnr_pooled']:.4f}")
    print(f"SSIM mean    {val_metrics['ssim_mean']:.4f}")
    print(f"clean_lr_psnr {val_metrics.get('clean_lr_psnr', float('nan')):.4f}")
    for band in ("low", "mid", "high", "nyq"):
        print(f"rho_{band:<9} {val_metrics.get(f'rho_{band}', float('nan')):.4f}")
    print(f"grad finite  {finite}")
    print(f"missing keys {len(missing)}: {missing[:4]}{' ...' if len(missing) > 4 else ''}")
    print(f"unexpected   {len(unexpected)}")
    print("\nAll checks:")
    for name, status in _results.items():
        print(f"  {status.split()[0]:<5} {name}")


if __name__ == "__main__":
    main()
