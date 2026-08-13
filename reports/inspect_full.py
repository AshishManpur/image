"""Exhaustive, assumption-free inspection of the semiconductor restoration dataset.

Read-only. Writes report artifacts to OUT (scratchpad), never touches Data/.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(r"c:\Users\Ashish Kumar\Desktop\Image restoration\Data")
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
OUT.mkdir(parents=True, exist_ok=True)

report: dict = {}

# ---------------------------------------------------------------- 1. hierarchy
tree = []
ext_counter = Counter()
dir_ext = defaultdict(Counter)
total_bytes = 0
for dirpath, dirnames, filenames in os.walk(ROOT):
    dirnames.sort()
    rel = str(Path(dirpath).relative_to(ROOT))
    exts = Counter()
    size = 0
    for f in filenames:
        e = Path(f).suffix.lower() or "<noext>"
        exts[e] += 1
        ext_counter[e] += 1
        try:
            size += (Path(dirpath) / f).stat().st_size
        except OSError:
            pass
    total_bytes += size
    dir_ext[rel] = exts
    tree.append({"dir": rel, "n_files": len(filenames), "bytes": size,
                 "exts": dict(exts), "subdirs": list(dirnames)})
report["tree"] = tree
report["ext_counter"] = dict(ext_counter)
report["total_bytes"] = total_bytes

# naming conventions -----------------------------------------------------------
def sample_names(d: Path, k=6):
    try:
        fs = sorted(p.name for p in d.iterdir() if p.is_file())
    except OSError:
        return []
    return fs[:k] + (["..."] + fs[-2:] if len(fs) > k else [])

roles = {
    "train_lr": ROOT / "train" / "train" / "NoisyLR",
    "train_gt": ROOT / "train" / "train" / "GT",
    "test_lr": ROOT / "Test_NoisyLR (1)" / "NoisyLR",
    "macosx_train_lr": ROOT / "train" / "__MACOSX" / "train" / "NoisyLR",
    "macosx_test_lr": ROOT / "Test_NoisyLR (1)" / "__MACOSX" / "NoisyLR",
}
report["naming"] = {k: sample_names(v) for k, v in roles.items() if v.exists()}

# --------------------------------------------------- 2/3/4. per-file full scan
def scan(d: Path, role: str):
    recs = {}
    bad = []
    for p in sorted(d.glob("*")):
        if not p.is_file():
            continue
        rec = {"name": p.name, "stem": p.stem, "suffix": p.suffix, "size": p.stat().st_size}
        try:
            a = np.load(p, allow_pickle=False)
        except Exception as exc:
            rec["status"] = "UNREADABLE"
            rec["error"] = f"{type(exc).__name__}: {exc}"
            bad.append(rec)
            recs[p.name] = rec
            continue
        rec["status"] = "ok"
        rec["shape"] = tuple(int(x) for x in a.shape)
        rec["ndim"] = int(a.ndim)
        rec["dtype"] = str(a.dtype)
        af = a.astype(np.float64, copy=False)
        rec["min"] = float(af.min()); rec["max"] = float(af.max())
        rec["mean"] = float(af.mean()); rec["std"] = float(af.std())
        rec["p1"], rec["p50"], rec["p99"] = [float(v) for v in np.percentile(af, [1, 50, 99])]
        rec["nan"] = bool(np.isnan(af).any()); rec["inf"] = bool(np.isinf(af).any())
        rec["hash"] = hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()
        recs[p.name] = rec
    return recs, bad

scans = {}
for role, d in [("train_lr", roles["train_lr"]), ("train_gt", roles["train_gt"]),
                ("test_lr", roles["test_lr"])]:
    print("scanning", role, flush=True)
    scans[role], badr = scan(d, role)
    report.setdefault("unreadable", {})[role] = badr

# macOS resource-fork probe (do not full-scan; just show what they are)
mac = roles.get("macosx_train_lr")
if mac and mac.exists():
    ps = sorted(mac.glob("*"))[:3]
    probe = []
    for p in ps:
        head = p.open("rb").read(16)
        probe.append({"name": p.name, "size": p.stat().st_size, "head_hex": head.hex()})
    report["macosx_probe"] = probe

# aggregate stats --------------------------------------------------------------
def agg(recs):
    ok = [r for r in recs.values() if r["status"] == "ok"]
    shapes = Counter(r["shape"] for r in ok)
    dtypes = Counter(r["dtype"] for r in ok)
    arr = lambda k: np.array([r[k] for r in ok], dtype=np.float64)
    out = {
        "n_files": len(recs), "n_ok": len(ok),
        "shapes": {str(k): v for k, v in shapes.items()},
        "dtypes": dict(dtypes),
        "min_of_min": float(arr("min").min()), "max_of_max": float(arr("max").max()),
        "mean_of_mean": float(arr("mean").mean()), "std_of_mean": float(arr("mean").std()),
        "mean_of_std": float(arr("std").mean()),
        "global_min_p1": float(arr("p1").min()), "global_max_p99": float(arr("p99").max()),
        "n_nan": sum(r["nan"] for r in ok), "n_inf": sum(r["inf"] for r in ok),
        "n_unique_hashes": len(set(r["hash"] for r in ok)),
        "per_image_min_range": [float(arr("min").min()), float(arr("min").max())],
        "per_image_max_range": [float(arr("max").min()), float(arr("max").max())],
        "per_image_mean_range": [float(arr("mean").min()), float(arr("mean").max())],
        "per_image_std_range": [float(arr("std").min()), float(arr("std").max())],
    }
    # duplicates
    h = defaultdict(list)
    for r in ok:
        h[r["hash"]].append(r["name"])
    out["duplicate_groups"] = {k: v for k, v in h.items() if len(v) > 1}
    return out

report["stats"] = {k: agg(v) for k, v in scans.items()}

# ------------------------------------------------------------- 7. pairing
lr = scans["train_lr"]; gt = scans["train_gt"]
lr_stems = {r["stem"] for r in lr.values()}
gt_stems = {r["stem"] for r in gt.values()}
report["pairing"] = {
    "n_lr": len(lr_stems), "n_gt": len(gt_stems),
    "matched": len(lr_stems & gt_stems),
    "lr_without_gt": sorted(lr_stems - gt_stems)[:20],
    "gt_without_lr": sorted(gt_stems - lr_stems)[:20],
    "n_lr_without_gt": len(lr_stems - gt_stems),
    "n_gt_without_lr": len(gt_stems - lr_stems),
    "stem_pattern_ok": all(s.isdigit() and len(s) == 6 for s in lr_stems),
    "stem_min": min(lr_stems), "stem_max": max(lr_stems),
    "contiguous_ids": sorted(int(s) for s in lr_stems) == list(range(len(lr_stems))),
}
# cross-split leakage: identical arrays between train LR and test LR
test_h = {r["hash"] for r in scans["test_lr"].values() if r["status"] == "ok"}
train_h = {r["hash"] for r in lr.values() if r["status"] == "ok"}
report["pairing"]["train_test_hash_overlap"] = len(test_h & train_h)

# scale factor consistency
sc = Counter()
for name, r in lr.items():
    g = gt.get(name)
    if g and r["status"] == "ok" and g["status"] == "ok":
        sc[(r["shape"], g["shape"])] += 1
report["pairing"]["shape_pairs"] = {str(k): v for k, v in sc.items()}

json.dump(report, (OUT / "report_stage1.json").open("w"), indent=1)
print("stage1 written")
