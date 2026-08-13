"""Stage 5: filesystem artefacts, extended operator search (Lanczos), SNR,
normalisation ordering, quantitative distribution shift, outliers."""
import json, os, sys, stat
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from scipy import stats

D = Path(r"c:\Users\Ashish Kumar\Desktop\Image restoration\Data")
TR, TE = D / "train/train", D / "Test_NoisyLR (1)/NoisyLR"
OUT = Path(sys.argv[1]); R = {}

# ------------------------------------------------- 1. filesystem forensics
fs = {"symlinks": [], "hidden": [], "zero_byte": [], "readonly": [], "reparse": [],
      "thumbs_db": [], "desktop_ini": [], "size_hist": {}}
sizes = {}
for dp, dn, fn in os.walk(D):
    for f in fn:
        p = Path(dp) / f
        try:
            st = p.lstat()
        except OSError:
            continue
        if stat.S_ISLNK(st.st_mode): fs["symlinks"].append(str(p.relative_to(D)))
        if getattr(st, "st_file_attributes", 0) & 0x400: fs["reparse"].append(str(p.relative_to(D)))
        if getattr(st, "st_file_attributes", 0) & 0x2: fs["hidden"].append(str(p.relative_to(D)))
        if f.startswith("."): fs["hidden"].append(str(p.relative_to(D)))
        if st.st_size == 0: fs["zero_byte"].append(str(p.relative_to(D)))
        if f.lower() in ("thumbs.db",): fs["thumbs_db"].append(str(p))
        if f.lower() == "desktop.ini": fs["desktop_ini"].append(str(p))
        if f.endswith(".npy") and "__MACOSX" not in p.parts:
            sizes.setdefault(p.parent.name + "|" + str(p.parents[1].name), []).append(st.st_size)
fs["hidden"] = sorted(set(fs["hidden"]))[:10]
fs["n_hidden"] = len(set(fs["hidden"]))
fs["size_uniformity"] = {k: {"unique_sizes": sorted(set(v)), "n": len(v)} for k, v in sizes.items()}
R["filesystem"] = fs

# ------------------------------------------------- 2. extended operator search
rng = np.random.default_rng(3); ids = rng.choice(3200, 200, replace=False)
GT = np.stack([np.load(TR / "GT" / f"{i:06d}.npy") for i in ids])
LR = np.stack([np.load(TR / "NoisyLR" / f"{i:06d}.npy") for i in ids])
gt = torch.from_numpy(GT)[:, None]; lr = torch.from_numpy(LR)[:, None]
lap_k = torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0.]]]])

def sep_conv(x, k1d):
    n = len(k1d); a, b = n // 2, n - 1 - n // 2   # asymmetric pad -> even kernels keep size
    k = torch.as_tensor(k1d, dtype=torch.float32)
    x = F.conv2d(F.pad(x, (a, b, 0, 0), mode="reflect"), k.view(1, 1, 1, n))
    return F.conv2d(F.pad(x, (0, 0, a, b), mode="reflect"), k.view(1, 1, n, 1))

def lanczos1d(a, phase):
    """Lanczos-a resampling weights for a half-pixel-phase 2x decimation."""
    t = np.arange(-a * 2 + 1, a * 2 + 1) + phase
    w = np.sinc(t / 2) * np.sinc(t / (2 * a))     # cutoff at 0.5 (2x decimation)
    return (w / w.sum()).astype(np.float32)

def gauss1d(s, ks=9):
    x = np.arange(ks) - ks // 2
    g = np.exp(-x ** 2 / (2 * s ** 2)); return (g / g.sum()).astype(np.float32)

def leak(clean, obs):
    r = obs - clean
    l = F.conv2d(F.pad(clean, (1,) * 4, mode="reflect"), lap_k)
    return float(np.corrcoef(r.flatten().numpy(), l.flatten().numpy())[0, 1]), float(r.std())

ops = {}
def down(g, mode):
    if mode == "area": return F.avg_pool2d(g, 2)
    if mode == "stride": return g[:, :, ::2, ::2]
    if mode.startswith("lanczos"):
        a = int(mode[-1]); return sep_conv(g, lanczos1d(a, 0.5))[:, :, ::2, ::2]
    kw = {"bicubic_noAA": dict(mode="bicubic", antialias=False),
          "bicubic_AA": dict(mode="bicubic", antialias=True),
          "bilinear_noAA": dict(mode="bilinear", antialias=False),
          "bilinear_AA": dict(mode="bilinear", antialias=True),
          "nearest": dict(mode="nearest-exact")}[mode]
    if mode == "nearest": return F.interpolate(g, scale_factor=0.5, **kw)
    return F.interpolate(g, scale_factor=0.5, align_corners=False, **kw)

for sig in [0.0, 0.3, 0.4, 0.5, 0.7]:
    g = gt if sig == 0 else sep_conv(gt, gauss1d(sig))
    for m in ["bicubic_noAA", "bicubic_AA", "bilinear_noAA", "bilinear_AA", "area",
              "stride", "nearest", "lanczos2", "lanczos3", "lanczos4"]:
        ops[f"blur{sig}_{m}"] = leak(down(g, m), lr)
R["operator_search"] = {k: {"leak": round(v[0], 4), "rmse": round(v[1], 5)}
                        for k, v in sorted(ops.items(), key=lambda x: (abs(x[1][0]) + 8 * x[1][1]))}
R["operator_best_by_rmse"] = min(ops, key=lambda k: ops[k][1])
R["operator_best_by_leak"] = min(ops, key=lambda k: abs(ops[k][0]))

# ------------------------------------------------- 3. SNR
clean = down(gt, "bicubic_noAA"); r = (lr - clean).numpy()[:, 0]; c = clean.numpy()[:, 0]
snr = 10 * np.log10(c.reshape(200, -1).var(1) / r.reshape(200, -1).var(1))
psnr_lr = 10 * np.log10(1.0 / (r ** 2).reshape(200, -1).mean(1))
R["snr_db"] = {"med": float(np.median(snr)), "p10": float(np.percentile(snr, 10)),
               "p90": float(np.percentile(snr, 90)), "min": float(snr.min()), "max": float(snr.max())}
R["input_psnr_vs_cleanLR_db"] = {"med": float(np.median(psnr_lr)),
                                 "p10": float(np.percentile(psnr_lr, 10)),
                                 "p90": float(np.percentile(psnr_lr, 90))}

# ------------------------------------------------- 4. normalisation ordering
# If min-max normalisation had been applied AFTER degradation, LR would also be
# exactly [0,1]. It is not. If applied to the clean HR before degradation, LR
# should equal D(GT)+noise with unit gain and zero offset -> test that, and test
# whether speckle strength is independent of per-image mean (scale-equivariance).
sl = np.array([np.polyfit(c[i].ravel(), LR[i].ravel(), 1) for i in range(200)])
R["norm_order"] = {
    "lr_is_exactly_01": bool(np.allclose(LR.min((1, 2)), 0) and np.allclose(LR.max((1, 2)), 1)),
    "gain_mean_std": [float(sl[:, 0].mean()), float(sl[:, 0].std())],
    "offset_mean_std": [float(sl[:, 1].mean()), float(sl[:, 1].std())],
}
sp = []
for i in range(200):
    m = c[i] > 0.3
    sp.append((r[i][m] / c[i][m]).std() if m.sum() > 500 else np.nan)
sp = np.array(sp); ok = ~np.isnan(sp)
R["norm_order"]["corr_speckle_vs_imgmean"] = float(np.corrcoef(sp[ok], c.reshape(200, -1).mean(1)[ok])[0, 1])

# ------------------------------------------------- 5. distribution shift (blind)
TEST = np.stack([np.load(TE / f"{i:06d}.npy") for i in range(400)])
def feats(x):
    t = torch.from_numpy(x)[:, None]
    lap = F.conv2d(F.pad(t, (1,) * 4, mode="reflect"), lap_k)
    hp = (t - F.avg_pool2d(F.pad(t, (2,) * 4, mode="reflect"), 5, 1)).std((1, 2, 3)).numpy()
    return {"mean": x.mean((1, 2)), "std": x.std((1, 2)),
            "immerkaer": (lap.abs().mean((1, 2, 3)) * np.sqrt(np.pi / 2) / 6).numpy(),
            "hp_std": hp, "grad": np.abs(np.diff(x, axis=2)).mean((1, 2)),
            "entropy": np.array([stats.entropy(np.histogram(im, 64, (-0.3, 1.6))[0] + 1) for im in x])}
ftr, fte = feats(LR if False else np.stack([np.load(TR / "NoisyLR" / f"{i:06d}.npy") for i in range(3200)])), feats(TEST)
R["shift"] = {}
for k in ftr:
    ks = stats.ks_2samp(ftr[k], fte[k])
    R["shift"][k] = {"train_med": float(np.median(ftr[k])), "test_med": float(np.median(fte[k])),
                     "ratio": float(np.median(fte[k]) / (np.median(ftr[k]) + 1e-12)),
                     "KS_D": float(ks.statistic), "p": float(ks.pvalue),
                     "wasserstein": float(stats.wasserstein_distance(ftr[k], fte[k]))}

# ------------------------------------------------- 6. outliers + histograms
GTall = np.stack([np.load(TR / "GT" / f"{i:06d}.npy") for i in range(3200)])
mu, sd = GTall.mean((1, 2)), GTall.std((1, 2))
z = lambda a: (a - a.mean()) / a.std()
R["outliers"] = {
    "low_contrast_ids": [int(i) for i in np.argsort(sd)[:10]],
    "low_contrast_std": [float(v) for v in np.sort(sd)[:10]],
    "dark_ids": [int(i) for i in np.argsort(mu)[:6]], "bright_ids": [int(i) for i in np.argsort(-mu)[:6]],
    "n_std_below_0.05": int((sd < 0.05).sum()), "n_std_below_0.03": int((sd < 0.03).sum()),
    "n_mean_outside_0.1_0.9": int(((mu < 0.1) | (mu > 0.9)).sum()),
}
h_gt, e = np.histogram(GTall, 64, (0, 1))
h_lr, e2 = np.histogram(np.stack([np.load(TR / "NoisyLR" / f"{i:06d}.npy") for i in range(3200)]), 64, (-0.4, 1.6))
R["hist_gt_64bins_0_1"] = h_gt.tolist()
R["hist_lr_64bins_m04_16"] = h_lr.tolist()
R["gt_unique_values"] = {"med_per_image": int(np.median([len(np.unique(GTall[i])) for i in range(0, 3200, 200)]))}
json.dump(R, (OUT / "report_stage5.json").open("w"), indent=1)
print("OK")
