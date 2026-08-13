"""Stage 4: pin down the exact forward operator + noise distribution + GT provenance artefacts."""
import json, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from scipy import stats

TR = Path(r"c:\Users\Ashish Kumar\Desktop\Image restoration\Data\train\train")
OUT = Path(sys.argv[1]); R = {}
rng = np.random.default_rng(1)
ids = rng.choice(3200, 200, replace=False)
GT = np.stack([np.load(TR / "GT" / f"{i:06d}.npy") for i in ids])
LR = np.stack([np.load(TR / "NoisyLR" / f"{i:06d}.npy") for i in ids])
gt = torch.from_numpy(GT)[:, None]; lr = torch.from_numpy(LR)[:, None]
lap_k = torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0.]]]])

def gauss(sig, ks=9):
    x = torch.arange(ks) - ks // 2
    g = torch.exp(-x.float() ** 2 / (2 * sig ** 2)); g /= g.sum()
    return (g[:, None] * g[None, :])[None, None]

def structure_leak(clean, obs):
    """corr(residual, laplacian(clean)) -- 0 iff the forward model is right."""
    r = (obs - clean)
    l = F.conv2d(F.pad(clean, (1,)*4, mode="reflect"), lap_k)
    a, b = r.flatten().numpy(), l.flatten().numpy()
    return float(np.corrcoef(a, b)[0, 1]), float(r.std())

cands = {}
for sig in [0.0, 0.4, 0.6, 0.8, 1.0, 1.4]:
    g = gt if sig == 0 else F.conv2d(F.pad(gt, (4,)*4, mode="reflect"), gauss(max(sig, 1e-3)))
    for mode, kw in [("bicubic_noaa", dict(mode="bicubic", antialias=False)),
                     ("bicubic_aa", dict(mode="bicubic", antialias=True)),
                     ("area", None), ("stride", None)]:
        if mode == "area": d = F.avg_pool2d(g, 2)
        elif mode == "stride": d = g[:, :, ::2, ::2]
        else: d = F.interpolate(g, scale_factor=0.5, align_corners=False, **kw)
        cands[f"blur{sig}_{mode}"] = structure_leak(d, lr)
R["forward_model_search"] = {k: {"leak_corr": round(v[0], 4), "resid_std": round(v[1], 4)}
                             for k, v in sorted(cands.items(), key=lambda x: abs(x[1][0]))}
best = min(cands, key=lambda k: abs(cands[k][0]))
R["best_forward_model"] = best

# --- noise distribution given best model
clean = F.interpolate(gt, scale_factor=0.5, mode="bicubic", align_corners=False, antialias=False)
r = (lr - clean).numpy()[:, 0]; I = clean.numpy()[:, 0]
w = r / np.sqrt(np.maximum(2.5e-4 + 0.0064 * I + 0.0209 * I ** 2, 1e-8))   # variance-normalised
R["normalised_noise"] = {"std": float(w.std()), "skew": float(stats.skew(w.ravel())),
                         "kurtosis_excess": float(stats.kurtosis(w.ravel())),
                         "gaussian_ref_kurt": 0.0}
# multiplicative test: is r/I homoscedastic?
m = I > 0.15
R["ratio_r_over_I"] = {"std": float((r[m] / I[m]).std()), "skew": float(stats.skew((r[m] / I[m]).ravel()))}
# per-image speckle sigma via 2-param fit var = a + c I^2
cs = []
for i in range(len(ids)):
    Ii, ri = I[i].ravel(), r[i].ravel()
    q = np.quantile(Ii, np.linspace(0, 1, 13)); b = np.clip(np.digitize(Ii, q) - 1, 0, 11)
    cen, var = [], []
    for j in range(12):
        s = b == j
        if s.sum() > 80: cen.append(Ii[s].mean()); var.append(ri[s].var())
    cen = np.array(cen); A = np.stack([np.ones_like(cen), cen ** 2], 1)
    cs.append(np.linalg.lstsq(A, np.array(var), rcond=None)[0])
cs = np.array(cs)
R["per_image_2param"] = {
    "sigma_gauss_med": float(np.sqrt(max(np.median(cs[:, 0]), 0))),
    "sigma_gauss_p10_p90": [float(np.sqrt(max(np.percentile(cs[:, 0], 10), 0))),
                            float(np.sqrt(max(np.percentile(cs[:, 0], 90), 0)))],
    "sigma_speckle_med": float(np.sqrt(max(np.median(cs[:, 1]), 0))),
    "sigma_speckle_p10_p90": [float(np.sqrt(max(np.percentile(cs[:, 1], 10), 0))),
                              float(np.sqrt(max(np.percentile(cs[:, 1], 90), 0)))],
    "corr_gauss_speckle": float(np.corrcoef(cs[:, 0], cs[:, 1])[0, 1]),
}

# --- GT provenance: 8x8 JPEG blockiness score
def blockiness(x, n=256):
    d = np.abs(np.diff(x, axis=2))
    on = d[:, :, 7::8].mean(); off = np.delete(d, np.s_[7::8], axis=2).mean()
    return float(on / off)
R["gt_blockiness_8x8"] = blockiness(GT)
R["lr_blockiness_8x8"] = blockiness(LR)
# GT high-freq energy at exactly Nyquist (aliasing / upsampled-source check)
Pw = (torch.fft.rfft2(gt[:, 0]).abs() ** 2).mean(0).numpy()
R["gt_power_at_nyquist_vs_half"] = float(Pw[128, 128] / Pw[64, 64])
# is GT itself already blurry (was it upsampled from smaller)?
R["gt_hf_frac_0.4_0.5"] = float(Pw[(np.sqrt(np.fft.fftfreq(256)[:, None]**2 +
                                    np.fft.rfftfreq(256)[None, :]**2) > 0.4)].sum() / Pw[1:].sum())

# --- clipping check: does LR clip at 0 / 1?
R["lr_frac_exact0"] = float((LR == 0).mean()); R["lr_frac_exact1"] = float((LR == 1).mean())
R["lr_frac_negative"] = float((LR < 0).mean()); R["lr_frac_gt1"] = float((LR > 1).mean())
R["gt_frac_exact0"] = float((GT == 0).mean()); R["gt_frac_exact1"] = float((GT == 1).mean())
json.dump(R, (OUT / "report_stage4.json").open("w"), indent=1)
print(json.dumps(R, indent=1))
