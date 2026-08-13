"""Stage 2: forensic characterisation of the degradation operator + image content."""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(r"c:\Users\Ashish Kumar\Desktop\Image restoration\Data\train\train")
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
N = 300
ids = rng.choice(3200, N, replace=False)

GT = np.stack([np.load(ROOT / "GT" / f"{i:06d}.npy") for i in ids]).astype(np.float32)
LR = np.stack([np.load(ROOT / "NoisyLR" / f"{i:06d}.npy") for i in ids]).astype(np.float32)
gt_t = torch.from_numpy(GT)[:, None]
lr_t = torch.from_numpy(LR)[:, None]
R = {}

# ---------------------------------------------------------- content statistics
hist, edges = np.histogram(GT, bins=100, range=(0, 1))
R["gt_hist"] = hist.tolist()
R["gt_frac_below_0.05"] = float((GT < 0.05).mean())
R["gt_frac_above_0.95"] = float((GT > 0.95).mean())
R["gt_frac_mid"] = float(((GT > 0.2) & (GT < 0.8)).mean())
# how many distinct values -> is GT quantised / binary?
u = [len(np.unique(GT[i])) for i in range(20)]
R["gt_unique_values_first20"] = u

# gradient / edge density
gx = np.abs(np.diff(GT, axis=2)).mean(); gy = np.abs(np.diff(GT, axis=1)).mean()
R["gt_grad_x"] = float(gx); R["gt_grad_y"] = float(gy); R["gt_grad_anisotropy"] = float(gx / gy)

# ------------------------------------------------ which downsampling operator?
cands = {}
cands["area"] = F.avg_pool2d(gt_t, 2)
cands["bicubic"] = F.interpolate(gt_t, scale_factor=0.5, mode="bicubic", align_corners=False, antialias=True)
cands["bicubic_noaa"] = F.interpolate(gt_t, scale_factor=0.5, mode="bicubic", align_corners=False, antialias=False)
cands["bilinear_aa"] = F.interpolate(gt_t, scale_factor=0.5, mode="bilinear", align_corners=False, antialias=True)
cands["stride00"] = gt_t[:, :, 0::2, 0::2]
cands["stride11"] = gt_t[:, :, 1::2, 1::2]

def fit_fir(x, y, ks=9):
    """least-squares FIR kernel k minimising ||conv(x,k)-y||^2, averaged over batch."""
    p = ks // 2
    xp = F.pad(x, (p, p, p, p), mode="reflect")
    cols = F.unfold(xp, ks)                      # B, ks*ks, L
    A = torch.einsum("bkl,bml->km", cols, cols)
    b = torch.einsum("bkl,bl->k", cols, y.flatten(2)[:, 0])
    k = torch.linalg.solve(A + 1e-3 * torch.eye(ks * ks), b)
    pred = torch.einsum("bkl,k->bl", cols, k).reshape_as(y)
    return k.reshape(ks, ks), pred

fits = {}
for name, xd in cands.items():
    # plain residual (no kernel)
    r0 = (lr_t - xd)
    k, pred = fit_fir(xd, lr_t)
    r1 = lr_t - pred
    fits[name] = {
        "rmse_direct": float(r0.std()),
        "rmse_after_fir": float(r1.std()),
        "kernel_center": float(k[4, 4]),
        "kernel_sum": float(k.sum()),
        "kernel": k.numpy().round(4).tolist(),
    }
R["downsample_candidates"] = fits
best = min(fits, key=lambda n: fits[n]["rmse_after_fir"])
R["best_downsampler"] = best
xd = cands[best]
k, pred = fit_fir(xd, lr_t)
resid = (lr_t - pred).numpy()[:, 0]
clean = pred.numpy()[:, 0]

# effective MTF: radially averaged |H(f)| from cross-spectrum, LR grid
Xd = torch.fft.rfft2(xd[:, 0]); Y = torch.fft.rfft2(lr_t[:, 0])
H = (Y * Xd.conj()).mean(0) / ((Xd.abs() ** 2).mean(0) + 1e-8)
Hm = H.abs().numpy()
fy = np.fft.fftfreq(128)[:, None]; fx = np.fft.rfftfreq(128)[None, :]
fr = np.sqrt(fy ** 2 + fx ** 2)
bins = np.linspace(0, 0.5, 26)
idx = np.digitize(fr.ravel(), bins) - 1
mtf = [float(Hm.ravel()[idx == i].mean()) for i in range(25)]
R["effective_MTF_radial"] = mtf
R["MTF_freq_bins"] = ((bins[:-1] + bins[1:]) / 2).tolist()

# --------------------------------------------------------------- noise model
I = clean.ravel(); r = resid.ravel()
qs = np.quantile(I, np.linspace(0, 1, 33))
bi = np.clip(np.digitize(I, qs) - 1, 0, 31)
mu, var, cen = [], [], []
for j in range(32):
    m = bi == j
    if m.sum() > 500:
        cen.append(float(I[m].mean())); mu.append(float(r[m].mean())); var.append(float(r[m].var()))
R["noise_var_vs_intensity"] = {"I": cen, "mean_resid": mu, "var": var}
A = np.stack([np.ones(len(cen)), np.array(cen), np.array(cen) ** 2], 1)
coef, *_ = np.linalg.lstsq(A, np.array(var), rcond=None)
R["noise_var_fit_a_bI_cI2"] = coef.tolist()
R["noise_global_std"] = float(r.std())

# per-image noise level spread (high-freq residual std per image)
per = resid.reshape(N, -1).std(1)
R["per_image_resid_std"] = {"min": float(per.min()), "p10": float(np.percentile(per, 10)),
                            "med": float(np.median(per)), "p90": float(np.percentile(per, 90)),
                            "max": float(per.max()), "cv": float(per.std() / per.mean())}
# correlate noise level with image contrast -> multiplicative?
R["corr_noiselevel_vs_imgstd"] = float(np.corrcoef(per, clean.reshape(N, -1).std(1))[0, 1])

# noise whiteness: normalised autocorrelation of residual
rc = torch.from_numpy(resid)[:, None]
Rf = torch.fft.rfft2(rc[:, 0])
ac = torch.fft.irfft2((Rf * Rf.conj())).mean(0).numpy()
ac = ac / ac[0, 0]
R["resid_autocorr_3x3"] = np.roll(np.roll(ac, 1, 0), 1, 1)[:3, :3].round(4).tolist()

# per-image scale/offset relation between LR and downsampled GT (photometric)
sl, off = [], []
for i in range(N):
    a = np.polyfit(clean[i].ravel(), LR[i].ravel(), 1)
    sl.append(a[0]); off.append(a[1])
R["photometric_slope"] = [float(np.mean(sl)), float(np.std(sl))]
R["photometric_offset"] = [float(np.mean(off)), float(np.std(off))]

# ------------------------------------------------- spectral content of GT
G = torch.fft.rfft2(gt_t[:, 0])
P = (G.abs() ** 2).mean(0).numpy()
fy = np.fft.fftfreq(256)[:, None]; fx = np.fft.rfftfreq(256)[None, :]
fr = np.sqrt(fy ** 2 + fx ** 2); th = np.arctan2(np.abs(fy) * np.ones_like(fx), fx + 1e-9)
bins = np.linspace(0, 0.5, 41); idx = np.digitize(fr.ravel(), bins) - 1
R["gt_radial_power"] = [float(P.ravel()[idx == i].mean()) for i in range(40)]
R["gt_radial_bins"] = ((bins[:-1] + bins[1:]) / 2).tolist()
# angular energy (excluding DC) -> Manhattan/periodic structure?
m = (fr > 0.05)
ab = np.linspace(0, np.pi / 2, 19); ai = np.clip(np.digitize(th[m], ab) - 1, 0, 17)
R["gt_angular_power"] = [float(P[m][ai == i].mean()) for i in range(18)]
# fraction of GT energy above LR-Nyquist (0.25) = information destroyed by 2x decimation
tot = P[fr > 0].sum(); R["gt_energy_above_nyquist_frac"] = float(P[(fr > 0.25)].sum() / tot)

# periodicity: strength of dominant non-DC peak per image
peaks = []
for i in range(60):
    p = np.abs(np.fft.rfft2(GT[i] - GT[i].mean())) ** 2
    p[0, 0] = 0
    peaks.append(float(p.max() / (p.sum() + 1e-9)))
R["gt_dominant_peak_energy_frac"] = {"med": float(np.median(peaks)), "p90": float(np.percentile(peaks, 90))}

# ---------------------------------------------- baseline metric floors
def psnr(a, b):
    return float(10 * np.log10(1.0 / np.mean((a - b) ** 2)))
up_b = F.interpolate(lr_t, scale_factor=2, mode="bicubic", align_corners=False).clamp(0, 1).numpy()[:, 0]
up_n = F.interpolate(lr_t, scale_factor=2, mode="nearest").clamp(0, 1).numpy()[:, 0]
R["baseline_psnr_bicubic"] = psnr(up_b, GT)
R["baseline_psnr_nearest"] = psnr(up_n, GT)
R["baseline_psnr_meanimg"] = psnr(np.broadcast_to(GT.mean((1, 2))[:, None, None], GT.shape), GT)
# oracle: noiseless downsample upsampled (bound from SR alone, no noise)
xd_up = F.interpolate(xd, scale_factor=2, mode="bicubic", align_corners=False).clamp(0, 1).numpy()[:, 0]
R["oracle_psnr_cleanLR_bicubic_up"] = psnr(xd_up, GT)

json.dump(R, (OUT / "report_deg.json").open("w"), indent=1)
np.save(OUT / "ids.npy", ids)
print("done", best)
