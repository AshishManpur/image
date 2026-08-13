"""Stage 3: per-image noise model, train/test domain gap, near-duplicate scenes, visuals."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = Path(r"c:\Users\Ashish Kumar\Desktop\Image restoration\Data")
TR, TE = D / "train/train", D / "Test_NoisyLR (1)/NoisyLR"
OUT = Path(sys.argv[1]); R = {}
rng = np.random.default_rng(0)
ids = rng.choice(3200, 300, replace=False)
GT = np.stack([np.load(TR / "GT" / f"{i:06d}.npy") for i in ids])
LR = np.stack([np.load(TR / "NoisyLR" / f"{i:06d}.npy") for i in ids])
gt = torch.from_numpy(GT)[:, None]
clean = F.interpolate(gt, scale_factor=0.5, mode="bicubic", align_corners=False, antialias=False).numpy()[:, 0]
res = LR - clean

# ---- per-image noise model  var = a + b*I + c*I^2
P = []
for i in range(len(ids)):
    I, r = clean[i].ravel(), res[i].ravel()
    q = np.quantile(I, np.linspace(0, 1, 17)); b = np.clip(np.digitize(I, q) - 1, 0, 15)
    cen, var = [], []
    for j in range(16):
        m = b == j
        if m.sum() > 60: cen.append(I[m].mean()); var.append(r[m].var())
    cen = np.array(cen); var = np.array(var)
    A = np.stack([np.ones_like(cen), cen, cen ** 2], 1)
    P.append(np.linalg.lstsq(A, var, rcond=None)[0])
P = np.array(P)
R["per_image_noise_coeffs"] = {
    n: {"med": float(np.median(P[:, j])), "p10": float(np.percentile(P[:, j], 10)),
        "p90": float(np.percentile(P[:, j], 90))} for j, n in enumerate(["a_add", "b_poisson", "c_speckle"])}
R["speckle_sigma_med"] = float(np.sqrt(max(np.median(P[:, 2]), 0)))
R["speckle_sigma_p10_p90"] = [float(np.sqrt(max(np.percentile(P[:, 2], 10), 0))),
                              float(np.sqrt(max(np.percentile(P[:, 2], 90), 0)))]
R["gauss_sigma_med"] = float(np.sqrt(max(np.median(P[:, 0]), 0)))
R["frac_images_negative_a"] = float((P[:, 0] < 0).mean())

# residual variance explained by pure-speckle model r = I*n
R["corr_c_vs_b"] = float(np.corrcoef(P[:, 1], P[:, 2])[0, 1])

# ---- train vs test domain gap (no GT for test -> use blind estimators)
def blind_stats(x):
    t = torch.from_numpy(x)[:, None]
    hp = t - F.avg_pool2d(F.pad(t, (1, 1, 1, 1), mode="reflect"), 3, 1)   # high-pass
    lap = F.conv2d(F.pad(t, (1,)*4, mode="reflect"),
                   torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0.]]]]))
    # Immerkaer sigma estimate
    sig = (lap.abs().mean((1, 2, 3)) * np.sqrt(np.pi / 2) / 6).numpy()
    return {"mean": x.mean((1, 2)), "std": x.std((1, 2)), "hp": hp.std((1, 2, 3)).numpy(),
            "immerkaer": sig, "p99": np.percentile(x, 99, axis=(1, 2)),
            "p1": np.percentile(x, 1, axis=(1, 2))}
TEST = np.stack([np.load(TE / f"{i:06d}.npy") for i in range(400)])
tr_b, te_b = blind_stats(LR), blind_stats(TEST)
R["domain_gap"] = {k: {"train": [float(np.mean(tr_b[k])), float(np.std(tr_b[k]))],
                       "test": [float(np.mean(te_b[k])), float(np.std(te_b[k]))]} for k in tr_b}

# spectral domain gap
def rad(x, n):
    Pw = (torch.fft.rfft2(torch.from_numpy(x)).abs() ** 2).mean(0).numpy()
    fy = np.fft.fftfreq(n)[:, None]; fx = np.fft.rfftfreq(n)[None, :]
    fr = np.sqrt(fy**2 + fx**2); bb = np.linspace(0, .5, 21); ix = np.digitize(fr.ravel(), bb) - 1
    return [float(Pw.ravel()[ix == i].mean()) for i in range(20)]
R["radial_power_train_lr"] = rad(LR, 128); R["radial_power_test_lr"] = rad(TEST, 128)

# ---- near-duplicate scene detection (for leak-free validation split)
def sig(x):  # 8x8 downsampled signature of full train set
    return F.adaptive_avg_pool2d(torch.from_numpy(x)[:, None], 8).flatten(1).numpy()
ALL = np.stack([np.load(TR / "GT" / f"{i:06d}.npy") for i in range(0, 3200)])
S = sig(ALL); S = (S - S.mean(1, keepdims=True)); S /= (np.linalg.norm(S, axis=1, keepdims=True) + 1e-8)
C = S @ S.T; np.fill_diagonal(C, -1)
mx = C.max(1)
R["near_dup"] = {"frac_cos>0.99": float((mx > .99).mean()), "frac_cos>0.95": float((mx > .95).mean()),
                 "frac_cos>0.9": float((mx > .9).mean()), "median_max_cos": float(np.median(mx))}
# structural diversity: k-means-ish on signatures
R["gt_global_mean_spread"] = [float(ALL.mean((1, 2)).min()), float(ALL.mean((1, 2)).max())]

json.dump(R, (OUT / "report_stage3.json").open("w"), indent=1)

# ---------------------------------------------------------------- visuals
sel = [int(i) for i in ids[:6]]
fig, ax = plt.subplots(6, 6, figsize=(19, 19))
for r_, i in enumerate(sel):
    g = np.load(TR / "GT" / f"{i:06d}.npy"); l = np.load(TR / "NoisyLR" / f"{i:06d}.npy")
    c = F.interpolate(torch.from_numpy(g)[None, None], scale_factor=.5, mode="bicubic",
                      align_corners=False, antialias=False).numpy()[0, 0]
    up = F.interpolate(torch.from_numpy(l)[None, None], scale_factor=2, mode="bicubic",
                       align_corners=False).numpy()[0, 0]
    im = [(g, f"GT {i:06d} 256"), (l, "NoisyLR 128"), (up, "bicubic x2"),
          (l - c, "noise residual"), (np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(g - g.mean())))), "log|FFT| GT"),
          (g[96:160, 96:160], "GT crop 64")]
    for c_, (a_, t) in enumerate(im):
        ax[r_, c_].imshow(a_, cmap="gray" if c_ != 4 else "magma")
        ax[r_, c_].set_title(t, fontsize=8); ax[r_, c_].axis("off")
plt.tight_layout(); plt.savefig(OUT / "samples.png", dpi=70); plt.close()

fig, ax = plt.subplots(4, 8, figsize=(20, 10))
for j in range(16):
    i = int(rng.integers(3200)); g = np.load(TR / "GT" / f"{i:06d}.npy")
    l = np.load(TR / "NoisyLR" / f"{i:06d}.npy")
    ax.ravel()[2*j].imshow(g, cmap="gray"); ax.ravel()[2*j].set_title(f"GT {i}", fontsize=7)
    ax.ravel()[2*j+1].imshow(l, cmap="gray"); ax.ravel()[2*j+1].set_title("LR", fontsize=7)
for a_ in ax.ravel(): a_.axis("off")
plt.tight_layout(); plt.savefig(OUT / "diversity.png", dpi=65); plt.close()

fig, ax = plt.subplots(2, 8, figsize=(20, 5.5))
for j in range(16):
    t = np.load(TE / f"{rng.integers(400):06d}.npy")
    ax.ravel()[j].imshow(t, cmap="gray"); ax.ravel()[j].axis("off")
plt.suptitle("TEST NoisyLR samples"); plt.tight_layout(); plt.savefig(OUT / "test.png", dpi=65); plt.close()
print("ok")
