# SPARC-Net — joint 2× super-resolution and denoising

Restoration of 128×128 low-resolution, noisy observations to 256×256 ground truth.
Single joint end-to-end model, PSNR-first, trained under a ~3.8 GB VRAM budget.

## The forward problem

The degradation recovered from the data is

```
y = bicubic_down2( gaussian_blur(x, sigma) ) * gamma_speckle(L) + N(0, sigma_g^2)
```

with `sigma ~ 0.4 px`, speckle `L ~ 26-51 looks` (`sigma_s = L^-0.5 ~ 0.14-0.20`) and an
additive floor `sigma_g ~ 0.024`. Noise is injected **after** decimation and nothing is
clipped: 3.36 % of real LR pixels sit above 1.0. See `datasets/degradation.py`.

## Architecture

A Haar-wavelet U-Net: every resample is a lossless DWT/IDWT rather than a strided
convolution or pixel-shuffle, and no convolution ever runs at 256×256.

```
LR 128x128
  -> NoiseHead (blind sigma_g, sigma_s)  -> VST -> RobustNormalizer
  -> Encoder  L0 48@64^2 / L1 96@32^2 / L2 160@16^2   (NAF blocks, optional GSA)
  -> Decoder  D1 -> D0                                (gated fusion + NAF)
  -> trunk features 48@64^2
       |-> CleanLRBranch  -> Delta -> clean_norm = y_hat + Delta -> bicubic x2 -> base
       |-> ReconstructionHead -> [LL,LH,HL,HH] -> IDWT -> detail
  -> out = base + detail -> denormalize -> VST^-1 -> clamp
```

| variant | params | GMAC @128² |
|---|---:|---:|
| `sparc-base` | 2,345,650 | 2.4055 |
| `sparc-clean-lr` | 2,417,811 | 2.8569 |

## Why the clean-LR branch exists

The V1 output path was `out = head + bicubic(noisy_LR)`, which hands the network the
observation noise and requires it to synthesise an exact cancelling signal. Measured on
the 320-image validation split:

| band | model err | oracle err | ratio | corr(R, R*) |
|---|---:|---:|---:|---:|
| LOW  (0–.125 Nyquist) | 1125.9 | 7.6 | 148× | 0.599 |
| MID  (.125–.35) | 594.0 | 52.7 | 11.3× | 0.831 |
| HIGH (.35–.7) | 276.2 | 170.0 | 1.62× | 0.815 |
| NYQ  (.7–1.0) | 97.6 | 97.2 | 1.00× | 0.302 |

Low frequencies carry the most signal energy and average down fastest under noise, so
they should be the *easiest* band — the network being worst there is the failure the
branch targets. It predicts a clean-LR estimate supervised against the exactly-known
`A(GT) = bicubic_down2(g_0.4 * GT)`, and that estimate replaces the noisy residual base.

`to_clean` is zero-initialised, so at step 0 the model is **bit-identical** to
`sparc-base` (verified `max|new - old| = 0`) and warm-starts from an existing checkpoint
with no regression.

## Measurement note

PSNR is reported in three reductions — `psnr_mean` (per-image mean), `psnr_median` and
`psnr_pooled` (MSE pooled over all pixels). **They are not interchangeable.** An earlier
analysis compared a pooled-PSNR oracle (27.36 dB) against a per-image-mean model score
(27.43 dB) and concluded the model had hit a ceiling; in matched units the model is
4.62 dB *below* the denoise-only oracle (27.43 vs 32.05). All three are logged every
validation epoch, along with `clean_lr_psnr` and the per-band correlations above.

## Layout

```
configs/      frozen architecture, loss, data and training configuration
datasets/     packed .npy loader, forward degradation operator, group-aware split
models/       SPARCNet, encoder/decoder, NAF + GSA blocks, Haar transforms, noise head
losses/       Charbonnier, MS-SSIM, wavelet, FFT, gradient, noise-aux, clean-LR
trainer/      training loop, EMA, telemetry
evaluation/   PSNR/SSIM/LPIPS, band-correlation diagnostics
scripts/      inference, smoke tests, profiling, ablation harnesses
reports/      phase reports and measured analyses
```

## Running

```bash
pip install -r requirements.txt
python scripts/smoke_clean_lr.py --device cuda    # 7 checks + 2 equivalence assertions
python train.py                                   # see configs/sparc_config.py
python -m pytest tests/ -q
```

## Not in this repository

`Data/` (1.6 GB), `checkpoints/` (591 MB) and `outputs/` (645 MB) are excluded: several
files exceed GitHub's 100 MB limit, and the imagery is grayscale-converted natural
photography that carries photographer watermarks. `scripts/pack_dataset.py` rebuilds the
packed arrays from a source dataset.
