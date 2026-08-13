# SPARC-Net inference contract

What the model actually consumes and produces, established by reading the pipeline
(`datasets/packed_dataset.py`, `datasets/degradation.py`, `scripts/pack_dataset.py`,
`models/sparc_net.py`) rather than by assumption. **An arbitrary PNG is not a valid
input.** Read §1 before running anything.

---

## 1. What the LR image is

| Property | Value | Source |
|---|---|---|
| Shape | `(128, 128)`, single channel | `DataConfig.lr_size`; packed array `(3200, 128, 128)` |
| Stored dtype | `float16` on disk, cast to `float32` on read | `pack_dataset.py`, `PackedRestorationDataset.__getitem__` |
| Value range | **nominally `[0, 1]` but NOT clipped** — measured `[-0.20, 1.70]` over train+test | packed arrays; Phase 1 measured 3.36 % of pixels > 1.0, 0.30 % < 0 |
| Already degraded? | **Yes.** The supplied `NoisyLR` is the real degraded observation. | Contract Part 6 |
| Degradation synthesised at train time? | **Sometimes.** With `p = 0.5` the loader *replaces* the supplied LR with one re-synthesised from GT. This is augmentation only — it never happens at inference. | `PackedRestorationDataset.__getitem__`, `resynth_prob = 0.5` |
| Normalisation applied by the dataset | **None.** | `__getitem__` returns the raw array as a tensor |

The degradation operator, for reference (`datasets/degradation.py`, Amendment A-001):

```
y = bicubic_down2( gaussian_blur(x, sigma ~ U(0.3, 0.5)) )      # antialias=False
y = y * Gamma(L ~ U(26, 51)) / L                                # speckle, at LR
y = y + N(0, sigma_g), sigma_g ~ U(0, 0.04)                     # additive, at LR
# no clipping at any stage
```

**You do not apply this at inference.** It exists to manufacture training pairs from GT.
The inference input is an already-degraded observation.

## 2. What the model does with it

All pre/post-processing lives *inside* `SPARCNet.forward` (Contract Part 1):

1. `RobustNormalizer`: `m = mean(y)`, `s = clamp_min(std(y), 0.02)`, `ŷ = (y - m) / s`.
   Statistics come from the **raw, unclipped** input — this is why clipping the input
   before handing it over changes the answer.
2. `NoiseHead` estimates `(σ_g, σ_s)` from the same raw input and builds a σ-map.
3. Trunk, decoder, sub-band reconstruction head.
4. `+ bicubic_up2(ŷ)` global residual, `* s + m` de-normalisation,
   **`clamp(0, 1)`** as the final op.

So: **do not normalise the input, and do not clamp the output.** Both are already done,
and doing them again is a bug. `scripts/infer.py` does neither.

## 3. Ground truth

| Property | Value |
|---|---|
| Shape | `(256, 256)`, single channel |
| Range | exactly `[0, 1]` for all 3200 images (verified over the packed array) |
| Relation to LR | `LR ≈ D(GT)` under the operator above, 2× decimation |

## 4. Input formats accepted by `scripts/infer.py`

| Format | Handling | Fidelity |
|---|---|---|
| `.npy` | passed through untouched | **exact** — bit-for-bit the training-time input |
| `.png` / `.tif` uint8 | divided by 255 | approximate: the `[-0.20, 1.70]` tail was already clipped and quantised when the PNG was written |
| `.png` / `.tif` uint16 | divided by 65535 | approximate, same reason, finer quantisation |
| float TIFF | passed through untouched | exact, if it was written unclipped |

Integer input logs a warning. Use `.npy` for anything you intend to score.

## 5. Size rules — never silently resized

`scripts/infer.py::check_input_size` enforces two independent constraints and raises
rather than resizing:

* **Divisible by 8.** The stem DWT plus two encoder DWT stages decimate three times.
* **Exactly 128×128 when attention is enabled.** Each `GSABlock` builds its
  relative-position index buffer for one grid at construction time (Contract Part 2.6),
  so a model with attention accepts precisely the size it was built for. The
  pre-attention checkpoint is fully convolutional and accepts any multiple of 8, but a
  different input size changes the noise statistics the noise head is conditioned on,
  so results at other sizes are not comparable to the reported PSNR.

Output is always exactly 2× the input in each dimension.

## 6. Output

Network output is float32 in `[0, 1]` at `(256, 256)`.

| `--output` suffix | Written as |
|---|---|
| `.png` / `.tif` | `round(x * (2^bits - 1))`, `--bit-depth 16` by default |
| `.npy` | float32, lossless |

16-bit is the default because 8-bit quantisation costs roughly 0.1 dB against the
metric the model is scored on. Use `.npy` when the output feeds another metric.

## 7. Weights

SPARC checkpoints (`trainer.Trainer.save`) contain `model`, `ema`, `optimizer`,
`scheduler`, `scaler`, `state` — **and no architecture config**. `infer.py` therefore
reconstructs the configuration from the state-dict's key names and tensor shapes
(`infer_config_from_state_dict`), then loads strictly: a mismatch is an error, never a
partial load. It detects the noise head, gated vs. concat fusion, attention, the layer
widths and every block depth, so pre-attention, SPARC-Tiny and full SPARC-Base
checkpoints all load without a flag.

EMA weights are used by default (Contract Part 5 evaluates them separately and they are
normally the better arm); `--no-ema` selects the live weights.

## 8. Example commands

Single image, exact input, GPU with bf16:

```bash
python scripts/infer.py \
    --weights outputs/integration_ckpt/integration_4_11_gated/best_psnr.pt \
    --input   Data/train/train/NoisyLR/000000.npy \
    --output  outputs/infer_demo/000000_restored.png \
    --device cuda --amp-dtype bf16
```

Folder:

```bash
python scripts/infer.py \
    --weights checkpoints/sparc-base/best_ema_psnr.pt \
    --input-dir  "Data/Test_NoisyLR (1)/NoisyLR" \
    --output-dir outputs/test_restored \
    --output-suffix .png --device cuda --amp-dtype bf16
```

Visual comparison with metrics:

```bash
python scripts/visualize_restoration.py \
    --weights outputs/integration_ckpt/integration_4_11_gated/best_psnr.pt \
    --input Data/train/train/NoisyLR/000000.npy \
    --gt    Data/train/train/GT/000000.npy \
    --output outputs/infer_demo/compare_000000.png --lpips
```

Prints model variant, parameter count, epoch, device, AMP dtype, input/output
dimensions, inference time, and PSNR/SSIM/LPIPS when `--gt` is supplied.

## 9. Flags

| Flag | Values | Default |
|---|---|---|
| `--device` | `auto`, `cuda`, `cpu` | `auto` (CUDA when visible) |
| `--amp-dtype` | `fp32`, `bf16`, `fp16` | `fp32` |
| `--bit-depth` | `8`, `16` | `16` |
| `--channels-last` | flag | off (Contract Part 5 has it on for training) |
| `--no-ema` | flag | off, i.e. EMA weights are used |

`--amp-dtype fp16` is rejected on CPU: PyTorch CPU autocast does not support it.
