# SPARC-Base V1 — Developer Onboarding

Everything you need to go from a fresh clone to confidently modifying any part of the
codebase. Written against the state of the repo at **Phase 4.10 complete**.

**Read this first:** the project is governed by
`SPARC_BASE_V1_IMPLEMENTATION_CONTRACT.md`, a frozen document that fixes the
architecture, every hyperparameter, and every acceptance test. It is not a design
sketch — it is a contract. Changing a number in it requires a written amendment in
`AMENDMENTS.md` (Part 16). Two amendments exist so far: **A-001** (degradation
statistics) and **A-002** (the step-6 overfit gate). When code and contract disagree,
that is a bug in one of them, and you resolve it by measurement, not preference.

**What the model does.** Input: a 128×128 grayscale image, degraded by blur, speckle
(multiplicative) noise, and additive Gaussian noise. Output: a clean 256×256 image. So
it is joint denoising **and** 2× super-resolution. Phase 1 measured that ~84 % of the
recoverable quality is denoising, not upsampling — which is why the architecture is
built around a denoiser with an upsampling head, rather than the reverse.

---

## Section 1 — Project structure

### `configs/`
**Why:** one place where every frozen constant lives, so no magic number is ever typed
twice. **Important file:** `sparc_config.py`. It defines four frozen dataclasses:

| Class | Holds |
|---|---|
| `SparcConfig` | Architecture: widths, block counts, attention heads, feature flags |
| `TrainingConfig` | Optimiser, schedule, EMA, AMP, checkpoint paths, dataloader settings |
| `LossConfig` | The six loss weights and their hyperparameters |
| `DataConfig` | Dataset paths, sizes, augmentation probabilities, degradation ranges |

All are `@dataclass(frozen=True, slots=True)`. **This bites people twice**, so learn it
now: frozen means you cannot assign to a field, and `slots=True` means instances have
**no `__dict__`**. To make a modified copy, use `dataclasses.replace(cfg, epochs=50)`.
`TrainingConfig(**cfg.__dict__)` raises `AttributeError` — that exact bug shipped in
`train.py` and crashed every CLI override until it was found in the Phase 4.7 audit.

`SparcConfig.validate()` runs in `__post_init__` and enforces the structural invariants:
3 trunk levels, even widths, attention head dimension exactly 16, scale factor 2. An
illegal config fails at construction, not 40 minutes into training.

**Depends on:** nothing. **Depended on by:** everything.

### `datasets/`
**Why:** turn 3200 raw `.npy` pairs into batched, augmented tensors with a leak-free
split. **Files:**

- `packed_dataset.py` — memory-mapped `Dataset`. Reads from the packed `.npy` arrays
  rather than thousands of small files, so the OS page cache does the work.
- `transforms.py` — paired geometric augmentation **only** (h-flip, v-flip, rot90).
  There is deliberately no photometric jitter: the input's intensity statistics *are*
  the noise signal the model must estimate, so brightness jitter would corrupt the
  thing being learned.
- `degradation.py` — the Phase 1 forward model. Blur → speckle → additive noise →
  bicubic ↓2. Also holds `fit_noise_parameters` and `analytic_sigma_map`, which produce
  the noise head's supervision target.
- `splits.py` — group-aware split. **Read the reasoning:** consecutive image IDs are
  near-duplicate scenes. A random split would put near-identical images in both train
  and val, and your validation PSNR would be a fiction. The split therefore moves
  contiguous 32-ID *blocks*, sending every 10th block to validation: 2880 train / 320
  val, with a hard assertion that no block straddles the boundary.

**Used:** every training step. **Depends on:** `configs`. **Depended on by:** `train.py`,
`trainer`, and `losses/noise_loss.py` (for the analytic target).

### `models/`
**Why:** the network. **Files:**

- `sparc_net.py` — top-level assembly and the only file that knows the full data flow.
- `normalization.py` — `RobustNormalizer`, per-image mean/std, exactly invertible.
- `encoder.py` — DWT stem, encoder levels, downsampling.
- `blocks/` — `LayerNorm2d`, `SimpleGate`, `LayerScale`, `NAFBlock`.
- `wavelet/haar.py` — `HaarDWT` / `HaarIDWT`, the lossless resampler used everywhere.
- `decoder/` — `Decoder` and `ReconstructionHead`.
- `fusion/` — `GatedFuse` (the V1 module) and `ConcatFusion` (retained as the ablation
  A3 control arm), selected by `build_fusion` on `config.use_gated_fusion`.
- `noise/` — `NoiseHead` and σ-map assembly.
- `attention/` — `GSABlock` (**Phase 4.12, not yet written** — the package is an
  empty placeholder).

**Depends on:** `configs`, `utils`. **Depended on by:** `train.py`, `trainer`, `scripts`.

### `losses/`
**Why:** the training objective, one file per term so each is independently testable.
`charbonnier.py`, `ms_ssim.py`, `wavelet_loss.py`, `fft_loss.py`, `gradient.py`,
`noise_loss.py`, and `composite_loss.py` which weights and sums them.

> **Naming note:** contract Part 14 lists these as `fft.py`, `wavelet.py`,
> `noise_aux.py`, `composite.py`. The current names came from a later direct
> instruction. Behaviourally identical; flagged for a Part 14 amendment or a rename.

**Depends on:** `configs` (weights), `models/wavelet` (Haar), `datasets/degradation`
(noise target). **Depended on by:** `trainer`, `train.py`.

### `trainer/`
**Why:** the training loop, isolated from both the model and the data so it can be
tested with a 3-layer toy network in milliseconds. **Files:** `trainer.py` (the loop),
`ema.py` (weight averaging), `tb_layout.py` (TensorBoard tag vocabulary and grouping).

### `evaluation/`
**Why:** scored metrics, kept separate from the losses on purpose. `metrics.py` has
PSNR (two reductions), SSIM, LPIPS, and `MetricAccumulator`.

> **Important:** the metric SSIM and the MS-SSIM *loss* are separate implementations.
> That is deliberate, not duplication — coupling the reported metric to the trained
> objective means changing one silently changes the other.
>
> **`evaluation/evaluate.py` does not exist yet.** There is no standalone evaluation
> script, no submission assembly, and no image saving. See Section 7.

### `utils/`
Cross-cutting helpers: `logging_utils` (console + CSV + JSONL loggers), `init`
(weight initialisation), `complexity` (parameter and FLOP counting via
`FlopCounterMode`), `profiling` (latency/memory), `checkpoint` (save/load), `seed`,
`io`.

### `scripts/`
Standalone entry points: `pack_dataset.py`, `baselines.py`, `overfit.py`,
`integration_check.py`, `benchmark.py`, `export_onnx.py`, `export_trt.py`.

### `outputs/` and `checkpoints/`
Run artefacts. `outputs/logs/<run_name>/` holds TensorBoard events plus `metrics.csv`
and `metrics.jsonl`; `checkpoints/<run_name>/` holds the three checkpoint kinds.

### `reports/`
Phase reports and the audit trail. `PHASE4_7_AUDIT.md` is the most useful document in
the repo after the contract — it records every known discrepancy and why it is not a
bug.

---

## Section 2 — Entry point: what `python train.py` does

### Call hierarchy

```
train.py :: main()
  │
  ├─ configure_logging()                        utils/logging_utils.py
  ├─ TrainingConfig()                           configs/sparc_config.py
  ├─ dataclasses.replace(cfg, **cli_overrides)  ← CLI wins over defaults
  ├─ cfg.validate()                             ← rejects illegal batch size etc.
  ├─ set_seed(cfg.seed)                         python + numpy + torch + cudnn
  │
  ├─ build_sparc_config(variant, **overrides)   configs/sparc_config.py
  ├─ SPARCNet(model_config)                     models/sparc_net.py
  │     ├─ RobustNormalizer
  │     ├─ NoiseHead            (if use_noise_head)
  │     ├─ Encoder → Decoder → ReconstructionHead
  │     └─ self.apply(default_init)
  ├─ count_parameters() / measure_complexity()  ← logs params and GMACs
  │
  ├─ build_loaders(DataConfig(), train_config)
  │     ├─ group_aware_split(3200, 32, 10)      datasets/splits.py
  │     ├─ verify_no_group_overlap(...)         ← hard assertion, not a warning
  │     ├─ build_datasets(...)                  datasets/packed_dataset.py
  │     └─ DataLoader × 2
  │
  ├─ Trainer(model, criterion, loaders, config, device, run_name)
  │     ├─ model.to(device) [+ channels_last if CUDA]
  │     ├─ build_param_groups(model, weight_decay)
  │     ├─ torch.optim.AdamW(groups, lr, betas, eps)
  │     ├─ LambdaLR(warmup_cosine_lambda(...))
  │     ├─ GradScaler(init_scale=2**14, enabled=cuda and amp)
  │     ├─ ModelEma(model, decay=0.999)
  │     ├─ JsonlLogger / CsvLogger / SummaryWriter
  │     └─ writer.add_custom_scalars(build_layout())
  │
  ├─ trainer.load(resume_path, resume=True)     ← only with --resume
  └─ trainer.fit()
```

### Step by step

**1. Configuration.** `TrainingConfig()` instantiates the frozen defaults from Part 5:
400 epochs, batch 8, lr 3e-4, AdamW β=(0.9, 0.9), weight decay 1e-4, EMA 0.999, seed
1337.

**2. CLI overrides.** Any of `--epochs --batch-size --lr --num-workers --seed` that is
not `None` is collected into a dict and applied with `dataclasses.replace`. Then
`validate()` runs, which enforces the *sanctioned* deviations only: batch size must be
8 or 16, and batch 16 requires lr 4.2e-4 (= 3e-4·√2). Anything else raises.

**3. Datasets.** `build_datasets` opens the packed arrays with `mmap_mode="r"` and wraps
them in two `PackedRestorationDataset` instances. The training set has augmentation on;
the validation set has it **off** — validation must be a fixed target.

**4. Splits.** `group_aware_split(3200, block_size=32, every_n=10)` → 2880/320.
`verify_no_group_overlap` then re-derives the block membership and asserts disjointness.

**5. DataLoader.** Batch 8, shuffle on train, `drop_last=True` (so every step has the
same batch size, which keeps BN-free normalisation statistics consistent),
`num_workers=4`, `pin_memory=True`, `persistent_workers=True`. Note the guard: worker
options are only passed when `num_workers > 0`, because PyTorch errors otherwise.

**6. Model.** `SPARCNet(config)` builds sub-modules, then `self.apply(default_init)`
sweeps the tree. Modules that manage their own initialisation set `_custom_init = True`
to opt out — `LayerNorm2d`, `LayerScale`, and critically the noise head's final linear
layer, whose zero weight and calibrated biases would otherwise be destroyed.

**7. Losses.** `train.py` currently constructs `CharbonnierLoss()`. To train the real
V1 objective, pass `CompositeLoss()` instead (see Section 7 — this is a one-line change
and is the main thing a new developer will want to do).

**8. Optimiser.** `build_param_groups` splits parameters in two: weights get decay,
while LayerNorm affine params, LayerScale gammas, relative-position tables and **all**
biases get `weight_decay=0`. Decaying a normalisation scale shrinks the network toward
a degenerate map rather than a simpler one.

**9. Scheduler.** `LambdaLR` stepped **per optimisation step**, not per epoch: linear
warmup from 1e-6 over `steps_per_epoch × 5`, then cosine down to 1e-6.

**10. Trainer.** `fit()` loops epochs; each does `train_epoch()`, then `evaluate()`
twice — once on live weights, once on EMA weights — then records, checkpoints, and
tests the early-stopping rule.

---

## Section 3 — Data flow for one training sample

```
packed train_gt.npy[i]          float16 on disk, memory-mapped
  ↓ np.asarray(..., float32) + unsqueeze
GT                              (1, 256, 256)   values exactly in [0,1]
  ↓ with probability 0.5: on-the-fly LR re-synthesis
  │   gaussian_blur(σ ~ U(0.3, 0.5))            (1, 256, 256)
  │   bicubic ↓2                                (1, 128, 128)
  │   × Gamma(L ~ U(26, 51))/L      speckle     (1, 128, 128)
  │   + N(0, σ_g), σ_g ~ U(0, 0.04)             (1, 128, 128)   ← NOT clipped
  │ otherwise: use the supplied LR from train_lr.npy
LR                              (1, 128, 128)
  ↓ apply_geometric_pair — the SAME op applied to LR and GT
LR (1,128,128), GT (1,256,256)
  ↓ DataLoader collate, batch 8
batch = {"lr": (8,1,128,128), "gt": (8,1,256,256), "index": (8,), "resynth": (8,)}
  ↓ Trainer._to_device
same shapes, on device, channels_last if CUDA
  ↓ model.forward_with_aux(batch["lr"])
SparcOutput(image=(8,1,256,256), sigma=(8,1,128,128), stats=..., noise=...)
  ↓ CompositeLoss(output, batch)
total: scalar        terms: dict of 13 floats
  ↓ scaler.scale(loss).backward()
gradients on every parameter
  ↓ unscale_ → clip_grad_norm_(1.0) → scaler.step() → scaler.update()
weights updated
  ↓ scheduler.step() ; ema.update(model)
```

**Why the noise is not clipped:** clipping would destroy the very tail statistics the
noise head is trained to estimate. The model's `RobustNormalizer` reads its statistics
from the raw unclipped input for the same reason.

---

## Section 4 — Model flow for one image

Shapes are for `B=1`. Parameter counts are the contract's Part 3 values, which the
implementation matches exactly.

| # | Module | Purpose | In | Out | Params |
|---|---|---|---|---|---|
| 0 | `RobustNormalizer` | Remove per-image exposure/contrast. Phase 1 found image means spanning 0.016–0.959; without this the network wastes capacity modelling brightness. Exactly invertible. | (1,1,128,128) | (1,1,128,128) + stats | 0 |
| 1 | `NoiseHead` | Blind estimate of (σ_gauss, σ_speckle) → per-pixel σ map. Tells the trunk *how much* to denoise. | (1,1,128,128) | (1,1,128,128) | 42,050 |
| 2 | concat | Give the trunk image and noise level in the same tensor | 2×(1,1,128,128) | (1,2,128,128) | 0 |
| 3 | `HaarDWT` (stem) | Lossless 2× downsample. **The single most important efficiency decision:** it cuts compute and activation memory 4× at the most expensive resolution and, being lossless, gives up nothing. | (1,2,128,128) | (1,8,64,64) | 0 |
| 4 | `Conv3×3(8→48)` | Project to trunk width | (1,8,64,64) | (1,48,64,64) | 3,504 |
| 5 | Enc L0: 4× `NAFBlock` | Local denoising at the finest trunk level | (1,48,64,64) | (1,48,64,64) → **skip₀** | 70,848 |
| 6 | `HaarDWT` + `Conv1×1(192→96)` | Downsample to level 1 | (1,48,64,64) | (1,96,32,32) | 18,528 |
| 7 | Enc L1: 4× NAF + 2× GSA | Mid-scale context. Attention starts here — never at 64² | (1,96,32,32) | (1,96,32,32) → **skip₁** | 395,622 |
| 8 | `HaarDWT` + `Conv1×1(384→160)` | Downsample to bottleneck | (1,96,32,32) | (1,160,16,16) | 61,600 |
| 9 | Enc L2: 4× NAF + 3× GSA | Bottleneck; global reasoning over 256 tokens | (1,160,16,16) | (1,160,16,16) | 1,158,655 |
| 10 | `Conv1×1(160→384)` + `HaarIDWT` | Upsample to level 1 | (1,160,16,16) | (1,96,32,32) | 61,824 |
| 11 | `GatedFuse(96)` + skip₁ | Adaptively mix skip and decoder. Noise level varies 8.5× across images, so a fixed mix is wrong | (1,96,32,32)×2 | (1,96,32,32) | 23,256 |
| 12 | Dec D1: 1× GSA + 4× NAF | Refine at level 1 | (1,96,32,32) | (1,96,32,32) | 333,171 |
| 13 | `Conv1×1(96→192)` + `HaarIDWT` | Upsample to level 0 | (1,96,32,32) | (1,48,64,64) | 18,624 |
| 14 | `GatedFuse(48)` + skip₀ | Adaptive skip fusion | (1,48,64,64)×2 | (1,48,64,64) | 5,868 |
| 15 | Dec D0: 4× NAF | Refine at level 0 | (1,48,64,64) | (1,48,64,64) | 70,848 |
| 16 | `Conv3×3(48→128)` + `HaarIDWT` | Head projection and first upsample | (1,48,64,64) | (1,32,128,128) | 55,424 |
| 17 | 3× `NAFBlock` (C=32) | Refine at 128². 35 % of all activations live here | (1,32,128,128) | (1,32,128,128) | 24,672 |
| 18 | `Conv3×3(32→4)` | Predict the four Haar sub-bands [LL,LH,HL,HH] | (1,32,128,128) | (1,4,128,128) | 1,156 |
| 19 | `HaarIDWT` | Final 2× upsample | (1,4,128,128) | (1,1,256,256) | 0 |
| 20 | `+ bicubic_up2(ŷ)` | Global residual: the network predicts a *correction*, not the image | (1,1,256,256) | (1,1,256,256) | 0 |
| 21 | de-normalise, `clamp(0,1)` | Undo step 0; GT is exactly [0,1] | (1,1,256,256) | (1,1,256,256) | 0 |
| | **Total** | | | | **2,345,650** |

**Why predict sub-bands instead of using PixelShuffle?** The four output channels get a
defined meaning, the band-weighted wavelet loss applies directly, and checkerboard
artefacts become *impossible* — an orthogonal basis has no preferred sub-pixel position,
so no ICNR initialisation is needed.

**Why no convolution ever runs at 256²?** The head is already 24 % of MACs and 35 % of
activations. On a 4 GB card, output resolution is pure memory bandwidth.

---

## Section 5 — One training iteration

From `Trainer.train_epoch()`:

```python
for batch in self.train_loader:
    batch = self._to_device(batch)                        # H2D + channels_last
    self.optimizer.zero_grad(set_to_none=True)            # set_to_none frees memory

    with autocast("cuda", float16, enabled=amp_enabled):  # 1. FORWARD
        loss, terms = self._compute_loss(batch)           #    → wants_aux routing

    if not torch.isfinite(loss):                          # 2. GUARD
        self.scheduler.step(); self.state.global_step += 1
        self.skipped_batches += 1
        continue                                          #    schedule still advances

    self.scaler.scale(loss).backward()                    # 3. BACKWARD (scaled)
    self.scaler.unscale_(self.optimizer)                  # 4. UNSCALE before clipping
    grad_norm = clip_grad_norm_(params, 1.0)              # 5. CLIP on true gradients
    self.scaler.step(self.optimizer)                      # 6. STEP (skipped if inf)
    self.scaler.update()                                  # 7. adapt the scale factor
    self.scheduler.step()                                 # 8. per-step LR update
    self.ema.update(self.model)                           # 9. shadow weights
    self._log_step(terms, grad_norm)                      # 10. per-step TensorBoard
```

**The ordering in steps 3–6 is the part people get wrong.** With AMP the loss is scaled
up before backward to stop small gradients underflowing fp16. If you clipped at that
point you would clip *scaled* gradients, so the clip threshold would depend on the
scaler's current value — meaningless. `unscale_` must come first. `scaler.step` then
skips the update entirely if it finds inf/NaN.

**`_compute_loss` routing** (this is how the composite loss gets what it needs):

```python
wants_aux = getattr(self.criterion, "wants_aux", False)
if wants_aux and hasattr(self.model, "forward_with_aux"):
    result = self.criterion(model.forward_with_aux(batch["lr"]), batch)  # full batch
else:
    result = self.criterion(self.model(batch["lr"]), batch["gt"])        # legacy path
```

`CompositeLoss` sets `wants_aux = True` and needs the whole batch because the noise
term derives its target from `(D(gt), lr)`. Anything without the flag keeps the old
two-tensor signature — that is what makes the change backward compatible.

**Per epoch**, after the loop: `evaluate(self.model)` and `evaluate(self.ema.module)`,
then `_record` writes to JSONL/CSV/TensorBoard, then three checkpoints are considered
(`best_psnr.pt`, `best_ema_psnr.pt`, `last.pt`), then early stopping is tested against
EMA validation PSNR (patience 40, min delta 0.01 dB).

**EMA subtlety:** `epoch 0` always counts as an improvement, because `best_ema_psnr`
starts at `-inf` and `-inf + min_delta` is still `-inf`. The first epoch establishes the
baseline rather than beating one.

---

## Section 6 — Outputs

| Artefact | Location |
|---|---|
| Checkpoints | `checkpoints/<run_name>/{best_psnr,best_ema_psnr,last}.pt` |
| TensorBoard events | `outputs/logs/<run_name>/events.out.tfevents.*` |
| Per-epoch metrics | `outputs/logs/<run_name>/metrics.csv` and `metrics.jsonl` |
| Baseline reference | `outputs/baselines.json` (bicubic 21.67 dB) |
| Overfit-gate reports | `outputs/overfit_*_report.json`, `outputs/overfit_*_trace.jsonl` |
| Integration checks | `outputs/integration_4_10.json`, `outputs/integration_4_11_{gated,concat}.json` |
| Fusion verification | `reports/report_fusion.json` (from `reports/inspect_fusion.py`) |
| Figures | `reports/figures/*.png` |
| Phase reports | `reports/*.md` |
| **Restored images** | **nowhere — image saving is not implemented yet** |

Each checkpoint contains `model`, `ema`, `optimizer`, `scheduler`, `scaler`, and a
`state` dict (epoch, global_step, best metrics, patience counter) — everything needed
to resume bit-exactly.

---

## Section 7 — How to run

### Step 0 — Environment
```bash
pip install -r requirements.txt
```

### Step 1 — Pack the dataset (once)
```bash
python scripts/pack_dataset.py
```
Converts 3200 + 3200 + 400 raw `.npy` files into three memory-mapped arrays plus a
manifest. It verifies file counts, LR↔GT stem pairing, and that the float16 round-trip
stays above 70 dB PSNR — it aborts rather than silently packing lossy data.

### Step 2 — Reproduce the baselines (recommended sanity check)
```bash
python scripts/baselines.py
```
Should reproduce **bicubic 21.67 dB**. If this number is wrong, your data pipeline is
wrong, and nothing downstream is trustworthy.

### Step 3 — Sanity-check the backbone
```bash
python scripts/overfit.py --variant sparc-tiny --images 2 --steps 2000
```
The step-6 gate under amendment A-002. Should exceed 45 dB within 2000 steps (it
currently reaches it at step 517). This proves the data path, gradient flow, head
expressiveness and optimiser are all sane.

### Step 4 — Train
```bash
python train.py --variant sparc-tiny --stage6 --epochs 5 --num-workers 0   # smoke test
python train.py --variant sparc-base --epochs 400                          # real run
```
`--stage6` disables the modules deferred past contract step 6 (noise head, gated
fusion, attention).

> **Two caveats you will hit immediately.**
> 1. `train.py` currently instantiates `CharbonnierLoss()`. To train the real V1
>    objective, change that line to `CompositeLoss()`. This is intentional — the
>    composite loss landed in Phase 4.10 and wiring it into the default path is a
>    deliberate, reviewable change rather than a silent one.
> 2. `--variant sparc-base` **fails today** with `ModuleNotFoundError` because
>    `use_attention=True` imports `models/attention/gsa_block.py`, which is Phase 4.12
>    and does not exist. Until then use `sparc-tiny`, or `sparc-base` with attention
>    disabled. Gated fusion is built and on by default — it no longer needs disabling.

### Step 5 — Resume
```bash
python train.py --variant sparc-base --resume checkpoints/sparc-base/last.pt
```
Restores model, EMA, optimiser, scheduler, scaler and all counters, and continues from
the following epoch.

### Step 6 — TensorBoard
```bash
tensorboard --logdir outputs/logs
```
Grouped by `trainer/tb_layout.py`: Overview, per-epoch loss terms, per-step loss terms,
Optimisation, Noise head, Attention, System. Per-step and per-epoch scalars are kept in
separate tag groups on purpose — they use different step bases and would otherwise share
a misleading x-axis.

### Step 7 — Integration check
```bash
python scripts/integration_check.py --epochs 5 --subset 128
```
Short end-to-end run verifying no NaN, every loss term decreasing, the noise head
learning, and PSNR improving.

### Step 8 — Benchmark
```bash
python scripts/benchmark.py --variant sparc-tiny
```
Measures parameters, MACs, GFLOPs, disk size, latency, throughput and VRAM against all
eight Part 10 budgets. On a CPU-only host it prints `UNMEASURED (no CUDA)` for the
GPU-denominated budgets rather than substituting CPU numbers.

### Step 9 — Export
```bash
python scripts/export_onnx.py --variant sparc-tiny --checkpoint checkpoints/.../best_ema_psnr.pt
python scripts/export_trt.py  --precision fp16       # requires CUDA; target machine only
```
Export forces `use_sdpa=False` so attention takes the explicit path the contract
requires for export, and deep-copies the model so exporting cannot mutate training state.

### Not yet implemented
These commands **do not exist**; do not go looking for them:

| Wanted | Status |
|---|---|
| Evaluation script | `evaluation/evaluate.py` — **missing** |
| Inference on one image | **missing** |
| Inference on a folder | **missing** |
| Saving restored images | **missing** |
| Submission assembly | **missing** (checklist item 24) |
| σ-/texture-stratified reporting | **missing** (required by Part 6) |

---

## Section 8 — Debugging checklist

Work top-down; each step assumes the ones above passed.

**1. Environment.**
`python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`.
This repo is developed CPU-only; AMP silently disables itself off CUDA
(`trainer.amp_enabled`), so an "AMP works" result on CPU means nothing.

**2. Data.** Does `data/packed/manifest.json` exist? Re-run `pack_dataset.py --force`
if counts look wrong. Then run `scripts/baselines.py` — if bicubic ≠ 21.67 dB, stop and
fix the data before touching the model.

**3. Configuration.** Did you construct a config with `dataclasses.replace`? A
`TypeError: 'member_descriptor'` or `AttributeError: no attribute '__dict__'` means you
hit the `slots=True` trap. A `ValueError` from `validate()` means you asked for an
unsanctioned batch size or `warmup_epochs >= epochs`.

**4. Model construction.** `ModuleNotFoundError: models.attention.gsa_block` is
expected — that module is Phase 4.12. Use `sparc-tiny` or disable attention.

**5. Shapes.** `SPARCNet.forward_with_aux` validates rank, channel count and
divisibility by `2**num_levels` up front, so a shape error should name the problem
directly. Input spatial size must be divisible by 8.

**6. Loss.** `ValueError: MS-SSIM ... needs images of at least 161 px` means you fed it
a small tensor — 5 pyramid scales of an 11×11 window need 161 px. Disable the term for
toy tests: `CompositeLoss(enabled={"ms_ssim": False})`.

**7. NaN.** The trainer logs `Non-finite loss at step N; skipping batch` and increments
`trainer.skipped_batches`. If that fires repeatedly: check the AMP scale in TensorBoard
(`optim/amp_scale` collapsing toward zero means persistent overflow), then the gradient
norm (`grad/global_norm`). Charbonnier, FFT, MS-SSIM and the noise loss all force
float32 internally precisely because they are the fp16-fragile ones.

**8. Not learning.** Compare `loss/*` terms individually in TensorBoard — an unweighted
`raw_*` term that is flat while others fall localises the problem immediately. Check
`optim/lr` is non-zero (a finished cosine schedule reaches 1e-6). Confirm the split is
sane: 360 train batches / 40 val batches at batch 8.

**9. Checkpoint/resume.** `load(..., resume=True)` sets `state.epoch = saved + 1`. If
resume appears to restart from scratch, check you did not pass `resume=False`.

**10. Reproducibility.** `set_seed(1337)` covers python, numpy, torch and cudnn. Dataset
augmentation is seeded *per index* (`seed * 1000003 + index`), so a given image gets the
same augmentation regardless of batch composition or worker count.

---

## Section 9 — Code walkthrough of the important functions

**`configs.sparc_config.build_sparc_config(variant, **overrides)`** — returns a
validated `SparcConfig` for `sparc-tiny` or `sparc-base`. Called by `train.py`,
`overfit.py`, `benchmark.py`, all tests. Overrides go through `with_overrides` →
`dataclasses.replace`, so validation re-runs.

**`datasets.splits.group_aware_split(n, block_size, every_n)`** — returns
`SplitIndices(train, val)`. Exists because a random split leaks near-duplicate scenes
across the boundary and inflates validation PSNR.

**`datasets.degradation.synthesize_lr(gt, params, generator, noise_at_lr)`** — applies
the Phase 1 forward model. The `noise_at_lr` flag is amendment A-001: injecting noise
before decimation delivers only 50 % of the measured speckle variance, because bicubic
decimation attenuates white noise.

**`datasets.degradation.fit_noise_parameters(noisy, clean)`** — closed-form per-image
least squares of `Var(r|I) = a + cI²`. Returns `(σ_gauss, σ_speckle)`, each `(B,)`.
Used by the noise-auxiliary loss. The three-parameter form was measured ill-conditioned
(corr(b,c) = −0.90), which is why only two parameters are fitted.

**`models.sparc_net.SPARCNet.forward_with_aux(y)`** — the real forward pass. Returns
`SparcOutput(image, sigma, stats, noise)`. `forward()` is a thin wrapper returning
`.image` only, so inference is unchanged by anything the loss needs.

**`models.noise.noise_head.NoiseHead.predict_parameters(y)`** — trunk → GAP → MLP →
softplus → clamp. Returns `(σ_g, σ_s)` each `(B,1)`. **Gotcha:** at initialisation the
final layer's weight is zero by contract, so the trunk receives *zero gradient* for
exactly one optimiser step. That is intended, and there is a test pinning it so a real
detachment bug can be told apart from it.

**`losses.composite_loss.CompositeLoss.forward(output, batch)`** — accepts either
`(SparcOutput, batch_dict)` or `(pred_tensor, target_tensor)`, so it is usable as a
drop-in reconstruction criterion in tests. Returns `(total, terms)` where `terms` has
each term's weighted contribution, its unweighted `raw_*` value, and `total`. The noise
term is skipped silently when the head is disabled — an ablation must not crash the
objective.

**`trainer.trainer.build_param_groups(model, weight_decay)`** — two groups, decayed and
not. Exclusion is by `ndim <= 1` plus name match on `norm`/`gamma`/`rel_pos`/`bias`.
`LayerScale`'s parameter is literally named `gamma` and is 4-D, so the keyword match is
load-bearing.

**`trainer.trainer.warmup_cosine_lambda(warmup_steps, total_steps, min_ratio)`** —
returns the LR multiplier callable. Step 0 gives exactly 1e-6; the end of schedule gives
exactly 1e-6.

**`trainer.ema.ModelEma.update(model)`** — `shadow = decay·shadow + (1−decay)·live`,
with a ramp for the first `warmup_steps` so the average is not dominated by the random
initialisation. Buffers are *copied*, not averaged — averaging running statistics is
meaningless.

**`utils.complexity.measure_complexity(module, inputs)`** — measures real dispatched
FLOPs with `FlopCounterMode`. A measurement, not an estimate. Note `macs = flops // 2`.

---

## Section 10 — The five ideas that explain the whole design

**1. Lossless resampling everywhere.** Every downsample is a Haar DWT and every upsample
a Haar IDWT. Strided convolutions and pooling throw information away; the Haar transform
is orthonormal and exactly invertible, so the 4× compute saving at each level costs
nothing in principle. This is why the model fits in 2.35 M parameters and 1.17 GB of
training VRAM.

**2. No activation functions.** There is no ReLU or GELU anywhere. Every nonlinearity is
a `SimpleGate`: split the channels in half and multiply. Combined with `LayerScale`
initialised to 1e-2, the network starts near the identity and trains without warmup
tricks. If you add a module, use SimpleGate; introducing an activation would be a
contract violation.

**3. The model predicts a correction, not an image.** The final output is
`network_output + bicubic_upsample(normalised_input)`. The network only has to learn the
*difference* from a decent classical baseline, which is a far easier function.

**4. Noise level is an input, not an assumption.** Phase 1 measured noise varying 8.5×
across images, and σ scales with intensity so it varies spatially too. The noise head
estimates it blindly and feeds a per-pixel σ map to the trunk as a second input channel.
`GatedFuse` exists for the same reason: how much to trust the skip connection depends on
how noisy this particular image is.

**5. Measurement beats opinion.** Every constant traces to a Phase 1 measurement, and
every acceptance test is a number. When the step-6 gate failed at 33.16 dB, the
resolution was not to lower the bar but to run controlled experiments — vary data at
fixed model, vary model at fixed data — establish capacity as the cause, and record an
amendment with the evidence. Follow that pattern: if you want to change something,
measure first, write the amendment, then change the code.

### Where to look when you want to change X

| You want to… | Go to |
|---|---|
| Change a hyperparameter | `configs/sparc_config.py` — and write an amendment |
| Add/modify a loss | `losses/<name>.py`, register in `composite_loss.py`, test in `tests/test_losses.py` |
| Change augmentation | `datasets/transforms.py` (geometric) or `degradation.py` (synthesis) |
| Change the training loop | `trainer/trainer.py` |
| Add a logged scalar | `trainer/tb_layout.py` first, then emit via `Trainer._log_scalar` |
| Change the architecture | Don't — the contract is frozen. Amend Part 16 first. |

### What is left to build

| Phase | Item | Status |
|---|---|---|
| 4.11 | `GatedFuse` | **Done** — 23,256 / 5,868 params, exact; `reports/PHASE4_11_VERIFICATION.md` |
| 4.12 | `GSABlock` + relative position bias | Not started; highest risk |
| 4.13 | Full integration, 2.346 M params, 2.449 G MACs | Blocked on 4.11–4.12 |
| 4.14 | Profiling | GPU numbers need the RTX A400 |
| 4.15 | ONNX / TensorRT export | Scripts prepared, not run |
| — | `evaluation/evaluate.py`, image saving, stratified reporting | Not started |
| — | `README.md` | Missing (checklist item 25) |
