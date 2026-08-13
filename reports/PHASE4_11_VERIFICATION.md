# Phase 4.11 — Gated Feature Fusion: Verification Report

**Date:** 2026-08-06
**Status:** **COMPLETE.** Contract Part 8 step 11 implemented; ablation A3 outstanding (training-time).
**Scope:** replace the temporary concatenation skip fusion with the contract-defined
`GatedFuse` (Contract Part 2.7, Part 3 stages 15 and 20).

Governed by `SPARC_BASE_V1_IMPLEMENTATION_CONTRACT.md` (frozen), `AMENDMENTS.md`
(A-001, A-002), `reports/PHASE4_7_AUDIT.md` (finding V-1), and
`reports/PHASE4_TEST_SPECIFICATIONS.md` (Phase 4.11 section).

**No amendment was required.** The module was implemented exactly as Part 2.7 specifies.
Every figure below is measured, not derived; reproduce with:

```bash
python reports/inspect_fusion.py            # -> reports/report_fusion.json
python -m pytest tests/test_fusion.py -q    # 38 passed, 1 skipped (CUDA)
```

---

## 1. Files added

| Path | Purpose |
|---|---|
| `models/fusion/gated_fuse.py` | `GatedFuse` — the Part 2.7 module |
| `tests/test_fusion.py` | 39 tests (Part 14 requires this file; Part 9 defines its matrix) |
| `reports/inspect_fusion.py` | Measurement harness producing every number in this report |
| `reports/report_fusion.json` | Machine-readable artefact of that harness |
| `reports/PHASE4_11_VERIFICATION.md` | This report |
| `outputs/integration_4_11_gated.{json,log}` | Integration run, gated arm |
| `outputs/integration_4_11_concat.{json,log}` | Integration run, concat control arm |

## 2. Files modified

| Path | Change | Why |
|---|---|---|
| `models/fusion/__init__.py` | Removed the dead deferred `import GatedFuse` inside `build_fusion` | The module is now imported at the top of the file; the inner import was a placeholder from when `gated_fuse.py` did not exist. **`build_fusion`'s signature and behaviour are unchanged.** |
| `scripts/integration_check.py` | Added `--concat-fusion`; report now records `fusion` and `parameters` | Lets the same harness run both A3 arms |
| `docs/ONBOARDING.md` | Marked 4.11 done; corrected the "not yet written" and `--stage6` notes | The doc stated `GatedFuse` did not exist |

**No public interface changed.** `SPARCNet.forward`, `SPARCNet.forward_with_aux`,
`build_fusion(config, channels)`, `Decoder`, `DecoderLevel` and every config field are
byte-for-byte as they were. Only the implementation behind the factory was replaced.

## 3. Classes added

| Class | Module | Notes |
|---|---|---|
| `GatedFuse(channels, reduction=4)` | `models/fusion/gated_fuse.py` | Signature was already fixed by the existing `build_fusion` call site; matched exactly. |

No other class was added, removed or altered. `ConcatFusion` is retained unchanged as the
ablation A3 control arm (Contract Part 11).

## 4. Functions added

| Function | Signature | Purpose |
|---|---|---|
| `GatedFuse.__init__` | `(channels: int, reduction: int = 4)` | Builds `project`, `squeeze`, `excite`; rejects non-positive channels/reduction and a collapsed bottleneck |
| `GatedFuse.parameter_count` | `staticmethod(channels, reduction=4) -> int` | Analytic count, asserted equal to the module's real count and to Part 3 |
| `GatedFuse.gate` | `(skip, decoded) -> (B,C,1,1)` | The per-channel mixing weight, exposed so tests and diagnostics can inspect it without re-deriving the fusion |
| `GatedFuse.forward` | `(skip, decoded) -> (B,C,H,W)` | The fusion |
| `GatedFuse.extra_repr` | `() -> str` | `channels`, `reduction`, `hidden` in the module tree |

`reports/inspect_fusion.py` additionally defines nine measurement functions
(`measure_shapes`, `measure_parameters`, `measure_module_flops`,
`measure_model_complexity`, `measure_activation_memory`, `measure_gradient_reach`,
`measure_stability`, `measure_readiness`, `measure_latency`, `measure_factory`). These
are reporting tools and are not imported by the model, the trainer or the losses.

### Implementation, verbatim against Part 2.7

```python
u   = Conv1x1(2C -> C)(concat([skip, dec], dim=1))
g   = sigmoid( Conv1x1(C//4 -> C)( Conv1x1(C -> C//4)( GAP(u) ) ) )
out = g * skip + (1 - g) * dec
```

Two deliberate, precedented notes:

* **`mean(dim=[2,3], keepdim=True)` in place of `AdaptiveAvgPool2d(1)`.** Numerically
  identical, one operator fewer, and it exports without a resize node. This is the same
  substitution `NAFBlock`'s SCA already uses, which `PHASE4_7_AUDIT.md` Task 1 assessed
  as "conformant and more export-friendly".
* **No activation between the two gate convolutions.** Part 5 fixes the activation set
  to "none (SimpleGate only)", and a SimpleGate there would halve the bottleneck and
  break the parameter budget. The composition is a low-rank linear map followed by the
  sigmoid, which supplies the nonlinearity. Part 3's counts confirm exactly two
  convolutions in the gate — see §6.

---

## 5. Tensor shape verification

Both fusion sites, `B ∈ {1, 2, 8}`. All 6 rows: output shape equals input shape.

| Site | B | `skip` | `decoded` | gate | output |
|---|---|---|---|---|---|
| Dec D1 (stage 15) | 1 | `(1,96,32,32)` | `(1,96,32,32)` | `(1,96,1,1)` | `(1,96,32,32)` |
| Dec D1 (stage 15) | 2 | `(2,96,32,32)` | `(2,96,32,32)` | `(2,96,1,1)` | `(2,96,32,32)` |
| Dec D1 (stage 15) | 8 | `(8,96,32,32)` | `(8,96,32,32)` | `(8,96,1,1)` | `(8,96,32,32)` |
| Dec D0 (stage 20) | 1 | `(1,48,64,64)` | `(1,48,64,64)` | `(1,48,1,1)` | `(1,48,64,64)` |
| Dec D0 (stage 20) | 2 | `(2,48,64,64)` | `(2,48,64,64)` | `(2,48,1,1)` | `(2,48,64,64)` |
| Dec D0 (stage 20) | 8 | `(8,48,64,64)` | `(8,48,64,64)` | `(8,48,1,1)` | `(8,48,64,64)` |

The gate is `(B,C,1,1)`, confirming it is **global per channel** — "how much do I trust
this channel of the skip for this image" — and not a per-pixel mask, which would be a
different module than the contract specifies.

Whole-model shape is unchanged: `(B,1,128,128) → (B,1,256,256)`, output in `[0,1]`.

## 6. Exact parameter count

Part 9 requires an **exact** match, not a tolerance.

| Site | Measured | Analytic | Contract (Part 3) | Verdict |
|---|---|---|---|---|
| Dec D1, `C=96` | **23,256** | 23,256 | 23,256 | **EXACT** |
| Dec D0, `C=48` | **5,868** | 5,868 | 5,868 | **EXACT** |

Breakdown at `C=96`: `project` 2·96·96+96 = 18,528 · `squeeze` 96·24+24 = 2,328 ·
`excite` 24·96+96 = 2,400. Total 23,256. This reproduces audit finding **V-1**, which
predicted both figures analytically before a line was written.

**Model level** (`sparc-base`, attention off — Phase 4.12 pending):

| Fusion | Parameters | Disk fp32 |
|---|---|---|
| GatedFuse | 1,737,562 | 6.628 MB |
| ConcatFusion | 1,731,622 | 6.606 MB |
| **Delta** | **+5,940** | +0.022 MB |

`+5,940` = `(23,256 + 5,868) − (18,528 + 4,656)`, i.e. exactly the two gate MLPs and
nothing else. **No parameter inflation.**

## 7. MAC / FLOP impact

Measured with `FlopCounterMode` (real dispatched operators, not an estimate).

| Site | Gated MACs | Concat MACs | Delta | Contract `2C²HW + C²/2` |
|---|---|---|---|---|
| Dec D1 (`C=96`, 32²) | 18,878,976 | 18,874,368 | +4,608 | 18,878,976 |
| Dec D0 (`C=48`, 64²) | 18,875,520 | 18,874,368 | +1,152 | 18,875,520 |

Both land on **18.88 MMAC**, exactly the Part 3 stage 15 and stage 20 cells — to the
last digit, not within ±5 %. (The two sites cost the same because `2C²HW` is invariant
when `C` halves and `H·W` quadruples.)

| Model | MACs | GFLOPs |
|---|---|---|
| GatedFuse | 1,819,042,496 | 3.6381 |
| ConcatFusion | 1,819,036,736 | 3.6381 |
| **Delta** | **+5,760** | **+0.0000058** |

**+0.0003 % MACs.** The gate's two convolutions run on a pooled `(B,C,1,1)` tensor, so
their cost is `C²/2` per site and independent of resolution — which is precisely why
Part 2.7 pools before gating rather than after.

## 8. Memory impact

Measured with `torch.autograd.graph.saved_tensors_hooks`: the tensors autograd retains
for backward, minus the module's own inputs and parameters, plus the output — i.e. the
tensors this module **allocates**. `skip` and `decoded` are charged to the stages that
produced them, and the two `(B,C,1,1)` gate tensors are negligible. Part 3 quotes fp16
decimal megabytes.

| Site | Elements | × `C·H·W` | Measured | Contract | Deviation |
|---|---|---|---|---|---|
| Dec D1 (stage 15) | 295,224 | **3.00** | 0.590 MB | 0.590 MB | **+0.08 %** |
| Dec D0 (stage 20) | 589,980 | **3.00** | 1.180 MB | 1.180 MB | **−0.00 %** |

Part 2.7 budgets `3·C·H·W` — the concatenated tensor (`2·C·H·W`) plus the output
(`C·H·W`). Measurement confirms exactly that, well inside Part 9's ±15 %. The overhead
versus `ConcatFusion` is **312 and 156 elements** respectively (the pooled and gate
tensors), i.e. **0.1 % and 0.03 %**. At batch 8 the whole-model increase is under 8 kB.

Part 10's activation budget (160 MB/image) is unaffected.

## 9. Training impact

`python scripts/integration_check.py --epochs 5 --subset 128`, `sparc-base` with
attention off, full `CompositeLoss` (all 6 terms), noise head active, seed 1337,
batch 8, 16 steps/epoch, CPU.

| | concat (Phase 4.10 baseline) | concat (re-run, 4.11 code) | **gated** |
|---|---|---|---|
| Parameters | — | 1,731,622 | 1,737,562 |
| val PSNR, epoch 0 | 19.8281 | 19.8281 | 21.8720 |
| **val PSNR, epoch 4** | **24.3868** | **24.3868** | **24.7026** |
| Charbonnier | 0.051435 | 0.051435 | 0.049431 |
| MS-SSIM | 0.151539 | 0.151539 | 0.148179 |
| Wavelet | 0.053250 | 0.053250 | 0.051422 |
| FFT | 0.031574 | 0.031574 | 0.029886 |
| Gradient | 0.025674 | 0.025674 | 0.024963 |
| Noise aux | 0.204107 | 0.204107 | 0.204012 |
| **Total** | **0.086435** | **0.086435** | **0.083622** |
| Noise-head log-σ MAE | 0.184427 | 0.184427 | 0.184249 |
| NaN encountered | False | False | **False** |

Every checked property holds:

* **No NaNs** — `any_nan: false`; zero skipped batches.
* **Stable optimisation** — all 6 terms and the total decrease monotonically in the
  first/last comparison; no loss spike; PSNR rises every epoch
  (21.872 → 23.445 → 24.270 → 24.636 → 24.703).
* **Composite Loss works** — all 6 terms active and independently decreasing.
* **Noise Head works** — log-σ MAE improves (0.187088 → 0.184249), and it improves
  slightly *more* than under concat.
* **TensorBoard logging unchanged** — see §12.
* **No regression from Phase 4.10** — see §12.

**Determinism datum:** an earlier interrupted run of the identical gated configuration
reproduced epochs 0–1 bit-for-bit (loss 0.129694, 21.8720 dB, 23.4453 dB).

## 10. Inference impact

Whole-model CPU latency, batch 1, `sparc-base` (attention off), 40 interleaved repeats.

| Fusion | median | p10 | p90 | min |
|---|---|---|---|---|
| GatedFuse | 218.0 ms | 127.2 | 243.4 | 114.8 |
| ConcatFusion | 215.7 ms | 122.1 | 242.3 | 117.4 |
| **Overhead** | **+1.06 %** | | | **−2.14 %** |

The overhead is **indistinguishable from zero** — its sign flips between repeats of the
measurement — which is what `+0.0003 %` MACs and `+0.1 %` activations predict.

> **Methodological note, recorded because it nearly became a false finding.** Timing one
> model to completion and then the other reported **+11.8 %** for the gated arm. That
> figure is an artefact: per-iteration spread on this host is enormous (p10 ~115 ms,
> p90 ~275 ms), so sequential measurement attributes all drift to whichever model ran
> second. Interleaving the arms and alternating their order removes it. The harness now
> interleaves by construction.

CPU latency is **not** a Part 10 budget — those limits are GPU-denominated. See §14.

## 11. Unit tests added

`tests/test_fusion.py` — **39 tests, 38 passed, 1 skipped** (CUDA, no device).

| Group | Tests | Covers |
|---|---|---|
| Parameters | 5 | Exact 23,256 / 5,868; analytic == measured; invalid-config rejection |
| Shapes | 5 | `B ∈ {1,2,8}`; gate is `(B,C,1,1)`; mismatched/wrong-channel inputs rejected |
| Gating | 7 | `g ∈ (0,1)`; safe saturation; **convex combination**; `g→1` returns `skip` exactly and `g→0` returns `dec`; identical inputs returned unchanged; gate adapts to its input |
| Gradients | 2 | Every parameter and both inputs get finite non-zero grads, in isolation and inside the assembled `SPARCNet` |
| Stability | 2 | 100 random batches at ×1e-3, ×1, ×1e3; no BatchNorm (Part 7) |
| MACs / memory | 4 | 18.88 MMAC ±5 %; 0.590 / 1.180 MB ±15 % |
| GPU / export | 6 | autocast, channels-last, TorchScript, `torch.compile`, ONNX, CUDA (skipped) |
| Integration | 5 | `build_fusion` dispatch; both decoder stages; model forward shape/range; state-dict round-trip with concat checkpoints correctly **rejected**; CompositeLoss + NoiseHead end-to-end |
| Regression vs concat | 3 | Parameter delta; model-level delta; FLOP overhead < 1 % |

Three tests exist specifically to catch bugs that would otherwise pass everything else:

* `test_saturated_gate_selects_one_input_exactly` — swapping the `skip`/`dec` operands
  still trains and still produces plausible numbers, but inverts the module's meaning.
* `test_gate_adapts_to_the_input` — a gate that ignored its inputs is still a valid
  convex combination and would leave every other test green.
* `test_full_model_gradients_reach_both_fusion_modules` — a gate that receives no
  gradient inside the assembled network passes every isolated test while silently
  degenerating to a constant mix.

### Verification matrix (Part 9 + the task's list)

| Requirement | Result |
|---|---|
| Exact tensor shapes | ✓ 6/6 configurations |
| Exact parameter count | ✓ 23,256 / 5,868, exact |
| MAC count | ✓ 18.88 MMAC, exact |
| FLOPs | ✓ model +0.0000058 GFLOP (+0.0003 %) |
| Gradient propagation | ✓ every parameter, both inputs, isolated and in-model |
| Numerical stability | ✓ 100 batches, ×1e-3 to ×1e3, all finite |
| AMP | ✓ bf16 autocast finite, max diff 2.85e-03 (CPU; fp16 needs CUDA — §14) |
| channels-last | ✓ max diff 1.19e-07 |
| `torch.compile` | ✓ `fullgraph=True`, max diff **0.0** |
| TorchScript | ✓ max diff **0.0** |
| ONNX | ✓ max diff 1.19e-07 (≪ 1e-3); opset caveat in §14 |
| CPU | ✓ all of the above |
| CUDA | ⏭ auto-skipped, no device (§14) |

## 12. Regression test results

**Full suite: 336 passed, 2 skipped, 0 failed** (was 298 before Phase 4.11; +38 fusion
tests). No pre-existing test was modified, disabled or re-baselined.

**The strongest regression evidence** is the concat control arm. Re-running the Phase
4.10 integration configuration against the Phase 4.11 codebase reproduces the stored
baseline **to within 1e-12 on every one of the 7 loss terms and on val PSNR**
(24.386797… dB in both). Phase 4.11 therefore changed nothing on the path it did not
touch.

**TensorBoard logging unchanged:** the gated run emits **56 scalar tags**, the concat run
emits **56**, and the Phase 4.10 baseline emitted **56** — the three sets are identical.
Nothing added, nothing lost, no tag renamed. (`train/*`, `loss/*`, `step_loss/*` — with
per-term and `raw_*` variants — plus `val/*`, `optim/*`, `throughput/*`.)

## 13. Contract clauses satisfied

| Clause | Requirement | Evidence |
|---|---|---|
| **Part 2.7** | The three-line `GatedFuse` definition | Implemented verbatim (§4) |
| **Part 2.7** | Params `2C² + C + 2·(C·C/4) + C/4 + C` | 23,256 / 5,868, exact (§6) |
| **Part 2.7** | MACs `2C²·H·W + C²/2` | 18,878,976 / 18,875,520, exact (§7) |
| **Part 2.7** | Activations `3·C·H·W` | Measured 3.00 × `C·H·W` (§8) |
| **Part 2.7** | "Output range: convex combination of inputs" | Asserted elementwise; 100-batch worst breach 2.44e-04 at ×1e3 (§14) |
| **Part 1** | `out = g·skip + (1−g)·dec`, `g` from `sigmoid` | Operand order pinned by a dedicated test |
| **Part 1** | Exactly 2 long skips, GatedFuse(96) and GatedFuse(48) | `decoder_fusion_channels == [96, 48]` |
| **Part 1** | "No concatenations anywhere except inside `GatedFuse`" | Satisfied — the only `cat` is `GatedFuse.project`'s input |
| **Part 3** | Stage 15 = 23,256 params / 18.88 MMAC / 0.590 MB | All three exact (§6–8) |
| **Part 3** | Stage 20 = 5,868 params / 18.88 MMAC / 1.180 MB | All three exact (§6–8) |
| **Part 5** | Fusion reduction = 4 | Read from `config.fusion_reduction`, asserted |
| **Part 5** | Dropout 0.0; activation "none (SimpleGate only)" | Neither appears (§4) |
| **Part 5** | channels-last ON | Verified, 1.19e-07 (§11) |
| **Part 7** | `GatedFuse` is **CORE**, in V1 | On by default (`use_gated_fusion=True`) |
| **Part 7** | BatchNorm forbidden | Asserted absent |
| **Part 8 step 11** | `models/fusion/gated_fuse.py`, test T11, **gate ∈ (0,1)** | Implemented; gate range asserted (§14) |
| **Part 9** | Shape / grad / stability / params / TorchScript / ONNX / memory | Full matrix (§11) |
| **Part 10** | No budget exceeded | Params 1.738 M < 2.60 M; MACs 1.819 G < 2.80 G; GFLOPs 3.638 < 5.60; disk 6.63 MB < 12 MB. VRAM/latency unmeasurable (§14) |
| **Part 11 (A3)** | GatedFuse vs concatenation | Both arms implemented and runnable; the scored ablation is outstanding (§14) |
| **Part 14** | File at `models/fusion/gated_fuse.py`, tests at `tests/test_fusion.py` | Both as specified |
| **Part 15 item 16** | "Fusion verified (gate range, ablation A3)" | Gate range ✓; A3 outstanding |

**Compatibility, as required by the task:** Noise Head ✓ (integration §9 and a dedicated
test) · Composite Loss ✓ (all 6 terms, §9) · current Trainer ✓ (unmodified) · existing
configuration files ✓ (`use_gated_fusion` / `fusion_reduction` already present, no new
field) · existing TensorBoard logging ✓ (§12) · existing checkpoints — see §14.

## 14. Known limitations

1. **Ablation A3 is not run.** The +0.316 dB in §9 is **not** ablation A3 and must not be
   quoted as one. It is a single seed, 5 epochs, a 128-image subset, with attention
   disabled. Contract Part 11 requires 3 seeds on the full schedule and split, reporting
   mean *and* median PSNR/SSIM/LPIPS, latency and peak VRAM. **A3 remains open** and is
   the only outstanding Part 8 step-11 acceptance item.
   There is also a confound worth stating: the two arms **cannot** share an
   initialisation, because the gated model draws more random numbers at construction, so
   the RNG stream diverges and every downstream layer initialises differently. Their
   epoch-0 PSNRs differ (21.87 vs 19.83) for that reason alone. A3's 3-seed protocol
   exists precisely to average this out.

2. **CUDA is unverified — no device on this host.** The CUDA test is written and
   auto-skips. Consequently, on the RTX A400 it remains to confirm: **fp16** autocast
   (only bf16-on-CPU is verified here), channels-last on real CUDA kernels, peak VRAM,
   and GPU latency at batch 16 for Part 12's promotion rule. Part 10's VRAM and latency
   budgets stay **UNMEASURED**, consistent with `PHASE4_7_AUDIT.md`.

3. **ONNX exports at opset 18, not the contract's 17.** Torch 2.10's exporter implements
   opset ≥ 18; requesting 17 triggers an automatic down-conversion that fails inside the
   ONNX C API (`axes_input_to_attribute.h: No initializer or constant input to node
   found`) and the file is kept at 18. This is a **toolchain** property, not a module
   property — the export itself succeeds and parity is **1.19e-07**, four orders inside
   the 1e-3 requirement. Flagged for Phase 4.15, where it affects the whole model, not
   just this module.

4. **`torch.compile` verified with `backend="eager"` only.** That exercises Dynamo
   tracing and passes `fullgraph=True` with **0.0** difference. Inductor's C++ codegen
   needs an MSVC toolchain this host lacks. Part 5 keeps `torch.compile` **OFF** in V1
   regardless, so this is a readiness check, not a shipped path.

5. **The gate saturates to exactly 0.0 / 1.0 at feature magnitudes near 1e3.** That is
   float32 arithmetic, not a defect: the gate stays in `[0,1]`, output stays finite, and
   the convex-combination guarantee holds — a saturated gate simply selects one input
   outright. The consequence worth knowing is that the *gate's own* gradient vanishes
   there, so a run whose features reached that magnitude would stop adapting the mix. The
   trunk operates on normalised features, so this is a stress condition, not an operating
   point. Documented in the module and pinned by a test. Strictly, then, Part 8's
   "gate ∈ (0,1)" holds on the open interval at realistic magnitudes and closes to
   `[0,1]` under float32 saturation.

6. **Worst convexity breach 2.44e-04**, at input magnitude 1e3 — a relative error of
   2.4e-07, i.e. float32 rounding on the `g·skip + (1−g)·dec` sum. At magnitude 1 the
   breach is below 1e-06.

7. **Existing checkpoints: gated and concat checkpoints are not interchangeable, by
   design.** A `GatedFuse` checkpoint round-trips exactly. Loading a **concat**
   checkpoint into a gated model raises `RuntimeError: Missing key(s)` for
   `squeeze.*`/`excite.*` — asserted by a test. This is the correct behaviour: silently
   accepting it would leave the gate at random initialisation while reporting a
   successful resume. Pre-4.11 checkpoints must therefore be re-trained or loaded into a
   `use_gated_fusion=False` model. No checkpoint written by any completed SPARC phase is
   invalidated, since the model total changes for Phase 4.12 anyway.

8. **Model total is 1,737,562 — not the contract's 2,345,650.** Expected: attention is
   Phase 4.12. The 608,088 difference is exactly the three GSA groups
   (124,902 + 420,735 + 62,451 = 608,088). Part 3's totals become checkable at Phase 4.13.

9. **CPU latency on this host is too noisy for a 10 %-level judgement** (p10 115 ms,
   p90 275 ms — a 2.4× spread). §10's conclusion is only that the overhead is within
   noise. Part 12 condition 4 ("runtime increase ≤ 10 %") must be evaluated on the A400
   at batch 16, not here.

## 15. Comparison against the temporary concat fusion

| | `ConcatFusion` (temporary) | `GatedFuse` (contract) |
|---|---|---|
| Mixing rule | `Conv1x1(2C→C)(concat)` — weights **fixed** after training | `g·skip + (1−g)·dec`, `g` computed **per image, per channel** |
| Params, C=96 / C=48 | 18,528 / 4,656 | **23,256 / 5,868** (+4,728 / +1,212) |
| Model params | 1,731,622 | 1,737,562 (**+5,940**, +0.34 %) |
| Model MACs | 1,819,036,736 | 1,819,042,496 (**+5,760**, +0.0003 %) |
| Activations, stage 15 / 20 | 0.590 / 1.180 MB | 0.590 / 1.180 MB (+0.1 % / +0.03 %) |
| CPU latency, batch 1 | 215.7 ms | 218.0 ms (+1.06 %, within noise) |
| Output bound | Unbounded — a projection can amplify | **Convex combination** — bounded elementwise by its two inputs |
| 5-epoch val PSNR | 24.3868 dB | **24.7026 dB** (+0.3158 dB) |
| 5-epoch total loss | 0.086435 | **0.083622** (−3.3 %) |
| Contract status | Ablation A3 control arm (Part 11) | **CORE**, V1 (Part 7) |

**Why the contract specifies it.** Phase 1 measured the noise level varying **8.5×
across images**, and because σ scales with intensity it varies spatially within each
image too. A fixed mixing rule cannot express "trust the skip more on *this* image": on
a clean image the skip carries recoverable high-frequency detail worth keeping, while on
a noisy one it largely carries noise the decoder has already suppressed. `GatedFuse`
makes that decision per channel from the joint statistics of both inputs, which is
exactly the four behaviours the task specifies — which encoder features to preserve,
which decoder features dominate, which channels to suppress, and adaptive blending.

The convex-combination property is what keeps it safe: the coefficients sum to exactly 1,
so the fusion can never amplify, whatever the gate learns. That is a stronger stability
guarantee than the projection it replaces, at +0.34 % parameters and no measurable
compute or latency cost.

## 16. Readiness for Phase 4.12

**Confirmed ready.** Specifically:

* **No public interface moved.** `SPARCNet.forward` / `forward_with_aux`,
  `build_fusion(config, channels)`, `Decoder`, `DecoderLevel` and every config field are
  unchanged. Phase 4.12 touches `models/attention/`, which does not import
  `models/fusion/`.
* **`DecoderLevel` already sequences `upsample → fusion → gsa → naf`**, so the GSA slot
  at D1 is wired and currently an empty `nn.Sequential`. Nothing about the fusion needs
  revisiting when it is filled.
* **Parameter accounting is closed.** With fusion exact at 23,256 + 5,868 and the model
  at 1,737,562, the remaining gap to Part 3's 2,345,650 is **exactly 608,088** — the
  three GSA groups audit finding V-1 already derived (124,902 + 420,735 + 62,451). Any
  Phase 4.12 drift will show up as a discrepancy against a known number, with one
  candidate cause.
* **The trainer, composite loss and noise head are verified against the gated model**,
  so a Phase 4.12 regression cannot be confused with a 4.11 one.
* **The A3 control arm is preserved and exercised.** `--concat-fusion` still runs and
  still reproduces the 4.10 baseline exactly, so the ablation remains available after
  attention lands.

**Carried into later phases:** ablation A3 (Part 11, training-time); CUDA/AMP-fp16,
VRAM and GPU-latency verification (Phase 4.14, needs the A400); the opset-17 export
question (Phase 4.15).

**Phase 4.12 was not begun.** No `GSABlock`, no `rel_pos.py`, no attention redesign, no
inference optimisation and no profiling work was performed.
