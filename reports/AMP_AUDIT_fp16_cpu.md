# AMP audit — device `cpu`, autocast dtype `torch.float16`

## 1. Operation policy

| Operation | No autocast | autocast(fp32 in) | autocast(reduced in) | Policy |
| --- | --- | --- | --- | --- |
| `conv2d` | torch.float32 | **torch.float16** | torch.float16 | DEMOTES to reduced (.float() does not survive) |
| `conv2d(.float() inputs)` | torch.float32 | **torch.float16** | torch.float16 | DEMOTES to reduced (.float() does not survive) |
| `conv2d depthwise` | torch.float32 | **torch.float16** | torch.float16 | DEMOTES to reduced (.float() does not survive) |
| `conv_transpose2d` | torch.float32 | **torch.float16** | torch.float16 | DEMOTES to reduced (.float() does not survive) |
| `linear` | torch.float32 | **torch.float16** | torch.float16 | DEMOTES to reduced (.float() does not survive) |
| `matmul` | torch.float32 | **torch.float16** | torch.float16 | DEMOTES to reduced (.float() does not survive) |
| `einsum` | torch.float32 | **torch.float16** | torch.float16 | DEMOTES to reduced (.float() does not survive) |
| `scaled_dot_product_attention` | torch.float32 | **torch.float16** | torch.float16 | DEMOTES to reduced (.float() does not survive) |
| `layer_norm (F.layer_norm)` | torch.float32 | **torch.float32** | torch.float16 | dtype follows input |
| `softmax` | torch.float32 | **torch.float32** | torch.float16 | dtype follows input |
| `log_softmax` | torch.float32 | **torch.float32** | torch.float16 | dtype follows input |
| `softplus` | torch.float32 | **torch.float32** | torch.float16 | dtype follows input |
| `fft.rfft2` | torch.complex64 | **torch.complex64** | torch.complex64 | no policy (dtype follows input) |
| `l1_loss` | torch.float32 | **torch.float32** | torch.float32 | promotes to fp32 (safe) |
| `mse_loss` | torch.float32 | **torch.float32** | torch.float32 | promotes to fp32 (safe) |
| `mean(dim=1)` | torch.float32 | **torch.float32** | torch.float16 | dtype follows input |
| `var(dim=1)` | torch.float32 | **torch.float32** | torch.float16 | dtype follows input |
| `rsqrt` | torch.float32 | **torch.float32** | torch.float16 | dtype follows input |
| `sqrt` | torch.float32 | **torch.float32** | torch.float16 | dtype follows input |
| `log` | torch.float32 | **torch.float32** | torch.float16 | dtype follows input |
| `pow(2)` | torch.float32 | **torch.float32** | torch.float16 | dtype follows input |
| `elementwise mul` | torch.float32 | **torch.float32** | torch.float16 | dtype follows input |
| `avg_pool2d` | torch.float32 | **torch.float32** | torch.float16 | dtype follows input |
| `interpolate bicubic` | torch.float32 | **torch.float32** | torch.float16 | dtype follows input |
| `pad replicate` | torch.float32 | **torch.float32** | torch.float32 | promotes to fp32 (safe) |

## 2. Model modules

- Reduced-precision outputs: **212**
- fp32 outputs: **105**

### LayerNorm2d sites receiving reduced-precision input

These compute mean, variance and rsqrt in reduced precision. The output dtype
is fp32 only because the affine parameters are fp32 — the moments are not.

- `noise_head.stages.0.norm`
- `noise_head.stages.1.norm`
- `noise_head.stages.2.norm`
- `noise_head.stages.3.norm`
- `encoder.levels.0.naf.0.norm1`
- `encoder.levels.1.naf.0.norm1`
- `encoder.levels.2.naf.0.norm1`
- `head.blocks.0.norm1`

### Largest activations (fp16 ceiling is 65504)

| Module | Class | dtype | absmax |
| --- | --- | --- | --- |
| `encoder.levels.0.naf.3.gate1` | SimpleGate | torch.float16 | 17.48 |
| `encoder.levels.1.naf.2.gate1` | SimpleGate | torch.float16 | 15.63 |
| `decoder.stages.0.naf.1.gate1` | SimpleGate | torch.float16 | 15.24 |
| `encoder.levels.0.naf.2.gate1` | SimpleGate | torch.float16 | 15.02 |
| `head.blocks.2.gate1` | SimpleGate | torch.float16 | 14.9 |
| `head.blocks.1.gate1` | SimpleGate | torch.float16 | 14.75 |
| `encoder.levels.0.naf.0.gate1` | SimpleGate | torch.float16 | 14.73 |
| `decoder.stages.1.naf.2.gate1` | SimpleGate | torch.float16 | 14.34 |
| `encoder.levels.2.naf.0.gate1` | SimpleGate | torch.float16 | 14.3 |
| `encoder.levels.0.naf.1.gate1` | SimpleGate | torch.float16 | 14.01 |
| `encoder.levels.1.naf.1.gate1` | SimpleGate | torch.float16 | 13.88 |
| `decoder.stages.1.naf.0.gate1` | SimpleGate | torch.float16 | 13.52 |

## 3. Loss terms

| Term | Returned dtype | Internal conv2d dtype | Finite |
| --- | --- | --- | --- |
| charbonnier | torch.float32 | n/a (no conv2d) | True |
| ms_ssim | torch.float32 | torch.float16 | True |
| wavelet | torch.float32 | n/a (no conv2d) | True |
| fft | torch.float32 | n/a (no conv2d) | True |
| gradient | torch.float32 | torch.float16 | True |
