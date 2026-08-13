"""GSABlock — exact global self-attention (Contract Part 2.6).

The NAF blocks are purely local: a 3x3 depthwise kernel and a global-average channel
scale. What they cannot do is relate two distant regions that happen to share texture
statistics, which is exactly what speckle suppression needs — the evidence for "this is
a real edge, not noise" often lives elsewhere in the image.

Part 2.6 answers that with **unrestricted** multi-head self-attention: every token
attends to every other token, with a learned relative-position bias on the logits. No
windows, no sparsity, no approximation. What is sparse is *where* the block is
instantiated — Enc L1 (32x32), Enc L2 (16x16) and Dec D1 (32x32) only. At 64x64 the
``(HW)^2`` term is 16x larger and the block is prohibited.

Structure, exactly as specified::

    t       = LayerNorm2d(x)
    qkv     = Conv1x1(C -> 3d) -> DWConv3x3(3d)          d = C // 2
    q,k,v   = split(qkv, d)     reshape (B, heads, HW, 16)
    t       = SDPA(q, k, v, attn_mask=rel_pos_bias)      head_dim == 16 always
    t       = Conv1x1(d -> C)
    x       = x + LayerScale_1 * t
    u       = LayerNorm2d(x)                             GDFN
    u       = Conv1x1(C -> 2C) -> DWConv3x3(2C) -> SimpleGate -> Conv1x1(C -> C)
    x       = x + LayerScale_2 * u

Two attention paths, mathematically identical, selected by ``use_sdpa``:

* **SDPA** (training, mandatory). ``F.scaled_dot_product_attention`` fuses the softmax
  and never materialises the ``(heads, HW, HW)`` logit tensor. Part 2.6 records the
  naive path as costing **+20.8 MB/image**, which at batch 8 on a 4 GB A400 is ~166 MB.
* **Explicit** ``matmul -> add bias -> softmax -> matmul`` (export only). SDPA does not
  lower to ONNX opset 17, so ``scripts/export_onnx.py`` flips this flag. The two paths
  must agree to **1e-4** — see ``tests/test_attention.py``.

Parameter count is ``5C^2 + 40C + heads * (2n-1)^2``: **62,451** at ``C=96, h=3, n=32``
and **140,245** at ``C=160, h=5, n=16``, matching Part 3 stages 8, 12 and 16 exactly.

.. note::
   Contract Part 2.6 prints the formula as ``~5C^2 + 8C + heads*(2n-1)^2`` and marks it
   approximate. The exact linear term is 40C: two LayerNorms (4C), two LayerScales (2C),
   the two depthwise 3x3 kernels (30C for the 3d and 2C groups combined) and the
   remaining convolution biases (4C). The Part 3 table is authoritative and agrees with
   the arithmetic implemented here.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from models.attention.rel_pos import RelativePositionBias
from models.blocks.layer_norm import LayerNorm2d
from models.blocks.simple_gate import LayerScale, SimpleGate

ATTENTION_HEAD_DIM = 16
"""Contract Part 5: the attention head dimension is invariant at 16."""


class GSABlock(nn.Module):
    """Exact global multi-head self-attention with a GDFN feed-forward branch.

    Args:
        channels: Channel count ``C`` of the input and output.
        heads: Number of attention heads.
        spatial_size: Side length ``n`` of the (square) feature map this block runs on.
            Fixed at construction because the relative-position tables are sized from it.
        attn_dim_ratio: ``d = int(C * attn_dim_ratio)``. The contract fixes this at 0.5.
        ffn_expansion: Expansion of the GDFN branch. The contract fixes this at 2.
        layer_scale_init: Initial value of both residual scales.
        layer_norm_eps: Epsilon of the two LayerNorms.
        use_sdpa: Use the fused SDPA kernel. ``False`` selects the explicit export path.
        use_relative_position_bias: Add the learned relative-position bias to the logits.
        use_attention_branch: Instantiate the self-attention sub-branch. ``False`` is the
            GDFN-only ablation arm: the block keeps its feed-forward branch and drops
            attention entirely, isolating attention as the single independent variable.
            The attention parameters are not constructed, so they are absent from the
            state dict rather than present-but-frozen — a dead parameter would still be
            decayed by the optimiser and would corrupt the parameter-count comparison.

    Raises:
        ValueError: If any dimension is non-positive, if ``d`` is not divisible by
            ``heads``, or if the resulting head dimension is not 16.
    """

    def __init__(
        self,
        channels: int,
        heads: int,
        spatial_size: int,
        attn_dim_ratio: float = 0.5,
        ffn_expansion: int = 2,
        layer_scale_init: float = 1e-2,
        layer_norm_eps: float = 1e-6,
        use_sdpa: bool = True,
        use_relative_position_bias: bool = True,
        use_attention_branch: bool = True,
    ) -> None:
        super().__init__()
        if channels <= 0 or channels % 2 != 0:
            raise ValueError(f"channels must be a positive even number, got {channels}.")
        if heads <= 0:
            raise ValueError(f"heads must be positive, got {heads}.")
        if spatial_size <= 0:
            raise ValueError(f"spatial_size must be positive, got {spatial_size}.")
        if ffn_expansion <= 0:
            raise ValueError(f"ffn_expansion must be positive, got {ffn_expansion}.")

        attn_dim = int(channels * attn_dim_ratio)
        if attn_dim <= 0 or attn_dim % heads != 0:
            raise ValueError(
                f"attention dim {attn_dim} (channels={channels}, "
                f"ratio={attn_dim_ratio}) is not divisible by {heads} heads."
            )
        head_dim = attn_dim // heads
        if head_dim != ATTENTION_HEAD_DIM:
            raise ValueError(
                f"head_dim is {head_dim}, Contract Part 5 requires "
                f"{ATTENTION_HEAD_DIM} (channels={channels}, heads={heads})."
            )

        self.channels = channels
        self.heads = heads
        self.spatial_size = spatial_size
        self.num_tokens = spatial_size * spatial_size
        self.attn_dim = attn_dim
        self.head_dim = head_dim
        self.scale = float(head_dim) ** -0.5
        self.use_sdpa = use_sdpa
        self.use_attention_branch = use_attention_branch

        # --- attention branch
        # Registered as None rather than omitted so `self.qkv` etc. always resolve;
        # `nn.Module.__setattr__` keeps None out of the state dict, so a GDFN-only
        # checkpoint simply has no attention keys.
        if use_attention_branch:
            self.norm1 = LayerNorm2d(channels, eps=layer_norm_eps)
            self.qkv = nn.Conv2d(channels, 3 * attn_dim, kernel_size=1)
            self.qkv_dwconv = nn.Conv2d(
                3 * attn_dim, 3 * attn_dim, kernel_size=3, padding=1, groups=3 * attn_dim
            )
            self.project = nn.Conv2d(attn_dim, channels, kernel_size=1)
            self.scale1 = LayerScale(channels, layer_scale_init)
            self.rel_pos = (
                RelativePositionBias(heads, spatial_size)
                if use_relative_position_bias
                else None
            )
        else:
            self.norm1 = None
            self.qkv = None
            self.qkv_dwconv = None
            self.project = None
            self.scale1 = None
            self.rel_pos = None

        # --- GDFN branch
        hidden = channels * ffn_expansion
        self.norm2 = LayerNorm2d(channels, eps=layer_norm_eps)
        self.ffn_in = nn.Conv2d(channels, hidden, kernel_size=1)
        self.ffn_dwconv = nn.Conv2d(
            hidden, hidden, kernel_size=3, padding=1, groups=hidden
        )
        self.gate = SimpleGate()
        self.ffn_out = nn.Conv2d(hidden // 2, channels, kernel_size=1)
        self.scale2 = LayerScale(channels, layer_scale_init)

    @staticmethod
    def parameter_count(
        channels: int,
        heads: int,
        spatial_size: int,
        attn_dim_ratio: float = 0.5,
        ffn_expansion: int = 2,
        use_relative_position_bias: bool = True,
        use_attention_branch: bool = True,
    ) -> int:
        """Analytic parameter count, used by the Part 3 budget assertions.

        Args:
            channels: Channel count ``C``.
            heads: Number of attention heads.
            spatial_size: Side length ``n`` of the feature map.
            attn_dim_ratio: ``d / C``.
            ffn_expansion: GDFN expansion factor.
            use_relative_position_bias: Whether the rel-pos table is present.
            use_attention_branch: Whether the attention sub-branch is present.

        Returns:
            Number of parameters.
        """
        c = channels
        d = int(c * attn_dim_ratio)
        hidden = c * ffn_expansion
        total = (
            2 * c  # norm2
            + (c * hidden + hidden)  # ffn_in
            + (9 * hidden + hidden)  # ffn depthwise
            + ((hidden // 2) * c + c)  # ffn_out
            + c  # scale2
        )
        if use_attention_branch:
            total += (
                2 * c  # norm1
                + (c * 3 * d + 3 * d)  # qkv
                + (9 * 3 * d + 3 * d)  # qkv depthwise
                + (d * c + c)  # project
                + c  # scale1
            )
            if use_relative_position_bias:
                total += RelativePositionBias.parameter_count(heads, spatial_size)
        return total

    # ------------------------------------------------------------------ attention
    def _attend(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor | None
    ) -> torch.Tensor:
        """Run one attention op over ``(B, heads, N, head_dim)`` tensors.

        Args:
            q: Queries.
            k: Keys.
            v: Values.
            bias: Additive logit bias broadcastable to ``(B, heads, N, N)``, or ``None``.

        Returns:
            Tensor of shape ``(B, heads, N, head_dim)``.
        """
        if self.use_sdpa:
            return F.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=self.scale)
        logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if bias is not None:
            logits = logits + bias
        return torch.matmul(logits.softmax(dim=-1), v)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the block.

        Args:
            x: Tensor of shape ``(B, C, n, n)``.

        Returns:
            Tensor of the same shape.

        Raises:
            ValueError: If the channel count or spatial size does not match the
                configuration. The spatial size is checked because the relative-position
                index buffer is built for one grid only.
        """
        # String concatenation rather than f-strings keeps `forward` scriptable
        # (Contract Part 9): TorchScript cannot format `tuple(tensor.shape)`.
        if x.dim() != 4:
            raise ValueError(
                "GSABlock expects a 4-D tensor, got " + str(x.dim()) + " dims."
            )
        if x.shape[1] != self.channels:
            raise ValueError(
                "GSABlock configured for " + str(self.channels)
                + " channels, got " + str(x.shape[1]) + "."
            )
        if x.shape[2] != self.spatial_size or x.shape[3] != self.spatial_size:
            raise ValueError(
                "GSABlock configured for a " + str(self.spatial_size) + "x"
                + str(self.spatial_size) + " grid, got " + str(x.shape[2]) + "x"
                + str(x.shape[3]) + "."
            )

        if self.use_attention_branch:
            x = x + self.scale1(self.project(self._attention(x)))

        u = self.ffn_dwconv(self.ffn_in(self.norm2(x)))
        u = self.ffn_out(self.gate(u))
        return x + self.scale2(u)

    def _attention(self, x: torch.Tensor) -> torch.Tensor:
        """Run the self-attention sub-branch.

        Args:
            x: Tensor of shape ``(B, C, n, n)``.

        Returns:
            Tensor of shape ``(B, d, n, n)``, before the output projection.
        """
        batch = x.shape[0]
        tokens = self.num_tokens

        t = self.qkv_dwconv(self.qkv(self.norm1(x)))
        q, k, v = t.split(self.attn_dim, dim=1)
        q = q.reshape(batch, self.heads, self.head_dim, tokens).transpose(-2, -1)
        k = k.reshape(batch, self.heads, self.head_dim, tokens).transpose(-2, -1)
        v = v.reshape(batch, self.heads, self.head_dim, tokens).transpose(-2, -1)

        bias: torch.Tensor | None = None
        if self.rel_pos is not None:
            # (heads, N, N) -> (1, heads, N, N); cast so the fused kernel accepts the
            # mask under autocast, where q is half precision and the table is fp32.
            bias = self.rel_pos().unsqueeze(0).to(q.dtype)

        attended = self._attend(q, k, v, bias)
        return attended.transpose(-2, -1).reshape(
            batch, self.attn_dim, self.spatial_size, self.spatial_size
        )

    def extra_repr(self) -> str:
        return (
            f"channels={self.channels}, heads={self.heads}, "
            f"head_dim={self.head_dim}, attn_dim={self.attn_dim}, "
            f"spatial_size={self.spatial_size}, use_sdpa={self.use_sdpa}, "
            f"use_attention_branch={self.use_attention_branch}"
        )
