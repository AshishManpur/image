"""Relative-position bias for the GSA blocks (Contract Part 2.6).

Attention is permutation-equivariant: without a positional term, shuffling the tokens
shuffles the outputs identically, and the block cannot tell a pixel's neighbour from a
pixel on the far side of the image. Restoration needs that distinction, so Part 2.6
adds a learned bias to the logits before the softmax.

The bias is *relative*, not absolute: what it encodes is the offset ``(di, dj)`` between
two tokens rather than their coordinates. On an ``n x n`` grid there are ``(2n-1)^2``
distinct offsets, so the learned table has that many entries per head and is gathered
into the ``(N, N)`` logit matrix by a fixed integer index::

    table : (heads, (2n-1)^2)      trainable, trunc_normal_(std=0.02)
    index : (N, N) int64           registered buffer, NOT a parameter
    bias  = table[:, index]        (heads, N, N)

Two facts about the split matter and are pinned by tests:

* the table is a ``Parameter`` and counts toward the Part 3 budget
  (``3 * 63^2 = 11,907`` at ``n=32``, ``5 * 31^2 = 4,805`` at ``n=16``);
* the index is a **non-trainable int64 buffer**. Were it a parameter it would add
  ``N^2`` (1,048,576 at ``n=32``) to the count, land in the weight-decay group, and be
  silently corrupted by the optimiser.
"""

from __future__ import annotations

import torch
from torch import nn

from utils.init import trunc_normal_


_shared_index_cache: dict[tuple[int, str], torch.Tensor] = {}
"""Module-level cache so bit-identical index buffers share storage.

The index depends only on ``size``, not on any learned state, so every GSA block at
the same grid resolution is holding a bit-identical copy of the other. At batch 8 the
six ``rel_pos_index`` buffers cost 26.74 MB; deduplicating the three at 32x32 and the
three at 16x16 leaves 8.91 MB, saving ~8.65 MB of measured peak training VRAM with no
change to the arithmetic (see ``reports/phase4_12_vram.json``, Phase 4.12 Part 10).
"""


def _shared_relative_position_index(size: int, device: torch.device) -> torch.Tensor:
    """Return the cached index buffer for ``size`` on ``device``, building it once.

    The entry is built with ``inference_mode`` explicitly disabled. A cache populated
    from inside ``torch.inference_mode()`` — e.g. the first block moved to CUDA by an
    inference test — would otherwise hold an *inference tensor*, and every later block
    sharing it would fail with "Inference tensors cannot be saved for backward" the
    moment the bias is gathered under autograd.
    """
    key = (size, str(device))
    cached = _shared_index_cache.get(key)
    if cached is None or torch.is_inference(cached):
        with torch.inference_mode(False):
            _shared_index_cache[key] = relative_position_index(size).to(device)
    return _shared_index_cache[key]


def relative_position_index(size: int) -> torch.Tensor:
    """Build the ``(N, N)`` gather index for an ``size x size`` token grid.

    Args:
        size: Side length ``n`` of the square grid; ``N = n * n``.

    Returns:
        Int64 tensor of shape ``(N, N)`` with values in ``[0, (2n-1)^2)``. Entry
        ``[a, b]`` encodes the offset from token ``b`` to token ``a``.

    Raises:
        ValueError: If ``size`` is not positive.
    """
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}.")
    coords = torch.stack(
        torch.meshgrid(
            torch.arange(size), torch.arange(size), indexing="ij"
        )
    ).flatten(1)  # (2, N)
    relative = coords[:, :, None] - coords[:, None, :]  # (2, N, N)
    relative = relative.permute(1, 2, 0).contiguous()  # (N, N, 2)
    relative[..., 0] += size - 1
    relative[..., 1] += size - 1
    relative[..., 0] *= 2 * size - 1
    return relative.sum(-1).to(torch.int64)


class RelativePositionBias(nn.Module):
    """Learned relative-position bias added to the attention logits.

    Args:
        heads: Number of attention heads.
        size: Side length ``n`` of the square token grid.

    Raises:
        ValueError: If ``heads`` or ``size`` is not positive.
    """

    def __init__(self, heads: int, size: int) -> None:
        super().__init__()
        if heads <= 0:
            raise ValueError(f"heads must be positive, got {heads}.")
        if size <= 0:
            raise ValueError(f"size must be positive, got {size}.")
        self.heads = heads
        self.size = size
        self.num_tokens = size * size
        self.num_offsets = (2 * size - 1) ** 2

        self.rel_pos_table = nn.Parameter(torch.zeros(heads, self.num_offsets))
        trunc_normal_(self.rel_pos_table, std=0.02)
        self.register_buffer(
            "rel_pos_index",
            _shared_relative_position_index(size, torch.device("cpu")),
            persistent=False,
        )
        self._custom_init = True

    def _apply(self, fn, *args, **kwargs):
        """Re-establish index sharing after ``Module._apply`` (e.g. ``.to(device)``).

        ``_apply`` maps every buffer independently, so a move to a new device would
        otherwise hand each block its own fresh copy of the index again.
        """
        module = super()._apply(fn, *args, **kwargs)
        module.rel_pos_index = _shared_relative_position_index(
            module.size, module.rel_pos_index.device
        )
        return module

    @staticmethod
    def parameter_count(heads: int, size: int) -> int:
        """Analytic parameter count: ``heads * (2 * size - 1) ** 2``."""
        return heads * (2 * size - 1) ** 2

    @property
    def center_index(self) -> int:
        """Table index of the zero offset, i.e. a token attending to itself."""
        return (self.size - 1) * (2 * self.size - 1) + (self.size - 1)

    def forward(self) -> torch.Tensor:
        """Gather the bias matrix.

        Returns:
            Tensor of shape ``(heads, N, N)`` in the table's dtype.
        """
        return self.rel_pos_table[:, self.rel_pos_index]

    def extra_repr(self) -> str:
        return (
            f"heads={self.heads}, size={self.size}, "
            f"num_tokens={self.num_tokens}, num_offsets={self.num_offsets}"
        )
