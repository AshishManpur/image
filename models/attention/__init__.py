"""Global self-attention (Contract Part 2.6, Part 14).

* ``rel_pos.py``   — relative-position bias table plus its non-trainable index buffer.
* ``gsa_block.py`` — :class:`GSABlock`, exact unrestricted multi-head self-attention.

``models.encoder.build_gsa_blocks`` still imports ``models.attention.gsa_block``
lazily, so the attention-free variants (SPARC-Tiny, the pre-attention baseline) build
without touching this package at all.

.. note::
   Contract Part 2.6 specifies attention that is **exact and unrestricted** — "no
   windows, no sparsity, no approximation". The word "sparse" in the phase name refers
   to *where* the blocks are instantiated (3 of 5 stages: Enc L1, Enc L2, Dec D1;
   never at 64x64 or 128x128), never to sparsifying the attention matrix itself.
"""

from __future__ import annotations

from models.attention.gsa_block import GSABlock
from models.attention.rel_pos import RelativePositionBias, relative_position_index

__all__ = ["GSABlock", "RelativePositionBias", "relative_position_index"]
