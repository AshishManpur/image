"""Checkpoint serialisation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from utils.logging_utils import get_logger

_LOGGER = get_logger(__name__)


def save_checkpoint(state: dict[str, Any], path: Path) -> None:
    """Write a checkpoint, creating the parent directory if needed.

    Args:
        state: Payload to serialise.
        path: Destination file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(
    path: Path, map_location: str | torch.device = "cpu"
) -> dict[str, Any]:
    """Read a checkpoint written by :func:`save_checkpoint`.

    ``weights_only=False`` is passed explicitly. Torch >= 2.6 flipped this default to
    ``True``, restricting unpickling to plain tensors and containers; SPARC checkpoints
    also carry optimiser, scheduler and scaler state. We only ever load our own
    checkpoints, so the restriction buys nothing here, and relying on the default would
    break silently the moment a non-tensor object enters the payload.

    Args:
        path: Checkpoint file.
        map_location: Device to map storages onto.

    Returns:
        The deserialised payload.
    """
    return torch.load(path, map_location=map_location, weights_only=False)


def load_backbone_weights(
    model: "torch.nn.Module",
    path: Path,
    allowed_new_prefixes: tuple[str, ...] = ("clean_branch.",),
    map_location: str | torch.device = "cpu",
    key: str = "model",
) -> tuple[list[str], list[str]]:
    """Warm-start a model from a checkpoint that predates its newest sub-modules.

    This is deliberately stricter than ``load_state_dict(..., strict=False)``. A silent
    partial load is the failure mode that matters here: dropping a trained tensor
    because a key was renamed produces a model that trains without error and is quietly
    worse than the checkpoint it claims to continue. Both key lists are logged in full
    and any violation raises.

    Args:
        model: Destination module.
        path: Checkpoint written by :func:`save_checkpoint`.
        allowed_new_prefixes: Key prefixes that are permitted to be missing, i.e. the
            genuinely new sub-modules that the checkpoint cannot contain.
        map_location: Device to map storages onto.
        key: Payload entry holding the state dict.

    Returns:
        ``(missing, unexpected)`` as returned by ``load_state_dict``.

    Raises:
        KeyError: If the payload has no ``key`` entry.
        RuntimeError: If any key is unexpected, or any missing key falls outside
            ``allowed_new_prefixes``.
    """
    payload = load_checkpoint(path, map_location=map_location)
    if key not in payload:
        raise KeyError(f"Checkpoint {path} has no '{key}' entry; got {list(payload)}.")

    incompatible = model.load_state_dict(payload[key], strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)

    _LOGGER.info("Warm-start from %s", path)
    _LOGGER.info("  missing keys (%d): %s", len(missing), missing)
    _LOGGER.info("  unexpected keys (%d): %s", len(unexpected), unexpected)

    if unexpected:
        raise RuntimeError(
            f"{len(unexpected)} unexpected key(s) in {path}: {unexpected[:10]}. "
            "The checkpoint holds trained weights this model has no home for; loading "
            "would silently discard them."
        )
    disallowed = [
        k for k in missing if not any(k.startswith(p) for p in allowed_new_prefixes)
    ]
    if disallowed:
        raise RuntimeError(
            f"{len(disallowed)} missing key(s) outside {allowed_new_prefixes}: "
            f"{disallowed[:10]}. Only genuinely new sub-modules may be uninitialised."
        )
    return missing, unexpected
