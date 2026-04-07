"""
Checkpoint I/O utilities.

load_sae() and save_sae() are first-class package exports so users never
need to manually manage the checkpoint format.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from meta_sae.sae import BatchTopKSAE, JumpReLUSAE, TopKSAE, VanillaSAE

_SAE_REGISTRY = {
    "batchtopk": BatchTopKSAE,
    "topk": TopKSAE,
    "vanilla": VanillaSAE,
    "jumprelu": JumpReLUSAE,
}


def load_sae(
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> BatchTopKSAE:
    """Load a primary SAE from a .pt checkpoint.

    The checkpoint must contain at minimum ``cfg`` and ``state_dict`` keys,
    as written by :func:`save_sae` or ``scripts/train.py``.

    Args:
        checkpoint_path: Path to the ``.pt`` checkpoint file.
        device: Device to load the model onto (e.g. ``"cpu"``, ``"cuda"``).

    Returns:
        Loaded SAE in eval mode.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg: Dict[str, Any] = ckpt["cfg"]
    cfg["device"] = device

    sae_type = cfg.get("sae_type", "batchtopk").lower()
    sae_cls = _SAE_REGISTRY.get(sae_type, BatchTopKSAE)

    sae = sae_cls(cfg)
    sae.load_state_dict(ckpt["state_dict"])
    sae.eval()
    sae.to(device)
    return sae


def save_sae(
    sae,
    path: str | Path,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Save an SAE checkpoint.

    Args:
        sae: The SAE to save (any subclass of BaseAutoencoder, or
             SAEWithPenalty / BatchTopKSAEWithPenalty).
        path: Destination ``.pt`` file path.
        extra: Optional dict of additional keys to store alongside
               ``cfg`` and ``state_dict`` (e.g. ``penalty_cfg``, ``meta_cfg``).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Unwrap penalty wrapper if present
    cfg = sae.cfg if hasattr(sae, "cfg") else sae.inner_sae.cfg

    payload: Dict[str, Any] = {
        "cfg": cfg,
        "state_dict": sae.state_dict(),
    }
    if extra:
        payload.update(extra)

    torch.save(payload, path)


def load_thresholds(path: str | Path) -> np.ndarray:
    """Load calibrated per-feature firing thresholds from a .npy file.

    Args:
        path: Path to ``thresholds_{label}.npy`` as produced by
              :func:`meta_sae.extension.calibrate_thresholds` or
              ``scripts/calibrate.py``.

    Returns:
        float32 array of shape ``(dict_size,)``.
    """
    return np.load(path).astype(np.float32)


def save_thresholds(thresholds: np.ndarray, path: str | Path) -> None:
    """Save per-feature thresholds to a .npy file."""
    np.save(path, thresholds.astype(np.float32))
