"""
meta_sae — pip-installable package for MetaSAE training and inference.

Key exports
-----------
Training:
    MetaSAEWrapper              — thin wrapper for training a meta SAE on decoder vectors
    SAEWithPenalty              — composition wrapper adding decomposability penalty to any SAE
    BatchTopKSAEWithPenalty     — BatchTopKSAE subclass with built-in penalty
    train_sae_with_meta         — alternating joint training loop
    train_primary_sae_solo      — solo primary SAE training (no penalty)
    train_meta_sae_on_frozen_primary — sequential meta SAE training

Calibration:
    calibrate_thresholds        — EER histogram calibration → per-feature thresholds
    calibrate_checkpoint        — load checkpoint, calibrate, save thresholds (one call)

Inference:
    InferenceSAE                — threshold-based deterministic inference wrapper

Evaluation:
    collect_activations         — collect JumpReLU contexts for auto-interp scoring
    compute_cooccurrence        — compute cross-SAE phi matrix and candidates

I/O:
    load_sae                    — load any supported SAE from a .pt checkpoint
    save_sae                    — save SAE state_dict + cfg to a .pt file
    load_thresholds             — load calibrated thresholds from a .npy file
    save_thresholds             — save thresholds to a .npy file
"""

from meta_sae.extension import (
    MetaSAEWrapper,
    SAEWithPenalty,
    BatchTopKSAEWithPenalty,
    InferenceSAE,
    get_sae_class,
    calibrate_thresholds,
    calibrate_checkpoint,
    train_sae_with_meta,
    train_primary_sae_solo,
    train_meta_sae_on_frozen_primary,
)

from meta_sae.io import (
    load_sae,
    save_sae,
    load_thresholds,
    save_thresholds,
)

from meta_sae.sae import (
    BatchTopKSAE,
    TopKSAE,
    VanillaSAE,
    JumpReLUSAE,
)

from meta_sae.activation_store import ActivationsStore
from meta_sae.collect import collect_activations
from meta_sae.cooccurrence import compute_cooccurrence

__all__ = [
    # Training
    "MetaSAEWrapper",
    "SAEWithPenalty",
    "BatchTopKSAEWithPenalty",
    "train_sae_with_meta",
    "train_primary_sae_solo",
    "train_meta_sae_on_frozen_primary",
    # Calibration
    "calibrate_thresholds",
    "calibrate_checkpoint",
    # Inference
    "InferenceSAE",
    # Evaluation
    "collect_activations",
    "compute_cooccurrence",
    # I/O
    "load_sae",
    "save_sae",
    "load_thresholds",
    "save_thresholds",
    # SAE classes
    "BatchTopKSAE",
    "TopKSAE",
    "VanillaSAE",
    "JumpReLUSAE",
    "get_sae_class",
    # Activation store
    "ActivationsStore",
]
