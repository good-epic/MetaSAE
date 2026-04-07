#!/usr/bin/env python3
"""
Assess trained SAE checkpoints: threshold analysis, L0 sparsity, reconstruction quality.

Loads joint and/or solo primary SAE checkpoints, runs assessments, and saves results.

Usage::

    python scripts/assess.py \\
        --joint-checkpoint  outputs/run1/joint_primary_sae.pt \\
        --solo-checkpoint   outputs/run1/solo_primary_sae.pt \\
        --output-dir        outputs/run1/assessment/ \\
        --num_batches       1000 \\
        --device            cuda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from functools import partial

import numpy as np
import torch

from meta_sae.io import load_sae
from meta_sae.utils import load_model_and_set_sizes


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------

def reconstruction_hook(value, hook, sae):
    B, T, D = value.shape
    x = value.reshape(B * T, D)
    with torch.no_grad():
        out = sae(x)["sae_out"]
    return out.reshape(B, T, D)


def zero_ablation_hook(value, hook):
    return torch.zeros_like(value)


# ---------------------------------------------------------------------------
# Assessment helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def assess_sae(sae, activation_store, hook_point, model, n_batches: int, device: str) -> dict:
    """Compute L0, L2, dead features, and CE metrics for one SAE."""
    from meta_sae.activation_store import ActivationsStore

    l0_list, l2_list, dead_list = [], [], []
    ce_orig_list, ce_recon_list, ce_zero_list = [], [], []

    for _ in range(n_batches):
        batch_tokens = activation_store.get_batch_tokens()
        x = activation_store.get_activations(batch_tokens).reshape(-1, sae.cfg["act_size"])

        out = sae(x)
        l0_list.append(out["l0_norm"].item())
        l2_list.append(out["l2_loss"].item())
        dead_list.append(out["num_dead_features"].item())

        sae_output = out["sae_out"].reshape(
            batch_tokens.shape[0], batch_tokens.shape[1], -1
        )
        ce_orig  = model(batch_tokens, return_type="loss").item()
        ce_recon = model.run_with_hooks(
            batch_tokens,
            fwd_hooks=[(hook_point, partial(reconstruction_hook, sae=sae))],
            return_type="loss",
        ).item()
        ce_zero  = model.run_with_hooks(
            batch_tokens,
            fwd_hooks=[(hook_point, zero_ablation_hook)],
            return_type="loss",
        ).item()

        ce_orig_list.append(ce_orig)
        ce_recon_list.append(ce_recon)
        ce_zero_list.append(ce_zero)

        del batch_tokens, x, out, sae_output

    mean_l0   = float(np.mean(l0_list))
    mean_l2   = float(np.mean(l2_list))
    mean_dead = float(np.mean(dead_list))
    mean_ce_orig  = float(np.mean(ce_orig_list))
    mean_ce_recon = float(np.mean(ce_recon_list))
    mean_ce_zero  = float(np.mean(ce_zero_list))
    delta_ce      = mean_ce_recon - mean_ce_orig
    frac_destroyed = delta_ce / max((mean_ce_zero - mean_ce_orig), 1e-9)

    return {
        "l0":                mean_l0,
        "l2":                mean_l2,
        "dead_features":     mean_dead,
        "ce_original":       mean_ce_orig,
        "ce_reconstructed":  mean_ce_recon,
        "ce_zero_ablated":   mean_ce_zero,
        "delta_ce":          delta_ce,
        "fraction_destroyed": frac_destroyed,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Assess trained SAE checkpoints.")
    p.add_argument("--joint-checkpoint", default=None)
    p.add_argument("--solo-checkpoint",  default=None)
    p.add_argument("--output-dir",       required=True)
    p.add_argument("--num-batches",      type=int,   default=500)
    p.add_argument("--model-batch-size", type=int,   default=32)
    p.add_argument("--seq-len",          type=int,   default=128)
    p.add_argument("--num-batches-in-buffer", type=int, default=3)
    p.add_argument("--device",           default="cuda")
    return p.parse_args()


def main():
    args       = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results    = {}

    checkpoints = {}
    if args.joint_checkpoint:
        checkpoints["joint"] = args.joint_checkpoint
    if args.solo_checkpoint:
        checkpoints["solo"]  = args.solo_checkpoint

    if not checkpoints:
        print("No checkpoints specified. Use --joint-checkpoint and/or --solo-checkpoint.")
        return

    # Load from first checkpoint to get model name / layer
    first_path = next(iter(checkpoints.values()))
    ref_sae    = load_sae(first_path, args.device)
    sae_cfg    = ref_sae.cfg

    cfg = {
        "model_name":            sae_cfg["model_name"],
        "dataset_path":          sae_cfg.get("dataset_path", "HuggingFaceFW/fineweb"),
        "dataset_name":          sae_cfg.get("dataset_name", "sample-10BT"),
        "hook_point":            f"blocks.{sae_cfg['layer']}.hook_resid_pre",
        "seq_len":               sae_cfg.get("seq_len", args.seq_len),
        "model_batch_size":      args.model_batch_size,
        "num_batches_in_buffer": args.num_batches_in_buffer,
        "device":                args.device,
        "layer":                 sae_cfg["layer"],
        "act_size":              sae_cfg["act_size"],
        "batch_size":            args.model_batch_size * sae_cfg.get("seq_len", args.seq_len),
        "site":                  "resid_pre",
        "model_dtype":           sae_cfg.get("model_dtype", "float32"),
    }

    from transformer_lens import HookedTransformer
    model_dtype = torch.bfloat16 if cfg["model_dtype"] == "bfloat16" else torch.float32
    print(f"Loading {cfg['model_name']} ...")
    model = HookedTransformer.from_pretrained(cfg["model_name"], dtype=model_dtype)
    model.eval()
    model.to(args.device)

    from meta_sae.activation_store import ActivationsStore
    store = ActivationsStore(model, cfg)

    for label, ckpt_path in checkpoints.items():
        print(f"\nAssessing {label} SAE from {ckpt_path} ...")
        sae = load_sae(ckpt_path, args.device)
        sae.eval()
        metrics = assess_sae(
            sae, store, cfg["hook_point"], model, args.num_batches, args.device
        )
        results[label] = metrics
        print(f"  L0={metrics['l0']:.1f}  L2={metrics['l2']:.4f}  "
              f"dead={metrics['dead_features']:.0f}  "
              f"ΔCE={metrics['delta_ce']:.4f}  "
              f"frac_destroyed={metrics['fraction_destroyed']:.3f}")
        del sae

    out_path = output_dir / "assessment_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
