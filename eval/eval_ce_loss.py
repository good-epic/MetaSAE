#!/usr/bin/env python3
"""
Estimate the cross-entropy (CE) loss impact of inserting a MetaSAE into the
residual stream, comparing joint vs. solo trained primary SAEs.

Three conditions:
  - original:      baseline CE loss (no SAE)
  - reconstructed: activations replaced by SAE reconstruction
  - zero_ablated:  activations replaced by zeros

Primary metric:  ΔCE = CE_reconstructed − CE_original
Normalised:      fraction_destroyed = ΔCE / (CE_zero_ablated − CE_original)

Usage::

    python eval/eval_ce_loss.py \\
        --joint-checkpoint outputs/run1/joint_primary_sae.pt \\
        --solo-checkpoint  outputs/run1/solo_primary_sae.pt \\
        --output-dir       outputs/run1/ce_loss/ \\
        --n-batches        200 \\
        --model-batch-size 16 \\
        --device           cuda
"""

import argparse
import json
from functools import partial
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer

from meta_sae.io import load_sae
from meta_sae.activation_store import ActivationsStore


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------

def reconstruction_hook(value, hook, sae):
    B, T, D = value.shape
    x = value.reshape(B * T, D)
    with torch.no_grad():
        sae_out = sae(x)["sae_out"]
    return sae_out.reshape(B, T, D)


def zero_ablation_hook(value, hook):
    return torch.zeros_like(value)


# ---------------------------------------------------------------------------
# CE loss estimation
# ---------------------------------------------------------------------------

def estimate_ce_loss(model, joint_sae, solo_sae, store_cfg: dict, n_batches: int, device: str) -> dict:
    """Stream n_batches through the model under four conditions and return CE arrays."""
    hook_point = store_cfg["hook_point"]
    store      = ActivationsStore(model, store_cfg)

    ce_orig   = np.zeros(n_batches, dtype=np.float32)
    ce_joint  = np.zeros(n_batches, dtype=np.float32)
    ce_solo   = np.zeros(n_batches, dtype=np.float32)
    ce_zero   = np.zeros(n_batches, dtype=np.float32)
    l0_joint  = np.zeros(n_batches, dtype=np.float32)
    l0_solo   = np.zeros(n_batches, dtype=np.float32)

    for i in tqdm(range(n_batches), desc="Batches"):
        tokens = store.get_batch_tokens().to(device)

        with torch.no_grad():
            ce_orig[i]  = model(tokens, return_type="loss").item()
            ce_joint[i] = model.run_with_hooks(
                tokens,
                fwd_hooks=[(hook_point, partial(reconstruction_hook, sae=joint_sae))],
                return_type="loss",
            ).item()
            ce_solo[i]  = model.run_with_hooks(
                tokens,
                fwd_hooks=[(hook_point, partial(reconstruction_hook, sae=solo_sae))],
                return_type="loss",
            ).item()
            ce_zero[i]  = model.run_with_hooks(
                tokens, fwd_hooks=[(hook_point, zero_ablation_hook)], return_type="loss",
            ).item()

            _, cache = model.run_with_cache(
                tokens, names_filter=[hook_point], stop_at_layer=store_cfg["layer"] + 1,
            )
            x = cache[hook_point].reshape(-1, store_cfg["act_size"])
            l0_joint[i] = (joint_sae(x)["feature_acts"] > 0).float().sum(dim=-1).mean().item()
            l0_solo[i]  = (solo_sae(x)["feature_acts"]  > 0).float().sum(dim=-1).mean().item()
            del cache

    return {
        "original":    ce_orig,
        "joint_recon": ce_joint,
        "solo_recon":  ce_solo,
        "zero_ablated": ce_zero,
        "joint_l0":    l0_joint,
        "solo_l0":     l0_solo,
    }


# ---------------------------------------------------------------------------
# Summary and plots
# ---------------------------------------------------------------------------

def summarise(results: dict) -> dict:
    orig  = results["original"]
    joint = results["joint_recon"]
    solo  = results["solo_recon"]
    zero  = results["zero_ablated"]

    def stats(arr):
        return {"mean": float(arr.mean()), "std": float(arr.std()),
                "sem":  float(arr.std() / np.sqrt(len(arr)))}

    delta_joint = joint - orig
    delta_solo  = solo  - orig
    delta_zero  = zero  - orig

    frac_joint = delta_joint / np.where(delta_zero > 0, delta_zero, np.nan)
    frac_solo  = delta_solo  / np.where(delta_zero > 0, delta_zero, np.nan)

    return {
        "n_batches":                    len(orig),
        "CE_original":                  stats(orig),
        "CE_joint_recon":               stats(joint),
        "CE_solo_recon":                stats(solo),
        "CE_zero_ablated":              stats(zero),
        "delta_CE_joint":               stats(delta_joint),
        "delta_CE_solo":                stats(delta_solo),
        "delta_CE_zero":                stats(delta_zero),
        "fraction_destroyed_joint":     stats(frac_joint[~np.isnan(frac_joint)]),
        "fraction_destroyed_solo":      stats(frac_solo[~np.isnan(frac_solo)]),
        "L0_joint":                     stats(results["joint_l0"]),
        "L0_solo":                      stats(results["solo_l0"]),
    }


def print_summary(summary: dict):
    orig = summary["CE_original"]["mean"]
    j_ce = summary["CE_joint_recon"]["mean"]
    s_ce = summary["CE_solo_recon"]["mean"]
    z_ce = summary["CE_zero_ablated"]["mean"]
    dj   = summary["delta_CE_joint"]["mean"]
    ds   = summary["delta_CE_solo"]["mean"]
    fj   = summary["fraction_destroyed_joint"]["mean"]
    fs   = summary["fraction_destroyed_solo"]["mean"]
    lj   = summary["L0_joint"]["mean"]
    ls   = summary["L0_solo"]["mean"]

    print("\n=== CE Loss Results ===")
    print(f"  Baseline CE (no SAE):   {orig:.4f}")
    print(f"  Zero-ablation CE:       {z_ce:.4f}  (Δ = +{z_ce - orig:.4f})")
    print()
    print(f"  Joint SAE CE:           {j_ce:.4f}  (Δ = +{dj:.4f},  "
          f"fraction_destroyed = {fj:.3%},  L0 = {lj:.1f})")
    print(f"  Solo  SAE CE:           {s_ce:.4f}  (Δ = +{ds:.4f},  "
          f"fraction_destroyed = {fs:.3%},  L0 = {ls:.1f})")
    print()
    print(f"  Joint vs Solo ΔCE:      {dj - ds:+.4f}  "
          f"({'joint costs more' if dj > ds else 'joint costs less'})")


def plot_results(results: dict, summary: dict, output_dir: Path, title: str = "CE Loss"):
    orig  = results["original"]
    joint = results["joint_recon"]
    solo  = results["solo_recon"]
    zero  = results["zero_ablated"]
    n     = len(orig)
    steps = np.arange(n)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(title, fontsize=12)

    ax = axes[0]
    ax.plot(steps, orig,  label="original",    color="black",     lw=1.2, alpha=0.7)
    ax.plot(steps, joint, label="joint SAE",   color="steelblue", lw=1.2, alpha=0.8)
    ax.plot(steps, solo,  label="solo SAE",    color="tomato",    lw=1.2, alpha=0.8)
    ax.plot(steps, zero,  label="zero ablated",color="gray",      lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("Batch"); ax.set_ylabel("CE loss (nats)"); ax.set_title("A: CE loss per batch")
    ax.legend(fontsize=8)

    ax = axes[1]
    delta_j = joint - orig
    delta_s = solo  - orig
    bins = np.linspace(min(delta_j.min(), delta_s.min()) - 0.01,
                       max(delta_j.max(), delta_s.max()) + 0.01, 40)
    ax.hist(delta_s, bins=bins, color="tomato",    alpha=0.6, label="solo",  edgecolor="none")
    ax.hist(delta_j, bins=bins, color="steelblue", alpha=0.6, label="joint", edgecolor="none")
    ax.axvline(delta_j.mean(), color="steelblue", ls="--", lw=1.5,
               label=f"joint mean={delta_j.mean():.4f}")
    ax.axvline(delta_s.mean(), color="tomato",    ls="--", lw=1.5,
               label=f"solo mean={delta_s.mean():.4f}")
    ax.set_xlabel("ΔCE"); ax.set_ylabel("Batch count"); ax.set_title("B: ΔCE distribution")
    ax.legend(fontsize=8)

    ax = axes[2]
    delta_z = zero - orig
    frac_j = np.where(delta_z > 0, delta_j / delta_z, np.nan)
    frac_s = np.where(delta_z > 0, delta_s / delta_z, np.nan)
    valid  = ~(np.isnan(frac_j) | np.isnan(frac_s))
    bins2  = np.linspace(0, max(np.nanmax(frac_j), np.nanmax(frac_s)) + 0.01, 40)
    ax.hist(frac_s[valid], bins=bins2, color="tomato",    alpha=0.6, label="solo",  edgecolor="none")
    ax.hist(frac_j[valid], bins=bins2, color="steelblue", alpha=0.6, label="joint", edgecolor="none")
    ax.axvline(np.nanmean(frac_j), color="steelblue", ls="--", lw=1.5,
               label=f"joint mean={np.nanmean(frac_j):.3%}")
    ax.axvline(np.nanmean(frac_s), color="tomato",    ls="--", lw=1.5,
               label=f"solo mean={np.nanmean(frac_s):.3%}")
    ax.set_xlabel("Fraction destroyed"); ax.set_ylabel("Batch count")
    ax.set_title("C: fraction of information destroyed")
    ax.legend(fontsize=8)

    plt.tight_layout()
    out = output_dir / "ce_loss_eval.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved figure → {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="CE loss impact of joint vs solo SAE insertion.")
    p.add_argument("--joint-checkpoint",    required=True)
    p.add_argument("--solo-checkpoint",     required=True)
    p.add_argument("--output-dir",          required=True)
    p.add_argument("--n-batches",           type=int, default=200)
    p.add_argument("--model-batch-size",    type=int, default=16)
    p.add_argument("--num-batches-in-buffer", type=int, default=4)
    p.add_argument("--dtype",              default="float32",
                   choices=["float32", "bfloat16", "float16"])
    p.add_argument("--device",             default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args       = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading SAEs...")
    joint_sae = load_sae(args.joint_checkpoint, args.device)
    solo_sae  = load_sae(args.solo_checkpoint,  args.device)
    joint_sae.eval(); solo_sae.eval()

    layer = joint_sae.cfg["layer"]
    print(f"  Joint: dict_size={joint_sae.cfg['dict_size']}  top_k={joint_sae.cfg['top_k']}  layer={layer}")
    print(f"  Solo:  dict_size={solo_sae.cfg['dict_size']}   top_k={solo_sae.cfg['top_k']}")

    _dtype_map  = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    model_dtype = _dtype_map[args.dtype]
    print(f"Loading {joint_sae.cfg['model_name']} (dtype={args.dtype})...")
    model = HookedTransformer.from_pretrained(joint_sae.cfg["model_name"], dtype=model_dtype)
    model.eval(); model.to(args.device)
    joint_sae.to(model_dtype); solo_sae.to(model_dtype)

    store_cfg = {
        "model_name":            joint_sae.cfg["model_name"],
        "dataset_path":          joint_sae.cfg["dataset_path"],
        "dataset_name":          joint_sae.cfg.get("dataset_name", None),
        "hook_point":            f"blocks.{layer}.hook_resid_pre",
        "seq_len":               joint_sae.cfg["seq_len"],
        "model_batch_size":      args.model_batch_size,
        "num_batches_in_buffer": args.num_batches_in_buffer,
        "device":                args.device,
        "layer":                 layer,
        "batch_size":            args.model_batch_size * joint_sae.cfg["seq_len"],
        "act_size":              joint_sae.cfg["act_size"],
    }

    print(f"\nEstimating CE loss over {args.n_batches} batches ...")
    results = estimate_ce_loss(model, joint_sae, solo_sae, store_cfg, args.n_batches, args.device)
    summary = summarise(results)
    print_summary(summary)

    with open(output_dir / "ce_loss_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    np.savez(output_dir / "ce_loss_raw.npz", **results)
    plot_results(results, summary, output_dir,
                 title=f"CE Loss: {joint_sae.cfg['model_name']} layer {layer}")
    print(f"\nAll results → {output_dir}")


if __name__ == "__main__":
    main()
