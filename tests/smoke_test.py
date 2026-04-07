#!/usr/bin/env python3
"""
End-to-end pipeline smoke test on GPT-2 Small.

Covers the four highest-risk unknowns identified in the pre-release audit:

  [5] VanillaSAE and JumpReLUSAE inference paths
  [1] train_sae_with_meta() device handling and training loop
  [2] collect_activations()  (heap, reservoir, matching logic)
  [3] compute_cooccurrence() (Pass 1, Pass 2, --load-topk shortcut)

GPT-2 Small (d_model=768, 12 layers) fits comfortably in 8 GB VRAM.
All parameters are deliberately tiny so the full suite runs in ~10-15 min
on an RTX 3070 Ti.

Usage::

    # from the repo root:
    python tests/smoke_test.py
    python tests/smoke_test.py --output-dir /tmp/my_smoke_run
    python tests/smoke_test.py --cpu          # force CPU (slow but works)
"""

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Smoke-test constants  (GPT-2 Small on a single 8 GB GPU)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent

# Model
MODEL_NAME  = "gpt2"       # GPT-2 Small  (d_model=768, 12 layers)
ACT_SIZE    = 768
LAYER       = 6            # middle layer
SITE        = "resid_pre"
DATASET      = "HuggingFaceFW/fineweb"
DATASET_NAME = "sample-10BT"

# SAE sizes — intentionally tiny so the phi cross-matrix is tiny too
DICT_SIZE      = 64
TOP_K          = 8         # ~12.5% sparsity
META_DICT_SIZE = 16
META_TOP_K     = 2

# Training volume — enough to exercise the loop without waiting long
NUM_TOKENS       = 50_000
BATCH_SIZE       = 256
MODEL_BATCH_SIZE = 8
SEQ_LEN          = 64
BUFFER_DEPTH     = 2

# MetaSAE penalty
LAMBDA2          = 0.3
SIGMA_SQ         = 1.0
N_PRIMARY_STEPS  = 5
N_META_STEPS     = 2

# Calibration / eval token counts — tiny, just enough to exercise code paths
CALIB_TOKENS     = 20_000
COLLECT_TOKENS   = 20_000
PHI_PASS1_TOKENS = 20_000
PHI_PASS2_TOKENS = 5_000

# collect_activations knobs
COLLECT_MIN_CONTEXTS   = 2    # very low — many features will be sparse after tiny training
COLLECT_N_TRAIN_TOP    = 3
COLLECT_N_FUZZ         = 1
COLLECT_N_DETECT       = 2
COLLECT_N_RANDOM       = 10
COLLECT_CTX_LEN        = 32

# compute_cooccurrence knobs
PHI_N_CANDIDATES = 5

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def banner(msg: str):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}", flush=True)


def check(cond: bool, msg: str):
    if not cond:
        print(f"\nFAILED: {msg}", file=sys.stderr)
        sys.exit(1)


@contextlib.contextmanager
def _restore_stderr():
    """Save and restore C-level fd 2 around calls to library orchestrators.

    Each orchestrator (calibrate_checkpoint, collect_activations,
    compute_cooccurrence) redirects fd 2 to /dev/null as its final action so
    that streaming-dataset teardown threads write silently.  This context
    manager restores the original fd so subsequent smoke-test output is visible.
    """
    saved = os.dup(2)
    try:
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)


# ---------------------------------------------------------------------------
# Step 0 — Pre-flight
# ---------------------------------------------------------------------------

def step_preflight(device: str):
    banner("Step 0: Pre-flight checks")

    missing = []
    for pkg in ["meta_sae", "transformer_lens", "datasets", "safetensors", "scipy"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    try:
        import lapjv  # noqa: F401
    except ImportError:
        missing.append("lapjv  (pip install lapjv)")

    if missing:
        print("Missing packages:\n  " + "\n  ".join(missing))
        print("\nInstall: pip install -e '.' && pip install lapjv")
        sys.exit(1)

    print(f"  device : {device}")
    if device.startswith("cuda"):
        props = torch.cuda.get_device_properties(0)
        gb    = props.total_memory / 1e9
        print(f"  GPU    : {props.name}  ({gb:.1f} GB)")
    print("  OK")


# ---------------------------------------------------------------------------
# Step 1 — VanillaSAE and JumpReLUSAE inference (unknown 5)
# ---------------------------------------------------------------------------

def step_alt_sae_types(device: str):
    banner("Step 1: VanillaSAE / JumpReLUSAE inference (unknown 5)")

    from meta_sae.config import get_default_cfg
    from meta_sae.sae import VanillaSAE, JumpReLUSAE
    from meta_sae import InferenceSAE

    base = get_default_cfg()
    base.update(dict(
        act_size  = ACT_SIZE,
        dict_size = 32,
        device    = device,
        dtype     = torch.float32,
        l1_coeff  = 0.001,
        seed      = 42,
    ))

    x = torch.randn(4, ACT_SIZE, device=device)

    # --- VanillaSAE ---
    vanilla = VanillaSAE(base).eval()
    out = vanilla(x)
    check(not torch.isnan(out["loss"]),           "VanillaSAE forward: NaN loss")
    check(out["sae_out"].shape == (4, ACT_SIZE),  "VanillaSAE: bad sae_out shape")

    feat   = vanilla.encode(x)
    check(feat.shape == (4, 32),                  "VanillaSAE encode: bad shape")
    x_hat  = vanilla.decode(feat)
    check(x_hat.shape == (4, ACT_SIZE),           "VanillaSAE decode: bad shape")
    feat2, x_hat2 = vanilla.encode_decode(x)
    check(feat2.shape == feat.shape,              "VanillaSAE encode_decode: feat shape")
    check(x_hat2.shape == x_hat.shape,            "VanillaSAE encode_decode: xhat shape")
    print("  VanillaSAE: forward, encode, decode, encode_decode — OK")

    # --- JumpReLUSAE ---
    jump_cfg = {**base, "dict_size": 32, "top_k": 8}
    jump = JumpReLUSAE(jump_cfg).eval()
    out2 = jump(x)
    check(not torch.isnan(out2["loss"]),          "JumpReLUSAE forward: NaN loss")
    feat3 = jump.encode(x)
    check(feat3.shape == (4, 32),                 "JumpReLUSAE encode: bad shape")
    x_hat3 = jump.decode(feat3)
    check(x_hat3.shape == (4, ACT_SIZE),          "JumpReLUSAE decode: bad shape")
    feat4, x_hat4 = jump.encode_decode(x)
    check(feat4.shape == feat3.shape,             "JumpReLUSAE encode_decode: feat shape")
    print("  JumpReLUSAE: forward, encode, decode, encode_decode — OK")

    # --- InferenceSAE wrapping BatchTopKSAE (the primary use case) ---
    # Use VanillaSAE here since BatchTopKSAE needs a real trained model for
    # meaningful thresholds; we just want shape / no-crash checks.
    thresholds = np.zeros(32, dtype=np.float32)   # zero = everything fires
    inf_sae = InferenceSAE.from_sae(vanilla, thresholds)
    feat5, x_hat5 = inf_sae(x)
    check(feat5.shape == (4, 32),                 "InferenceSAE forward: feat shape")
    check(x_hat5.shape == (4, ACT_SIZE),          "InferenceSAE forward: xhat shape")

    # steer() with identity modify_fn should reproduce forward output
    x_steered = inf_sae.steer(x, lambda a: a)
    check(
        torch.allclose(x_steered, x_hat5, atol=1e-4),
        f"InferenceSAE.steer(identity) mismatch: max_diff="
        f"{(x_steered - x_hat5).abs().max().item():.2e}",
    )

    # top_activating_examples returns the correct shapes
    vals, idxs = inf_sae.top_activating_examples(feature_idx=0, activations=x, k=3)
    check(vals.shape == idxs.shape,               "top_activating_examples: shape mismatch")
    check(len(vals) <= 3,                         "top_activating_examples: too many results")
    print("  InferenceSAE: forward, steer(identity), top_activating_examples — OK")

    print("PASSED")


# ---------------------------------------------------------------------------
# Step 2 — Train joint + solo SAEs (unknown 1)
# ---------------------------------------------------------------------------

def step_training(out_dir: Path, device: str):
    banner("Step 2: train_sae_with_meta() + solo training (unknown 1)")

    from meta_sae.config import get_default_cfg, post_init_cfg
    from meta_sae.extension import (
        MetaSAEWrapper, SAEWithPenalty, get_sae_class,
        train_sae_with_meta, train_primary_sae_solo,
    )
    from meta_sae.activation_store import ActivationsStore
    from meta_sae.utils import load_model_and_set_sizes
    from meta_sae.io import save_sae

    cfg = get_default_cfg()
    cfg.update(dict(
        model_name            = MODEL_NAME,
        dataset_path          = DATASET,
        dataset_name          = DATASET_NAME,
        layer                 = LAYER,
        site                  = SITE,
        device                = device,
        model_dtype           = "float32",
        dict_size             = DICT_SIZE,
        top_k                 = TOP_K,
        top_k_aux             = DICT_SIZE // 2,
        aux_penalty           = 1 / TOP_K,
        num_tokens            = NUM_TOKENS,
        batch_size            = BATCH_SIZE,
        seq_len               = SEQ_LEN,
        model_batch_size      = MODEL_BATCH_SIZE,
        num_batches_in_buffer = BUFFER_DEPTH,
        lr                    = 3e-4,
    ))

    meta_cfg = {**cfg, "dict_size": META_DICT_SIZE, "top_k": META_TOP_K}
    penalty_cfg = dict(
        lambda2         = LAMBDA2,
        sigma_sq        = SIGMA_SQ,
        n_primary_steps = N_PRIMARY_STEPS,
        n_meta_steps    = N_META_STEPS,
    )

    print(f"  Loading {MODEL_NAME} ...")
    model = load_model_and_set_sizes(cfg)
    meta_cfg["act_size"]   = cfg["act_size"]
    meta_cfg["model_name"] = cfg["model_name"]
    post_init_cfg(cfg)
    post_init_cfg(meta_cfg)

    # ── Joint ────────────────────────────────────────────────────────────────
    print("  Joint training ...")
    store       = ActivationsStore(model, cfg)
    meta_sae    = MetaSAEWrapper(get_sae_class("batchtopk"), meta_cfg)
    primary_sae = SAEWithPenalty(get_sae_class("batchtopk"), cfg, meta_sae, penalty_cfg)

    m = train_sae_with_meta(primary_sae, meta_sae, store, model,
                             cfg, meta_cfg, penalty_cfg)

    check("loss"  in m and not np.isnan(m["loss"]),
          f"Joint: bad final metrics: {m}")
    check("decomp" in m and m["decomp"] > 0,
          f"Decomp penalty is zero — meta SAE not influencing training. metrics={m}")
    check("l0" in m and 0 < m["l0"] <= TOP_K * 3,
          f"Joint L0 out of range: {m}")
    print(f"  Joint — loss={m['loss']:.4f}  L0={m['l0']:.1f}  decomp={m['decomp']:.6f}")

    joint_path = out_dir / "joint_primary_sae.pt"
    save_sae(primary_sae, joint_path,
             extra={"meta_cfg": meta_cfg, "penalty_cfg": penalty_cfg})
    print(f"  Saved → {joint_path}")

    del store, primary_sae, meta_sae
    torch.cuda.empty_cache() if device.startswith("cuda") else None

    # ── Solo ─────────────────────────────────────────────────────────────────
    print("  Solo training ...")
    solo_store = ActivationsStore(model, cfg)
    solo_sae   = get_sae_class("batchtopk")(cfg)

    sm = train_primary_sae_solo(solo_sae, solo_store, cfg)

    check("loss" in sm and not np.isnan(sm["loss"]),
          f"Solo: bad final metrics: {sm}")
    check("l0" in sm and 0 < sm["l0"] <= TOP_K * 3,
          f"Solo L0 out of range: {sm}")
    print(f"  Solo  — loss={sm['loss']:.4f}  L0={sm['l0']:.1f}")

    solo_path = out_dir / "solo_primary_sae.pt"
    save_sae(solo_sae, solo_path)
    print(f"  Saved → {solo_path}")

    del solo_store, solo_sae, model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    print("PASSED")
    return joint_path, solo_path


# ---------------------------------------------------------------------------
# Step 3 — Calibrate thresholds
# ---------------------------------------------------------------------------

def step_calibrate(out_dir: Path, joint_path: Path, solo_path: Path, device: str):
    banner("Step 3: Threshold calibration")

    from meta_sae import calibrate_checkpoint

    for label, ckpt in [("joint", joint_path), ("solo", solo_path)]:
        with _restore_stderr():
            calibrate_checkpoint(
                checkpoint=str(ckpt),
                output_dir=str(out_dir),
                label=label,
                n_tokens=CALIB_TOKENS,
                batch_size=2048,
                model_batch_size=MODEL_BATCH_SIZE,
                num_batches_in_buffer=BUFFER_DEPTH,
                max_val=10.0,
                n_bins=200,
                min_active_samples=100,
                undercalibrated_midpoint=False,
                device=device,
            )

        thresh_path = out_dir / f"thresholds_{label}.npy"
        check(thresh_path.exists(), f"Missing {thresh_path}")
        thresholds = np.load(thresh_path)
        check(thresholds.shape == (DICT_SIZE,),
              f"{label} threshold shape: expected ({DICT_SIZE},) got {thresholds.shape}")
        n_dead = int((thresholds >= 10.0).sum())
        print(f"  {label}: {DICT_SIZE - n_dead}/{DICT_SIZE} features calibrated,"
              f" {n_dead} dead")

    print("PASSED")


# ---------------------------------------------------------------------------
# Step 4 — collect_activations (unknown 2)
# ---------------------------------------------------------------------------

def step_collect(out_dir: Path, joint_path: Path, solo_path: Path, device: str):
    banner("Step 4: collect_activations() (unknown 2)")

    from safetensors.torch import load_file as st_load
    from meta_sae import collect_activations

    cache_dir = out_dir / "delphi_cache"
    with _restore_stderr():
        collect_activations(
            joint_checkpoint=str(joint_path),
            solo_checkpoint=str(solo_path),
            thresholds_joint=str(out_dir / "thresholds_joint.npy"),
            thresholds_solo=str(out_dir / "thresholds_solo.npy"),
            output_dir=str(cache_dir),
            n_tokens=COLLECT_TOKENS,
            n_train_top=COLLECT_N_TRAIN_TOP,
            n_train_per_lower=5,
            n_fuzz_per_quintile=COLLECT_N_FUZZ,
            n_detect=COLLECT_N_DETECT,
            n_boundary=0,
            ctx_len=COLLECT_CTX_LEN,
            n_random=COLLECT_N_RANDOM,
            min_contexts=COLLECT_MIN_CONTEXTS,
            model_batch_size=MODEL_BATCH_SIZE,
            num_batches_in_buffer=BUFFER_DEPTH,
            max_val=10.0,
            seed=42,
            device=device,
        )

    for label in ["joint", "solo"]:
        st_path = cache_dir / f"{label}.safetensors"
        check(st_path.exists(), f"Missing {st_path}")
        t = st_load(str(st_path))
        for key in ("tokens", "peak_vals", "feature_indices"):
            check(key in t, f"{label}.safetensors missing key '{key}'")
        n_feats, k_total, ctx_len = t["tokens"].shape
        check(n_feats > 0,
              f"{label}: 0 features collected — every feature may be dead after "
              f"tiny training. Try increasing COLLECT_TOKENS or decreasing DICT_SIZE.")
        check(ctx_len == COLLECT_CTX_LEN,
              f"{label}: ctx_len={ctx_len}, expected {COLLECT_CTX_LEN}")
        print(f"  {label}: {n_feats} features × {k_total} contexts × {ctx_len} ctx_len")

    rand_path = cache_dir / "random_contexts.safetensors"
    check(rand_path.exists(), f"Missing {rand_path}")
    rand_t = st_load(str(rand_path))
    check("tokens" in rand_t, "random_contexts.safetensors missing 'tokens'")
    print(f"  random contexts: {rand_t['tokens'].shape[0]} samples")

    print("PASSED")
    return cache_dir


# ---------------------------------------------------------------------------
# Step 5 — compute_cooccurrence (unknown 3)
# ---------------------------------------------------------------------------

def step_cooccurrence(out_dir: Path, joint_path: Path, solo_path: Path,
                      cache_dir: Path, device: str):
    banner("Step 5: compute_cooccurrence() (unknown 3)")

    from safetensors.torch import load_file as st_load
    from meta_sae import compute_cooccurrence

    # Build a minimal mock scores.json whose feature indices match the cache.
    features = []
    for label in ["joint", "solo"]:
        t = st_load(str(cache_dir / f"{label}.safetensors"))
        for idx in t["feature_indices"].tolist():
            features.append({
                "label":           label,
                "feature_idx":     int(idx),
                "explanation":     f"Smoke-test mock explanation for {label} #{int(idx)}",
                "fuzzing_score":   0.3,
                "detection_score": None,
                "n_contexts_used": int(t["tokens"].shape[1]),
            })
    scores_path = out_dir / "mock_scores.json"
    scores_path.write_text(json.dumps({"features": features}))
    print(f"  Mock scores.json: {len(features)} entries")

    shared_kwargs = dict(
        joint_checkpoint=str(joint_path),
        solo_checkpoint=str(solo_path),
        joint_thresholds=str(out_dir / "thresholds_joint.npy"),
        solo_thresholds=str(out_dir / "thresholds_solo.npy"),
        scores_json=str(scores_path),
        n_tokens_pass1=PHI_PASS1_TOKENS,
        n_tokens_pass2=PHI_PASS2_TOKENS,
        skip_documents=0,
        n_candidates=PHI_N_CANDIDATES,
        candidate_max_fr=0.05,
        top_k_save=50,
        top_matches=10,
        top_examples=5,
        flush_every=20,
        chunk_size=512,
        model_batch_size=MODEL_BATCH_SIZE,
        seq_len=SEQ_LEN,
        dataset_path=DATASET,
        dataset_name=DATASET_NAME,
        device=device,
    )

    # Pass 1 + Pass 2 (full run)
    phi_dir = out_dir / "phi"
    phi_dir.mkdir(parents=True, exist_ok=True)
    with _restore_stderr():
        compute_cooccurrence(output_dir=str(phi_dir), **shared_kwargs)
    topk_path = phi_dir / "cross_phi_topk.npz"
    check(topk_path.exists(), f"Missing {topk_path}")
    data = np.load(topk_path, allow_pickle=True)
    print(f"  Pass 1+2 — cross_phi_topk.npz keys: {sorted(data.files)}")

    # --load-topk shortcut: skip Pass 1, run only Pass 2 from the saved npz
    phi_dir2 = out_dir / "phi_reload"
    phi_dir2.mkdir(parents=True, exist_ok=True)
    with _restore_stderr():
        compute_cooccurrence(output_dir=str(phi_dir2), load_topk=str(topk_path),
                             **shared_kwargs)
    check((phi_dir2 / "splitting_candidates.json").exists(),
          "phi_reload: Missing splitting_candidates.json")
    check((phi_dir2 / "merging_candidates.json").exists(),
          "phi_reload: Missing merging_candidates.json")
    print("  --load-topk shortcut (skip Pass 1): OK")

    print("PASSED")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="MetaSAE pipeline smoke test")
    p.add_argument("--output-dir", default="/tmp/metasae_smoke",
                   help="Directory for checkpoints and outputs (default: /tmp/metasae_smoke)")
    p.add_argument("--cpu", action="store_true",
                   help="Force CPU even if a GPU is available (slow)")
    return p.parse_args()


def main():
    args    = parse_args()
    device  = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    step_preflight(device)
    step_alt_sae_types(device)
    joint_path, solo_path = step_training(out_dir, device)
    step_calibrate(out_dir, joint_path, solo_path, device)
    cache_dir = step_collect(out_dir, joint_path, solo_path, device)
    step_cooccurrence(out_dir, joint_path, solo_path, cache_dir, device)

    elapsed = time.perf_counter() - t0
    print(f"\n{'='*60}")
    print(f"  ALL STEPS PASSED  ({elapsed / 60:.1f} min total)")
    print(f"  Outputs in: {out_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
