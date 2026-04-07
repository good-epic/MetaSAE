"""
Collect JumpReLU activation contexts for auto-interp scoring.

Library entry point: ``collect_activations()``.
Individual helper functions are also importable for custom pipelines.

Context format: pre-context ending at the firing token (the FINAL TOKEN in each
window is where the latent activated).

Output pools per feature (second axis order):
  [0          : n_train         ] — explanation examples (Q5-biased)
  [n_train    : n_train+n_fuzz  ] — fuzzing examples    (stratified, shuffled)
  [n_train+n_fuzz : ...+n_detect] — detection examples  (Q5)
  [...        : K_total         ] — boundary examples   (Q1, if n_boundary > 0)

Two modes:
  Matching mode (default): Hungarian-matched joint/solo decoder pairs.
  Targeted mode (targets_joint/solo paths provided): explicit feature index lists.
"""

import gc
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer

try:
    from safetensors.torch import save_file
except ImportError:
    raise ImportError("Install safetensors: pip install safetensors")

from meta_sae.io import load_sae
from meta_sae.activation_store import ActivationsStore


# ---------------------------------------------------------------------------
# Hungarian matching (requires lapjv: pip install lapjv)
# ---------------------------------------------------------------------------

def compute_hungarian_matching(sae_a, sae_b):
    """Compute Hungarian matching between two SAE decoders.

    Returns (row_indices, col_indices, cosine_similarities) where
    row=sae_a feature index, col=matched sae_b feature index.
    Requires: pip install lapjv
    """
    from lapjv import lapjv

    dec_a  = sae_a.W_dec.detach().float()
    dec_b  = sae_b.W_dec.detach().float()
    norm_a = dec_a / (dec_a.norm(dim=1, keepdim=True) + 1e-10)
    norm_b = dec_b / (dec_b.norm(dim=1, keepdim=True) + 1e-10)

    print("Computing cosine similarity matrix for Hungarian matching...")
    sim_matrix = (norm_a @ norm_b.T).cpu().numpy()
    print("Running Hungarian matching (lapjv)...")
    row_to_col, _, _ = lapjv(-sim_matrix)
    row_ind  = np.arange(len(row_to_col))
    col_ind  = row_to_col
    cosines  = sim_matrix[row_ind, col_ind]
    print(f"Hungarian matching done. Mean cosine: {cosines.mean():.4f}, "
          f"Median: {np.median(cosines):.4f}")
    return row_ind, col_ind, cosines


def load_or_compute_matching(joint_sae, solo_sae, matching_path: Path):
    if matching_path.exists():
        print(f"Loading cached matching from {matching_path}")
        d = np.load(matching_path)
        return d["row_ind"], d["col_ind"], d["cosines"]
    print("Computing Hungarian decoder matching (may take a few minutes)...")
    row_ind, col_ind, cosines = compute_hungarian_matching(solo_sae, joint_sae)
    matching_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(matching_path, row_ind=row_ind, col_ind=col_ind, cosines=cosines)
    print(f"Saved matching → {matching_path}")
    return row_ind, col_ind, cosines


# ---------------------------------------------------------------------------
# Context collection
# ---------------------------------------------------------------------------

def collect_contexts(
    sae,
    thresholds: np.ndarray,
    sampled_feat_idx: np.ndarray,
    model,
    n_tokens: int,
    n_train_top: int,
    n_train_per_lower: int,
    n_fuzz_per_quintile: int,
    n_detect: int,
    n_boundary: int,
    ctx_len: int,
    n_random: int,
    model_batch_size: int,
    num_batches_in_buffer: int,
    device: str,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_features      = len(sampled_feat_idx)
    n_train         = n_train_top + 4 * n_train_per_lower
    n_fuzz          = 5 * n_fuzz_per_quintile
    K_total         = n_train + n_fuzz + n_detect + n_boundary

    layer           = sae.cfg["layer"]
    seq_len         = sae.cfg["seq_len"]
    input_unit_norm = sae.cfg.get("input_unit_norm", False)

    feat_idx_t = torch.tensor(sampled_feat_idx, dtype=torch.long, device=device)
    W_enc_s    = sae.W_enc[:, feat_idx_t].detach().to(device)
    b_dec      = sae.b_dec.detach().to(device)
    thresh_s   = torch.tensor(thresholds[sampled_feat_idx], dtype=torch.float32, device=device)

    store_cfg = {
        "model_name":            sae.cfg["model_name"],
        "dataset_path":          sae.cfg["dataset_path"],
        "dataset_name":          sae.cfg.get("dataset_name", None),
        "hook_point":            f"blocks.{layer}.hook_resid_pre",
        "seq_len":               seq_len,
        "model_batch_size":      model_batch_size,
        "num_batches_in_buffer": num_batches_in_buffer,
        "device":                device,
        "batch_size":            model_batch_size * seq_len,
        "act_size":              sae.cfg["act_size"],
        "layer":                 layer,
    }
    store = ActivationsStore(model, store_cfg)

    all_tokens = np.zeros(n_tokens, dtype=np.int32)

    acc_feat_idx:   List[np.ndarray] = []
    acc_global_pos: List[np.ndarray] = []
    acc_vals:       List[np.ndarray] = []

    random_pos_reservoir: List[int] = []
    random_total = 0
    stream_pos   = 0
    batch_num    = 0
    T            = seq_len

    print(f"  Streaming {n_tokens:,} tokens for {n_features} features "
          f"(ctx_len={ctx_len}, K_total={K_total}, n_random={n_random})...")

    while stream_pos < n_tokens:
        batch_tokens = store.get_batch_tokens()
        acts_raw     = store.get_activations(batch_tokens)
        B       = batch_tokens.shape[0]
        end_pos = min(stream_pos + B * T, n_tokens)

        flat_toks = batch_tokens.cpu().numpy().reshape(-1)
        all_tokens[stream_pos:end_pos] = flat_toks[:end_pos - stream_pos]

        with torch.no_grad():
            x = acts_raw.reshape(B * T, -1)
            if input_unit_norm:
                x_mean = x.mean(dim=-1, keepdim=True)
                x_std  = x.std(dim=-1, keepdim=True)
                x = (x - x_mean) / (x_std + 1e-5)

            pre_s  = F.relu((x - b_dec) @ W_enc_s)
            acts_s = pre_s * (pre_s > thresh_s)

            t_in_seq   = torch.arange(B * T, device=device) % T
            valid_mask = (t_in_seq >= ctx_len - 1).unsqueeze(1)
            acts_s     = acts_s * valid_mask

            fired_rows, fired_feats = torch.where(acts_s > 0)
            if fired_rows.numel() > 0:
                g_pos   = (stream_pos + fired_rows.cpu().numpy()).astype(np.int32)
                fi_np   = fired_feats.cpu().numpy().astype(np.int32)
                val_np  = acts_s[fired_rows, fired_feats].cpu().numpy().astype(np.float32)
                in_range = g_pos < n_tokens
                if in_range.any():
                    acc_feat_idx.append(fi_np[in_range])
                    acc_global_pos.append(g_pos[in_range])
                    acc_vals.append(val_np[in_range])

        rand_local  = random.randrange(B * T)
        rand_global = stream_pos + rand_local
        if rand_global < n_tokens:
            random_total += 1
            if len(random_pos_reservoir) < n_random:
                random_pos_reservoir.append(rand_global)
            else:
                j = random.randrange(random_total)
                if j < n_random:
                    random_pos_reservoir[j] = rand_global

        stream_pos = end_pos
        batch_num += 1
        if batch_num % 10 == 0:
            print(f"    {stream_pos:>10,} / {n_tokens:,} tokens", flush=True)

    print(f"    {stream_pos:>10,} / {n_tokens:,} tokens — done.", flush=True)
    del store

    if not acc_feat_idx:
        raise RuntimeError("No firings recorded — check thresholds and token count.")

    all_fi  = np.concatenate(acc_feat_idx)
    all_pos = np.concatenate(acc_global_pos)
    all_val = np.concatenate(acc_vals)

    order   = np.argsort(all_fi, kind="stable")
    all_fi  = all_fi[order]
    all_pos = all_pos[order]
    all_val = all_val[order]

    feat_bounds = np.searchsorted(all_fi, np.arange(n_features + 1))

    tokens_out    = np.zeros((n_features, K_total, ctx_len), dtype=np.int32)
    peak_vals_out = np.zeros((n_features, K_total),          dtype=np.float32)
    n_complete    = 0

    for fi in range(n_features):
        lo = int(feat_bounds[fi])
        hi = int(feat_bounds[fi + 1])
        if hi == lo:
            continue

        mags   = all_val[lo:hi]
        pos    = all_pos[lo:hi]
        desc   = np.argsort(-mags)
        mags_d = mags[desc]
        pos_d  = pos[desc]
        n_tot  = len(mags_d)

        if n_tot < K_total:
            continue

        qs       = max(1, n_tot // 5)
        q_bounds = [0, qs, 2 * qs, 3 * qs, 4 * qs, n_tot]
        q_ranges = [np.arange(q_bounds[i], q_bounds[i + 1]) for i in range(5)]

        q5_train    = rng.choice(q_ranges[0], size=min(n_train_top, len(q_ranges[0])), replace=False)
        train_q5_set = set(q5_train.tolist())

        train_lower_sets: List[set] = []
        train_lower_idx:  List[int] = []
        for qi in range(1, 5):
            chosen = rng.choice(q_ranges[qi], size=min(n_train_per_lower, len(q_ranges[qi])), replace=False)
            train_lower_sets.append(set(chosen.tolist()))
            train_lower_idx.extend(chosen.tolist())
        train_idx = q5_train.tolist() + train_lower_idx

        fuzz_idx:  List[int] = []
        remaining_q5 = np.setdiff1d(q_ranges[0], list(train_q5_set))
        fuzz_q5  = rng.choice(remaining_q5, size=min(n_fuzz_per_quintile, len(remaining_q5)), replace=False)
        fuzz_q5_set = set(fuzz_q5.tolist())
        fuzz_idx.extend(fuzz_q5.tolist())

        fuzz_lower_sets: List[set] = []
        for qi in range(1, 5):
            remaining = np.setdiff1d(q_ranges[qi], list(train_lower_sets[qi - 1]))
            chosen = rng.choice(remaining, size=min(n_fuzz_per_quintile, len(remaining)), replace=False)
            fuzz_lower_sets.append(set(chosen.tolist()))
            fuzz_idx.extend(chosen.tolist())

        fuzz_arr = np.array(fuzz_idx, dtype=np.int64)
        rng.shuffle(fuzz_arr)
        fuzz_idx = fuzz_arr.tolist()

        used_q5             = train_q5_set | fuzz_q5_set
        remaining_q5_detect = np.setdiff1d(q_ranges[0], list(used_q5))
        detect_idx = rng.choice(remaining_q5_detect, size=min(n_detect, len(remaining_q5_detect)), replace=False).tolist()

        boundary_idx: List[int] = []
        if n_boundary > 0:
            used_q1      = train_lower_sets[3] | fuzz_lower_sets[3]
            remaining_q1 = np.setdiff1d(q_ranges[4], list(used_q1))
            boundary_idx = rng.choice(remaining_q1, size=min(n_boundary, len(remaining_q1)), replace=False).tolist()

        all_sample_idx = train_idx + fuzz_idx + detect_idx + boundary_idx
        for k, idx in enumerate(all_sample_idx[:K_total]):
            gpos   = int(pos_d[idx])
            c_start = gpos - ctx_len + 1
            tokens_out[fi, k]    = all_tokens[c_start : gpos + 1]
            peak_vals_out[fi, k] = float(mags_d[idx])

        if min(len(all_sample_idx), K_total) >= K_total:
            n_complete += 1

    print(f"  Features with full {K_total} contexts: {n_complete} / {n_features}")

    random_out = np.zeros((len(random_pos_reservoir), ctx_len), dtype=np.int32)
    for i, gpos in enumerate(random_pos_reservoir):
        c_start = max(0, gpos - ctx_len + 1)
        c_end   = c_start + ctx_len
        if c_end > n_tokens:
            c_end   = n_tokens
            c_start = max(0, c_end - ctx_len)
        window = all_tokens[c_start:c_end]
        random_out[i, :len(window)] = window

    return tokens_out, peak_vals_out, random_out


# ---------------------------------------------------------------------------
# Save cache
# ---------------------------------------------------------------------------

def save_cache(output_dir, label, tokens, peak_vals, feature_indices, random_tokens,
               n_train, n_fuzz, n_detect, n_boundary, ctx_len,
               feature_tags: Optional[List[str]] = None):
    tensors = {
        "tokens":          torch.tensor(tokens,          dtype=torch.int32),
        "peak_vals":       torch.tensor(peak_vals,       dtype=torch.float32),
        "feature_indices": torch.tensor(feature_indices, dtype=torch.int32),
    }
    save_file(tensors, str(output_dir / f"{label}.safetensors"))
    print(f"  Saved {label} cache → {output_dir / f'{label}.safetensors'}")

    rand_path = output_dir / "random_contexts.safetensors"
    if not rand_path.exists():
        save_file({"tokens": torch.tensor(random_tokens, dtype=torch.int32)}, str(rand_path))
        print(f"  Saved random contexts → {rand_path}")

    meta = {
        "label":           label,
        "n_features":      int(tokens.shape[0]),
        "K_total":         int(tokens.shape[1]),
        "n_train":         n_train,
        "n_fuzz":          n_fuzz,
        "n_detect":        n_detect,
        "n_boundary":      n_boundary,
        "ctx_len":         ctx_len,
        "feature_indices": feature_indices.tolist(),
    }
    if feature_tags is not None:
        meta["feature_tags"] = feature_tags
    (output_dir / f"{label}_meta.json").write_text(json.dumps(meta, indent=2))


# ---------------------------------------------------------------------------
# High-level orchestrator
# ---------------------------------------------------------------------------

def collect_activations(
    joint_checkpoint: str,
    solo_checkpoint: str,
    thresholds_joint: str,
    thresholds_solo: str,
    output_dir: str,
    *,
    targets_joint: Optional[str] = None,
    targets_solo: Optional[str] = None,
    matching_path: Optional[str] = None,
    n_features: int = 20480,
    n_tokens: int = 5_000_000,
    n_train_top: int = 30,
    n_train_per_lower: int = 5,
    n_fuzz_per_quintile: int = 5,
    n_detect: int = 25,
    n_boundary: int = 0,
    ctx_len: int = 64,
    n_random: int = 200,
    min_contexts: int = 60,
    model_batch_size: int = 64,
    num_batches_in_buffer: int = 3,
    max_val: float = 10.0,
    seed: int = 42,
    device: str = "cuda",
) -> None:
    """Collect JumpReLU activation contexts and save a delphi-format cache.

    Loads the SAE pair, performs Hungarian matching (or uses provided targets),
    streams tokens to collect stratified context windows per feature, and saves
    safetensors + JSON metadata files to ``output_dir``.

    The final action redirects C-level stderr to suppress streaming-dataset
    teardown noise. Callers that need stderr restored afterwards (e.g. tests)
    should wrap this call in a ``_restore_stderr()`` context manager.
    """
    targeted_mode = targets_joint is not None
    if targeted_mode and targets_solo is None:
        raise ValueError("targets_solo required when targets_joint is provided")
    if not targeted_mode and targets_solo is not None:
        raise ValueError("targets_joint required when targets_solo is provided")

    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed=seed)

    n_train = n_train_top + 4 * n_train_per_lower
    n_fuzz  = 5 * n_fuzz_per_quintile
    K_total = n_train + n_fuzz + n_detect + n_boundary
    print(f"Context pools: n_train={n_train}, n_fuzz={n_fuzz}, "
          f"n_detect={n_detect}, n_boundary={n_boundary} → K_total={K_total}")
    print(f"Mode: {'TARGETED' if targeted_mode else 'MATCHING'}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading SAEs...")
    joint_sae = load_sae(joint_checkpoint, device)
    solo_sae  = load_sae(solo_checkpoint,  device)
    joint_sae.eval(); solo_sae.eval()

    thresh_joint = np.load(thresholds_joint)
    thresh_solo  = np.load(thresholds_solo)
    print(f"  Joint dead: {(thresh_joint >= max_val).sum()} / {len(thresh_joint)}")
    print(f"  Solo  dead: {(thresh_solo  >= max_val).sum()} / {len(thresh_solo)}")

    if targeted_mode:
        with open(targets_joint) as f:
            joint_target_dict: Dict[str, str] = json.load(f)
        with open(targets_solo) as f:
            solo_target_dict:  Dict[str, str] = json.load(f)

        live_joint = thresh_joint < max_val
        live_solo  = thresh_solo  < max_val

        joint_items = [(int(k), v) for k, v in joint_target_dict.items() if live_joint[int(k)]]
        solo_items  = [(int(k), v) for k, v in solo_target_dict.items()  if live_solo[int(k)]]

        selected_joint = np.array([fi for fi, _ in joint_items], dtype=np.int32)
        joint_tags     = [tag for _, tag in joint_items]
        selected_solo  = np.array([fi for fi, _ in solo_items],  dtype=np.int32)
        solo_tags      = [tag for _, tag in solo_items]
        print(f"  Joint targets: {len(selected_joint):,}  Solo targets: {len(selected_solo):,}")
    else:
        match_path = (Path(matching_path) if matching_path else out_dir / "matching.npz")
        row_ind, col_ind, cosines = load_or_compute_matching(joint_sae, solo_sae, match_path)
        solo_matched  = row_ind
        joint_matched = col_ind

        live_joint = thresh_joint < max_val
        live_solo  = thresh_solo  < max_val
        live_mask  = live_joint[joint_matched] & live_solo[solo_matched]
        live_positions = np.where(live_mask)[0]
        print(f"\n{live_mask.sum()} / {len(live_mask)} matched pairs are live.")

        if len(live_positions) <= n_features:
            selected = live_positions
        else:
            selected = rng.choice(live_positions, size=n_features, replace=False)
            selected = np.sort(selected)

        selected_joint = joint_matched[selected]
        selected_solo  = solo_matched[selected]
        selected_cos   = cosines[selected]
        joint_tags     = None
        solo_tags      = None
        print(f"Selected {len(selected)} pairs (mean cos: {selected_cos.mean():.4f})")

        np.savez(out_dir / "feature_pairs.npz",
                 joint_indices=selected_joint.astype(np.int32),
                 solo_indices=selected_solo.astype(np.int32),
                 cosines=selected_cos.astype(np.float32))

    model_name = joint_sae.cfg["model_name"]
    print(f"\nLoading {model_name} via TransformerLens...")
    model = HookedTransformer.from_pretrained(model_name)
    model.eval()
    model.to(device)

    collect_kwargs = dict(
        model=model, n_tokens=n_tokens,
        n_train_top=n_train_top, n_train_per_lower=n_train_per_lower,
        n_fuzz_per_quintile=n_fuzz_per_quintile, n_detect=n_detect,
        n_boundary=n_boundary, ctx_len=ctx_len, n_random=n_random,
        model_batch_size=model_batch_size,
        num_batches_in_buffer=num_batches_in_buffer,
        device=device, rng=rng,
    )

    print("\n=== Collecting JOINT SAE contexts ===")
    j_tokens, j_peaks, random_tokens = collect_contexts(
        sae=joint_sae, thresholds=thresh_joint, sampled_feat_idx=selected_joint,
        **collect_kwargs,
    )
    print("\n=== Collecting SOLO SAE contexts ===")
    s_tokens, s_peaks, _ = collect_contexts(
        sae=solo_sae, thresholds=thresh_solo, sampled_feat_idx=selected_solo,
        **collect_kwargs,
    )

    j_n_ctxs = (j_peaks > 0).sum(axis=1)
    s_n_ctxs = (s_peaks > 0).sum(axis=1)

    if targeted_mode:
        j_keep = np.where(j_n_ctxs >= min_contexts)[0]
        s_keep = np.where(s_n_ctxs >= min_contexts)[0]
        print(f"\nJoint with >= {min_contexts} contexts: {len(j_keep)} / {len(selected_joint)}")
        print(f"Solo  with >= {min_contexts} contexts: {len(s_keep)} / {len(selected_solo)}")
        if len(j_keep) == 0 or len(s_keep) == 0:
            raise RuntimeError("No features passed min_contexts. Try min_contexts=1 or n_tokens larger.")
        j_tags_kept = [joint_tags[i] for i in j_keep] if joint_tags else None
        s_tags_kept = [solo_tags[i]  for i in s_keep] if solo_tags  else None
    else:
        both_ok = (j_n_ctxs >= min_contexts) & (s_n_ctxs >= min_contexts)
        keep    = np.where(both_ok)[0]
        print(f"\nFeatures with >= {min_contexts} contexts in BOTH: {len(keep)} / {len(selected_joint)}")
        if len(keep) == 0:
            raise RuntimeError("No features passed min_contexts. Try min_contexts=1 or n_tokens larger.")
        j_keep = s_keep = keep
        j_tags_kept = s_tags_kept = None
        np.savez(out_dir / "feature_pairs.npz",
                 joint_indices=selected_joint[keep].astype(np.int32),
                 solo_indices=selected_solo[keep].astype(np.int32),
                 cosines=selected_cos[keep].astype(np.float32))

    print("\nSaving caches...")
    save_cache(out_dir, "joint",
               j_tokens[j_keep], j_peaks[j_keep], selected_joint[j_keep],
               random_tokens, n_train, n_fuzz, n_detect, n_boundary,
               ctx_len, feature_tags=j_tags_kept)
    save_cache(out_dir, "solo",
               s_tokens[s_keep], s_peaks[s_keep], selected_solo[s_keep],
               random_tokens, n_train, n_fuzz, n_detect, n_boundary,
               ctx_len, feature_tags=s_tags_kept)
    print("\nDone.")

    # Redirect C-level stderr before local variable teardown triggers streaming
    # dataset background threads, which otherwise print spurious messages on exit.
    sys.stdout.flush()
    sys.stderr.flush()
    _devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(_devnull, 2)
    os.close(_devnull)
