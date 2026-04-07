"""
Compute the cross-SAE phi (φ) co-occurrence matrix between joint and solo primary SAEs.

Library entry point: ``compute_cooccurrence()``.
Individual helper functions are also importable for custom pipelines.

N11[i, j] = number of tokens where joint feature i AND solo feature j both fire.
φ(i, j) = (N11 * n − N1i * N1j) / sqrt(N1i*(n−N1i)*N1j*(n−N1j))

Firing uses per-feature EER thresholds (from scripts/calibrate.py) for
reproducible per-token firing decisions.

Two-pass approach:
  Pass 1 (default 20M tokens): accumulate N11 cross-matrix, compute phi.
  Pass 2 (default 2M tokens): collect top-K activation text examples for candidates.
"""

import gc
import heapq
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from meta_sae.io import load_sae
from meta_sae.activation_store import ActivationsStore
from meta_sae.utils import load_model_and_set_sizes


# ---------------------------------------------------------------------------
# Encoder helper
# ---------------------------------------------------------------------------

def get_pre_activations(sae, x: torch.Tensor) -> torch.Tensor:
    """Compute ReLU((x_norm - b_dec) @ W_enc), applying input_unit_norm if set."""
    with torch.no_grad():
        if sae.cfg.get("input_unit_norm", False):
            x_mean = x.mean(dim=-1, keepdim=True)
            x_std  = x.std(dim=-1, keepdim=True)
            x = (x - x_mean) / (x_std + 1e-5)
        return F.relu((x - sae.b_dec) @ sae.W_enc)


# ---------------------------------------------------------------------------
# Activation store builder
# ---------------------------------------------------------------------------

def build_store(model, layer: int, seq_len: int, model_batch_size: int,
                device: str, dataset_path: str, dataset_name: Optional[str],
                skip: int = 0) -> ActivationsStore:
    store_cfg = {
        "model_name":            model.cfg.model_name,
        "dataset_path":          dataset_path,
        "dataset_name":          dataset_name,
        "hook_point":            f"blocks.{layer}.hook_resid_pre",
        "layer":                 layer,
        "seq_len":               seq_len,
        "model_batch_size":      model_batch_size,
        "device":                device,
        "num_batches_in_buffer": 1,
        "batch_size":            model_batch_size * seq_len,
        "act_size":              model.cfg.d_model,
        "skip_documents":        skip,
    }
    return ActivationsStore(model, store_cfg)


# ---------------------------------------------------------------------------
# Descriptions loader
# ---------------------------------------------------------------------------

def load_auto_interp_descriptions(scores_json: Path) -> Dict:
    with open(scores_json) as f:
        data = json.load(f)
    desc = {}
    for feat in data["features"]:
        desc[(feat["label"], feat["feature_idx"])] = feat.get("explanation", "")
    return desc


# ---------------------------------------------------------------------------
# Pass 1: accumulate N11 cross-matrix
# ---------------------------------------------------------------------------

def run_pass1(
    model,
    joint_sae,
    solo_sae,
    joint_thresh: torch.Tensor,
    solo_thresh: torch.Tensor,
    device: str,
    *,
    n_tokens: int,
    model_batch_size: int,
    seq_len: int,
    layer: int,
    dataset_path: str,
    dataset_name: Optional[str],
    skip_documents: int = 0,
    flush_every: int = 20,
):
    """Accumulate N11, N1_joint, N1_solo over ``n_tokens`` tokens.

    Returns:
        (N11_cpu, N1_joint, N1_solo, n_total)
    """
    dict_size        = joint_sae.cfg["dict_size"]
    tokens_per_batch = model_batch_size * seq_len
    n_batches        = (n_tokens + tokens_per_batch - 1) // tokens_per_batch

    print(f"\nPass 1: {n_tokens:,} tokens → {n_batches} batches")

    N11_cpu  = torch.zeros(dict_size, dict_size, dtype=torch.float32)
    N1_joint = torch.zeros(dict_size, dtype=torch.float64)
    N1_solo  = torch.zeros(dict_size, dtype=torch.float64)
    n_total  = 0

    N11_gpu = torch.zeros(dict_size, dict_size, dtype=torch.float16, device=device)
    store   = build_store(model, layer, seq_len, model_batch_size, device,
                          dataset_path, dataset_name, skip=skip_documents)

    for batch_idx in tqdm(range(n_batches), desc="  Pass 1", unit="batch"):
        batch_tokens = store.get_batch_tokens()
        acts         = store.get_activations(batch_tokens)
        flat         = acts.reshape(-1, acts.shape[-1]).to(device)
        del batch_tokens, acts

        n_toks  = flat.shape[0]
        n_total += n_toks

        with torch.no_grad():
            joint_pre   = get_pre_activations(joint_sae, flat)
            solo_pre    = get_pre_activations(solo_sae,  flat)
            joint_fired = (joint_pre > joint_thresh).half()
            solo_fired  = (solo_pre  > solo_thresh).half()

            N1_joint += joint_fired.float().sum(dim=0).cpu().double()
            N1_solo  += solo_fired.float().sum(dim=0).cpu().double()
            N11_gpu  += joint_fired.T @ solo_fired

        del flat, joint_pre, solo_pre, joint_fired, solo_fired

        if (batch_idx + 1) % flush_every == 0 or batch_idx == n_batches - 1:
            N11_cpu += N11_gpu.cpu().float()
            N11_gpu.zero_()
            torch.cuda.empty_cache()

    del N11_gpu, store
    gc.collect(); torch.cuda.empty_cache()

    print(f"  Processed {n_total:,} tokens")
    print(f"  Mean joint firing rate: {float(N1_joint.mean() / n_total):.4f}")
    print(f"  Mean solo  firing rate: {float(N1_solo.mean()  / n_total):.4f}")
    return N11_cpu, N1_joint.float(), N1_solo.float(), n_total


# ---------------------------------------------------------------------------
# Phi computation (chunked)
# ---------------------------------------------------------------------------

def compute_phi_topk(N11_cpu, N1_joint, N1_solo, n_total, top_k, chunk_size, device):
    dict_size = N11_cpu.shape[0]
    n         = float(n_total)
    n_chunks  = (dict_size + chunk_size - 1) // chunk_size

    N1j      = N1_joint.to(device)
    N1s      = N1_solo.to(device)
    denom_j  = (N1j * (n - N1j)).sqrt()
    denom_s  = (N1s * (n - N1s)).sqrt()

    def phi_chunk(n11, N1_row, N1_col, denom_row, denom_col):
        numer = n11 * n - N1_row.unsqueeze(1) * N1_col.unsqueeze(0)
        denom = denom_row.unsqueeze(1) * denom_col.unsqueeze(0)
        return torch.where(denom > 0, numer / denom.clamp(min=1e-12), torch.zeros_like(numer))

    phi_j2s_vals = torch.zeros(dict_size, top_k, dtype=torch.float32)
    phi_j2s_idx  = torch.zeros(dict_size, top_k, dtype=torch.int32)
    print(f"\nComputing phi — Pass A (joint→solo), {n_chunks} chunks...")
    for j_start in tqdm(range(0, dict_size, chunk_size), desc="  Pass A", unit="chunk"):
        j_end = min(j_start + chunk_size, dict_size)
        n11   = N11_cpu[j_start:j_end].to(device)
        phi   = phi_chunk(n11, N1j[j_start:j_end], N1s, denom_j[j_start:j_end], denom_s)
        k     = min(top_k, phi.shape[1])
        topv, topi = torch.topk(phi, k=k, dim=1)
        phi_j2s_vals[j_start:j_end, :k] = topv.cpu()
        phi_j2s_idx[j_start:j_end,  :k] = topi.cpu().int()
        del n11, phi, topv, topi

    phi_s2j_vals = torch.zeros(dict_size, top_k, dtype=torch.float32)
    phi_s2j_idx  = torch.zeros(dict_size, top_k, dtype=torch.int32)
    print(f"Computing phi — Pass B (solo→joint), {n_chunks} chunks...")
    for s_start in tqdm(range(0, dict_size, chunk_size), desc="  Pass B", unit="chunk"):
        s_end = min(s_start + chunk_size, dict_size)
        n11   = N11_cpu[:, s_start:s_end].to(device)
        phi   = phi_chunk(n11.T, N1s[s_start:s_end], N1j, denom_s[s_start:s_end], denom_j)
        k     = min(top_k, phi.shape[1])
        topv, topi = torch.topk(phi, k=k, dim=1)
        phi_s2j_vals[s_start:s_end, :k] = topv.cpu()
        phi_s2j_idx[s_start:s_end,  :k] = topi.cpu().int()
        del n11, phi, topv, topi

    return phi_j2s_vals, phi_j2s_idx, phi_s2j_vals, phi_s2j_idx


# ---------------------------------------------------------------------------
# Nearest geometric neighbor
# ---------------------------------------------------------------------------

def nearest_geom_neighbors(candidate_indices, W_dec_query, W_dec_key, chunk=512):
    W_q = F.normalize(W_dec_query[candidate_indices], dim=1)
    W_k = F.normalize(W_dec_key, dim=1)
    result = {}
    for i, cand_idx in enumerate(candidate_indices):
        cosines = W_q[i] @ W_k.T
        result[int(cand_idx)] = int(cosines.argmax())
    return result


# ---------------------------------------------------------------------------
# Pass 2: activation text examples
# ---------------------------------------------------------------------------

def run_pass2(
    model,
    joint_sae,
    solo_sae,
    joint_thresh: torch.Tensor,
    solo_thresh: torch.Tensor,
    joint_targets: List[int],
    solo_targets: List[int],
    device: str,
    *,
    n_tokens: int,
    model_batch_size: int,
    seq_len: int,
    layer: int,
    dataset_path: str,
    dataset_name: Optional[str],
    skip_documents: int = 0,
    top_examples: int = 5,
):
    """Collect top activation text examples for candidate features.

    Returns:
        (joint_examples, solo_examples) — dicts mapping feature_idx → list of
        {"activation": float, "text": str, "dominant_token": str}.
    """
    tokenizer        = model.tokenizer
    tokens_per_batch = model_batch_size * seq_len
    n_batches        = (n_tokens + tokens_per_batch - 1) // tokens_per_batch

    joint_targets_t = torch.tensor(list(joint_targets), dtype=torch.long, device=device)
    solo_targets_t  = torch.tensor(list(solo_targets),  dtype=torch.long, device=device)

    joint_heaps: Dict[int, list] = {i: [] for i in joint_targets}
    solo_heaps:  Dict[int, list] = {i: [] for i in solo_targets}

    print(f"\nPass 2: {n_tokens:,} tokens → {n_batches} batches")
    print(f"  Collecting for {len(joint_targets)} joint + {len(solo_targets)} solo features")

    pass2_skip = skip_documents + 500_000
    store = build_store(model, layer, seq_len, model_batch_size, device,
                        dataset_path, dataset_name, skip=pass2_skip)

    for _ in tqdm(range(n_batches), desc="  Pass 2", unit="batch"):
        batch_tokens = store.get_batch_tokens()
        acts         = store.get_activations(batch_tokens)
        B, T         = batch_tokens.shape
        flat         = acts.reshape(-1, acts.shape[-1]).to(device)

        with torch.no_grad():
            joint_pre = get_pre_activations(joint_sae, flat)
            solo_pre  = get_pre_activations(solo_sae,  flat)

        for targets_t, targets, thresh, heaps in [
            (joint_targets_t, joint_targets, joint_thresh, joint_heaps),
            (solo_targets_t,  solo_targets,  solo_thresh,  solo_heaps),
        ]:
            acts_sub = (joint_pre if heaps is joint_heaps else solo_pre)[:, targets_t]
            for k, feat_idx in enumerate(targets):
                col  = acts_sub[:, k]
                for pos in col.gt(thresh[feat_idx]).nonzero(as_tuple=True)[0].cpu().tolist():
                    val   = float(col[pos].item())
                    b_idx = pos // T
                    t_idx = pos %  T
                    dominant_tok = tokenizer.decode(
                        [batch_tokens[b_idx, t_idx].item()], skip_special_tokens=False
                    )
                    snip = batch_tokens[b_idx, max(0, t_idx - 5):min(T, t_idx + 6)].tolist()
                    text = tokenizer.decode(snip, skip_special_tokens=True)
                    h = heaps[feat_idx]
                    if len(h) < top_examples:
                        heapq.heappush(h, (val, text, dominant_tok))
                    elif val > h[0][0]:
                        heapq.heapreplace(h, (val, text, dominant_tok))

        del flat, acts, batch_tokens, joint_pre, solo_pre

    def heap_to_sorted(h):
        return [{"activation": float(v), "text": t, "dominant_token": dt}
                for v, t, dt in sorted(h, key=lambda x: -x[0])]

    return (
        {i: heap_to_sorted(h) for i, h in joint_heaps.items()},
        {i: heap_to_sorted(h) for i, h in solo_heaps.items()},
    )


# ---------------------------------------------------------------------------
# Candidate assembly
# ---------------------------------------------------------------------------

def is_junk_token(tok: str) -> bool:
    tok = tok.strip()
    if len(tok) == 0 or len(tok) == 1:
        return True
    return all(not c.isalnum() for c in tok)


def dominant_token_is_junk(examples: list) -> bool:
    if not examples:
        return False
    junk = sum(1 for ex in examples if is_junk_token(ex.get("dominant_token", "")))
    return junk > len(examples) / 2


def assemble_candidates(cand_indices, cand_label, phi_topk_vals, phi_topk_idx,
                        match_label, descriptions, examples_dict, match_descriptions,
                        geom_neighbors, firing_rates_cand, firing_rates_match,
                        n_matches, n_candidates):
    candidates = []
    n_junk_skipped = 0
    for feat_idx in cand_indices:
        if len(candidates) >= n_candidates:
            break
        feat_idx = int(feat_idx)
        exs = examples_dict.get(feat_idx, [])
        if dominant_token_is_junk(exs):
            n_junk_skipped += 1
            continue

        top_vals = phi_topk_vals[feat_idx]
        top_inds = phi_topk_idx[feat_idx]
        nearest  = geom_neighbors.get(feat_idx, -1)
        phi_top2 = float(top_vals[1]) if len(top_vals) > 1 else 0.0

        match_entries = [
            {
                "phi":                      float(top_vals[rank]),
                "match_feature_idx":        int(top_inds[rank]),
                "match_description":        match_descriptions.get((match_label, int(top_inds[rank])), ""),
                "match_firing_rate":        float(firing_rates_match[int(top_inds[rank])]),
                "is_nearest_geom_neighbor": (int(top_inds[rank]) == nearest),
            }
            for rank in range(min(n_matches, len(top_vals)))
        ]
        candidates.append({
            "feature_idx":     feat_idx,
            "description":     descriptions.get((cand_label, feat_idx), ""),
            "firing_rate":     float(firing_rates_cand[feat_idx]),
            "phi_top2":        phi_top2,
            "top_examples":    exs,
            "top_phi_matches": match_entries,
        })
    if n_junk_skipped:
        print(f"  Skipped {n_junk_skipped} junk-token features")
    return candidates


# ---------------------------------------------------------------------------
# High-level orchestrator
# ---------------------------------------------------------------------------

def compute_cooccurrence(
    joint_checkpoint: str,
    solo_checkpoint: str,
    joint_thresholds: str,
    solo_thresholds: str,
    scores_json: str,
    output_dir: str,
    *,
    n_tokens_pass1: int = 20_000_000,
    n_tokens_pass2: int = 2_000_000,
    model_batch_size: int = 32,
    seq_len: int = 128,
    top_k_save: int = 50,
    n_candidates: int = 200,
    candidate_max_fr: float = 0.05,
    load_topk: Optional[str] = None,
    top_matches: int = 10,
    top_examples: int = 5,
    flush_every: int = 20,
    dataset_path: str = "HuggingFaceFW/fineweb",
    dataset_name: Optional[str] = "sample-10BT",
    skip_documents: int = 5_000_000,
    chunk_size: int = 512,
    device: str = "cuda:0",
) -> None:
    """Compute cross-SAE phi matrix and candidate lists, saving to ``output_dir``.

    Outputs:
        cross_phi_topk.npz         top-K phi matches each direction
        splitting_candidates.json  top-N solo features by phi_top2
        merging_candidates.json    top-N joint features by phi_top2

    The final action redirects C-level stderr to suppress streaming-dataset
    teardown noise. Callers that need stderr restored afterwards (e.g. tests)
    should wrap this call in a ``_restore_stderr()`` context manager.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading SAEs...")
    joint_sae = load_sae(joint_checkpoint, device)
    solo_sae  = load_sae(solo_checkpoint,  device)
    dict_size = joint_sae.cfg["dict_size"]
    layer     = joint_sae.cfg["layer"]

    print("Loading calibration thresholds...")
    joint_thresh = torch.from_numpy(np.load(joint_thresholds)).to(device)
    solo_thresh  = torch.from_numpy(np.load(solo_thresholds)).to(device)

    print("Loading model...")
    model_cfg = {
        "model_name":  joint_sae.cfg["model_name"],
        "layer":       layer,
        "site":        joint_sae.cfg.get("site", "resid_pre"),
        "device":      device,
        "model_dtype": joint_sae.cfg.get("model_dtype", "float32"),
    }
    model = load_model_and_set_sizes(model_cfg)
    model.eval()

    print("Loading auto-interp descriptions...")
    descriptions = load_auto_interp_descriptions(Path(scores_json))

    topk_path = out_dir / "cross_phi_topk.npz"

    pass1_kwargs = dict(
        n_tokens=n_tokens_pass1,
        model_batch_size=model_batch_size,
        seq_len=seq_len,
        layer=layer,
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        skip_documents=skip_documents,
        flush_every=flush_every,
    )

    if load_topk:
        print(f"Loading phi top-k arrays from {load_topk} (skipping Pass 1)...")
        d            = np.load(load_topk)
        phi_j2s_vals = torch.from_numpy(d["phi_j2s_vals"])
        phi_j2s_idx  = torch.from_numpy(d["phi_j2s_idx"])
        phi_s2j_vals = torch.from_numpy(d["phi_s2j_vals"])
        phi_s2j_idx  = torch.from_numpy(d["phi_s2j_idx"])
        N1_joint     = torch.from_numpy(d["N1_joint"])
        N1_solo      = torch.from_numpy(d["N1_solo"])
        n_total      = int(d["n_total"][0])
    else:
        N11_cpu, N1_joint, N1_solo, n_total = run_pass1(
            model, joint_sae, solo_sae, joint_thresh, solo_thresh, device,
            **pass1_kwargs,
        )
        phi_j2s_vals, phi_j2s_idx, phi_s2j_vals, phi_s2j_idx = compute_phi_topk(
            N11_cpu, N1_joint, N1_solo, n_total,
            top_k=top_k_save, chunk_size=chunk_size, device=device,
        )
        np.savez_compressed(
            topk_path,
            phi_j2s_vals=phi_j2s_vals.numpy(), phi_j2s_idx=phi_j2s_idx.numpy(),
            phi_s2j_vals=phi_s2j_vals.numpy(), phi_s2j_idx=phi_s2j_idx.numpy(),
            N1_joint=N1_joint.numpy(), N1_solo=N1_solo.numpy(),
            n_total=np.array([n_total]),
        )
        print(f"Saved phi top-k arrays → {topk_path}")

    phi_top2_solo  = phi_s2j_vals[:, 1] if top_k_save > 1 else phi_s2j_vals[:, 0]
    phi_top2_joint = phi_j2s_vals[:, 1] if top_k_save > 1 else phi_j2s_vals[:, 0]

    fr_joint = N1_joint / n_total
    fr_solo  = N1_solo  / n_total

    split_valid = torch.isfinite(phi_s2j_vals[:, 0]) & (fr_solo  <= candidate_max_fr)
    merge_valid = torch.isfinite(phi_j2s_vals[:, 0]) & (fr_joint <= candidate_max_fr)

    oversample = n_candidates * 3

    split_valid_idxs = split_valid.nonzero(as_tuple=True)[0]
    merge_valid_idxs = merge_valid.nonzero(as_tuple=True)[0]

    split_cand_indices = split_valid_idxs[
        phi_top2_solo[split_valid_idxs].argsort(descending=True)
    ][:oversample].tolist()
    merge_cand_indices = merge_valid_idxs[
        phi_top2_joint[merge_valid_idxs].argsort(descending=True)
    ][:oversample].tolist()

    print("\nComputing nearest geometric neighbors...")
    W_dec_joint = joint_sae.state_dict()["W_dec"].cpu()
    W_dec_solo  = solo_sae.state_dict()["W_dec"].cpu()
    split_geom  = nearest_geom_neighbors(split_cand_indices, W_dec_solo,  W_dec_joint)
    merge_geom  = nearest_geom_neighbors(merge_cand_indices, W_dec_joint, W_dec_solo)

    joint_examples, solo_examples = run_pass2(
        model, joint_sae, solo_sae, joint_thresh, solo_thresh,
        list(set(merge_cand_indices)), list(set(split_cand_indices)), device,
        n_tokens=n_tokens_pass2,
        model_batch_size=model_batch_size,
        seq_len=seq_len,
        layer=layer,
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        skip_documents=skip_documents,
        top_examples=top_examples,
    )

    fr_joint_np = fr_joint.numpy()
    fr_solo_np  = fr_solo.numpy()

    split_candidates = assemble_candidates(
        split_cand_indices, "solo", phi_s2j_vals, phi_s2j_idx,
        "joint", descriptions, solo_examples, descriptions,
        split_geom, fr_solo_np, fr_joint_np, top_matches, n_candidates,
    )
    split_candidates.sort(key=lambda x: -x["phi_top2"])

    merge_candidates = assemble_candidates(
        merge_cand_indices, "joint", phi_j2s_vals, phi_j2s_idx,
        "solo", descriptions, joint_examples, descriptions,
        merge_geom, fr_joint_np, fr_solo_np, top_matches, n_candidates,
    )
    merge_candidates.sort(key=lambda x: -x["phi_top2"])

    split_path = out_dir / "splitting_candidates.json"
    merge_path = out_dir / "merging_candidates.json"
    with open(split_path, "w") as f:
        json.dump({"n_tokens": n_total, "candidates": split_candidates}, f, indent=2)
    with open(merge_path, "w") as f:
        json.dump({"n_tokens": n_total, "candidates": merge_candidates}, f, indent=2)

    print(f"\nDone:")
    print(f"  {topk_path}")
    print(f"  {split_path}  ({len(split_candidates)} splitting candidates)")
    print(f"  {merge_path}  ({len(merge_candidates)} merging candidates)")

    # Redirect C-level stderr before local variable teardown triggers streaming
    # dataset background threads, which otherwise print spurious messages on exit.
    sys.stdout.flush()
    sys.stderr.flush()
    _devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(_devnull, 2)
    os.close(_devnull)
