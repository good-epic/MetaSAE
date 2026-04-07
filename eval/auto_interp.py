"""
auto_interp.py

Generate and score feature explanations for matched joint vs solo SAE features
using vLLM offline inference.  Implements two complementary scoring methods:

  Fuzzing   — LLM predicts activation strength on held-out examples.
              Score = Spearman rank correlation between predictions and actuals.

  Detection — LLM identifies activating examples from a shuffled mix of
              activating and random contexts.
              Score = balanced accuracy (chance = 0.5).

Requires: pip install vllm safetensors scipy

The pipeline runs in 3 batched generate() calls (no server, no HTTP):
  1. Explanations  — all features in one batch
  2. Fuzzing       — all features in one batch
  3. Detection     — all features in one batch

Context format: each example is a pre-context window ending at the firing
token — the FINAL TOKEN shown is where the latent activated.

Usage:
    python eval/auto_interp.py \\
        --cache-dir  outputs/delphi_cache/ \\
        --output-dir outputs/auto_interp_results/ \\
        --model-name Qwen/Qwen2.5-72B-Instruct-AWQ \\
        --quantization awq_marlin

For a quick smoke-test:
    python eval/auto_interp.py ... --n-features 10 \\
        --model-name Qwen/Qwen2.5-7B-Instruct
"""

import argparse
import json
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr, ttest_rel
from safetensors.torch import load_file as st_load_file


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FeatureData:
    feature_idx: int         # SAE feature index
    pair_idx: int            # index into the matched-pair list
    label: str               # "joint" or "solo"
    tokens: np.ndarray       # int32  (K_total, ctx_len)
    peak_vals: np.ndarray    # float32 (K_total,)
    tag: Optional[str] = None  # from targeted mode (e.g. "split_candidate")


@dataclass
class ScoredFeature:
    feature_idx: int
    pair_idx: int
    label: str
    explanation: str
    fuzzing_score: Optional[float]
    detection_score: Optional[float]
    n_contexts_used: int
    tag: Optional[str] = None


# ---------------------------------------------------------------------------
# Cache loading
# ---------------------------------------------------------------------------

def load_cache(cache_dir: Path, label: str) -> Tuple[List[FeatureData], np.ndarray]:
    """Load feature contexts and random baseline contexts from the safetensors cache."""
    feat_tensors  = st_load_file(str(cache_dir / f"{label}.safetensors"))
    rand_tensors  = st_load_file(str(cache_dir / "random_contexts.safetensors"))

    tokens_all   = feat_tensors["tokens"].cpu().numpy()           # (n, K_total, ctx_len)
    peaks_all    = feat_tensors["peak_vals"].cpu().numpy()        # (n, K_total)
    feat_idx_all = feat_tensors["feature_indices"].cpu().numpy()  # (n,)
    random_tokens = rand_tensors["tokens"].cpu().numpy()          # (n_random, ctx_len)

    # Load tags from meta JSON (present in targeted mode)
    meta_path = cache_dir / f"{label}_meta.json"
    tags_list = None
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        tags_list = meta.get("feature_tags", None)

    features = [
        FeatureData(
            feature_idx=int(feat_idx_all[i]),
            pair_idx=i,
            label=label,
            tokens=tokens_all[i],
            peak_vals=peaks_all[i],
            tag=tags_list[i] if tags_list is not None else None,
        )
        for i in range(tokens_all.shape[0])
    ]
    return features, random_tokens


# ---------------------------------------------------------------------------
# Token display helpers
# ---------------------------------------------------------------------------

_TOKENIZER = None
_CONTEXT_TOKENIZER_NAME = "gpt2-large"  # set from CLI in main()

def _tok():
    """
    Tokenizer for decoding cached token IDs back to text.
    Must match the SAE's base model (gpt2-large by default), NOT the
    explainer LLM — the cached token IDs are from the SAE model.
    """
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer
        _TOKENIZER = AutoTokenizer.from_pretrained(_CONTEXT_TOKENIZER_NAME)
    return _TOKENIZER


def format_context_plain(token_ids: np.ndarray) -> str:
    return "".join(_tok().decode([int(t)]) for t in token_ids)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a precise scientific assistant helping to understand neural network features. "
    "Follow instructions exactly and respond only in the requested format."
)


def build_explanation_prompt(feat: FeatureData, n_train: int,
                             n_boundary: int = 0, boundary_start: int = 0) -> str:
    """
    Show n_train high-activation examples with strengths (normalized 0-10).
    Optionally appends n_boundary low-activation examples labeled as boundary.
    The final token in each example is where the latent fired.
    """
    K = feat.tokens.shape[0]
    n_shown = min(n_train, K)
    if n_shown == 0:
        return ""

    # Normalize activation strengths to 0-10 relative to this feature's max
    train_peaks = feat.peak_vals[:n_shown]
    max_peak = float(train_peaks.max()) if train_peaks.max() > 0 else 1.0

    lines = []
    for i in range(n_shown):
        ctx      = format_context_plain(feat.tokens[i])
        strength = min(10.0, (float(feat.peak_vals[i]) / max_peak) * 10.0)
        lines.append(f"Example {i + 1} (activation: {strength:.1f}/10):\n\"{ctx}\"")

    prompt = (
        "We are studying a latent (feature) in a sparse autoencoder that activates "
        "on certain patterns in text.\n\n"
        "Below are examples where the latent activated. "
        "THE FINAL TOKEN of each example is the token where the latent fired. "
        "The activation strength is shown on a 0–10 scale.\n\n"
        + "\n\n".join(lines)
    )

    # Add boundary (low-activation) examples if available
    n_boundary_shown = min(n_boundary, K - boundary_start) if n_boundary > 0 else 0
    if n_boundary_shown > 0:
        boundary_lines = []
        for i in range(n_boundary_shown):
            ctx = format_context_plain(feat.tokens[boundary_start + i])
            strength = min(10.0, (float(feat.peak_vals[boundary_start + i]) / max_peak) * 10.0)
            boundary_lines.append(
                f"Low-activation example {i + 1} (activation: {strength:.1f}/10):\n\"{ctx}\""
            )
        prompt += (
            "\n\nThe following are examples where the latent fired weakly "
            "(near the activation boundary):\n\n"
            + "\n\n".join(boundary_lines)
        )

    prompt += (
        "\n\nDescribe this latent's behaviour in ONE concise sentence. "
        "Focus on the linguistic or semantic pattern the final activating token "
        "responds to, and what distinguishes stronger from weaker activations."
    )
    return prompt


def build_fuzzing_prompt(explanation: str, feat: FeatureData,
                         fuzz_start: int, n_fuzz: int) -> str:
    """
    Show n_fuzz examples WITHOUT activation strengths (model must predict them).
    The final token in each example is where the latent fires.
    Examples are already shuffled in the cache (random activation-magnitude order).
    """
    K = feat.tokens.shape[0]
    n_shown = min(n_fuzz, K - fuzz_start)
    if n_shown == 0:
        return ""

    lines = []
    for i in range(n_shown):
        ctx = format_context_plain(feat.tokens[fuzz_start + i])
        lines.append(f"Example {i + 1}:\n\"{ctx}\"")

    return (
        f"This latent has been described as:\n\"{explanation}\"\n\n"
        f"For each of the {n_shown} examples below, THE FINAL TOKEN is where "
        "the latent fires. Predict how strongly the latent activates on a scale "
        "of 0 (no activation) to 10 (maximum activation).\n\n"
        + "\n\n".join(lines) + "\n\n"
        f"Respond with ONLY a JSON array of {n_shown} numbers, e.g.: [3.5, 0.0, 7.2]"
    )


def build_detection_prompt(
    explanation: str,
    act_contexts: List[np.ndarray],
    rand_contexts: List[np.ndarray],
) -> Tuple[str, List[int]]:
    """
    Returns (prompt, true_passage_numbers_1indexed).
    In real activation passages, the FINAL TOKEN is where the latent fired.
    Random passages are unrelated text where the latent did not fire.
    """
    combined = [(t, True) for t in act_contexts] + [(t, False) for t in rand_contexts]
    random.shuffle(combined)

    lines, true_set = [], []
    for i, (toks, is_act) in enumerate(combined):
        lines.append(f"Passage {i + 1}:\n\"{format_context_plain(toks)}\"")
        if is_act:
            true_set.append(i + 1)

    return (
        f"This latent has been described as:\n\"{explanation}\"\n\n"
        f"Below are {len(combined)} text passages. In passages where this latent "
        "fires, THE FINAL TOKEN is where the activation occurs. "
        "Other passages are random text where the latent did not fire.\n\n"
        + "\n\n".join(lines) + "\n\n"
        "Identify ALL passages where you think this latent fires. "
        "There may be any number.\n"
        "Respond with ONLY a JSON array of passage numbers (1-indexed), "
        "e.g.: [1, 3, 5]  — or [] if none."
    ), true_set


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def apply_chat_template(tokenizer, user_prompt: str) -> str:
    """Apply the model's chat template to a single user turn."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]
    if getattr(tokenizer, "chat_template", None) is not None:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return f"{SYSTEM_PROMPT}\n\n### User\n{user_prompt}\n\n### Assistant\n"


def batch_generate(llm, tokenizer, prompts: List[str],
                   max_tokens: int) -> List[str]:
    """Apply chat template to all prompts and run a single llm.generate() call."""
    from vllm import SamplingParams
    formatted = [apply_chat_template(tokenizer, p) for p in prompts]
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    outputs = llm.generate(formatted, params)
    return [o.outputs[0].text for o in outputs]


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def _try_parse_json(text: str):
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r'\[([^\[\]]*)\]', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# Score computation (pure Python, no LLM)
# ---------------------------------------------------------------------------

def compute_fuzzing_score(
    response: str, feat: FeatureData, fuzz_start: int, n_fuzz: int
) -> Optional[float]:
    K = feat.tokens.shape[0]
    n_avail = min(n_fuzz, K - fuzz_start)
    if n_avail < 2:
        return None

    predicted = _try_parse_json(response)
    if not isinstance(predicted, list):
        return None

    actual = feat.peak_vals[fuzz_start:fuzz_start + n_avail].tolist()
    pred   = []
    for v in predicted[:n_avail]:
        try:
            pred.append(float(v))
        except (TypeError, ValueError):
            pred.append(0.0)
    while len(pred) < n_avail:
        pred.append(0.0)

    if len(set(pred)) < 2 or len(set(actual)) < 2:
        return 0.0
    rho, _ = spearmanr(actual, pred)
    return float(rho) if not np.isnan(rho) else 0.0


def compute_detection_score(
    response: str, true_set: List[int], n_act: int, n_rand: int
) -> Optional[float]:
    guessed = _try_parse_json(response)
    if not isinstance(guessed, list):
        return None

    guessed_set = set()
    for v in guessed:
        try:
            guessed_set.add(int(v))
        except (TypeError, ValueError):
            pass

    true_set_s = set(true_set)
    all_idx    = set(range(1, n_act + n_rand + 1))
    rand_set   = all_idx - true_set_s

    tp  = len(guessed_set & true_set_s)
    tn  = len((all_idx - guessed_set) & rand_set)
    tpr = tp / n_act         if n_act     > 0 else 0.0
    tnr = tn / len(rand_set) if rand_set      else 0.0
    return (tpr + tnr) / 2.0


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(args, llm, tokenizer, all_features, random_tokens):
    n            = len(all_features)
    fuzz_start   = args.n_train
    detect_start = args.n_train + args.n_fuzz
    boundary_start = args.n_train + args.n_fuzz + args.n_detect_act

    n_ref = max(n // 2, 1)   # denominator for per-pair timing (or per-feature if not pairs)
    t_pipeline_start = time.perf_counter()
    n_batches = 2 if args.fuzzing_only else 3

    # ------------------------------------------------------------------
    # Batch 1: Explanations
    # ------------------------------------------------------------------
    print(f"\n[1/{n_batches}] Generating explanations ({n} prompts)...", flush=True)
    t0 = time.perf_counter()
    expl_prompts = [
        build_explanation_prompt(f, args.n_train,
                                 n_boundary=args.n_boundary,
                                 boundary_start=boundary_start)
        for f in all_features
    ]
    explanations = batch_generate(llm, tokenizer, expl_prompts, max_tokens=150)
    t1 = time.perf_counter()
    print(f"      Done. ({t1 - t0:.1f}s, {(t1 - t0) / n_ref:.2f}s/feature)")

    # ------------------------------------------------------------------
    # Batch 2: Fuzzing
    # ------------------------------------------------------------------
    print(f"[2/{n_batches}] Scoring fuzzing ({n} prompts)...", flush=True)
    t0 = time.perf_counter()
    fuzz_prompts = [
        build_fuzzing_prompt(expl, feat, fuzz_start, args.n_fuzz)
        for expl, feat in zip(explanations, all_features)
    ]
    fuzz_responses = batch_generate(llm, tokenizer, fuzz_prompts, max_tokens=200)
    t1 = time.perf_counter()
    print(f"      Done. ({t1 - t0:.1f}s, {(t1 - t0) / n_ref:.2f}s/feature)")

    # ------------------------------------------------------------------
    # Batch 3: Detection (skipped with --fuzzing-only)
    # ------------------------------------------------------------------
    if args.fuzzing_only:
        detect_responses  = [None] * n
        detect_true_sets  = [[] for _ in range(n)]
        detect_n_act      = [0] * n
        detect_n_rand     = [0] * n
    else:
        print(f"[3/{n_batches}] Scoring detection ({n} prompts)...", flush=True)
        pool_size = len(random_tokens)
        detect_prompts, detect_true_sets, detect_n_act, detect_n_rand = [], [], [], []

        for feat, expl in zip(all_features, explanations):
            K = feat.tokens.shape[0]
            n_avail_act = min(args.n_detect_act, K - detect_start)
            act_ctxs  = [feat.tokens[detect_start + i] for i in range(n_avail_act)]
            rand_idxs = random.sample(range(pool_size), min(args.n_detect_rand, pool_size))
            rand_ctxs = [random_tokens[j] for j in rand_idxs]

            prompt, true_set = build_detection_prompt(expl, act_ctxs, rand_ctxs)
            detect_prompts.append(prompt)
            detect_true_sets.append(true_set)
            detect_n_act.append(n_avail_act)
            detect_n_rand.append(len(rand_ctxs))

        t0 = time.perf_counter()
        detect_responses = batch_generate(llm, tokenizer, detect_prompts, max_tokens=200)
        t1 = time.perf_counter()
        print(f"      Done. ({t1 - t0:.1f}s, {(t1 - t0) / n_ref:.2f}s/feature)")

    t_total = time.perf_counter() - t_pipeline_start
    print(f"\n--- Timing summary ---")
    print(f"  Total wall time : {t_total:.1f}s  ({t_total / 60:.1f} min)")
    print(f"  Per feature     : {t_total / n_ref:.2f}s")
    print(f"  Projected 1000  : {t_total / n_ref * 1000 / 60:.0f} min  "
          f"({t_total / n_ref * 1000 / 3600:.1f} hr)")

    # ------------------------------------------------------------------
    # Assemble results
    # ------------------------------------------------------------------
    results: List[ScoredFeature] = []
    for i, feat in enumerate(all_features):
        fuzz_score = compute_fuzzing_score(
            fuzz_responses[i], feat, fuzz_start, args.n_fuzz
        )
        det_score = (
            None if args.fuzzing_only else
            compute_detection_score(
                detect_responses[i], detect_true_sets[i],
                detect_n_act[i], detect_n_rand[i],
            )
        )
        results.append(ScoredFeature(
            feature_idx=feat.feature_idx,
            pair_idx=feat.pair_idx,
            label=feat.label,
            explanation=explanations[i],
            fuzzing_score=fuzz_score,
            detection_score=det_score,
            n_contexts_used=int(feat.tokens.shape[0]),
            tag=feat.tag,
        ))

    return results


# ---------------------------------------------------------------------------
# Statistical summary + save
# ---------------------------------------------------------------------------

def summarise_and_save(results: List[ScoredFeature], output_dir: Path, model_name: str):
    joint_by_pair = {r.pair_idx: r for r in results if r.label == "joint"}
    solo_by_pair  = {r.pair_idx: r for r in results if r.label == "solo"}
    pairs = sorted(set(joint_by_pair) & set(solo_by_pair))

    def _paired(attr):
        j = np.array([getattr(joint_by_pair[i], attr) for i in pairs
                      if getattr(joint_by_pair[i], attr) is not None
                      and getattr(solo_by_pair[i], attr) is not None], dtype=np.float32)
        s = np.array([getattr(solo_by_pair[i],  attr) for i in pairs
                      if getattr(joint_by_pair[i], attr) is not None
                      and getattr(solo_by_pair[i], attr) is not None], dtype=np.float32)
        return j, s

    print("\n=== Results ===")
    for attr, lbl in [("fuzzing_score",   "Fuzzing (Spearman ρ)"),
                      ("detection_score", "Detection (balanced acc)")]:
        j, s = _paired(attr)
        n = len(j)
        if n < 2:
            # Fall back to independent group means when no matched pairs
            jv = np.array([getattr(r, attr) for r in results
                           if r.label == "joint" and getattr(r, attr) is not None],
                          dtype=np.float32)
            sv = np.array([getattr(r, attr) for r in results
                           if r.label == "solo"  and getattr(r, attr) is not None],
                          dtype=np.float32)
            if len(jv) == 0 and len(sv) == 0:
                print(f"  {lbl}: no data")
                continue
            jm = float(jv.mean()) if len(jv) else float("nan")
            sm = float(sv.mean()) if len(sv) else float("nan")
            print(f"  {lbl}: joint={jm:.4f} (n={len(jv)})  "
                  f"solo={sm:.4f} (n={len(sv)})  [independent, no matched pairs]")
            continue
        stat, p = ttest_rel(j, s)
        print(f"  {lbl}: joint={j.mean():.4f}  solo={s.mean():.4f}  "
              f"Δ={j.mean()-s.mean():+.4f}  t={stat:.3f}  p={p:.4f}  n={n}")

    # Per-tag group means (targeted mode)
    all_tags = sorted({r.tag for r in results if r.tag is not None})
    if all_tags:
        print("\n=== Per-tag fuzzing means ===")
        for tag in all_tags:
            tag_joint = [r.fuzzing_score for r in results
                         if r.tag == tag and r.label == "joint"
                         and r.fuzzing_score is not None]
            tag_solo  = [r.fuzzing_score for r in results
                         if r.tag == tag and r.label == "solo"
                         and r.fuzzing_score is not None]
            jm = float(np.mean(tag_joint)) if tag_joint else float("nan")
            sm = float(np.mean(tag_solo))  if tag_solo  else float("nan")
            print(f"  {tag:<40}  joint={jm:+.4f} (n={len(tag_joint)})  "
                  f"solo={sm:+.4f} (n={len(tag_solo)})")

    j_fuzz, s_fuzz = _paired("fuzzing_score")
    j_det,  s_det  = _paired("detection_score")

    # Independent group means (used when no matched pairs)
    def _group_mean(attr, label_filter):
        vals = np.array([getattr(r, attr) for r in results
                         if r.label == label_filter and getattr(r, attr) is not None],
                        dtype=np.float32)
        return float(vals.mean()) if len(vals) else None

    out = {
        "model_name": model_name,
        "n_pairs":    len(pairs),
        "summary": {
            "joint_fuzzing_mean":   float(j_fuzz.mean()) if len(j_fuzz) else _group_mean("fuzzing_score", "joint"),
            "solo_fuzzing_mean":    float(s_fuzz.mean()) if len(s_fuzz) else _group_mean("fuzzing_score", "solo"),
            "joint_detection_mean": float(j_det.mean())  if len(j_det)  else _group_mean("detection_score", "joint"),
            "solo_detection_mean":  float(s_det.mean())  if len(s_det)  else _group_mean("detection_score", "solo"),
        },
        "features": [
            {
                "pair_idx":        r.pair_idx,
                "label":           r.label,
                "feature_idx":     r.feature_idx,
                "tag":             r.tag,
                "explanation":     r.explanation,
                "fuzzing_score":   r.fuzzing_score,
                "detection_score": r.detection_score,
                "n_contexts_used": r.n_contexts_used,
            }
            for r in sorted(results, key=lambda x: (x.pair_idx, x.label))
        ],
    }
    out_path = output_dir / "scores.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved results → {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Auto-interp scoring: fuzzing + detection for joint vs solo SAEs."
    )
    parser.add_argument("--cache-dir",   required=True,
                        help="Directory with joint/solo safetensors caches")
    parser.add_argument("--output-dir",  required=True)
    parser.add_argument("--model-name",  required=True,
                        help="HuggingFace model ID (e.g. Qwen/Qwen2.5-72B-Instruct-AWQ)")
    parser.add_argument("--quantization", default=None,
                        help="vLLM quantization method (e.g. awq_marlin)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=8192,
                        help="Max total tokens per prompt+response (default: 8192)")
    parser.add_argument("--n-features",   type=int, default=20480,
                        help="Max feature pairs to score (default: all)")
    parser.add_argument("--n-train",      type=int, default=50,
                        help="Explanation training examples (default: 50)")
    parser.add_argument("--n-fuzz",       type=int, default=25,
                        help="Fuzzing test examples (default: 25)")
    parser.add_argument("--n-detect-act", type=int, default=25,
                        help="Activating examples for detection (default: 25)")
    parser.add_argument("--n-detect-rand",type=int, default=25,
                        help="Random examples for detection (default: 25)")
    parser.add_argument("--n-boundary",  type=int, default=0,
                        help="Boundary examples in cache (must match collect_activations.py "
                             "--n-boundary; used in explanation prompt, default: 0)")
    parser.add_argument("--fuzzing-only", action="store_true",
                        help="Skip detection scoring entirely. Only run explanation + fuzzing.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--context-tokenizer", default="gpt2-large",
                        help="Tokenizer used to decode cached token IDs for display. "
                             "Must match the SAE's base model, NOT the explainer LLM "
                             "(default: gpt2-large)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    global _CONTEXT_TOKENIZER_NAME
    _CONTEXT_TOKENIZER_NAME = args.context_tokenizer

    cache_dir  = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading caches...")
    joint_features, random_tokens = load_cache(cache_dir, "joint")
    solo_features,  _             = load_cache(cache_dir, "solo")

    K = joint_features[0].tokens.shape[0]
    contexts_needed = args.n_train + args.n_fuzz + (0 if args.fuzzing_only else args.n_detect_act)
    if contexts_needed > K:
        parser.error(
            f"n_train + n_fuzz + n_detect_act = {contexts_needed} "
            f"exceeds K_total = {K} (contexts per feature in cache)"
        )

    # In targeted mode joint/solo lists may have different lengths — process independently
    n_joint = min(args.n_features, len(joint_features))
    n_solo  = min(args.n_features, len(solo_features))
    joint_features = joint_features[:n_joint]
    solo_features  = solo_features[:n_solo]
    all_features   = joint_features + solo_features
    print(f"  {n_joint} joint + {n_solo} solo = {len(all_features)} features total")

    print(f"\nLoading {args.model_name} via vLLM...")
    from vllm import LLM
    llm_kwargs = dict(
        model=args.model_name,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
    )
    if args.quantization:
        llm_kwargs["quantization"] = args.quantization
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    print("  Model loaded.")

    results = run_pipeline(args, llm, tokenizer, all_features, random_tokens)
    summarise_and_save(results, output_dir, args.model_name)

    import gc, os
    del llm
    gc.collect()
    os._exit(0)


if __name__ == "__main__":
    main()
