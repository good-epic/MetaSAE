# MetaSAE

Code and paper for **"Meta-SAEs: Encouraging Atomic Features in Sparse Autoencoders"**.

MetaSAEs jointly train a primary SAE and a smaller *meta SAE*.  The meta SAE is trained on the
decoder columns of the primary, and its reconstruction quality is used as a penalty: the primary
SAE is discouraged from learning features that are easily decomposed into combinations of the meta
SAE's features.  This pushes the primary toward more *atomic* (less polysemantic) representations.

---

## Table of contents

1. [Installation](#installation)
2. [Quick start — training](#quick-start--training)
3. [Custom datasets](#custom-datasets)
4. [Training at multiple hook points](#training-at-multiple-hook-points)
5. [Monitoring training convergence](#monitoring-training-convergence)
6. [Hyperparameter search and retraining](#hyperparameter-search-and-retraining)
7. [Assessing checkpoints](#assessing-checkpoints)
8. [Threshold calibration](#threshold-calibration)
9. [Qualitative feature exploration](#qualitative-feature-exploration)
10. [Auto-interp fuzzing](#auto-interp-fuzzing)
11. [Phi matrix comparison](#phi-matrix-comparison)
12. [Inference](#inference)
13. [Activation steering](#activation-steering)
14. [Repository layout](#repository-layout)
15. [Key hyperparameters](#key-hyperparameters)
16. [Citation](#citation)

---

## Installation

```bash
git clone https://github.com/your-org/meta-sae
cd meta-sae
pip install -e ".[wandb]"          # include wandb for training logging
```

Requirements: Python ≥ 3.10, PyTorch ≥ 2.0, `transformer_lens`, `datasets`, `safetensors`,
`scipy`.  For phi matrix computation, also install `lapjv`:

```bash
pip install lapjv
```

For auto-interp fuzzing, vLLM must run in a separate virtual environment to avoid numpy version
conflicts with TransformerLens:

```bash
python3 -m venv ~/.vllm_venv
~/.vllm_venv/bin/pip install vllm scipy safetensors datasets
```

---

## Quick start — training

The main training script trains a joint MetaSAE + a standard solo BatchTopK SAE on the same
data in a single run.

```bash
python scripts/train.py \
    --model_name       gpt2-large \
    --dataset_path     HuggingFaceFW/fineweb \
    --dataset_name     sample-10BT \
    --layer            20 \
    --site             resid_pre \
    --dict_size        20480 \
    --meta_dict_size   1800 \
    --primary_top_k    64 \
    --meta_top_k       4 \
    --lambda2          0.3 \
    --sigma_sq         1.0 \
    --num_tokens       200000000 \
    --train_joint_saes \
    --train_sequential_saes \
    --joint_primary_path   outputs/run1/joint_primary_sae.pt \
    --joint_meta_path      outputs/run1/joint_meta_sae.pt \
    --solo_primary_path    outputs/run1/solo_primary_sae.pt \
    --sequential_meta_path outputs/run1/sequential_meta_sae.pt \
    --wandb_project        my-project
```

`--train_joint_saes` trains the joint MetaSAE.  `--train_sequential_saes` trains the
standard solo SAE (your control baseline) on identical data.  Both are always recommended.

See `examples/gpt2l_joint.sh` and `examples/gemma2_9b_joint.sh` for full reproduction
commands for the paper's best configs.

---

## Custom datasets

`ActivationsStore` accepts several data sources.

### HuggingFace Hub (default)

```bash
python scripts/train.py --dataset_path HuggingFaceFW/fineweb --dataset_name sample-10BT ...
```

### Local files

Pass a local path to `--dataset_path`:

| File type | Expected format |
|---|---|
| Directory saved with `dataset.save_to_disk()` | Arrow shards, any column schema |
| `.jsonl` / `.json` | One JSON object per line with a `"text"` or `"tokens"` key |
| `.txt` | One document per line (yielding `{"text": "..."}`) |

```bash
# JSONL example: each line is {"text": "..."}
python scripts/train.py --dataset_path /data/my_corpus.jsonl ...

# Pre-tokenized: each line is {"tokens": [101, 2054, ...]}
python scripts/train.py --dataset_path /data/pretokenized.jsonl ...

# HuggingFace Arrow directory
python scripts/train.py --dataset_path /data/my_arrow_dataset/ ...
```

### Python API — arbitrary iterables

```python
import datasets
from meta_sae import ActivationsStore

# From a list of dicts
rows  = [{"text": doc} for doc in my_documents]
hf_ds = datasets.Dataset.from_list(rows)
store = ActivationsStore.from_dataset(model, cfg, hf_ds)

# From a pre-tokenized tensor
rows  = [{"tokens": ids.tolist()} for ids in my_token_tensor]  # (N, seq_len)
hf_ds = datasets.Dataset.from_list(rows)
store = ActivationsStore.from_dataset(model, cfg, hf_ds)
```

> **Note:** datasets passed via `from_dataset()` do not support automatic restarts
> at exhaustion.  Make sure your dataset is large enough for `num_tokens`.

---

## Training at multiple hook points

Each `scripts/train.py` invocation trains one hook point.  To train across several
layers, run once per layer with a distinct `--layer` and output directory:

```bash
for LAYER in 12 16 20 24; do
  python scripts/train.py \
      --model_name gpt2-large \
      --layer $LAYER \
      --dict_size 20480 \
      --primary_top_k 64 \
      --lambda2 0.3 --sigma_sq 1.0 \
      --num_tokens 200000000 \
      --train_joint_saes --train_sequential_saes \
      --joint_primary_path   outputs/layer${LAYER}/joint_primary_sae.pt \
      --joint_meta_path      outputs/layer${LAYER}/joint_meta_sae.pt \
      --solo_primary_path    outputs/layer${LAYER}/solo_primary_sae.pt \
      --sequential_meta_path outputs/layer${LAYER}/sequential_meta_sae.pt
done
```

Each run is independent and can be launched on a separate GPU simultaneously.

---

## Monitoring training convergence

### JSON logs (always produced)

Training automatically writes structured JSON logs to the output directory:

```
outputs/run1/
  training_joint_primary_metrics.json   # per-step: loss, L2, L0, dead features, penalty
  training_solo_primary_metrics.json
  training_sequential_meta_metrics.json
  training_summary.json                 # final values + avg-last-100 for each phase
  training_all_metrics.json             # everything in one file
  metrics.json                          # high-level summary across all phases
```

Each step entry looks like:
```json
{
  "step": 1500,
  "phase": "joint_primary",
  "loss": 0.0412,
  "l2_loss": 0.0389,
  "l0_norm": 63.7,
  "num_dead_features": 0,
  "decomp_penalty": 0.0023,
  "gpu_memory_allocated_gb": 18.4
}
```

To plot training curves from the JSON (requires `matplotlib`):

```python
import json, matplotlib.pyplot as plt

with open("outputs/run1/training_joint_primary_metrics.json") as f:
    data = json.load(f)

steps  = [m["step"]   for m in data["metrics"]]
l2     = [m["l2_loss"] for m in data["metrics"]]
l0     = [m["l0_norm"] for m in data["metrics"]]
dead   = [m["num_dead_features"] for m in data["metrics"]]

fig, axes = plt.subplots(1, 3, figsize=(12, 3))
axes[0].plot(steps, l2);   axes[0].set_title("L2 loss")
axes[1].plot(steps, l0);   axes[1].set_title("L0 (sparsity)")
axes[2].plot(steps, dead); axes[2].set_title("Dead features")
plt.tight_layout(); plt.savefig("training_curves.png")
```

### W&B (optional)

Pass `--wandb_project my-project` to `scripts/train.py`.  All per-step metrics are
logged automatically.

### Convergence checks

A well-converged run typically shows:
- L2 loss flattening (no longer decreasing)
- L0 ≈ `primary_top_k` (within ~5%)
- Dead features → 0 or a stable low count
- `decomp_penalty` (joint only): small but non-zero, indicating the penalty is active

---

## Hyperparameter search and retraining

### Grid search over λ₂ × σ²

```bash
python scripts/grid_search.py \
    --lambda2   0.1 0.3 1.0 \
    --sigma_sq  1.0 3.0 \
    --model_name gpt2-large \
    --layer 20 \
    --dict_size 20480 --primary_top_k 64 \
    --meta_dict_size 1800 --meta_top_k 4 \
    --num_tokens 200000000 \
    --num_workers 6 \
    --output_dir outputs/grid_$(date +%Y%m%d_%H%M%S) \
    --train_joint_saes --train_sequential_saes

# Dry-run to preview commands without executing:
python scripts/grid_search.py ... --dry_run

# Resume incomplete runs after interruption:
python scripts/grid_search.py ... --resume
```

This spawns one subprocess per (λ₂, σ²) pair, managed to fill `--num_workers` slots.
Each run writes its checkpoints and logs to a hash-named subdirectory.

### Single retrain with different hyperparameters

Just re-run `scripts/train.py` with a new `--output_dir`.  All hyperparameters
(`--lr`, `--batch_size`, `--primary_top_k`, `--dict_size`, …) are CLI args.

---

## Assessing checkpoints

`scripts/assess.py` computes L0, L2, dead features, CE loss, and fraction-of-
information-destroyed for one or both checkpoints:

```bash
python scripts/assess.py \
    --joint-checkpoint outputs/run1/joint_primary_sae.pt \
    --solo-checkpoint  outputs/run1/solo_primary_sae.pt \
    --output-dir       outputs/run1/assessment/ \
    --num-batches      500 \
    --device           cuda
```

Results are saved to `outputs/run1/assessment/assessment_results.json`:

```json
{
  "joint": {
    "l0": 63.8,
    "l2": 0.0389,
    "dead_features": 0,
    "delta_ce": 0.041,
    "fraction_destroyed": 0.083
  },
  "solo": { ... }
}
```

For a more detailed CE-loss comparison with per-batch distributions and plots:

```bash
python eval/eval_ce_loss.py \
    --joint-checkpoint outputs/run1/joint_primary_sae.pt \
    --solo-checkpoint  outputs/run1/solo_primary_sae.pt \
    --output-dir       outputs/run1/ce_loss/ \
    --n-batches        200
```

---

## Threshold calibration

BatchTopK SAEs use batch-level top-k during training.  At inference time you need
per-feature thresholds (JumpReLU style) so that features fire deterministically
on single examples.  The EER (equal-error-rate) calibration procedure finds the
threshold for each feature that equalises false-positive and false-negative rates.

```bash
python scripts/calibrate.py \
    --checkpoint outputs/run1/joint_primary_sae.pt \
    --output-dir outputs/run1/ \
    --label      joint \
    --n-tokens   5000000 \
    --device     cuda
```

This writes `outputs/run1/thresholds_joint.npy` (float32 array of shape
`(dict_size,)`).  Run once per checkpoint.

Python API — one call (loads SAE + model, calibrates, saves):

```python
from meta_sae import calibrate_checkpoint

calibrate_checkpoint(
    checkpoint  = "outputs/run1/joint_primary_sae.pt",
    output_dir  = "outputs/run1/",
    label       = "joint",
    n_tokens    = 5_000_000,
    device      = "cuda",
)
# writes outputs/run1/thresholds_joint.npy
```

Python API — lower level (bring your own model):

```python
from meta_sae import calibrate_thresholds, load_sae, save_thresholds
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("gpt2-large").cuda()
sae   = load_sae("outputs/run1/joint_primary_sae.pt", device="cuda")

thresholds = calibrate_thresholds(sae, model, n_tokens=5_000_000, device="cuda")
save_thresholds(thresholds, "outputs/run1/thresholds_joint.npy")
```

---

## Qualitative feature exploration

After calibrating thresholds, you can interactively inspect features using the
`InferenceSAE` Python API.

### Find which features fire on a given input

```python
import torch
from transformer_lens import HookedTransformer
from meta_sae import InferenceSAE

model = HookedTransformer.from_pretrained("gpt2-large").cuda()
sae   = InferenceSAE.from_checkpoint(
    "outputs/run1/joint_primary_sae.pt",
    "outputs/run1/thresholds_joint.npy",
    device="cuda",
)

tokens = model.to_tokens("The Eiffel Tower was built in")
with torch.no_grad():
    _, cache = model.run_with_cache(tokens, names_filter=[sae.hook_point],
                                    stop_at_layer=sae.sae.cfg["layer"] + 1)
x = cache[sae.hook_point].reshape(-1, sae.sae.cfg["act_size"])

feature_acts = sae.encode(x)           # (seq_len, dict_size)

# Which features fire on the last token?
last = feature_acts[-1]
top  = torch.topk(last, 10)
print("Top features on last token:")
for val, idx in zip(top.values, top.indices):
    print(f"  feature {idx.item():5d}  activation={val.item():.3f}")
```

### Find which token positions activate a given feature most strongly

```python
# Collect activations over a larger corpus batch
tokens = model.to_tokens(my_texts, padding_side="right")  # (B, T)
with torch.no_grad():
    _, cache = model.run_with_cache(tokens, names_filter=[sae.hook_point],
                                    stop_at_layer=sae.sae.cfg["layer"] + 1)
x = cache[sae.hook_point].reshape(-1, sae.sae.cfg["act_size"])

values, flat_indices = sae.top_activating_examples(feature_idx=42, activations=x, k=20)

B, T = tokens.shape
for v, i in zip(values.tolist(), flat_indices.tolist()):
    b, t = divmod(i, T)
    ctx  = model.to_string(tokens[b, max(0, t-10):t+1])
    print(f"  act={v:.3f}  context: «{ctx}»")
```

### Compare joint vs. solo feature activations

```python
from meta_sae import InferenceSAE

joint_sae = InferenceSAE.from_checkpoint("outputs/run1/joint_primary_sae.pt",
                                          "outputs/run1/thresholds_joint.npy",
                                          device="cuda")
solo_sae  = InferenceSAE.from_checkpoint("outputs/run1/solo_primary_sae.pt",
                                          "outputs/run1/thresholds_solo.npy",
                                          device="cuda")

joint_acts = joint_sae.encode(x)
solo_acts  = solo_sae.encode(x)

# Check L0 (average features active per token)
print(f"Joint L0: {(joint_acts > 0).float().mean(-1).mean():.1f}")
print(f"Solo  L0: {(solo_acts  > 0).float().mean(-1).mean():.1f}")
```

---

## Auto-interp fuzzing

Fuzzing asks an LLM to predict activation strengths on held-out examples.  A higher
Spearman ρ (joint vs. solo) indicates that joint features are more interpretable.

This is a three-step pipeline:

### Step 1: Calibrate thresholds

See [Threshold calibration](#threshold-calibration) above.  Both checkpoints need thresholds:

```bash
python scripts/calibrate.py \
    --checkpoint outputs/run1/joint_primary_sae.pt \
    --output-dir outputs/run1/ --label joint \
    --n-tokens 5000000 --device cuda

python scripts/calibrate.py \
    --checkpoint outputs/run1/solo_primary_sae.pt \
    --output-dir outputs/run1/ --label solo \
    --n-tokens 5000000 --device cuda
```

### Step 2: Collect activation caches

```bash
python eval/collect_activations.py \
    --joint-checkpoint  outputs/run1/joint_primary_sae.pt \
    --solo-checkpoint   outputs/run1/solo_primary_sae.pt \
    --thresholds-joint  outputs/run1/thresholds_joint.npy \
    --thresholds-solo   outputs/run1/thresholds_solo.npy \
    --output-dir        outputs/run1/delphi_cache/ \
    --n-tokens          5000000 \
    --n-boundary        5 \
    --ctx-len           64 \
    --n-random          200 \
    --min-contexts      40 \
    --model-batch-size  64 \
    --device            cuda
```

For targeted evaluation (specific features, e.g. split candidates), pass
`--targets-joint targets_joint.json --targets-solo targets_solo.json` where each JSON
file maps feature indices to tag strings.

Python API:

```python
from meta_sae import collect_activations

collect_activations(
    joint_checkpoint = "outputs/run1/joint_primary_sae.pt",
    solo_checkpoint  = "outputs/run1/solo_primary_sae.pt",
    thresholds_joint = "outputs/run1/thresholds_joint.npy",
    thresholds_solo  = "outputs/run1/thresholds_solo.npy",
    output_dir       = "outputs/run1/delphi_cache/",
    n_tokens         = 5_000_000,
    n_boundary       = 5,
    ctx_len          = 64,
    min_contexts     = 40,
    device           = "cuda",
)
```

### Step 3: Score with an LLM

```bash
~/.vllm_venv/bin/python eval/auto_interp.py \
    --cache-dir              outputs/run1/delphi_cache/ \
    --output-dir             outputs/run1/auto_interp/ \
    --model-name             Qwen/Qwen2.5-72B-Instruct-AWQ \
    --quantization           awq_marlin \
    --gpu-memory-utilization 0.85 \
    --fuzzing-only \
    --context-tokenizer      gpt2-large
```

Results are written to `outputs/run1/auto_interp/scores.json`:

```json
{
  "summary": {
    "joint_fuzzing_mean": 0.312,
    "solo_fuzzing_mean":  0.274
  },
  "features": [...]
}
```

---

## Phi matrix comparison

The phi (φ) matrix measures co-occurrence between joint and solo SAE features,
giving a quantitative picture of how differently the two SAEs decompose the same
activations.

> **Note:** `compute_cooccurrence.py` annotates candidate features with their
> LLM-generated descriptions, so it reads `--scores-json` from a completed
> auto-interp run.  Complete the auto-interp fuzzing pipeline above first.

```bash
# Requires: thresholds for both SAEs + a completed auto_interp/scores.json
python eval/compute_cooccurrence.py \
    --joint-checkpoint  outputs/run1/joint_primary_sae.pt \
    --solo-checkpoint   outputs/run1/solo_primary_sae.pt \
    --joint-thresholds  outputs/run1/thresholds_joint.npy \
    --solo-thresholds   outputs/run1/thresholds_solo.npy \
    --scores-json       outputs/run1/auto_interp/scores.json \
    --output-dir        outputs/run1/phi/ \
    --n-tokens-pass1    20000000 \
    --n-tokens-pass2    2000000 \
    --dataset-path      HuggingFaceFW/fineweb \
    --dataset-name      sample-10BT \
    --device            cuda:0
```

Outputs in `outputs/run1/phi/`:
- `cross_phi_topk.npz` — top-50 φ matches in each direction (joint→solo, solo→joint)
- `splitting_candidates.json` — top solo features by φ_top2 (solo features that
  match multiple joint features — candidates for cases where MetaSAE split a concept)
- `merging_candidates.json` — top joint features by φ_top2 (joint features that
  match multiple solo features)

The key metric across configs is **mean|φ|**: lower means joint features co-occur
less with solo features — more atomic representations.

To skip the slow Pass 1 and reload a previously computed phi array:
```bash
python eval/compute_cooccurrence.py ... --load-topk outputs/run1/phi/cross_phi_topk.npz
```

Python API:

```python
from meta_sae import compute_cooccurrence

compute_cooccurrence(
    joint_checkpoint = "outputs/run1/joint_primary_sae.pt",
    solo_checkpoint  = "outputs/run1/solo_primary_sae.pt",
    joint_thresholds = "outputs/run1/thresholds_joint.npy",
    solo_thresholds  = "outputs/run1/thresholds_solo.npy",
    scores_json      = "outputs/run1/auto_interp/scores.json",
    output_dir       = "outputs/run1/phi/",
    n_tokens_pass1   = 20_000_000,
    n_tokens_pass2   = 2_000_000,
    device           = "cuda:0",
)

# Reload saved phi arrays to skip Pass 1 (e.g. for different candidate filtering):
compute_cooccurrence(
    ...,
    load_topk = "outputs/run1/phi/cross_phi_topk.npz",
)
```

---

## Inference

### Load a calibrated SAE

```python
from meta_sae import InferenceSAE

sae = InferenceSAE.from_checkpoint(
    "outputs/run1/joint_primary_sae.pt",
    "outputs/run1/thresholds_joint.npy",
    device="cuda",
)

feature_acts        = sae.encode(x)          # (batch, dict_size), sparse
x_hat               = sae.decode(feature_acts)
feature_acts, x_hat = sae(x)                 # both at once; x_hat is in original space
```

### Load any saved SAE (without thresholds)

```python
from meta_sae import load_sae

sae = load_sae("outputs/run1/joint_primary_sae.pt", device="cuda")
out = sae(x)   # returns a loss dict with "sae_out", "feature_acts", "loss", …
```

---

## Activation steering

The `InferenceSAE` provides a full steering API: encode activations into the SAE
feature space, modify the latent vector, then decode back into the residual stream
for the model to continue processing.

> **Why this requires special handling:** if the SAE was trained with
> `input_unit_norm=True` (the default for GPT-2L), activations are normalised
> before encoding and the normalisation must be reversed after decoding.
> `InferenceSAE.steer()` and `make_steering_hook()` handle this automatically.
> Using `encode()` → modify → `decode()` directly will produce wrong-scale
> activations for these models.

### One-shot: modify activations for a single tensor

```python
from meta_sae import InferenceSAE
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("gpt2-large").cuda()
sae   = InferenceSAE.from_checkpoint(
    "outputs/run1/joint_primary_sae.pt",
    "outputs/run1/thresholds_joint.npy",
    device="cuda",
)

tokens = model.to_tokens("The Eiffel Tower was built in")
with torch.no_grad():
    _, cache = model.run_with_cache(tokens, names_filter=[sae.hook_point],
                                    stop_at_layer=sae.sae.cfg["layer"] + 1)
x = cache[sae.hook_point].reshape(-1, sae.sae.cfg["act_size"])

def my_edits(acts):
    acts = acts.clone()
    acts[:, 42]   = 0.0    # suppress feature 42
    acts[:, 1337] *= 2.0   # amplify feature 1337
    return acts

x_patched = sae.steer(x, my_edits)   # same shape as x, correct residual-stream scale
```

### In-forward: steer during the full forward pass

`make_steering_hook()` returns a TransformerLens hook so the modification is seen
by all subsequent layers — the standard setting for causal steering experiments.

```python
import torch
from meta_sae import InferenceSAE
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("gpt2-large").cuda()
sae   = InferenceSAE.from_checkpoint(
    "outputs/run1/joint_primary_sae.pt",
    "outputs/run1/thresholds_joint.npy",
    device="cuda",
)

tokens = model.to_tokens("The Eiffel Tower was built in")

def suppress_feature_42(acts):
    acts = acts.clone()
    acts[:, 42] = 0.0
    return acts

hook = sae.make_steering_hook(suppress_feature_42)

with torch.no_grad():
    original_logits = model(tokens)
    steered_logits  = model.run_with_hooks(
        tokens,
        fwd_hooks=[(sae.hook_point, hook)],
    )

# Generate steered text
steered_str = model.run_with_hooks(
    tokens,
    fwd_hooks=[(sae.hook_point, hook)],
    return_type="str",
)
print(steered_str)
```

### Read, inspect, and alter latent vectors interactively

```python
# Encode to get the latent representation
with torch.no_grad():
    _, cache = model.run_with_cache(tokens, names_filter=[sae.hook_point],
                                    stop_at_layer=sae.sae.cfg["layer"] + 1)
x = cache[sae.hook_point].reshape(-1, sae.sae.cfg["act_size"])

# Inspect which features are active
feature_acts, x_hat = sae(x)                     # x_hat is a faithful reconstruction
active_features = feature_acts[-1].nonzero()[:, 0]  # features active on last token
print(f"Active features: {active_features.tolist()}")
print(f"Activation values: {feature_acts[-1, active_features].tolist()}")

# Edit specific features
feature_acts_modified = feature_acts.clone()
feature_acts_modified[:, 99]  = 5.0   # force feature 99 to fire everywhere
feature_acts_modified[:, 42]  = 0.0   # silence feature 42

# Decode back to residual-stream space (correctly un-normalises if needed)
_, x_mean, x_std = sae.encode_with_stats(x)
x_modified = sae.decode_from_stats(feature_acts_modified, x_mean, x_std)

# Patch into a TransformerLens run_with_hooks call
from functools import partial

def patch_hook(value, hook, x_patched):
    return x_patched.reshape(value.shape)

with torch.no_grad():
    output = model.run_with_hooks(
        tokens,
        fwd_hooks=[(sae.hook_point, partial(patch_hook, x_patched=x_modified))],
    )
```

### sae.hook_point

`InferenceSAE.hook_point` is a property that returns the correct TransformerLens
hook name for this SAE's layer (e.g. `"blocks.20.hook_resid_pre"`), computed from
the saved `layer` and `site` fields.  The `hook_point` value stored in the
checkpoint cfg can be stale; always use this property.

```python
print(sae.hook_point)   # "blocks.20.hook_resid_pre"
```

---

## Repository layout

```
meta_sae/                    pip-installable package
  __init__.py                all key exports
  extension.py               MetaSAEWrapper, penalty classes, training loops,
                               calibrate_thresholds(), calibrate_checkpoint(),
                               InferenceSAE (with steering API)
  collect.py                 collect_activations() — context collection for auto-interp
                               (Hungarian matching, stratified sampling, cache I/O)
  cooccurrence.py            compute_cooccurrence() — cross-SAE φ matrix computation
                               (pass 1 N11 accumulation, phi top-k, candidate assembly)
  sae.py                     BatchTopKSAE, TopKSAE, VanillaSAE, JumpReLUSAE
                               + encode() / decode() / encode_decode() inference methods
  io.py                      load_sae(), save_sae(), load_thresholds(), save_thresholds()
  config.py                  get_default_cfg(), post_init_cfg()
  activation_store.py        ActivationsStore — streaming activation buffer,
                               supports HF Hub, local files, and arbitrary datasets
  training_logger.py         structured JSON training logger
  logs.py                    logging utilities
  utils.py                   shared CLI utilities

scripts/                     thin CLI wrappers around meta_sae library functions
  train.py                   joint + solo + sequential training
  calibrate.py               EER threshold calibration → meta_sae.calibrate_checkpoint()
  assess.py                  L0 / L2 / CE loss assessment
  grid_search.py             parallel grid search over λ₂ × σ²

eval/                        thin CLI wrappers for paper reproducibility
  collect_activations.py     context collection → meta_sae.collect_activations()
  compute_cooccurrence.py    φ matrix → meta_sae.compute_cooccurrence()
  eval_ce_loss.py            CE loss impact (detailed, with plots)
  auto_interp.py             LLM fuzzing / detection scoring

examples/
  gpt2l_joint.sh             reproduce GPT-2L best config
  gemma2_9b_joint.sh         reproduce Gemma 2 9B best config
```

---

## Key hyperparameters

| Param          | Meaning                                         | GPT-2L best | Gemma 9B best |
|----------------|-------------------------------------------------|-------------|---------------|
| `lambda2`      | Weight for decomposability penalty              | 0.3         | 0.1           |
| `sigma_sq`     | Penalty bandwidth (σ² in exp(−err/σ²))         | 1.0         | 1.0           |
| `dict_size`    | Primary SAE dictionary size                    | 20480       | 65536         |
| `top_k`        | Primary SAE sparsity (L0)                      | 64          | 128           |
| `meta_dict_size` | Meta SAE dictionary size                     | 1800        | 5000          |
| `meta_top_k`   | Meta SAE sparsity                               | 4           | 8             |

---

## Citation

```bibtex
@inproceedings{metasae2026,
  title     = {Meta-SAEs: Encouraging Atomic Features in Sparse Autoencoders},
  author    = {...},
  booktitle = {Conference on Language Modeling (COLM)},
  year      = {2026},
}
```
