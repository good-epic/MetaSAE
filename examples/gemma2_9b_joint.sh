#!/usr/bin/env bash
# Reproduce the best Gemma 2 9B joint MetaSAE configuration from the paper.
# λ₂ = 0.1, σ² = 1.0, layer 31, dict_size = 65536, top_k = 128
# Requires an A100 80GB (or equivalent) — Gemma 2 9B in bfloat16 ~18 GB.
set -euo pipefail

OUTPUT_DIR="outputs/gemma2_9b_best"

python scripts/train.py \
    --model_name       google/gemma-2-9b \
    --dataset_path     HuggingFaceFW/fineweb \
    --dataset_name     sample-10BT \
    --layer            31 \
    --site             resid_pre \
    --model_dtype      bfloat16 \
    --dict_size        65536 \
    --meta_dict_size   5000 \
    --primary_top_k    128 \
    --meta_top_k       8 \
    --lambda2          0.1 \
    --sigma_sq         1.0 \
    --num_tokens       200000000 \
    --batch_size       4096 \
    --lr               3e-4 \
    --n_primary_steps  100 \
    --n_meta_steps     10 \
    --model_batch_size 32 \
    --num_batches_in_buffer_joint      3 \
    --num_batches_in_buffer_sequential 2 \
    --train_joint_saes \
    --train_sequential_saes \
    --joint_primary_path   "$OUTPUT_DIR/joint_primary_sae.pt" \
    --joint_meta_path      "$OUTPUT_DIR/joint_meta_sae.pt" \
    --solo_primary_path    "$OUTPUT_DIR/solo_primary_sae.pt" \
    --sequential_meta_path "$OUTPUT_DIR/sequential_meta_sae.pt"

# Calibrate thresholds for inference
python scripts/calibrate.py \
    --checkpoint   "$OUTPUT_DIR/joint_primary_sae.pt" \
    --output-dir   "$OUTPUT_DIR" \
    --label        joint \
    --n-tokens     5000000 \
    --model-batch-size 32 \
    --device       cuda

python scripts/calibrate.py \
    --checkpoint   "$OUTPUT_DIR/solo_primary_sae.pt" \
    --output-dir   "$OUTPUT_DIR" \
    --label        solo \
    --n-tokens     5000000 \
    --model-batch-size 32 \
    --device       cuda

echo "Training and calibration complete. Checkpoints in $OUTPUT_DIR/"
