#!/usr/bin/env bash
# Reproduce the best GPT-2L joint MetaSAE configuration from the paper.
# λ₂ = 0.3, σ² = 1.0, layer 20, dict_size = 20480, top_k = 64
# Requires: ~80 GB VRAM for 200M tokens, or reduce --num_tokens for a quick test.
set -euo pipefail

OUTPUT_DIR="outputs/gpt2l_best"

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
    --batch_size       4096 \
    --lr               3e-4 \
    --n_primary_steps  100 \
    --n_meta_steps     10 \
    --model_batch_size 64 \
    --num_batches_in_buffer_joint      5 \
    --num_batches_in_buffer_sequential 3 \
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
    --model-batch-size 64 \
    --device       cuda

python scripts/calibrate.py \
    --checkpoint   "$OUTPUT_DIR/solo_primary_sae.pt" \
    --output-dir   "$OUTPUT_DIR" \
    --label        solo \
    --n-tokens     5000000 \
    --model-batch-size 64 \
    --device       cuda

echo "Training and calibration complete. Checkpoints in $OUTPUT_DIR/"
