#!/usr/bin/env python3
"""
Collect JumpReLU activation contexts for auto-interp scoring.

Thin CLI wrapper around ``meta_sae.collect.collect_activations``.
See that module for full documentation.

Usage::

    # Matching mode
    python eval/collect_activations.py \\
        --joint-checkpoint outputs/run1/joint_primary_sae.pt \\
        --solo-checkpoint  outputs/run1/solo_primary_sae.pt \\
        --thresholds-joint outputs/run1/thresholds_joint.npy \\
        --thresholds-solo  outputs/run1/thresholds_solo.npy \\
        --output-dir       outputs/run1/delphi_cache/ \\
        --n-tokens         5000000 \\
        --device           cuda

    # Targeted mode (e.g. random feature samples for grid-fuzz)
    python eval/collect_activations.py \\
        --joint-checkpoint ... --solo-checkpoint ... \\
        --thresholds-joint ... --thresholds-solo ... \\
        --targets-joint    outputs/run1/targets_joint.json \\
        --targets-solo     outputs/run1/targets_solo.json \\
        --n-boundary 5 \\
        --output-dir       outputs/run1/delphi_cache/ \\
        --device           cuda
"""

import argparse
import os

from meta_sae.collect import collect_activations


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect JumpReLU activation contexts for auto-interp scoring."
    )
    parser.add_argument("--joint-checkpoint",      required=True)
    parser.add_argument("--solo-checkpoint",       required=True)
    parser.add_argument("--thresholds-joint",      required=True)
    parser.add_argument("--thresholds-solo",       required=True)
    parser.add_argument("--output-dir",            required=True)
    parser.add_argument("--matching-path",         default=None)
    parser.add_argument("--n-features",            type=int, default=20480)
    parser.add_argument("--targets-joint",         default=None)
    parser.add_argument("--targets-solo",          default=None)
    parser.add_argument("--n-tokens",              type=int, default=5_000_000)
    parser.add_argument("--n-train-top",           type=int, default=30)
    parser.add_argument("--n-train-per-lower",     type=int, default=5)
    parser.add_argument("--n-fuzz-per-quintile",   type=int, default=5)
    parser.add_argument("--n-detect",              type=int, default=25)
    parser.add_argument("--n-boundary",            type=int, default=0)
    parser.add_argument("--ctx-len",               type=int, default=64)
    parser.add_argument("--n-random",              type=int, default=200)
    parser.add_argument("--min-contexts",          type=int, default=60)
    parser.add_argument("--model-batch-size",      type=int, default=64)
    parser.add_argument("--num-batches-in-buffer", type=int, default=3)
    parser.add_argument("--max-val",               type=float, default=10.0)
    parser.add_argument("--seed",                  type=int, default=42)
    parser.add_argument("--device",               default="cuda")
    return parser.parse_args()


def run(args):
    collect_activations(
        joint_checkpoint=args.joint_checkpoint,
        solo_checkpoint=args.solo_checkpoint,
        thresholds_joint=args.thresholds_joint,
        thresholds_solo=args.thresholds_solo,
        output_dir=args.output_dir,
        targets_joint=args.targets_joint,
        targets_solo=args.targets_solo,
        matching_path=args.matching_path,
        n_features=args.n_features,
        n_tokens=args.n_tokens,
        n_train_top=args.n_train_top,
        n_train_per_lower=args.n_train_per_lower,
        n_fuzz_per_quintile=args.n_fuzz_per_quintile,
        n_detect=args.n_detect,
        n_boundary=args.n_boundary,
        ctx_len=args.ctx_len,
        n_random=args.n_random,
        min_contexts=args.min_contexts,
        model_batch_size=args.model_batch_size,
        num_batches_in_buffer=args.num_batches_in_buffer,
        max_val=args.max_val,
        seed=args.seed,
        device=args.device,
    )


def main():
    run(parse_args())
    os._exit(0)


if __name__ == "__main__":
    main()
