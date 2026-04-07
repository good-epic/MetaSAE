#!/usr/bin/env python3
"""
Calibrate per-feature JumpReLU thresholds from a trained SAE checkpoint.

Thin CLI wrapper around ``meta_sae.extension.calibrate_checkpoint``.

Uses an EER (equal-error-rate) histogram approach: streams tokens through the
SAE encoder, accumulates per-feature active/inactive pre-activation histograms,
and finds the threshold that minimises FP+FN relative to BatchTopK selection.

Usage::

    python scripts/calibrate.py \\
        --checkpoint outputs/run1/joint_primary_sae.pt \\
        --output-dir outputs/run1/ \\
        --label      joint \\
        --n-tokens   5000000 \\
        --device     cuda

Outputs:
    {output_dir}/thresholds_{label}.npy   shape (dict_size,) float32
"""

import argparse
import os

from meta_sae.extension import calibrate_checkpoint


def parse_args():
    p = argparse.ArgumentParser(
        description="Calibrate JumpReLU thresholds for a trained SAE."
    )
    p.add_argument("--checkpoint",            required=True,
                   help="Path to SAE .pt checkpoint")
    p.add_argument("--output-dir",            required=True,
                   help="Directory to write thresholds_{label}.npy")
    p.add_argument("--label",                 default="sae",
                   help="Label for the output filename (default: sae)")
    p.add_argument("--n-tokens",              type=int, default=5_000_000)
    p.add_argument("--batch-size",            type=int, default=2048)
    p.add_argument("--model-batch-size",      type=int, default=64)
    p.add_argument("--num-batches-in-buffer", type=int, default=4)
    p.add_argument("--max-val",               type=float, default=10.0,
                   help="Pre-activation ceiling; dead features get this value. "
                        "Must match --max-val used in collect_activations.py.")
    p.add_argument("--n-bins",                type=int, default=200)
    p.add_argument("--min-active-samples",    type=int, default=100)
    p.add_argument("--undercalibrated-midpoint", action="store_true",
                   help="Use midpoint of overlap region for undercalibrated features "
                        "instead of the conservative max. Reduces over-filtering when "
                        "calibrating on few tokens.")
    p.add_argument("--device",                default="cuda")
    return p.parse_args()


def run(args):
    calibrate_checkpoint(
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        label=args.label,
        n_tokens=args.n_tokens,
        batch_size=args.batch_size,
        model_batch_size=args.model_batch_size,
        num_batches_in_buffer=args.num_batches_in_buffer,
        max_val=args.max_val,
        n_bins=args.n_bins,
        min_active_samples=args.min_active_samples,
        undercalibrated_midpoint=args.undercalibrated_midpoint,
        device=args.device,
    )


def main():
    run(parse_args())
    os._exit(0)


if __name__ == "__main__":
    main()
