#!/usr/bin/env python3
"""
Compute the cross-SAE phi (φ) matrix between joint and solo primary SAEs.

Thin CLI wrapper around ``meta_sae.cooccurrence.compute_cooccurrence``.
See that module for full documentation.

Usage::

    python eval/compute_cooccurrence.py \\
        --joint-checkpoint outputs/run1/joint_primary_sae.pt \\
        --solo-checkpoint  outputs/run1/solo_primary_sae.pt \\
        --joint-thresholds outputs/run1/thresholds_joint.npy \\
        --solo-thresholds  outputs/run1/thresholds_solo.npy \\
        --scores-json      outputs/run1/auto_interp/scores.json \\
        --output-dir       outputs/run1/phi_results/

Outputs in --output-dir:
    cross_phi_topk.npz         top-50 phi matches each direction
    splitting_candidates.json  top-200 solo features by phi_top2
    merging_candidates.json    top-200 joint features by phi_top2
"""

import argparse
import os

from meta_sae.cooccurrence import compute_cooccurrence


def parse_args():
    p = argparse.ArgumentParser(description="Cross-SAE phi matrix computation")
    p.add_argument("--joint-checkpoint",  required=True)
    p.add_argument("--solo-checkpoint",   required=True)
    p.add_argument("--joint-thresholds",  required=True)
    p.add_argument("--solo-thresholds",   required=True)
    p.add_argument("--scores-json",       required=True)
    p.add_argument("--output-dir",        required=True)
    p.add_argument("--n-tokens-pass1",    type=int, default=20_000_000)
    p.add_argument("--n-tokens-pass2",    type=int, default=2_000_000)
    p.add_argument("--model-batch-size",  type=int, default=32)
    p.add_argument("--seq-len",           type=int, default=128)
    p.add_argument("--top-k-save",        type=int, default=50)
    p.add_argument("--n-candidates",      type=int, default=200)
    p.add_argument("--candidate-max-fr",  type=float, default=0.05)
    p.add_argument("--load-topk",         type=str, default=None)
    p.add_argument("--top-matches",       type=int, default=10)
    p.add_argument("--top-examples",      type=int, default=5)
    p.add_argument("--flush-every",       type=int, default=20)
    p.add_argument("--device",            type=str, default="cuda:0")
    p.add_argument("--dataset-path",      type=str, default="HuggingFaceFW/fineweb")
    p.add_argument("--dataset-name",      type=str, default="sample-10BT")
    p.add_argument("--skip-documents",    type=int, default=5_000_000)
    p.add_argument("--chunk-size",        type=int, default=512)
    return p.parse_args()


def run(args):
    compute_cooccurrence(
        joint_checkpoint=args.joint_checkpoint,
        solo_checkpoint=args.solo_checkpoint,
        joint_thresholds=args.joint_thresholds,
        solo_thresholds=args.solo_thresholds,
        scores_json=args.scores_json,
        output_dir=args.output_dir,
        n_tokens_pass1=args.n_tokens_pass1,
        n_tokens_pass2=args.n_tokens_pass2,
        model_batch_size=args.model_batch_size,
        seq_len=args.seq_len,
        top_k_save=args.top_k_save,
        n_candidates=args.n_candidates,
        candidate_max_fr=args.candidate_max_fr,
        load_topk=args.load_topk,
        top_matches=args.top_matches,
        top_examples=args.top_examples,
        flush_every=args.flush_every,
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        skip_documents=args.skip_documents,
        chunk_size=args.chunk_size,
        device=args.device,
    )


def main():
    run(parse_args())
    os._exit(0)


if __name__ == "__main__":
    main()
