#!/usr/bin/env python3
"""
Train a primary SAE with a meta-SAE decomposability penalty.

Supports three modes:
  --train_joint_saes       — joint training: primary + meta SAE together
  --train_sequential_saes  — solo primary, then meta SAE on frozen primary

Example::

    python scripts/train.py \\
        --model_name gpt2-large \\
        --layer 20 \\
        --dict_size 20480 \\
        --meta_dict_size 1024 \\
        --primary_top_k 64 \\
        --meta_top_k 4 \\
        --lambda2 0.3 \\
        --sigma_sq 1.0 \\
        --num_tokens 200000000 \\
        --train_joint_saes \\
        --joint_primary_path outputs/run1/joint_primary_sae.pt \\
        --joint_meta_path    outputs/run1/joint_meta_sae.pt \\
        --solo_primary_path  outputs/run1/solo_primary_sae.pt
"""

from __future__ import annotations

import argparse
import json
import gc
import sys
from pathlib import Path
from typing import Dict, Any

import torch

from meta_sae.config import get_default_cfg, post_init_cfg
from meta_sae.activation_store import ActivationsStore
from meta_sae.training_logger import TrainingLogger
from meta_sae.io import save_sae
from meta_sae.extension import (
    MetaSAEWrapper,
    SAEWithPenalty,
    get_sae_class,
    train_sae_with_meta,
    train_primary_sae_solo,
    train_meta_sae_on_frozen_primary,
)
from meta_sae.utils import (
    build_common_arg_parser,
    parse_args_common,
    load_model_and_set_sizes,
    print_gpu_memory_usage,
    print_configs,
)


def parse_args() -> argparse.Namespace:
    parser = build_common_arg_parser("Train a primary SAE with a meta-SAE penalty.")

    # Additional training-only args
    parser.add_argument(
        "--wandb_project", type=str, default=None,
        help="W&B project name (omit to skip W&B logging)",
    )
    parser.add_argument(
        "--checkpoint_freq", type=int, default=10_000,
        help="Save mid-run checkpoints every N primary steps (requires --joint_primary_path dir)",
    )

    filtered_argv = [arg for arg in sys.argv if not arg.startswith("--f=")]
    is_interactive = len(filtered_argv) == 1 or any("ipykernel" in a for a in sys.argv)
    args = parse_args_common(parser)

    if is_interactive:
        args.train_joint_saes      = True
        args.train_sequential_saes = True

    return args


def build_configs(args: argparse.Namespace):
    cfg = get_default_cfg()
    cfg.update(vars(args))
    post_init_cfg(cfg)
    cfg["top_k"]                  = args.primary_top_k
    cfg["sae_type"]               = args.primary_sae_type
    cfg["bandwidth"]              = args.bandwidth
    cfg["num_batches_in_buffer"]  = args.num_batches_in_buffer_joint

    meta_cfg              = cfg.copy()
    meta_cfg["dict_size"] = args.meta_dict_size
    meta_cfg["top_k"]     = args.meta_top_k
    meta_cfg["sae_type"]  = args.meta_sae_type
    meta_cfg["bandwidth"] = args.bandwidth

    penalty_cfg = {
        "lambda2":          args.lambda2,
        "sigma_sq":         args.sigma_sq,
        "n_primary_steps":  args.n_primary_steps,
        "n_meta_steps":     args.n_meta_steps,
        "start_with_primary": True,
    }
    return cfg, meta_cfg, penalty_cfg


def main():
    if torch.cuda.is_available():
        print(f"CUDA available — GPU: {torch.cuda.get_device_name(0)}, "
              f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("CUDA not available — training on CPU (slow).")

    args               = parse_args()
    cfg, meta_cfg, penalty_cfg = build_configs(args)
    model              = load_model_and_set_sizes(cfg)
    meta_cfg["act_size"]   = cfg["act_size"]
    meta_cfg["model_name"] = cfg["model_name"]
    meta_cfg["hook_point"] = cfg["hook_point"]
    meta_cfg["layer"]      = cfg["layer"]
    meta_cfg["site"]       = cfg["site"]

    print_configs(cfg, meta_cfg, penalty_cfg)
    print_gpu_memory_usage()

    output_dir = Path(args.joint_primary_path).parent
    logger     = TrainingLogger(output_dir=output_dir, run_name="training", save_freq=100)
    logger.set_config(cfg, meta_cfg, penalty_cfg)
    print(f"Training metrics → {output_dir}")

    all_metrics = {
        "joint": None,
        "solo":  None,
        "sequential_meta": None,
        "config": {
            "lambda2":          penalty_cfg["lambda2"],
            "sigma_sq":         penalty_cfg["sigma_sq"],
            "dict_size":        cfg["dict_size"],
            "meta_dict_size":   meta_cfg["dict_size"],
            "primary_sae_type": cfg["sae_type"],
            "meta_sae_type":    meta_cfg["sae_type"],
            "primary_top_k":    cfg["top_k"],
            "meta_top_k":       meta_cfg["top_k"],
            "num_tokens":       cfg["num_tokens"],
        },
    }

    # ── Joint training ────────────────────────────────────────────────────────
    if args.train_joint_saes:
        cfg["num_batches_in_buffer"] = args.num_batches_in_buffer_joint
        activation_store = ActivationsStore(model, cfg)

        primary_sae_cls = get_sae_class(cfg["sae_type"])
        meta_sae_cls    = get_sae_class(meta_cfg["sae_type"])
        meta_sae        = MetaSAEWrapper(meta_sae_cls, meta_cfg)
        primary_sae     = SAEWithPenalty(primary_sae_cls, cfg, meta_sae, penalty_cfg)

        print("\n" + "=" * 60)
        print("PHASE 1: JOINT TRAINING")
        print("=" * 60)
        joint_metrics = train_sae_with_meta(
            primary_sae, meta_sae, activation_store, model,
            cfg, meta_cfg, penalty_cfg,
            logger=logger,
            wandb_project=args.wandb_project,
            checkpoint_dir=output_dir,
            checkpoint_freq=args.checkpoint_freq,
        )
        all_metrics["joint"] = joint_metrics

        # Cleanup
        del activation_store
        for p in list(primary_sae.parameters()) + list(meta_sae.parameters()):
            p.grad = None
        gc.collect(); torch.cuda.empty_cache()

        print("Saving jointly trained models...")
        save_sae(primary_sae, args.joint_primary_path,
                 extra={"meta_cfg": meta_cfg, "penalty_cfg": penalty_cfg})
        save_sae(meta_sae, args.joint_meta_path, extra={"meta_cfg": meta_cfg})

    # ── Solo + sequential training ────────────────────────────────────────────
    if args.train_sequential_saes:
        cfg["num_batches_in_buffer"] = args.num_batches_in_buffer_sequential

        print("\n" + "=" * 60)
        print("PHASE 2: SOLO PRIMARY SAE TRAINING")
        print("=" * 60)
        solo_activation_store = ActivationsStore(model, cfg)
        solo_primary_sae_cls  = get_sae_class(cfg["sae_type"])
        solo_primary_sae      = solo_primary_sae_cls(cfg)

        solo_metrics          = train_primary_sae_solo(solo_primary_sae, solo_activation_store, cfg, logger=logger)
        all_metrics["solo"]   = solo_metrics

        del solo_activation_store
        for p in solo_primary_sae.parameters():
            p.grad = None
        gc.collect(); torch.cuda.empty_cache()

        print("Saving solo primary SAE...")
        save_sae(solo_primary_sae, args.solo_primary_path)

        print("\n" + "=" * 60)
        print("PHASE 3: SEQUENTIAL META SAE TRAINING")
        print("=" * 60)
        sequential_meta_sae_cls = get_sae_class(meta_cfg["sae_type"])
        sequential_meta_sae     = MetaSAEWrapper(sequential_meta_sae_cls, meta_cfg)

        seq_meta_metrics              = train_meta_sae_on_frozen_primary(
            sequential_meta_sae, solo_primary_sae, meta_cfg, penalty_cfg, logger=logger
        )
        all_metrics["sequential_meta"] = seq_meta_metrics

        for p in sequential_meta_sae.parameters():
            p.grad = None
        gc.collect(); torch.cuda.empty_cache()

        print("Saving sequential meta SAE...")
        save_sae(sequential_meta_sae, args.sequential_meta_path,
                 extra={"meta_cfg": meta_cfg, "penalty_cfg": penalty_cfg})

    print("\n" + "=" * 60)
    print("ALL TRAINING PHASES COMPLETED")
    print("=" * 60)

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Metrics → {metrics_path}")

    logger.save()
    logger.print_summary()


if __name__ == "__main__":
    main()
