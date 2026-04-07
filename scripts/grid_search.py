#!/usr/bin/env python3
"""
Grid search over MetaSAE hyperparameters (lambda2 × sigma_sq).

Launches one subprocess per configuration, managing concurrency to
fill available GPUs.  Each run calls ``scripts/train.py`` with a
unique output directory.

Usage::

    python scripts/grid_search.py \\
        --lambda2 0.1 0.3 1.0 \\
        --sigma_sq 1.0 3.0 \\
        --model_name gpt2-large \\
        --layer 20 \\
        --dict_size 20480 \\
        --meta_dict_size 1800 \\
        --primary_top_k 64 \\
        --meta_top_k 4 \\
        --num_tokens 200000000 \\
        --num_workers 6 \\
        --output_dir outputs/grid_search_$(date +%Y%m%d_%H%M%S) \\
        --train_joint_saes

    # Dry run (print commands without executing)
    python scripts/grid_search.py ... --dry_run

    # Resume incomplete runs
    python scripts/grid_search.py ... --resume
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grid search over MetaSAE hyperparameters.")

    # Grid axes
    p.add_argument("--lambda2",    type=float, nargs="+", default=[0.1, 0.3, 1.0])
    p.add_argument("--sigma_sq",   type=float, nargs="+", default=[1.0, 3.0])
    p.add_argument("--meta_dict_size", type=int, nargs="+", default=[1800])

    # Passed through to train.py
    p.add_argument("--model_name",       default="gpt2-large")
    p.add_argument("--dataset_path",     default="HuggingFaceFW/fineweb")
    p.add_argument("--dataset_name",     default="sample-10BT")
    p.add_argument("--layer",            type=int, default=20)
    p.add_argument("--site",             default="resid_pre")
    p.add_argument("--dict_size",        type=int, default=20480)
    p.add_argument("--primary_top_k",    type=int, default=64)
    p.add_argument("--meta_top_k",       type=int, default=4)
    p.add_argument("--num_tokens",       type=int, default=200_000_000)
    p.add_argument("--batch_size",       type=int, default=4096)
    p.add_argument("--lr",               type=float, default=3e-4)
    p.add_argument("--n_primary_steps",  type=int, default=100)
    p.add_argument("--n_meta_steps",     type=int, default=10)
    p.add_argument("--model_batch_size", type=int, default=64)
    p.add_argument("--model_dtype",      default="float32", choices=["float32", "bfloat16"])
    p.add_argument("--primary_sae_type", default="batchtopk")
    p.add_argument("--meta_sae_type",    default="batchtopk")
    p.add_argument("--num_batches_in_buffer_joint",      type=int, default=5)
    p.add_argument("--num_batches_in_buffer_sequential", type=int, default=3)

    # Phase toggles
    p.add_argument("--train_joint_saes",      action="store_true", default=False)
    p.add_argument("--train_sequential_saes", action="store_true", default=False)

    # Grid search control
    p.add_argument("--output_dir",   required=True, help="Base output directory")
    p.add_argument("--num_workers",  type=int, default=1,
                   help="Max concurrent subprocesses")
    p.add_argument("--gpu_ids",      type=str, default=None,
                   help="Comma-separated GPU IDs to cycle over (e.g. '0,1,2')")
    p.add_argument("--dry_run",      action="store_true",
                   help="Print commands without executing")
    p.add_argument("--resume",       action="store_true",
                   help="Skip runs whose output dir already exists")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Run helpers
# ---------------------------------------------------------------------------

def run_id(lambda2: float, sigma_sq: float, meta_dict_size: int, primary_sae_type: str, meta_sae_type: str) -> str:
    key = f"l2_{lambda2}_s2_{sigma_sq}_md_{meta_dict_size}_pt_{primary_sae_type[:3]}_mt_{meta_sae_type[:3]}"
    suffix = hashlib.md5(key.encode()).hexdigest()[:6]
    return f"{key}_{suffix}"


def build_command(args, lambda2: float, sigma_sq: float, meta_dict_size: int, run_dir: Path, gpu_id: Optional[int]) -> list:
    env_prefix = []
    if gpu_id is not None:
        env_prefix = [f"CUDA_VISIBLE_DEVICES={gpu_id}"]

    cmd = [
        sys.executable, str(Path(__file__).parent / "train.py"),
        "--model_name",       args.model_name,
        "--dataset_path",     args.dataset_path,
        "--dataset_name",     args.dataset_name,
        "--layer",            str(args.layer),
        "--site",             args.site,
        "--dict_size",        str(args.dict_size),
        "--meta_dict_size",   str(meta_dict_size),
        "--primary_top_k",    str(args.primary_top_k),
        "--meta_top_k",       str(args.meta_top_k),
        "--num_tokens",       str(args.num_tokens),
        "--batch_size",       str(args.batch_size),
        "--lr",               str(args.lr),
        "--lambda2",          str(lambda2),
        "--sigma_sq",         str(sigma_sq),
        "--n_primary_steps",  str(args.n_primary_steps),
        "--n_meta_steps",     str(args.n_meta_steps),
        "--model_batch_size", str(args.model_batch_size),
        "--model_dtype",      args.model_dtype,
        "--primary_sae_type", args.primary_sae_type,
        "--meta_sae_type",    args.meta_sae_type,
        "--num_batches_in_buffer_joint",      str(args.num_batches_in_buffer_joint),
        "--num_batches_in_buffer_sequential", str(args.num_batches_in_buffer_sequential),
        "--joint_primary_path",   str(run_dir / "joint_primary_sae.pt"),
        "--joint_meta_path",      str(run_dir / "joint_meta_sae.pt"),
        "--solo_primary_path",    str(run_dir / "solo_primary_sae.pt"),
        "--sequential_meta_path", str(run_dir / "sequential_meta_sae.pt"),
    ]

    if args.train_joint_saes:
        cmd.append("--train_joint_saes")
    if args.train_sequential_saes:
        cmd.append("--train_sequential_saes")

    if env_prefix:
        cmd = ["/usr/bin/env"] + env_prefix + cmd

    return cmd


def _run_one(cmd: list, run_dir: Path, log_path: Path) -> int:
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as log_f:
        proc = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT)
    return proc.returncode


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    base_dir = Path(args.output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    gpu_ids: Optional[List[int]] = None
    if args.gpu_ids is not None:
        gpu_ids = [int(g.strip()) for g in args.gpu_ids.split(",")]

    # Build all configs
    configs = list(itertools.product(args.lambda2, args.sigma_sq, args.meta_dict_size))
    print(f"Grid: {len(configs)} configurations")
    print(f"  lambda2      = {args.lambda2}")
    print(f"  sigma_sq     = {args.sigma_sq}")
    print(f"  meta_dict_size = {args.meta_dict_size}")

    tasks = []
    for i, (l2, s2, md) in enumerate(configs):
        rid     = run_id(l2, s2, md, args.primary_sae_type, args.meta_sae_type)
        run_dir = base_dir / rid
        log_path = run_dir / "train.log"

        if args.resume and (run_dir / "joint_primary_sae.pt").exists():
            print(f"  [skip] {rid} (already complete)")
            continue

        gpu_id = gpu_ids[i % len(gpu_ids)] if gpu_ids else None
        cmd    = build_command(args, l2, s2, md, run_dir, gpu_id)
        tasks.append((rid, run_dir, log_path, cmd))

    if not tasks:
        print("No runs to execute.")
        return

    print(f"\n{len(tasks)} runs to execute (workers={args.num_workers})")

    for rid, run_dir, log_path, cmd in tasks:
        print(f"  {rid}")
        if args.dry_run:
            print("    " + " ".join(cmd))

    if args.dry_run:
        print("\nDry run — no processes launched.")
        return

    # Save grid config
    grid_meta = {
        "launched": datetime.now().isoformat(),
        "n_configs": len(tasks),
        "lambda2":   args.lambda2,
        "sigma_sq":  args.sigma_sq,
        "meta_dict_size": args.meta_dict_size,
    }
    with open(base_dir / "grid_meta.json", "w") as f:
        json.dump(grid_meta, f, indent=2)

    completed = 0
    failed    = []

    with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
        futures = {
            pool.submit(_run_one, cmd, run_dir, log_path): rid
            for (rid, run_dir, log_path, cmd) in tasks
        }
        for future in as_completed(futures):
            rid  = futures[future]
            code = future.result()
            completed += 1
            status = "OK" if code == 0 else f"FAILED (rc={code})"
            print(f"  [{completed}/{len(tasks)}] {rid}: {status}")
            if code != 0:
                failed.append(rid)

    print(f"\nGrid search complete: {len(tasks) - len(failed)}/{len(tasks)} succeeded.")
    if failed:
        print(f"Failed runs: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
