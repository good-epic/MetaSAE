"""
Utility functions shared between training and assessment scripts.
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, Any

import torch


def build_common_arg_parser(description: str = "Common Arg Parser") -> argparse.ArgumentParser:
    """Build a superset argument parser for training and assessment scripts."""
    parser = argparse.ArgumentParser(description=description)

    # Data & model hooks
    parser.add_argument("--model_name", type=str, default="gpt2-small")
    parser.add_argument("--dataset_path", type=str, default="HuggingFaceFW/fineweb")
    parser.add_argument("--dataset_name", type=str, default="sample-10BT")
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--site", type=str, default="resid_pre")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--model_dtype", type=str, default="float32", choices=["float32", "bfloat16"]
    )

    # Model/dataset related optional knobs
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--model_batch_size", type=int, default=256)

    # SAE sizes & sparsity
    parser.add_argument("--dict_size", type=int, default=36864)
    parser.add_argument("--meta_dict_size", type=int, default=1536)
    parser.add_argument("--primary_top_k", type=int, default=32)
    parser.add_argument("--meta_top_k", type=int, default=4)

    # SAE types
    parser.add_argument(
        "--primary_sae_type", type=str, default="batchtopk",
        choices=["batchtopk", "jumprelu", "topk", "vanilla"],
    )
    parser.add_argument(
        "--meta_sae_type", type=str, default="batchtopk",
        choices=["batchtopk", "jumprelu", "topk", "vanilla"],
    )
    parser.add_argument("--bandwidth", type=float, default=0.001)
    parser.add_argument("--jumprelu_init_threshold", type=float, default=None)

    # JumpReLU sparsity control
    parser.add_argument("--l0_coeff", type=float, default=None)
    parser.add_argument("--target_l0", type=int, default=None)
    parser.add_argument("--l0_coeff_start", type=float, default=1e-5)
    parser.add_argument("--l0_coeff_lr", type=float, default=1e-4)
    parser.add_argument("--l0_Kp", type=float, default=0.001)
    parser.add_argument("--l0_Kd", type=float, default=0.01)
    parser.add_argument("--l0_below_target_multiplier", type=float, default=2.0)
    parser.add_argument("--l0_burnin_steps", type=int, default=1000)
    parser.add_argument("--l0_burnin_multiplier", type=float, default=0.5)

    # Training hyperparameters
    parser.add_argument("--num_tokens", type=int, default=100_000_000)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lambda2", type=float, default=1e-3)
    parser.add_argument("--sigma_sq", type=float, default=0.1)
    parser.add_argument("--n_primary_steps", type=int, default=10)
    parser.add_argument("--n_meta_steps", type=int, default=5)

    # Buffering/memory knobs
    parser.add_argument("--num_batches_in_buffer_joint", type=int, default=5)
    parser.add_argument("--num_batches_in_buffer_sequential", type=int, default=3)
    parser.add_argument("--num_batches_in_buffer", type=int, default=3)

    # Assessment knobs
    parser.add_argument("--num_assessment_batches", type=int, default=1000)

    # Phase toggles (training)
    parser.add_argument("--train_joint_saes", action="store_true", default=False)
    parser.add_argument("--train_sequential_saes", action="store_true", default=False)

    # Phase toggles (assessment)
    parser.add_argument("--assess_joint_saes", action="store_true", default=True)
    parser.add_argument("--assess_sequential_saes", action="store_true", default=True)

    # Checkpoint paths
    parser.add_argument("--joint_primary_path", type=str, default="joint_primary_sae.pt")
    parser.add_argument("--joint_meta_path", type=str, default="joint_meta_sae.pt")
    parser.add_argument("--solo_primary_path", type=str, default="solo_primary_sae.pt")
    parser.add_argument("--sequential_meta_path", type=str, default="sequential_meta_sae.pt")

    return parser


def parse_args_common(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """Common argument parsing that handles Jupyter / interactive mode."""
    DEFAULT_DATASET_PATH = "HuggingFaceFW/fineweb"
    DEFAULT_DATASET_NAME = "sample-10BT"

    try:
        filtered_argv = [arg for arg in sys.argv if not arg.startswith("--f=")]
        if len(filtered_argv) == 1 or any("ipykernel" in arg for arg in sys.argv):
            print("Interactive mode detected, using parser defaults")
            args = parser.parse_args([])
        else:
            original_argv = sys.argv
            sys.argv = filtered_argv
            try:
                args = parser.parse_args()
            finally:
                sys.argv = original_argv
    except SystemExit:
        print("Failed to parse command line args, using parser defaults")
        args = parser.parse_args([])

    if getattr(args, "dataset_path", None) is None:
        args.dataset_path = DEFAULT_DATASET_PATH
        args.dataset_name = DEFAULT_DATASET_NAME

    return args


def load_model_and_set_sizes(cfg: Dict[str, Any]) -> Any:
    """Load the transformer model specified by ``cfg['model_name']`` and update act_size."""
    try:
        from transformer_lens import HookedTransformer
    except ImportError as e:
        raise ImportError(
            "transformer_lens is required. Install via 'pip install transformer_lens'."
        ) from e

    model_name      = cfg["model_name"]
    model_dtype_str = cfg.get("model_dtype", "float32")
    model_dtype     = torch.bfloat16 if model_dtype_str == "bfloat16" else torch.float32
    model           = HookedTransformer.from_pretrained(model_name, device=cfg["device"], dtype=model_dtype)

    hook_point = cfg.get("hook_point")
    if hook_point is None:
        from transformer_lens.utils import get_act_name
        hook_point      = get_act_name(cfg["site"], cfg["layer"])
        cfg["hook_point"] = hook_point

    act_size = model.cfg.d_model
    try:
        dummy_tokens = model.tokenizer.encode("Hello world", return_tensors="pt").to(cfg["device"])
        _, cache     = model.run_with_cache(
            dummy_tokens, names_filter=[hook_point], stop_at_layer=cfg["layer"] + 1
        )
        act_size = cache[hook_point].shape[-1]
    except Exception:
        pass
    cfg["act_size"] = act_size
    return model


def print_gpu_memory_usage():
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        allocated     = torch.cuda.memory_allocated() / 1e9
        reserved      = torch.cuda.memory_reserved() / 1e9
        max_allocated = torch.cuda.max_memory_allocated() / 1e9
        print(
            f"GPU Memory — Allocated: {allocated:.1f}GB, "
            f"Reserved: {reserved:.1f}GB, Peak: {max_allocated:.1f}GB"
        )
    else:
        print("CUDA not available — using CPU")


def print_configs(cfg, meta_cfg, penalty_cfg=None):
    """Print all configuration dictionaries."""
    import pprint
    print("=== Primary SAE Config ===")
    pprint.pprint(cfg)
    print("\n=== Meta SAE Config ===")
    pprint.pprint(meta_cfg)
    if penalty_cfg is not None:
        print("\n=== Penalty Config ===")
        pprint.pprint(penalty_cfg)
    print("=" * 40)
