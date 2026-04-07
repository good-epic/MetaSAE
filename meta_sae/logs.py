import wandb
import torch
from functools import partial
import os
import json


def init_wandb(cfg):
    return wandb.init(project=cfg["wandb_project"], name=cfg["name"], config=cfg, reinit=True)


def log_wandb(output, step, wandb_run, index=None):
    metrics_to_log = [
        "loss", "l2_loss", "l1_loss", "l0_norm", "l1_norm", "aux_loss", "num_dead_features"
    ]
    log_dict = {k: output[k].item() for k in metrics_to_log if k in output}
    log_dict["n_dead_in_batch"] = (output["feature_acts"].sum(0) == 0).sum().item()

    if index is not None:
        log_dict = {f"{k}_{index}": v for k, v in log_dict.items()}

    wandb_run.log(log_dict, step=step)


# Hooks for model performance evaluation
def reconstr_hook(activation, hook, sae_out):
    return sae_out


def zero_abl_hook(activation, hook):
    return torch.zeros_like(activation)


def mean_abl_hook(activation, hook):
    return activation.mean([0, 1]).expand_as(activation)


@torch.no_grad()
def log_model_performance(wandb_run, step, model, activations_store, sae, index=None, batch_tokens=None):
    if batch_tokens is None:
        batch_tokens = activations_store.get_batch_tokens()[
            : sae.cfg["batch_size"] // sae.cfg["seq_len"]
        ]
    batch = activations_store.get_activations(batch_tokens).reshape(-1, sae.cfg["act_size"])

    sae_output = sae(batch)["sae_out"].reshape(
        batch_tokens.shape[0], batch_tokens.shape[1], -1
    )

    original_loss = model(batch_tokens, return_type="loss").item()
    reconstr_loss = model.run_with_hooks(
        batch_tokens,
        fwd_hooks=[(sae.cfg["hook_point"], partial(reconstr_hook, sae_out=sae_output))],
        return_type="loss",
    ).item()
    zero_loss = model.run_with_hooks(
        batch_tokens,
        fwd_hooks=[(sae.cfg["hook_point"], zero_abl_hook)],
        return_type="loss",
    ).item()
    mean_loss = model.run_with_hooks(
        batch_tokens,
        fwd_hooks=[(sae.cfg["hook_point"], mean_abl_hook)],
        return_type="loss",
    ).item()

    zero_degradation = original_loss - zero_loss
    mean_degradation = original_loss - mean_loss

    log_dict = {
        "performance/ce_degradation":      original_loss - reconstr_loss,
        "performance/recovery_from_zero":  (reconstr_loss - zero_loss) / zero_degradation,
        "performance/recovery_from_mean":  (reconstr_loss - mean_loss) / mean_degradation,
    }

    if index is not None:
        log_dict = {f"{k}_{index}": v for k, v in log_dict.items()}

    wandb_run.log(log_dict, step=step)
