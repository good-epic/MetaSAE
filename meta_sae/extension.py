"""
Extensions for MetaSAE: meta-SAE wrapper, decomposability penalty, training loops,
threshold calibration, and deterministic inference.

Classes:
    MetaSAEWrapper          — thin wrapper for training a meta SAE on decoder vectors
    SAEWithPenalty          — composition wrapper adding decomposability penalty to any SAE
    BatchTopKSAEWithPenalty — subclass of BatchTopKSAE with built-in penalty
    InferenceSAE            — threshold-based deterministic inference wrapper

Functions:
    get_sae_class               — registry lookup by name
    calibrate_thresholds        — EER histogram calibration (package API)
    train_sae_with_meta         — alternating joint training loop
    train_primary_sae_solo      — solo primary SAE training
    train_meta_sae_on_frozen_primary — sequential meta SAE training
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from typing import Callable, Dict, Any, Optional, Tuple
from pathlib import Path
import tqdm
import torch.optim as optim

from meta_sae.sae import BatchTopKSAE, TopKSAE, VanillaSAE, JumpReLUSAE

try:
    from meta_sae.training_logger import TrainingLogger
except ImportError:
    TrainingLogger = None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SAE_REGISTRY = {
    "batchtopk": BatchTopKSAE,
    "topk": TopKSAE,
    "vanilla": VanillaSAE,
    "jumprelu": JumpReLUSAE,
}


def get_sae_class(sae_type: str) -> type:
    """Get the SAE class for a given type name.

    Args:
        sae_type: One of ``'batchtopk'``, ``'topk'``, ``'vanilla'``, ``'jumprelu'``.

    Returns:
        The corresponding SAE class.
    """
    sae_type = sae_type.lower()
    if sae_type not in SAE_REGISTRY:
        raise ValueError(
            f"Unknown SAE type: {sae_type!r}. Valid options: {list(SAE_REGISTRY.keys())}"
        )
    return SAE_REGISTRY[sae_type]


# ---------------------------------------------------------------------------
# MetaSAEWrapper
# ---------------------------------------------------------------------------

class MetaSAEWrapper(nn.Module):
    """Wrap an SAE for training on arbitrary 2-D tensors.

    The meta SAE is trained on the decoder columns of a primary SAE rather
    than on model activations.  ``forward`` accepts a 2-D tensor of shape
    ``(num_vectors, act_size)`` and returns the standard SAE output dict.
    """

    def __init__(self, sae_cls: type, cfg: Dict[str, Any]):
        super().__init__()
        self.meta_sae: nn.Module = sae_cls(cfg)

    def forward(self, vectors: torch.Tensor) -> Dict[str, Any]:
        """Run the meta SAE on a batch of decoder vectors.

        Args:
            vectors: Tensor of shape ``(N, act_size)``.

        Returns:
            Output dict with at least ``"sae_out"`` and ``"loss"``.
        """
        return self.meta_sae(vectors)

    def forward_on_vectors(self, vectors: torch.Tensor) -> Dict[str, Any]:
        """Alias for ``forward`` (backward compatibility)."""
        return self.forward(vectors)


# ---------------------------------------------------------------------------
# SAEWithPenalty (composition wrapper)
# ---------------------------------------------------------------------------

class SAEWithPenalty(nn.Module):
    """Generic wrapper that adds the decomposability penalty to any SAE.

    Composes rather than inherits, so it works with any SAE class.
    """

    def __init__(
        self,
        sae_cls: type,
        cfg: Dict[str, Any],
        meta_sae: MetaSAEWrapper | None,
        penalty_cfg: Dict[str, Any],
    ):
        super().__init__()
        self.inner_sae = sae_cls(cfg)
        self.meta_sae = meta_sae
        self.penalty_lambda = penalty_cfg.get("lambda2", 0.0)
        self.sigma_sq = penalty_cfg.get("sigma_sq", 1.0)
        self.cfg = cfg

    @property
    def W_dec(self):
        return self.inner_sae.W_dec

    @property
    def W_enc(self):
        return self.inner_sae.W_enc

    @property
    def b_dec(self):
        return self.inner_sae.b_dec

    @property
    def b_enc(self):
        return self.inner_sae.b_enc

    def parameters(self, recurse: bool = True):
        return self.inner_sae.parameters(recurse=recurse)

    def state_dict(self, *args, **kwargs):
        return self.inner_sae.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, *args, **kwargs):
        return self.inner_sae.load_state_dict(state_dict, *args, **kwargs)

    def compute_decomposability_penalty(self) -> torch.Tensor:
        """Penalty based on how easily the meta SAE reconstructs W_dec.

        High penalty (= small reconstruction error) pushes decoder vectors
        away from the meta SAE's representational space.
        """
        if self.meta_sae is None:
            return torch.tensor(0.0, device=self.W_dec.device, dtype=self.W_dec.dtype)

        W_dec = self.W_dec
        with torch.no_grad():
            meta_output = self.meta_sae.forward_on_vectors(W_dec.detach())
            recon = meta_output["sae_out"].detach()

        errors = ((W_dec - recon).pow(2)).sum(dim=1)
        penalties = torch.exp(-errors / self.sigma_sq)
        return penalties.mean()

    def forward(self, x):
        output = self.inner_sae(x)
        decomp_penalty = self.compute_decomposability_penalty()
        output["decomp_penalty"] = decomp_penalty
        output["loss"] = output["loss"] + self.penalty_lambda * decomp_penalty
        return output

    def make_decoder_weights_and_grad_unit_norm(self):
        if hasattr(self.inner_sae, "make_decoder_weights_and_grad_unit_norm"):
            self.inner_sae.make_decoder_weights_and_grad_unit_norm()


# ---------------------------------------------------------------------------
# BatchTopKSAEWithPenalty (inheritance variant)
# ---------------------------------------------------------------------------

class BatchTopKSAEWithPenalty(BatchTopKSAE):
    """BatchTopKSAE with built-in decomposability penalty.

    Inherits from ``BatchTopKSAE``; the penalty is added inside
    ``get_loss_dict`` so it integrates naturally with the parent's forward.
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        meta_sae: MetaSAEWrapper | None,
        penalty_cfg: Dict[str, Any],
    ):
        super().__init__(cfg)
        self.meta_sae = meta_sae
        self.penalty_lambda = penalty_cfg.get("lambda2", 0.0)
        self.sigma_sq = penalty_cfg.get("sigma_sq", 1.0)

    def compute_decomposability_penalty(self) -> torch.Tensor:
        """Compute the penalty based on meta SAE reconstruction error.

        Gradients flow to W_dec (to push it away from reconstructible directions)
        but not through meta_sae (treated as frozen).
        """
        if self.meta_sae is None:
            return torch.tensor(0.0, device=self.W_dec.device, dtype=self.W_dec.dtype)

        W_dec = self.W_dec
        with torch.no_grad():
            meta_output = self.meta_sae.forward_on_vectors(W_dec.detach())
            recon = meta_output["sae_out"].detach()

        errors = ((W_dec - recon).pow(2)).sum(dim=1)
        penalties = torch.exp(-errors / self.sigma_sq)
        return penalties.mean()

    def get_loss_dict(self, x, x_reconstruct, acts, acts_topk, x_mean, x_std):
        base_output = super().get_loss_dict(x, x_reconstruct, acts, acts_topk, x_mean, x_std)
        decomp_penalty = self.compute_decomposability_penalty()
        base_output["decomp_penalty"] = decomp_penalty
        base_output["loss"] = base_output["loss"] + self.penalty_lambda * decomp_penalty
        return base_output


# ---------------------------------------------------------------------------
# calibrate_thresholds  (package API wrapping the EER histogram logic)
# ---------------------------------------------------------------------------

def calibrate_thresholds(
    sae,
    model,
    n_tokens: int,
    device: str,
    batch_size: int = 2048,
    model_batch_size: int = 64,
    num_batches_in_buffer: int = 4,
    max_val: float = 10.0,
    n_bins: int = 200,
    min_active_samples: int = 100,
    undercalibrated_midpoint: bool = False,
) -> np.ndarray:
    """Calibrate per-feature JumpReLU thresholds using EER histograms.

    Streams ``n_tokens`` through the SAE encoder and finds, for each feature,
    the threshold that minimises FP+FN relative to BatchTopK selection.

    Dead features (0 top-k appearances) receive ``max_val`` as a sentinel so
    that downstream scripts can filter them out.

    Args:
        sae: Trained SAE with ``cfg``, ``W_enc``, and ``b_dec`` attributes.
        model: Loaded HookedTransformer used to generate activations.
        n_tokens: Number of calibration tokens to stream.
        device: ``"cuda"`` or ``"cpu"``.
        batch_size: Tokens per ``ActivationsStore.next_batch()`` call.
        model_batch_size: Sequences per model forward pass.
        num_batches_in_buffer: Depth of the activation buffer.
        max_val: Pre-activation ceiling; bins span ``[0, max_val)``.
        n_bins: Number of histogram bins over ``[0, max_val)``.
        min_active_samples: Minimum top-k appearances for full EER calibration.
            Features below this get a conservative fallback threshold.
        undercalibrated_midpoint: If True, use midpoint of overlap region for
            undercalibrated features instead of the conservative max.

    Returns:
        float32 array of shape ``(dict_size,)`` with per-feature thresholds.
    """
    from meta_sae.activation_store import ActivationsStore

    dict_size       = sae.cfg["dict_size"]
    top_k           = sae.cfg["top_k"]
    input_unit_norm = sae.cfg.get("input_unit_norm", False)
    layer           = sae.cfg["layer"]

    W_enc = sae.W_enc.detach().to(device)
    b_dec = sae.b_dec.detach().to(device)

    active_hist_flat   = torch.zeros(dict_size * n_bins, dtype=torch.int64, device=device)
    inactive_hist_flat = torch.zeros(dict_size * n_bins, dtype=torch.int64, device=device)
    bin_scale = n_bins / max_val

    store_cfg = {
        "model_name":            sae.cfg["model_name"],
        "dataset_path":          sae.cfg["dataset_path"],
        "dataset_name":          sae.cfg.get("dataset_name", None),
        "hook_point":            f"blocks.{layer}.hook_{sae.cfg.get('site', 'resid_pre')}",
        "seq_len":               sae.cfg["seq_len"],
        "model_batch_size":      model_batch_size,
        "num_batches_in_buffer": num_batches_in_buffer,
        "device":                device,
        "layer":                 layer,
        "batch_size":            batch_size,
        "act_size":              sae.cfg["act_size"],
    }
    store = ActivationsStore(model, store_cfg)

    tokens_processed = 0
    batch_idx = 0

    print(
        f"Calibrating over {n_tokens:,} tokens "
        f"(batch_size={batch_size}, dict_size={dict_size}, "
        f"n_bins={n_bins}, max_val={max_val})..."
    )

    while tokens_processed < n_tokens:
        x = store.next_batch().to(device)
        N = x.shape[0]

        with torch.no_grad():
            if input_unit_norm:
                x_mean = x.mean(dim=-1, keepdim=True)
                x_std  = x.std(dim=-1, keepdim=True)
                x = (x - x_mean) / (x_std + 1e-5)

            pre = F.relu((x - b_dec) @ W_enc)   # (N, dict_size)

            k_total = top_k * N
            _, topk_idx = torch.topk(pre.flatten(), k_total)
            mask = torch.zeros(N * dict_size, dtype=torch.bool, device=device)
            mask[topk_idx] = True
            mask = mask.reshape(N, dict_size)

            bin_idx = (pre * bin_scale).long().clamp(0, n_bins - 1)

            act_rows, act_cols = torch.where(mask)
            act_bins = bin_idx[act_rows, act_cols]
            act_lin  = act_cols * n_bins + act_bins
            active_hist_flat.scatter_add_(
                0, act_lin, torch.ones(k_total, dtype=torch.int64, device=device)
            )

            inc_mask = (pre > 0) & ~mask
            inc_rows, inc_cols = torch.where(inc_mask)
            if inc_rows.numel() > 0:
                inc_bins = bin_idx[inc_rows, inc_cols]
                inc_lin  = inc_cols * n_bins + inc_bins
                inactive_hist_flat.scatter_add_(
                    0, inc_lin,
                    torch.ones(inc_rows.numel(), dtype=torch.int64, device=device),
                )

        tokens_processed += N
        batch_idx += 1
        if batch_idx % 50 == 0:
            print(f"  {tokens_processed:>10,} / {n_tokens:,} tokens", flush=True)

    print(f"  {tokens_processed:>10,} / {n_tokens:,} tokens — done.", flush=True)
    del store

    active_hist   = active_hist_flat.cpu().numpy().reshape(dict_size, n_bins)
    inactive_hist = inactive_hist_flat.cpu().numpy().reshape(dict_size, n_bins)

    bin_edges   = np.linspace(0.0, max_val, n_bins + 1, dtype=np.float64)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    thresholds = np.full(dict_size, max_val, dtype=np.float32)
    n_dead = 0
    n_undercalibrated = 0

    for f in range(dict_size):
        total_active = int(active_hist[f].sum())
        if total_active == 0:
            n_dead += 1
            continue

        if total_active < min_active_samples:
            first_active_bin = int(np.argmax(active_hist[f] > 0))
            min_active_val   = float(bin_centers[first_active_bin])
            total_inactive   = int(inactive_hist[f].sum())
            if total_inactive > 0:
                last_inactive_bin = int(n_bins - 1 - np.argmax(inactive_hist[f][::-1] > 0))
                max_inactive_val  = float(bin_centers[last_inactive_bin])
                if undercalibrated_midpoint and max_inactive_val > min_active_val:
                    thresholds[f] = (min_active_val + max_inactive_val) / 2.0
                else:
                    thresholds[f] = max(min_active_val, max_inactive_val)
            else:
                thresholds[f] = min_active_val
            n_undercalibrated += 1
            continue

        cdf_active   = np.cumsum(active_hist[f]).astype(np.float64) / total_active
        total_inactive = int(inactive_hist[f].sum())
        if total_inactive == 0:
            first_bin = int(np.argmax(active_hist[f] > 0))
            thresholds[f] = float(bin_centers[first_bin])
            continue

        cdf_inactive = np.cumsum(inactive_hist[f]).astype(np.float64) / total_inactive
        eer_idx = int(np.argmin(np.abs(cdf_active + cdf_inactive - 1.0)))
        thresholds[f] = float(bin_centers[eer_idx])

    n_calibrated = dict_size - n_dead - n_undercalibrated
    print(f"\nThreshold summary:")
    print(f"  Calibrated (EER)    : {n_calibrated:,} / {dict_size:,}")
    print(f"  Undercalibrated fallback: {n_undercalibrated:,}")
    print(f"  Dead (0 top-k appearances): {n_dead:,}")
    cal_thresh = thresholds[(thresholds < max_val) & (thresholds > 0.0)]
    if len(cal_thresh) > 0:
        print(
            f"  Threshold stats (calibrated+fallback): "
            f"min={cal_thresh.min():.3f}  "
            f"median={np.median(cal_thresh):.3f}  "
            f"max={cal_thresh.max():.3f}"
        )

    return thresholds


# ---------------------------------------------------------------------------
# InferenceSAE
# ---------------------------------------------------------------------------

class InferenceSAE(nn.Module):
    """Deterministic inference wrapper for a calibrated BatchTopKSAE.

    Replaces batch-level top-k with per-feature threshold firing:
    feature ``i`` fires iff ``pre_act_i > threshold_i``.

    Key methods:
      - ``encode(x)``                          → sparse feature activations
      - ``decode(feature_acts)``               → reconstructed activations (normalised space)
      - ``forward(x)``                         → (feature_acts, x_hat) in original space
      - ``steer(x, modify_fn)``                → modify latents, decode to original space
      - ``make_steering_hook(modify_fn)``      → TransformerLens hook for in-forward steering
      - ``top_activating_examples(feat, x)``  → inspect which positions activate a feature

    Usage::

        sae = InferenceSAE.from_checkpoint(
            "joint_primary_sae.pt", "thresholds_joint.npy"
        )
        feature_acts        = sae.encode(x)        # (batch, dict_size), sparse
        feature_acts, x_hat = sae(x)               # both at once, x_hat in original space

        # Steering
        def my_edits(acts):
            acts = acts.clone()
            acts[:, 42]   = 0.0    # suppress feature 42
            acts[:, 1337] *= 2.0   # amplify feature 1337
            return acts

        hook = sae.make_steering_hook(my_edits)
        output = model.run_with_hooks(tokens, fwd_hooks=[(sae.hook_point, hook)])
    """

    def __init__(self, sae: BatchTopKSAE, thresholds: np.ndarray):
        super().__init__()
        self.sae = sae
        sae_device = next(sae.parameters()).device
        self.register_buffer(
            "thresholds",
            torch.tensor(thresholds, dtype=torch.float32).to(sae_device),
        )
        self.input_unit_norm: bool = sae.cfg.get("input_unit_norm", False)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        thresholds_path: str | Path,
        device: str = "cpu",
    ) -> "InferenceSAE":
        """Load from a ``.pt`` checkpoint and a ``.npy`` thresholds file.

        Args:
            checkpoint_path: Path to the SAE ``.pt`` file.
            thresholds_path: Path to the ``thresholds_{label}.npy`` file.
            device: Device to load onto.

        Returns:
            ``InferenceSAE`` in eval mode on ``device``.
        """
        from meta_sae.io import load_sae, load_thresholds
        sae        = load_sae(checkpoint_path, device=device)
        thresholds = load_thresholds(thresholds_path)
        obj        = cls(sae, thresholds)
        obj.to(device)
        obj.eval()
        return obj

    @classmethod
    def from_sae(
        cls,
        sae: BatchTopKSAE,
        thresholds: np.ndarray,
    ) -> "InferenceSAE":
        """Wrap an already-loaded SAE and thresholds array.

        Args:
            sae: Trained SAE (already on the desired device).
            thresholds: float32 array of shape ``(dict_size,)``.

        Returns:
            ``InferenceSAE`` in eval mode.
        """
        obj = cls(sae, thresholds)
        obj.eval()
        return obj

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def hook_point(self) -> str:
        """The TransformerLens hook point name for this SAE's layer.

        Built from the checkpoint's ``layer`` and ``site`` fields rather than
        the (potentially stale) ``hook_point`` value stored in cfg.
        """
        layer = self.sae.cfg["layer"]
        site  = self.sae.cfg.get("site", "resid_pre")
        return f"blocks.{layer}.hook_{site}"

    # ------------------------------------------------------------------
    # Core encode / decode
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode ``x`` to sparse feature activations using per-feature thresholds.

        Args:
            x: Input tensor of shape ``(batch, act_size)``.

        Returns:
            Sparse feature activations of shape ``(batch, dict_size)``.
        """
        feature_acts, _, _ = self.encode_with_stats(x)
        return feature_acts

    @torch.no_grad()
    def encode_with_stats(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Encode ``x`` and return the normalisation statistics alongside.

        Required when you want to decode back to the *original* (unnormalised)
        residual-stream space, e.g. for steering or reconstruction hooks.

        Args:
            x: Input tensor of shape ``(batch, act_size)``.

        Returns:
            ``(feature_acts, x_mean, x_std)`` where ``x_mean`` and ``x_std``
            are ``None`` when ``input_unit_norm=False``.
        """
        W_enc = self.sae.W_enc   # (act_size, dict_size)
        b_dec = self.sae.b_dec   # (act_size,)

        x_mean, x_std = None, None
        if self.input_unit_norm:
            x_mean = x.mean(dim=-1, keepdim=True)
            x_std  = x.std(dim=-1, keepdim=True)
            x = (x - x_mean) / (x_std + 1e-5)

        pre = F.relu((x - b_dec) @ W_enc)   # (batch, dict_size)
        feature_acts = pre * (pre > self.thresholds)
        return feature_acts, x_mean, x_std

    @torch.no_grad()
    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        """Decode feature activations.

        .. note::
            If ``input_unit_norm=True`` (the default for GPT-2L checkpoints)
            this returns activations in the *normalised* space.  Use
            ``decode_from_stats()`` or ``steer()`` when you need the output in
            the original residual-stream space (e.g. for reconstruction hooks
            or steering).

        Args:
            feature_acts: Sparse activations of shape ``(batch, dict_size)``.

        Returns:
            Reconstructed activations of shape ``(batch, act_size)``.
        """
        return self.sae.decode(feature_acts)

    @torch.no_grad()
    def decode_from_stats(
        self,
        feature_acts: torch.Tensor,
        x_mean: Optional[torch.Tensor],
        x_std: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Decode and undo input normalisation, recovering original activation scale.

        Use the ``x_mean`` and ``x_std`` returned by ``encode_with_stats()``
        to correctly map back to the residual-stream space.

        Args:
            feature_acts: Sparse activations of shape ``(batch, dict_size)``.
            x_mean: Per-sample mean saved during encoding, or ``None``.
            x_std:  Per-sample std saved during encoding, or ``None``.

        Returns:
            Reconstructed activations in the original space, shape ``(batch, act_size)``.
        """
        x_hat = self.sae.decode(feature_acts)
        if self.input_unit_norm and x_mean is not None and x_std is not None:
            x_hat = x_hat * (x_std + 1e-5) + x_mean
        return x_hat

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode and decode in one call.

        Returns:
            ``(feature_acts, x_hat)`` where ``x_hat`` is in the original
            (unnormalised) activation space, matching the input ``x``.
        """
        feature_acts, x_mean, x_std = self.encode_with_stats(x)
        x_hat = self.decode_from_stats(feature_acts, x_mean, x_std)
        return feature_acts, x_hat

    # ------------------------------------------------------------------
    # Steering
    # ------------------------------------------------------------------

    @torch.no_grad()
    def steer(
        self,
        x: torch.Tensor,
        modify_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Encode, modify feature activations, and decode back to the original space.

        This is the correct way to do activation steering: it preserves the
        input normalisation statistics so that the patched activations are in
        the same scale as the original residual stream.

        Args:
            x: Residual-stream activations, shape ``(batch, act_size)``.
            modify_fn: A callable ``(feature_acts) -> feature_acts`` that edits
                the latent representation.  Receives and returns a tensor of
                shape ``(batch, dict_size)``.

        Returns:
            Patched activations in the original residual-stream space,
            shape ``(batch, act_size)``.

        Example::

            def suppress_and_amplify(acts):
                acts = acts.clone()
                acts[:, 42]   = 0.0    # suppress feature 42
                acts[:, 1337] *= 2.0   # amplify feature 1337
                return acts

            x_patched = sae.steer(x, suppress_and_amplify)
        """
        feature_acts, x_mean, x_std = self.encode_with_stats(x)
        feature_acts = modify_fn(feature_acts)
        return self.decode_from_stats(feature_acts, x_mean, x_std)

    def make_steering_hook(
        self,
        modify_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> Callable:
        """Return a TransformerLens forward hook that steers activations via this SAE.

        The hook encodes each token's residual-stream vector, calls ``modify_fn``
        on the feature activations, then decodes back — all within the forward
        pass so subsequent layers see the modified representation.

        Args:
            modify_fn: A callable ``(feature_acts) -> feature_acts``.
                Receives a tensor of shape ``(batch * seq_len, dict_size)``.

        Returns:
            A hook function compatible with
            ``model.run_with_hooks(fwd_hooks=[(hook_point, hook)])``.

        Example::

            sae = InferenceSAE.from_checkpoint("joint_primary_sae.pt",
                                               "thresholds_joint.npy",
                                               device="cuda")

            def suppress_feature(acts):
                acts = acts.clone()
                acts[:, 42] = 0.0
                return acts

            hook   = sae.make_steering_hook(suppress_feature)
            output = model.run_with_hooks(
                tokens,
                fwd_hooks=[(sae.hook_point, hook)],
                return_type="str",
            )
        """
        sae_self = self

        @torch.no_grad()
        def _hook(value, hook):
            B, T, D = value.shape
            x = value.reshape(B * T, D)
            x_patched = sae_self.steer(x, modify_fn)
            return x_patched.reshape(B, T, D)

        return _hook

    # ------------------------------------------------------------------
    # Qualitative exploration
    # ------------------------------------------------------------------

    @torch.no_grad()
    def top_activating_examples(
        self,
        feature_idx: int,
        activations: torch.Tensor,
        k: int = 20,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the k token positions where a feature activates most strongly.

        Args:
            feature_idx: Index of the SAE feature to inspect.
            activations: Pre-extracted residual-stream activations,
                shape ``(N, act_size)`` where N = batch × seq_len.
            k: Number of top positions to return.

        Returns:
            ``(values, indices)``: top-k activation strengths and their flat
            position indices into ``activations``.

        Example::

            with torch.no_grad():
                _, cache = model.run_with_cache(
                    tokens, names_filter=[sae.hook_point],
                    stop_at_layer=sae.sae.cfg["layer"] + 1,
                )
            x = cache[sae.hook_point].reshape(-1, sae.sae.cfg["act_size"]).to(device)

            values, indices = sae.top_activating_examples(feature_idx=42, activations=x, k=20)
            # Decode the context of each hit:
            # flat_idx → (batch_pos, seq_pos) = divmod(idx, seq_len)
        """
        feature_acts = self.encode(activations)        # (N, dict_size)
        col          = feature_acts[:, feature_idx]    # (N,)
        k_actual     = min(k, int((col > 0).sum().item()), col.numel())
        if k_actual == 0:
            return torch.tensor([]), torch.tensor([], dtype=torch.long)
        values, indices = torch.topk(col, k_actual)
        return values, indices


# ---------------------------------------------------------------------------
# Training functions
# ---------------------------------------------------------------------------

def train_sae_with_meta(
    primary_sae: BatchTopKSAEWithPenalty,
    meta_sae: MetaSAEWrapper,
    activation_store,
    model,
    primary_cfg: Dict[str, Any],
    meta_cfg: Dict[str, Any],
    penalty_cfg: Dict[str, Any],
    logger: Optional["TrainingLogger"] = None,
    checkpoint_fn=None,
    wandb_project: str | None = None,
    checkpoint_dir: Path | None = None,
    checkpoint_freq: int = 10_000,
) -> Dict[str, Any]:
    """Train a primary SAE with an alternating meta SAE training loop.

    Alternates between updating the primary SAE on model activations and
    updating the meta SAE on the primary SAE's decoder columns.

    Args:
        primary_sae: Primary SAE with an attached meta_sae and penalty.
        meta_sae: Wrapper around the meta SAE.
        activation_store: Provides batches via ``next_batch()``.
        model: Base transformer model (used only by logger callbacks).
        primary_cfg: Configuration dict for the primary SAE.
        meta_cfg: Configuration dict for the meta SAE.
        penalty_cfg: Penalty hyperparameters (``lambda2``, ``sigma_sq``,
            ``n_primary_steps``, ``n_meta_steps``).
        logger: Optional ``TrainingLogger`` for structured metric logging.
        checkpoint_fn: Optional callable ``(step: int) -> None``.  Called
            after each primary step; callers decide which steps to act on.
        wandb_project: If set, log metrics to this W&B project.
        checkpoint_dir: If set, save ``joint_primary_sae.pt`` and
            ``joint_meta_sae.pt`` every ``checkpoint_freq`` primary steps.
        checkpoint_freq: Checkpoint save interval in primary steps.

    Returns:
        Dict of final training metrics.
    """
    from meta_sae.io import save_sae

    device = primary_cfg.get("device", "cuda:0")
    print(f"   Device configuration: {device}")
    print(f"   Primary SAE W_enc device: {primary_sae.W_enc.device}")
    print(f"   Meta SAE W_enc device: {meta_sae.meta_sae.W_enc.device}")

    primary_sae.meta_sae = meta_sae
    lambda2          = penalty_cfg.get("lambda2", 0.0)
    sigma_sq         = penalty_cfg.get("sigma_sq", 1.0)
    n_primary_steps  = penalty_cfg.get("n_primary_steps", 100)
    n_meta_steps     = penalty_cfg.get("n_meta_steps", 10)

    primary_sae.penalty_lambda = lambda2
    primary_sae.sigma_sq       = sigma_sq

    primary_optimizer = torch.optim.Adam(
        primary_sae.parameters(),
        lr=primary_cfg["lr"],
        betas=(primary_cfg["beta1"], primary_cfg["beta2"]),
    )
    meta_optimizer = torch.optim.Adam(
        meta_sae.parameters(),
        lr=meta_cfg["lr"],
        betas=(meta_cfg["beta1"], meta_cfg["beta2"]),
    )

    total_batches    = primary_cfg["num_tokens"] // primary_cfg["batch_size"]
    batch_iter       = 0
    meta_step_global = 0

    # Optional W&B setup
    wandb_run = None
    if wandb_project is not None:
        try:
            import wandb
            wandb_run = wandb.init(project=wandb_project, config={**primary_cfg, **meta_cfg, **penalty_cfg})
        except ImportError:
            print("wandb not installed — skipping W&B logging.")

    pbar       = tqdm.tqdm(total=total_batches, desc="Training SAE + Meta SAE")
    log_freq   = 50
    print_freq = 200

    final_metrics     = {}
    first_batch_logged = False

    while batch_iter < total_batches:
        # Primary SAE phase
        for _ in range(n_primary_steps):
            if batch_iter >= total_batches:
                break
            batch = activation_store.next_batch()

            if not first_batch_logged:
                print(f"   First batch device: {batch.device}")
                first_batch_logged = True

            output = primary_sae(batch)
            loss   = output["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(primary_sae.parameters(), primary_cfg["max_grad_norm"])
            primary_sae.make_decoder_weights_and_grad_unit_norm()
            primary_optimizer.step()
            primary_optimizer.zero_grad()

            batch_iter += 1
            pbar.update(1)

            if checkpoint_fn is not None:
                checkpoint_fn(batch_iter)

            if checkpoint_dir is not None and batch_iter % checkpoint_freq == 0:
                ckpt_dir = Path(checkpoint_dir)
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                save_sae(
                    primary_sae,
                    ckpt_dir / f"joint_primary_sae_step{batch_iter}.pt",
                    extra={"meta_cfg": meta_cfg, "penalty_cfg": penalty_cfg},
                )
                save_sae(meta_sae, ckpt_dir / f"joint_meta_sae_step{batch_iter}.pt")

            if batch_iter % log_freq == 0:
                final_metrics = {
                    "loss":  loss.item(),
                    "l0":    output["l0_norm"].item(),
                    "l2":    output["l2_loss"].item(),
                    "decomp": output.get("decomp_penalty", torch.tensor(0.0)).item(),
                }
                pbar.set_postfix({
                    "Loss":   f"{final_metrics['loss']:.4f}",
                    "L0":     f"{final_metrics['l0']:.1f}",
                    "L2":     f"{final_metrics['l2']:.4f}",
                    "Decomp": f"{final_metrics['decomp']:.4f}",
                })

                if logger is not None:
                    logger.log_step(batch_iter, "joint_primary", output)

                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "joint_primary/loss":         final_metrics["loss"],
                            "joint_primary/l0":           final_metrics["l0"],
                            "joint_primary/l2":           final_metrics["l2"],
                            "joint_primary/decomp":       final_metrics["decomp"],
                        },
                        step=batch_iter,
                    )

                if batch_iter % print_freq == 0:
                    l0_coeff_str = (
                        f", l0_coeff={output['l0_coeff']:.2e}"
                        if "l0_coeff" in output
                        else ""
                    )
                    print(
                        f"[Joint Primary] step={batch_iter}/{total_batches} "
                        f"loss={final_metrics['loss']:.4f} "
                        f"L0={final_metrics['l0']:.1f} "
                        f"L2={final_metrics['l2']:.4f}"
                        f"{l0_coeff_str}",
                        flush=True,
                    )

            del batch, output, loss

            if batch_iter % 1000 == 0:
                torch.cuda.empty_cache()

        # Meta SAE phase
        for meta_step in range(n_meta_steps):
            W_dec      = primary_sae.W_dec.detach()
            meta_output = meta_sae(W_dec)
            meta_loss   = meta_output["loss"]
            meta_loss.backward()
            torch.nn.utils.clip_grad_norm_(meta_sae.parameters(), meta_cfg["max_grad_norm"])
            if hasattr(meta_sae.meta_sae, "make_decoder_weights_and_grad_unit_norm"):
                meta_sae.meta_sae.make_decoder_weights_and_grad_unit_norm()
            meta_optimizer.step()
            meta_optimizer.zero_grad()

            meta_step_global += 1

            if logger is not None and meta_step_global % log_freq == 0:
                logger.log_step(meta_step_global, "joint_meta", meta_output)

            if wandb_run is not None and meta_step_global % log_freq == 0:
                wandb_run.log(
                    {
                        "joint_meta/loss": meta_output["loss"].item(),
                        "joint_meta/l0":   meta_output["l0_norm"].item(),
                        "joint_meta/l2":   meta_output["l2_loss"].item(),
                    },
                    step=meta_step_global,
                )

            del W_dec, meta_output, meta_loss

    pbar.close()

    if wandb_run is not None:
        wandb_run.finish()

    return final_metrics


def train_primary_sae_solo(
    primary_sae,
    activation_store,
    cfg: Dict[str, Any],
    logger: Optional["TrainingLogger"] = None,
    checkpoint_fn=None,
) -> Dict[str, Any]:
    """Train a primary SAE without any meta SAE or decomposability penalty.

    Args:
        primary_sae: Any SAE model (``BatchTopKSAE`` or similar).
        activation_store: Provides batches via ``next_batch()``.
        cfg: Primary SAE configuration dict.
        logger: Optional ``TrainingLogger``.
        checkpoint_fn: Optional callable ``(step: int) -> None``.

    Returns:
        Dict of final training metrics.
    """
    print(f"Training solo primary SAE...")
    print(f"   Dictionary size: {cfg['dict_size']}")
    print(f"   Target tokens: {cfg['num_tokens']:,}")
    print(f"   Batch size: {cfg['batch_size']}")
    print(f"   Top-k: {cfg['top_k']}")
    print(f"   Device: {cfg.get('device', 'cuda:0')}")

    optimizer  = optim.Adam(primary_sae.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]))
    num_batches = cfg["num_tokens"] // cfg["batch_size"]
    pbar        = tqdm.tqdm(total=num_batches, desc="Solo Primary SAE Training", leave=True)

    primary_sae.train()
    log_freq   = 50
    print_freq = 200
    final_metrics = {}

    for step in range(num_batches):
        try:
            batch = activation_store.next_batch()
        except StopIteration:
            print(f"   Dataset exhausted at step {step}, stopping training.")
            break

        if step == 0:
            print(f"   First batch device: {batch.device}")

        optimizer.zero_grad()
        output = primary_sae(batch)
        loss   = output["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(primary_sae.parameters(), cfg["max_grad_norm"])

        if hasattr(primary_sae, "make_decoder_weights_and_grad_unit_norm"):
            primary_sae.make_decoder_weights_and_grad_unit_norm()

        optimizer.step()
        pbar.update(1)

        if checkpoint_fn is not None:
            checkpoint_fn(step + 1)

        if (step + 1) % log_freq == 0:
            final_metrics = {
                "loss": loss.item(),
                "l2":   output["l2_loss"].item(),
                "l1":   output["l1_loss"].item(),
                "l0":   output["l0_norm"].item(),
                "dead": output["num_dead_features"].item(),
            }
            pbar.set_postfix({
                "loss": f'{final_metrics["loss"]:.4f}',
                "l2":   f'{final_metrics["l2"]:.4f}',
                "l0":   f'{final_metrics["l0"]:.1f}',
                "dead": f'{final_metrics["dead"]}',
            })

            if logger is not None:
                logger.log_step(step + 1, "solo_primary", output)

            if (step + 1) % print_freq == 0:
                l0_coeff_str = (
                    f", l0_coeff={output['l0_coeff']:.2e}" if "l0_coeff" in output else ""
                )
                print(
                    f"[Solo Primary] step={step+1}/{num_batches} "
                    f"loss={final_metrics['loss']:.4f} "
                    f"L0={final_metrics['l0']:.1f} "
                    f"L2={final_metrics['l2']:.4f}"
                    f"{l0_coeff_str}",
                    flush=True,
                )

        del batch, output, loss

        if (step + 1) % 1000 == 0:
            torch.cuda.empty_cache()

    pbar.close()
    print("Solo primary SAE training completed!", flush=True)
    return final_metrics


def train_meta_sae_on_frozen_primary(
    meta_sae: MetaSAEWrapper,
    primary_sae,
    meta_cfg: Dict[str, Any],
    penalty_cfg: Dict[str, Any],
    logger: Optional["TrainingLogger"] = None,
) -> Dict[str, Any]:
    """Train a meta SAE on the frozen decoder weights of a pre-trained primary SAE.

    Args:
        meta_sae: Meta SAE wrapper to train.
        primary_sae: Pre-trained primary SAE (frozen during this phase).
        meta_cfg: Meta SAE configuration dict.
        penalty_cfg: Penalty configuration (provides ``num_tokens``,
            ``n_primary_steps``, ``n_meta_steps``).
        logger: Optional ``TrainingLogger``.

    Returns:
        Dict of final training metrics.
    """
    print(f"Training meta SAE on frozen primary SAE...")
    print(f"   Meta dictionary size: {meta_cfg['dict_size']}")
    print(f"   Primary dictionary size: {primary_sae.W_dec.shape[0]}")
    print(f"   Meta top-k: {meta_cfg['top_k']}")

    for param in primary_sae.parameters():
        param.requires_grad = False
    primary_sae.eval()

    meta_optimizer = optim.Adam(
        meta_sae.parameters(),
        lr=meta_cfg["lr"],
        betas=(meta_cfg["beta1"], meta_cfg["beta2"]),
    )

    primary_batches  = penalty_cfg.get("num_tokens", meta_cfg["num_tokens"]) // meta_cfg["batch_size"]
    cycles           = primary_batches // penalty_cfg["n_primary_steps"]
    total_meta_steps = cycles * penalty_cfg["n_meta_steps"]

    print(f"   Total meta training steps: {total_meta_steps}")

    pbar          = tqdm.tqdm(total=total_meta_steps, desc="Meta SAE Training", leave=True)
    log_freq      = 50
    print_freq    = 200
    final_metrics = {}

    meta_sae.train()

    for step in range(total_meta_steps):
        W_dec      = primary_sae.W_dec.detach()
        meta_optimizer.zero_grad()
        meta_output = meta_sae(W_dec)
        meta_loss   = meta_output["loss"]
        meta_loss.backward()

        if hasattr(meta_sae.meta_sae, "make_decoder_weights_and_grad_unit_norm"):
            meta_sae.meta_sae.make_decoder_weights_and_grad_unit_norm()

        meta_optimizer.step()
        pbar.update(1)

        if (step + 1) % log_freq == 0:
            final_metrics = {
                "loss": meta_loss.item(),
                "l2":   meta_output["l2_loss"].item(),
                "l0":   meta_output["l0_norm"].item(),
            }
            pbar.set_postfix({
                "loss": f'{final_metrics["loss"]:.4f}',
                "l2":   f'{final_metrics["l2"]:.4f}',
                "l0":   f'{final_metrics["l0"]:.1f}',
            })

            if logger is not None:
                logger.log_step(step + 1, "sequential_meta", meta_output)

            if (step + 1) % print_freq == 0:
                print(
                    f"[Sequential Meta] step={step+1}/{total_meta_steps} "
                    f"loss={final_metrics['loss']:.4f} "
                    f"L0={final_metrics['l0']:.1f} "
                    f"L2={final_metrics['l2']:.4f}",
                    flush=True,
                )

        del W_dec, meta_output, meta_loss

        if (step + 1) % 1000 == 0:
            torch.cuda.empty_cache()

    pbar.close()
    print("Meta SAE training on frozen primary completed!", flush=True)

    for param in primary_sae.parameters():
        param.requires_grad = True
    primary_sae.train()


# ---------------------------------------------------------------------------
# Checkpoint-level calibration helper
# ---------------------------------------------------------------------------

def calibrate_checkpoint(
    checkpoint: str,
    output_dir: str,
    label: str = "sae",
    *,
    n_tokens: int = 5_000_000,
    batch_size: int = 2048,
    model_batch_size: int = 64,
    num_batches_in_buffer: int = 4,
    max_val: float = 10.0,
    n_bins: int = 200,
    min_active_samples: int = 100,
    undercalibrated_midpoint: bool = False,
    device: str = "cuda",
) -> np.ndarray:
    """Load an SAE checkpoint, calibrate JumpReLU thresholds, and save them.

    Loads the SAE and its base language model, runs EER histogram calibration
    over ``n_tokens`` streamed tokens, writes
    ``{output_dir}/thresholds_{label}.npy``, and returns the threshold array.

    The final action redirects C-level stderr to suppress streaming-dataset
    teardown noise. Callers that need stderr restored afterwards (e.g. tests)
    should wrap this call in a ``_restore_stderr()`` context manager.
    """
    import os
    import sys
    from pathlib import Path as _Path

    from transformer_lens import HookedTransformer
    from meta_sae.io import load_sae, save_thresholds

    out_dir = _Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading SAE from {checkpoint} ...")
    sae = load_sae(checkpoint, device)
    sae.eval()
    print(f"  dict_size={sae.cfg['dict_size']}, top_k={sae.cfg['top_k']}, "
          f"act_size={sae.cfg['act_size']}, layer={sae.cfg['layer']}")

    model_name = sae.cfg["model_name"]
    print(f"\nLoading {model_name} via TransformerLens ...")
    model_dtype_str = sae.cfg.get("model_dtype", "float32")
    model_dtype     = torch.bfloat16 if model_dtype_str == "bfloat16" else torch.float32
    model = HookedTransformer.from_pretrained(model_name, dtype=model_dtype)
    model.eval()
    model.to(device)
    print(f"  Loaded {model_name}.")

    thresholds = calibrate_thresholds(
        sae=sae,
        model=model,
        n_tokens=n_tokens,
        device=device,
        batch_size=batch_size,
        model_batch_size=model_batch_size,
        num_batches_in_buffer=num_batches_in_buffer,
        max_val=max_val,
        n_bins=n_bins,
        min_active_samples=min_active_samples,
        undercalibrated_midpoint=undercalibrated_midpoint,
    )

    out_path = out_dir / f"thresholds_{label}.npy"
    save_thresholds(thresholds, out_path)
    print(f"\nSaved thresholds → {out_path}")

    # Redirect C-level stderr before local variable teardown triggers streaming
    # dataset background threads, which otherwise print spurious messages on exit.
    sys.stdout.flush()
    sys.stderr.flush()
    _devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(_devnull, 2)
    os.close(_devnull)

    return thresholds

    return final_metrics
