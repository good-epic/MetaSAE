"""
SAE architectures: BatchTopK, TopK, Vanilla, JumpReLU.

All classes extend BaseAutoencoder which provides encode() and decode()
methods for clean inference use. The training-oriented forward() method
returns a loss dict and is used during training only.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseAutoencoder(nn.Module):
    """Base class for autoencoder models."""

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        torch.manual_seed(self.cfg["seed"])

        self.b_dec = nn.Parameter(torch.zeros(self.cfg["act_size"]))
        self.b_enc = nn.Parameter(torch.zeros(self.cfg["dict_size"]))
        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.cfg["act_size"], self.cfg["dict_size"])
            )
        )
        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.cfg["dict_size"], self.cfg["act_size"])
            )
        )
        self.W_dec.data[:] = self.W_enc.t().data
        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        self.num_batches_not_active = torch.zeros((self.cfg["dict_size"],)).to(
            cfg["device"]
        )

        self.to(cfg["dtype"]).to(cfg["device"])

    def preprocess_input(self, x):
        if self.cfg.get("input_unit_norm", False):
            x_mean = x.mean(dim=-1, keepdim=True)
            x = x - x_mean
            x_std = x.std(dim=-1, keepdim=True)
            x = x / (x_std + 1e-5)
            return x, x_mean, x_std
        else:
            return x, None, None

    def postprocess_output(self, x_reconstruct, x_mean, x_std):
        if self.cfg.get("input_unit_norm", False):
            x_reconstruct = x_reconstruct * x_std + x_mean
        return x_reconstruct

    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(
            -1, keepdim=True
        ) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj
        self.W_dec.data = W_dec_normed

    def update_inactive_features(self, acts):
        self.num_batches_not_active += (acts.sum(0) == 0).float()
        self.num_batches_not_active[acts.sum(0) > 0] = 0

    # ------------------------------------------------------------------
    # Clean inference API (no loss computation)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode activations to sparse feature space.

        Args:
            x: Model activations, shape (batch, d_model).

        Returns:
            feature_acts: Sparse feature activations, shape (batch, dict_size).
            Uses the same sparsity mechanism as forward() but without loss.
        """
        raise NotImplementedError("Subclasses must implement encode()")

    @torch.no_grad()
    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        """Decode feature activations back to model activation space.

        Args:
            feature_acts: Sparse feature activations, shape (batch, dict_size).

        Returns:
            x_reconstruct: Reconstructed activations in original space,
                shape (batch, d_model). If input_unit_norm was used during
                training, note that decode() cannot undo the normalisation
                without the original mean/std — use encode_decode() for that.
        """
        return feature_acts @ self.W_dec + self.b_dec

    @torch.no_grad()
    def encode_decode(self, x: torch.Tensor):
        """Encode then decode in one call, correctly handling input_unit_norm.

        Args:
            x: Model activations, shape (batch, d_model).

        Returns:
            (feature_acts, x_reconstruct): feature activations and
                reconstructed activations in the original (unnormalised) space.
        """
        x_proc, x_mean, x_std = self.preprocess_input(x)
        feature_acts = self._encode_preprocessed(x_proc)
        x_hat_proc = feature_acts @ self.W_dec + self.b_dec
        x_hat = self.postprocess_output(x_hat_proc, x_mean, x_std)
        return feature_acts, x_hat

    def _encode_preprocessed(self, x: torch.Tensor) -> torch.Tensor:
        """Encode already-preprocessed activations. Override in subclasses."""
        raise NotImplementedError


class BatchTopKSAE(BaseAutoencoder):
    def __init__(self, cfg):
        super().__init__(cfg)

    def forward(self, x):
        x, x_mean, x_std = self.preprocess_input(x)

        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc)
        acts_topk = torch.topk(acts.flatten(), self.cfg["top_k"] * x.shape[0], dim=-1)
        acts_topk = (
            torch.zeros_like(acts.flatten())
            .scatter(-1, acts_topk.indices, acts_topk.values)
            .reshape(acts.shape)
        )
        x_reconstruct = acts_topk @ self.W_dec + self.b_dec

        self.update_inactive_features(acts_topk)
        output = self.get_loss_dict(x, x_reconstruct, acts, acts_topk, x_mean, x_std)
        return output

    def get_loss_dict(self, x, x_reconstruct, acts, acts_topk, x_mean, x_std):
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).mean()
        l1_norm = acts_topk.float().abs().sum(-1).mean()
        l1_loss = self.cfg["l1_coeff"] * l1_norm
        l0_norm = (acts_topk > 0).float().sum(-1).mean()
        aux_loss = self.get_auxiliary_loss(x, x_reconstruct, acts)
        loss = l2_loss + l1_loss + aux_loss
        num_dead_features = (
            self.num_batches_not_active > self.cfg["n_batches_to_dead"]
        ).sum()
        sae_out = self.postprocess_output(x_reconstruct, x_mean, x_std)
        output = {
            "sae_out": sae_out,
            "feature_acts": acts_topk,
            "num_dead_features": num_dead_features,
            "loss": loss,
            "l1_loss": l1_loss,
            "l2_loss": l2_loss,
            "l0_norm": l0_norm,
            "l1_norm": l1_norm,
            "aux_loss": aux_loss,
        }
        return output

    def get_auxiliary_loss(self, x, x_reconstruct, acts):
        dead_features = self.num_batches_not_active >= self.cfg["n_batches_to_dead"]
        if dead_features.sum() > 0:
            residual = x.float() - x_reconstruct.float()
            acts_topk_aux = torch.topk(
                acts[:, dead_features],
                min(self.cfg["top_k_aux"], dead_features.sum()),
                dim=-1,
            )
            acts_aux = torch.zeros_like(acts[:, dead_features]).scatter(
                -1, acts_topk_aux.indices, acts_topk_aux.values
            )
            x_reconstruct_aux = acts_aux @ self.W_dec[dead_features]
            l2_loss_aux = (
                self.cfg["aux_penalty"]
                * (x_reconstruct_aux.float() - residual.float()).pow(2).mean()
            )
            return l2_loss_aux
        else:
            return torch.tensor(0, dtype=x.dtype, device=x.device)

    # Inference API
    def _encode_preprocessed(self, x: torch.Tensor) -> torch.Tensor:
        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc)
        # Per-sample top-k (not batch top-k) — appropriate for single-sample inference
        acts_topk = torch.topk(acts, self.cfg["top_k"], dim=-1)
        return torch.zeros_like(acts).scatter(-1, acts_topk.indices, acts_topk.values)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode using per-sample top-k (inference mode).

        Note: training uses batch-level top-k for uniform sparsity across the
        batch. At inference time this falls back to per-sample top-k, which
        fires exactly cfg['top_k'] features per token. For threshold-based
        inference that better replicates training behaviour, use InferenceSAE.
        """
        x_proc, _, _ = self.preprocess_input(x)
        return self._encode_preprocessed(x_proc)


class TopKSAE(BaseAutoencoder):
    def __init__(self, cfg):
        super().__init__(cfg)

    def forward(self, x):
        x, x_mean, x_std = self.preprocess_input(x)

        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc)
        acts_topk = torch.topk(acts, self.cfg["top_k"], dim=-1)
        acts_topk = torch.zeros_like(acts).scatter(
            -1, acts_topk.indices, acts_topk.values
        )
        x_reconstruct = acts_topk @ self.W_dec + self.b_dec

        self.update_inactive_features(acts_topk)
        output = self.get_loss_dict(x, x_reconstruct, acts, acts_topk, x_mean, x_std)
        return output

    def get_loss_dict(self, x, x_reconstruct, acts, acts_topk, x_mean, x_std):
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).mean()
        l1_norm = acts_topk.float().abs().sum(-1).mean()
        l1_loss = self.cfg["l1_coeff"] * l1_norm
        l0_norm = (acts_topk > 0).float().sum(-1).mean()
        aux_loss = self.get_auxiliary_loss(x, x_reconstruct, acts)
        loss = l2_loss + l1_loss + aux_loss
        num_dead_features = (
            self.num_batches_not_active > self.cfg["n_batches_to_dead"]
        ).sum()
        sae_out = self.postprocess_output(x_reconstruct, x_mean, x_std)
        output = {
            "sae_out": sae_out,
            "feature_acts": acts_topk,
            "num_dead_features": num_dead_features,
            "loss": loss,
            "l1_loss": l1_loss,
            "l2_loss": l2_loss,
            "l0_norm": l0_norm,
            "l1_norm": l1_norm,
            "aux_loss": aux_loss,
        }
        return output

    def get_auxiliary_loss(self, x, x_reconstruct, acts):
        dead_features = self.num_batches_not_active >= self.cfg["n_batches_to_dead"]
        if dead_features.sum() > 0:
            residual = x.float() - x_reconstruct.float()
            acts_topk_aux = torch.topk(
                acts[:, dead_features],
                min(self.cfg["top_k_aux"], dead_features.sum()),
                dim=-1,
            )
            acts_aux = torch.zeros_like(acts[:, dead_features]).scatter(
                -1, acts_topk_aux.indices, acts_topk_aux.values
            )
            x_reconstruct_aux = acts_aux @ self.W_dec[dead_features]
            l2_loss_aux = (
                self.cfg["aux_penalty"]
                * (x_reconstruct_aux.float() - residual.float()).pow(2).mean()
            )
            return l2_loss_aux
        else:
            return torch.tensor(0, dtype=x.dtype, device=x.device)

    def _encode_preprocessed(self, x: torch.Tensor) -> torch.Tensor:
        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc)
        acts_topk = torch.topk(acts, self.cfg["top_k"], dim=-1)
        return torch.zeros_like(acts).scatter(-1, acts_topk.indices, acts_topk.values)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_proc, _, _ = self.preprocess_input(x)
        return self._encode_preprocessed(x_proc)


class VanillaSAE(BaseAutoencoder):
    def __init__(self, cfg):
        super().__init__(cfg)

    def forward(self, x):
        x, x_mean, x_std = self.preprocess_input(x)
        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc + self.b_enc)
        x_reconstruct = acts @ self.W_dec + self.b_dec
        self.update_inactive_features(acts)
        output = self.get_loss_dict(x, x_reconstruct, acts, x_mean, x_std)
        return output

    def get_loss_dict(self, x, x_reconstruct, acts, x_mean, x_std):
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).mean()
        l1_norm = acts.float().abs().sum(-1).mean()
        l1_loss = self.cfg["l1_coeff"] * l1_norm
        l0_norm = (acts > 0).float().sum(-1).mean()
        loss = l2_loss + l1_loss
        num_dead_features = (
            self.num_batches_not_active > self.cfg["n_batches_to_dead"]
        ).sum()
        sae_out = self.postprocess_output(x_reconstruct, x_mean, x_std)
        output = {
            "sae_out": sae_out,
            "feature_acts": acts,
            "num_dead_features": num_dead_features,
            "loss": loss,
            "l1_loss": l1_loss,
            "l2_loss": l2_loss,
            "l0_norm": l0_norm,
            "l1_norm": l1_norm,
        }
        return output

    def _encode_preprocessed(self, x: torch.Tensor) -> torch.Tensor:
        x_cent = x - self.b_dec
        return F.relu(x_cent @ self.W_enc + self.b_enc)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_proc, _, _ = self.preprocess_input(x)
        return self._encode_preprocessed(x_proc)


class JumpReLUSAE(BaseAutoencoder):
    """
    JumpReLU SAE with L0 sparsity penalty using sigmoid STE.

    Uses x * H(x - θ) activation where:
    - Forward: hard threshold (preserves full magnitude, no shrinkage)
    - Backward for activations: straight-through (gradient flows through pre_acts)
    - Backward for threshold: sigmoid surrogate provides smooth gradients

    Sparsity modes:
    1. Fixed mode: Set l0_coeff directly, sparsity penalty = l0_coeff * L0
    2. Dynamic mode: Set target_l0, coefficient adapts to achieve target sparsity
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        init_threshold = cfg.get("jumprelu_init_threshold")
        if init_threshold is None:
            init_threshold = 0.001
        self.threshold = nn.Parameter(
            torch.full((cfg["dict_size"],), init_threshold, device=cfg["device"], dtype=cfg["dtype"])
        )
        self.bandwidth = cfg.get("bandwidth", 0.001)

        self.target_l0 = cfg.get("target_l0", None)
        if self.target_l0 is not None:
            self.l0_coeff = cfg.get("l0_coeff_start", 1e-5)
            self.l0_stability_threshold = cfg.get("l0_stability_threshold", 0.02)
            self.l0_stability_window = cfg.get("l0_stability_window", 500)
            self.l0_adjustment_factor = cfg.get("l0_adjustment_factor", 0.1)
            self.l0_ema = None
            self.l0_ema_decay = 0.99
            self.l0_history = []
            self.l0_update_steps = 0
            self.l0_is_stable = False
            self.l0_last_adjustment_step = 0
        else:
            self.l0_coeff = cfg.get("l0_coeff", cfg.get("l1_coeff", 0.0))

    def update_l0_coeff(self, current_l0):
        if self.target_l0 is None:
            return
        self.l0_update_steps += 1
        current_l0_val = current_l0.item() if torch.is_tensor(current_l0) else current_l0
        if self.l0_ema is None:
            self.l0_ema = current_l0_val
        self.l0_ema = self.l0_ema_decay * self.l0_ema + (1 - self.l0_ema_decay) * current_l0_val
        self.l0_history.append(self.l0_ema)
        if len(self.l0_history) > self.l0_stability_window:
            self.l0_history.pop(0)
        if len(self.l0_history) < self.l0_stability_window:
            return
        window_start = self.l0_history[0]
        window_end = self.l0_history[-1]
        relative_change = abs(window_end - window_start) / max(window_start, 1.0)
        self.l0_is_stable = relative_change < self.l0_stability_threshold
        min_steps_between_adjustments = self.l0_stability_window
        steps_since_adjustment = self.l0_update_steps - self.l0_last_adjustment_step
        if self.l0_is_stable and steps_since_adjustment >= min_steps_between_adjustments:
            error = self.l0_ema - self.target_l0
            relative_error = error / self.target_l0
            adjustment = self.l0_adjustment_factor * relative_error
            adjustment = max(-0.5, min(0.5, adjustment))
            self.l0_coeff *= (1 + adjustment)
            self.l0_coeff = max(1e-8, min(1.0, self.l0_coeff))
            self.l0_last_adjustment_step = self.l0_update_steps
            self.l0_history.clear()

    def forward(self, x, use_pre_enc_bias=False):
        x, x_mean, x_std = self.preprocess_input(x)
        if use_pre_enc_bias:
            x = x - self.b_dec
        pre_acts = F.relu(x @ self.W_enc + self.b_enc)
        mask = (pre_acts > self.threshold).float()
        acts = pre_acts * mask
        x_reconstruct = acts @ self.W_dec + self.b_dec
        self.update_inactive_features(acts)
        return self.get_loss_dict(x, x_reconstruct, pre_acts, acts, x_mean, x_std)

    def get_loss_dict(self, x, x_reconstruct, pre_acts, acts, x_mean, x_std):
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).mean()
        l0_surrogate = torch.sigmoid(
            (pre_acts - self.threshold) / self.bandwidth
        ).sum(dim=-1).mean()
        l0_actual = (pre_acts > self.threshold).float().sum(dim=-1).mean()
        self.update_l0_coeff(l0_actual)
        sparsity_loss = self.l0_coeff * l0_surrogate
        loss = l2_loss + sparsity_loss
        num_dead_features = (
            self.num_batches_not_active > self.cfg["n_batches_to_dead"]
        ).sum()
        sae_out = self.postprocess_output(x_reconstruct, x_mean, x_std)
        output = {
            "sae_out": sae_out,
            "feature_acts": acts,
            "num_dead_features": num_dead_features,
            "loss": loss,
            "l0_loss": sparsity_loss,
            "l0_coeff": self.l0_coeff,
            "l2_loss": l2_loss,
            "l0_norm": l0_actual,
            "l1_loss": sparsity_loss,
            "l1_norm": l0_actual,
        }
        if self.target_l0 is not None:
            output["target_l0"] = self.target_l0
            output["l0_ema"] = self.l0_ema
        return output

    def _encode_preprocessed(self, x: torch.Tensor) -> torch.Tensor:
        pre_acts = F.relu(x @ self.W_enc + self.b_enc)
        mask = (pre_acts > self.threshold).float()
        return pre_acts * mask

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_proc, _, _ = self.preprocess_input(x)
        return self._encode_preprocessed(x_proc)
