from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from tabm_bridge.config import BridgeConfig, TaskType


@dataclass
class BridgeOutput:
    """Forward outputs of TabM-Bridge (v1 and v2)."""

    z: Tensor                           # (B, d_z) consensus single-vector embedding
    pred_logits: Tensor                 # (B, d_out) distilled prediction head output
    uq_pred: Tensor                     # (B, 1) predicted standardized uncertainty proxy
    attn_weights: Tensor                # (B, k) sample-adaptive branch attention weights
    z_anchor: Tensor | None = None      # (B, d_z) normalized Procrustes anchor (Bridge-v2 only)


@dataclass
class BridgeTrainResult:
    """Summary of TabM-Bridge training."""

    best_epoch: int
    best_val_loss: float
    train_history: list[dict[str, float]]
    val_history: list[dict[str, float]]
    elapsed_sec: float
    stopped_due_to_nan: bool = False


def compute_consensus_gram(
    h: Tensor,
    center: bool = True,
    eps: float = 1e-8,
) -> Tensor:
    """Compute rotation-invariant (and shift-invariant when centered) consensus Gram matrix G_bar.

    For each submodel i in {1..k}:
      - When center=True (and B > 1): h_i <- h_i - mean_b(h_i), removing post-ReLU positive cone
        anisotropy and yielding strict invariance under h_i -> h_i Q_i + c_i for any Q_i in O(d), c_i in R^d.
      - G_i = H_hat_i @ H_hat_i^T in R^{B x B}, averaged across all k submodels via batched GEMM.
    """
    h_proc = h - h.mean(dim=0, keepdim=True) if (center and h.shape[0] > 1) else h
    h_unit = h_proc / torch.linalg.norm(h_proc, dim=-1, keepdim=True).clamp_min(eps)
    h_k = h_unit.transpose(0, 1)
    return torch.matmul(h_k, h_k.transpose(-1, -2)).mean(dim=0)


def _gram_mse(g_z: Tensor, g_bar: Tensor, mask_diag: bool = False) -> Tensor:
    """MSE between student and teacher Gram matrices, optionally excluding trivial diagonal 1s."""
    if not mask_diag:
        return F.mse_loss(g_z, g_bar)
    b = g_z.shape[0]
    if b <= 1:
        return g_z.new_zeros(())
    diff2 = (g_z - g_bar).pow(2)
    off_diag = ~torch.eye(b, dtype=torch.bool, device=g_z.device)
    return diff2[off_diag].mean()


def _gram_warmup_weight(epoch: int, alpha_gram: float, warmup_epochs: int) -> float:
    """Cosine ramp from ~0 to alpha_gram over `warmup_epochs` epochs."""
    if warmup_epochs <= 0 or epoch >= warmup_epochs:
        return float(alpha_gram)
    progress = float(epoch + 1) / float(warmup_epochs)
    return float(alpha_gram) * 0.5 * (1.0 - math.cos(math.pi * progress))


def _make_bridge_param_groups(module: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    """Disable weight decay for biases, LayerNorm scales, and scalar residual gates."""
    zero_wd_ids: set[int] = set()
    for _, sub in module.named_modules():
        for param_name, param in sub.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            if (
                param_name.endswith("bias")
                or param_name == "res_gate"
                or isinstance(sub, nn.LayerNorm)
            ):
                zero_wd_ids.add(id(param))

    decay_params: list[nn.Parameter] = []
    no_decay_params: list[nn.Parameter] = []
    for param in module.parameters():
        if not param.requires_grad:
            continue
        if id(param) in zero_wd_ids:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    groups: list[dict[str, Any]] = []
    if decay_params:
        groups.append({"params": decay_params, "weight_decay": weight_decay})
    if no_decay_params:
        groups.append({"params": no_decay_params, "weight_decay": 0.0})
    return groups


class TabMBridge(nn.Module):
    """Consensus bottleneck bridging k TabM submodel embeddings into a single vector z(x) (v1)."""

    def __init__(
        self,
        k: int,
        d_in: int,
        d_out: int,
        config: BridgeConfig | None = None,
    ) -> None:
        super().__init__()
        self.k = k
        self.d_in = d_in
        self.d_out = d_out
        self.config = config or BridgeConfig()

        d_z = self.config.d_z
        d_hidden = self.config.d_hidden

        # 1. Per-branch alignment projection
        self.align_mode = self.config.align_mode
        if self.align_mode == "low_rank":
            r = self.config.align_rank
            self.shared_proj = nn.Linear(d_in, d_z, bias=False)
            self.lora_a = nn.Parameter(torch.empty(k, d_in, r))
            self.lora_b = nn.Parameter(torch.zeros(k, r, d_z))
            self.lora_scale = 1.0 / math.sqrt(float(r))
            bound = 1.0 / math.sqrt(float(d_in))
            nn.init.uniform_(self.lora_a, -bound, bound)
            self.align_weight: nn.Parameter | None = None
        elif self.align_mode == "full":
            self.shared_proj = None
            self.lora_a = None
            self.lora_b = None
            self.lora_scale = 1.0
            self.align_weight = nn.Parameter(torch.empty(k, d_in, d_z))
            bound = 1.0 / math.sqrt(float(d_in))
            nn.init.uniform_(self.align_weight, -bound, bound)
        else:
            raise ValueError(f"Unsupported align_mode: {self.align_mode}")

        self.align_bias = nn.Parameter(torch.zeros(k, d_z))
        self.align_norm = nn.LayerNorm(d_z)
        self.align_dropout = nn.Dropout(self.config.dropout)

        # 2. Sample-adaptive attention scorer over aligned branches
        self.attn_scorer = nn.Sequential(
            nn.Linear(d_z, max(32, d_z // 4)),
            nn.GELU(),
            nn.Linear(max(32, d_z // 4), 1),
        )

        # 3. Residual bottleneck producing final single vector z(x)
        self.refine = nn.Sequential(
            nn.Linear(d_z, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(d_hidden, d_z),
        )
        self.out_norm = nn.LayerNorm(d_z)

        # 4. Auxiliary distillation heads
        self.pred_head = nn.Linear(d_z, d_out)
        self.uq_head = nn.Sequential(
            nn.Linear(d_z, max(32, d_z // 2)),
            nn.GELU(),
            nn.Linear(max(32, d_z // 2), 1),
        )

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def make_parameter_groups(self, weight_decay: float) -> list[dict[str, Any]]:
        return _make_bridge_param_groups(self, weight_decay)

    def forward(self, h: Tensor, z_proc: Tensor | None = None) -> BridgeOutput:
        if h.ndim != 3 or h.shape[1] != self.k or h.shape[2] != self.d_in:
            raise ValueError(
                f"Expected h of shape (B, {self.k}, {self.d_in}), got {tuple(h.shape)}."
            )

        h_in = h.detach() if self.config.stop_gradient else h
        h_k = h_in.transpose(0, 1)
        if self.align_mode == "low_rank":
            assert self.shared_proj is not None and self.lora_a is not None and self.lora_b is not None
            base = self.shared_proj(h_in)
            delta = torch.matmul(torch.matmul(h_k, self.lora_a), self.lora_b).transpose(0, 1)
            aligned = base + delta * self.lora_scale + self.align_bias.unsqueeze(0)
        else:
            assert self.align_weight is not None
            aligned = torch.matmul(h_k, self.align_weight).transpose(0, 1) + self.align_bias.unsqueeze(0)

        aligned = self.align_dropout(F.gelu(self.align_norm(aligned)))
        attn_logits = self.attn_scorer(aligned).squeeze(-1)
        attn_weights = torch.softmax(attn_logits, dim=-1)
        pooled = (attn_weights.unsqueeze(-1) * aligned).sum(dim=1)

        z = self.out_norm(pooled + self.refine(pooled))
        pred_logits = self.pred_head(z)
        uq_pred = self.uq_head(z)

        return BridgeOutput(
            z=z,
            pred_logits=pred_logits,
            uq_pred=uq_pred,
            attn_weights=attn_weights,
            z_anchor=None,
        )


class ProcrustesAnchoredResidualBridge(nn.Module):
    """Bridge-v2: Procrustes-Anchored Residual Bridge with Zero-Init & Learnable Gate.

    Formulation:
        z_anc(x) = ((z_proc(x) - mu_anc) @ P_anc) / scale_anc
        z(x)     = z_anc(x) + gamma * Delta_theta(H_detach(x))

    Properties:
      - Exact isometric initialization at Epoch 0: final layer of Delta_theta is zero-initialized,
        so z(x) == z_anc(x) identically before training, preserving AuxR2 and Procrustes geometry.
      - Non-zero learnable `res_gate` (init_gate=0.05, excluded from WD) prevents a dead saddle
        point at Epoch 0 while keeping early updates smoothly bounded.
      - No post-sum LayerNorm, avoiding distortion of Euclidean/cosine anchor geometry.
      - P_anc is exact identity I when d_z == d_in, or orthonormal top-d_z SVD basis fitted
        strictly on train z_proc when d_z < d_in (handles N < d_z via full_matrices fallback).
    """

    def __init__(
        self,
        k: int,
        d_in: int,
        d_out: int,
        config: BridgeConfig | None = None,
    ) -> None:
        super().__init__()
        self.k = k
        self.d_in = d_in
        self.d_out = d_out
        self.config = config or BridgeConfig(
            alpha_gram=10.0,
            alpha_pred=1.0,
            alpha_uq=0.15,
            alpha_recon=1.0,
            init_gate=0.05,
            gram_warmup_epochs=10,
            use_cosine_lr=True,
            mask_gram_diag=True,
            center_z_gram=True,
        )

        d_z = self.config.d_z
        d_hidden = self.config.d_hidden

        # Train-only Procrustes anchor projection buffers
        self.register_buffer("anchor_mean", torch.zeros(1, d_in))
        self.register_buffer("anchor_proj", torch.eye(d_in, d_z))
        self.register_buffer("anchor_scale", torch.tensor(1.0))
        self.register_buffer("anchor_fitted", torch.tensor(False, dtype=torch.bool))

        # Per-branch residual alignment
        self.align_mode = self.config.align_mode
        if self.align_mode == "low_rank":
            r = self.config.align_rank
            self.shared_proj = nn.Linear(d_in, d_z, bias=False)
            self.lora_a = nn.Parameter(torch.empty(k, d_in, r))
            self.lora_b = nn.Parameter(torch.zeros(k, r, d_z))
            self.lora_scale = 1.0 / math.sqrt(float(r))
            bound = 1.0 / math.sqrt(float(d_in))
            nn.init.uniform_(self.lora_a, -bound, bound)
            self.align_weight: nn.Parameter | None = None
        elif self.align_mode == "full":
            self.shared_proj = None
            self.lora_a = None
            self.lora_b = None
            self.lora_scale = 1.0
            self.align_weight = nn.Parameter(torch.empty(k, d_in, d_z))
            bound = 1.0 / math.sqrt(float(d_in))
            nn.init.uniform_(self.align_weight, -bound, bound)
        else:
            raise ValueError(f"Unsupported align_mode: {self.align_mode}")

        self.align_bias = nn.Parameter(torch.zeros(k, d_z))
        self.align_norm = nn.LayerNorm(d_z)
        self.align_dropout = nn.Dropout(self.config.dropout)

        # Sample-adaptive attention over branches (captures cross-head disagreement)
        self.attn_scorer = nn.Sequential(
            nn.Linear(d_z, max(32, d_z // 4)),
            nn.GELU(),
            nn.Linear(max(32, d_z // 4), 1),
        )

        # Zero-initialized Pre-Norm residual MLP
        self.refine_fc1 = nn.Linear(d_z, d_hidden)
        self.refine_norm = nn.LayerNorm(d_hidden)
        self.refine_drop = nn.Dropout(self.config.dropout)
        self.refine_fc2 = nn.Linear(d_hidden, d_z)
        nn.init.zeros_(self.refine_fc2.weight)
        nn.init.zeros_(self.refine_fc2.bias)

        # Learnable residual gate (excluded from weight decay)
        self.res_gate = nn.Parameter(torch.tensor(float(self.config.init_gate)))

        # Auxiliary distillation heads
        self.pred_head = nn.Linear(d_z, d_out)
        self.uq_head = nn.Sequential(
            nn.Linear(d_z, max(32, d_z // 2)),
            nn.GELU(),
            nn.Linear(max(32, d_z // 2), 1),
        )

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def make_parameter_groups(self, weight_decay: float) -> list[dict[str, Any]]:
        return _make_bridge_param_groups(self, weight_decay)

    @torch.no_grad()
    def set_anchor_stats(
        self,
        z_proc_train: np.ndarray,
        max_svd_samples: int = 16384,
        seed: int = 42,
    ) -> None:
        """Fit isotropic centering, orthonormal basis P_anc, and RMS scale strictly on train."""
        arr = np.asarray(z_proc_train, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != self.d_in:
            raise ValueError(f"Expected z_proc_train shape (N, {self.d_in}), got {arr.shape}.")

        d_z = self.config.d_z
        mu = arr.mean(axis=0, keepdims=True)
        centered = arr - mu

        if d_z == self.d_in:
            proj = np.eye(self.d_in, dtype=np.float64)
        elif d_z < self.d_in:
            if len(centered) > max_svd_samples:
                idx = np.random.default_rng(seed).choice(
                    len(centered), size=max_svd_samples, replace=False
                )
                sub = centered[idx]
            else:
                sub = centered
            need_full = sub.shape[0] < d_z
            _, _, vt = np.linalg.svd(sub, full_matrices=need_full)
            proj = vt[:d_z].T  # (d_in, d_z) with P^T P = I_{d_z}
        else:
            proj = np.zeros((self.d_in, d_z), dtype=np.float64)
            proj[:, : self.d_in] = np.eye(self.d_in, dtype=np.float64)

        projected = centered @ proj
        scale = float(np.sqrt(np.mean(projected * projected)))
        if not np.isfinite(scale) or scale < 1e-8:
            scale = 1.0

        self.anchor_mean.copy_(torch.from_numpy(mu.astype(np.float32)).to(self.anchor_mean.device))
        self.anchor_proj.copy_(torch.from_numpy(proj.astype(np.float32)).to(self.anchor_proj.device))
        self.anchor_scale.fill_(scale)
        self.anchor_fitted.fill_(True)

    def project_anchor(self, z_proc: Tensor) -> Tensor:
        return ((z_proc - self.anchor_mean) @ self.anchor_proj) / self.anchor_scale

    def forward(self, h: Tensor, z_proc: Tensor | None = None) -> BridgeOutput:
        if h.ndim != 3 or h.shape[1] != self.k or h.shape[2] != self.d_in:
            raise ValueError(
                f"Expected h of shape (B, {self.k}, {self.d_in}), got {tuple(h.shape)}."
            )
        if z_proc is None:
            raise ValueError("ProcrustesAnchoredResidualBridge.forward requires z_proc anchor tensor.")
        if z_proc.ndim != 2 or z_proc.shape[0] != h.shape[0] or z_proc.shape[1] != self.d_in:
            raise ValueError(
                f"Expected z_proc of shape ({h.shape[0]}, {self.d_in}), got {tuple(z_proc.shape)}."
            )

        h_in = h.detach() if self.config.stop_gradient else h
        z_proc_in = z_proc.detach() if self.config.stop_gradient else z_proc
        z_anc = self.project_anchor(z_proc_in)

        h_k = h_in.transpose(0, 1)
        if self.align_mode == "low_rank":
            assert self.shared_proj is not None and self.lora_a is not None and self.lora_b is not None
            base = self.shared_proj(h_in)
            delta_a = torch.matmul(torch.matmul(h_k, self.lora_a), self.lora_b).transpose(0, 1)
            aligned = base + delta_a * self.lora_scale + self.align_bias.unsqueeze(0)
        else:
            assert self.align_weight is not None
            aligned = torch.matmul(h_k, self.align_weight).transpose(0, 1) + self.align_bias.unsqueeze(0)

        aligned = self.align_dropout(F.gelu(self.align_norm(aligned)))
        attn_weights = torch.softmax(self.attn_scorer(aligned).squeeze(-1), dim=-1)
        pooled = (attn_weights.unsqueeze(-1) * aligned).sum(dim=1)

        delta_z = self.refine_fc2(self.refine_drop(F.gelu(self.refine_norm(self.refine_fc1(pooled)))))
        z = z_anc + self.res_gate * delta_z

        return BridgeOutput(
            z=z,
            pred_logits=self.pred_head(z),
            uq_pred=self.uq_head(z),
            attn_weights=attn_weights,
            z_anchor=z_anc,
        )


def compute_bridge_losses(
    out: BridgeOutput,
    h_teacher: Tensor,
    target_mean_pred: Tensor,
    target_uq: Tensor,
    task_type: TaskType,
    config: BridgeConfig,
    alpha_gram_override: float | None = None,
    eps: float = 1e-8,
) -> tuple[Tensor, dict[str, float]]:
    """Compute multi-term TabM-Bridge loss: Gram + Prediction + Uncertainty (+ Anchor Recon)."""
    alpha_gram = float(config.alpha_gram if alpha_gram_override is None else alpha_gram_override)

    # 1. Rotation-invariant (and shift-invariant when centered) consensus Gram distillation
    if alpha_gram > 0.0:
        with torch.no_grad():
            g_bar = compute_consensus_gram(
                h_teacher.detach(), center=config.center_gram, eps=eps
            )
        z_for_gram = (
            out.z - out.z.mean(dim=0, keepdim=True)
            if (config.center_z_gram and out.z.shape[0] > 1)
            else out.z
        )
        z_unit = z_for_gram / torch.linalg.norm(z_for_gram, dim=-1, keepdim=True).clamp_min(eps)
        g_z = z_unit @ z_unit.T
        loss_gram = _gram_mse(g_z, g_bar, mask_diag=config.mask_gram_diag)
    else:
        loss_gram = out.z.new_zeros(())

    # 2. Ensemble consensus prediction distillation
    if config.alpha_pred > 0.0:
        if task_type == "regression":
            loss_pred = F.mse_loss(out.pred_logits, target_mean_pred)
        elif task_type in ("binclass", "multiclass"):
            if target_mean_pred.shape[-1] < 2 or out.pred_logits.shape[-1] < 2:
                raise ValueError(
                    f"Classification distillation requires d_out >= 2, got target={tuple(target_mean_pred.shape)}, "
                    f"logits={tuple(out.pred_logits.shape)}."
                )
            target_probs = target_mean_pred.clamp_min(eps)
            target_probs = target_probs / target_probs.sum(dim=-1, keepdim=True)
            log_probs = F.log_softmax(out.pred_logits, dim=-1)
            loss_pred = -(target_probs * log_probs).sum(dim=-1).mean()
        else:
            raise ValueError(f"Unsupported task_type: {task_type}")
    else:
        loss_pred = out.z.new_zeros(())

    # 3. Epistemic uncertainty distillation
    if config.alpha_uq > 0.0:
        loss_uq = F.smooth_l1_loss(out.uq_pred, target_uq)
    else:
        loss_uq = out.z.new_zeros(())

    # 4. Isometric anchor trust-region reconstruction (Bridge-v2)
    if config.alpha_recon > 0.0 and out.z_anchor is not None:
        loss_recon = F.mse_loss(out.z, out.z_anchor.detach())
    else:
        loss_recon = out.z.new_zeros(())

    total = (
        alpha_gram * loss_gram
        + config.alpha_pred * loss_pred
        + config.alpha_uq * loss_uq
        + config.alpha_recon * loss_recon
    )
    parts = {
        "total": float(total.detach().item()),
        "gram": float(loss_gram.detach().item()),
        "pred": float(loss_pred.detach().item()),
        "uq": float(loss_uq.detach().item()),
        "recon": float(loss_recon.detach().item()),
    }
    return total, parts


class TabMBridgeTrainer:
    """Standalone leak-free, memory-efficient trainer for TabMBridge (v1) and Bridge-v2.

    Supports:
      - Native float16 CPU storage of H tensors (cast to float32 per minibatch).
      - Optional Procrustes anchor arrays (z_proc_train, z_proc_val) for Bridge-v2.
      - Cosine warmup on alpha_gram (val Early Stopping ALWAYS uses full target alpha_gram).
      - Optional CosineAnnealingLR schedule and full checkpoint serialization including scalers.
    """

    def __init__(
        self,
        bridge: TabMBridge | ProcrustesAnchoredResidualBridge,
        task_type: TaskType,
        device: str = "cuda",
        seed: int = 42,
    ) -> None:
        self.bridge = bridge
        self.task_type = task_type
        self.config = bridge.config
        self.device = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
        self.seed = seed
        self.bridge.to(self.device)
        self.requires_anchor = isinstance(self.bridge, ProcrustesAnchoredResidualBridge)

        # Strictly train-fitted scalers for regression target & uncertainty target
        self.pred_mean_: np.ndarray | None = None
        self.pred_std_: np.ndarray | None = None
        self.uq_mean_: float = 0.0
        self.uq_std_: float = 1.0

    def _fit_target_scalers(self, mean_pred_train: np.ndarray, uq_train: np.ndarray) -> None:
        if self.task_type == "regression":
            self.pred_mean_ = mean_pred_train.mean(axis=0, keepdims=True).astype(np.float32)
            std = mean_pred_train.std(axis=0, keepdims=True, ddof=0).astype(np.float32)
            self.pred_std_ = np.where(std < 1e-6, 1.0, std).astype(np.float32)

        log_uq = np.log1p(np.clip(uq_train.astype(np.float32), 0.0, None))
        self.uq_mean_ = float(log_uq.mean())
        uq_std = float(log_uq.std(ddof=0))
        self.uq_std_ = uq_std if uq_std >= 1e-6 else 1.0

    def _transform_targets(
        self,
        mean_pred: np.ndarray,
        uq: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        mp = np.asarray(mean_pred, dtype=np.float32)
        if mp.ndim == 1:
            mp = mp[:, None]
        if self.task_type == "regression":
            assert self.pred_mean_ is not None and self.pred_std_ is not None
            mp = (mp - self.pred_mean_) / self.pred_std_

        uq_arr = np.asarray(uq, dtype=np.float32)
        if uq_arr.ndim == 1:
            uq_arr = uq_arr[:, None]
        uq_scaled = (np.log1p(np.clip(uq_arr, 0.0, None)) - self.uq_mean_) / self.uq_std_
        return mp, uq_scaled.astype(np.float32)

    @torch.no_grad()
    def _eval_epoch(
        self,
        h: Tensor,
        mp: Tensor,
        uq: Tensor,
        batch_size: int,
        z_proc: Tensor | None = None,
    ) -> tuple[dict[str, float], bool]:
        self.bridge.eval()
        n = h.shape[0]
        acc = {"total": 0.0, "gram": 0.0, "pred": 0.0, "uq": 0.0, "recon": 0.0}
        count = 0
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            hb = h[start:end].to(self.device, dtype=torch.float32)
            mpb = mp[start:end].to(self.device, dtype=torch.float32)
            uqb = uq[start:end].to(self.device, dtype=torch.float32)
            zpb = None if z_proc is None else z_proc[start:end].to(self.device, dtype=torch.float32)
            out = self.bridge(hb, z_proc=zpb)
            # Always evaluate validation using FULL target alpha_gram (never ramped warmup weight)
            loss, parts = compute_bridge_losses(
                out,
                hb,
                mpb,
                uqb,
                self.task_type,
                self.config,
                alpha_gram_override=self.config.alpha_gram,
            )
            if not torch.isfinite(loss):
                inf_parts = {k: float("inf") for k in acc}
                return inf_parts, False
            bs = end - start
            for key in acc:
                acc[key] += parts[key] * bs
            count += bs
        return {key: val / max(count, 1) for key, val in acc.items()}, True

    def fit(
        self,
        h_train: np.ndarray,
        mean_pred_train: np.ndarray,
        uq_train: np.ndarray,
        h_val: np.ndarray,
        mean_pred_val: np.ndarray,
        uq_val: np.ndarray,
        z_proc_train: np.ndarray | None = None,
        z_proc_val: np.ndarray | None = None,
    ) -> BridgeTrainResult:
        t0 = time.perf_counter()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        if self.requires_anchor:
            if z_proc_train is None or z_proc_val is None:
                raise ValueError("Bridge-v2 requires z_proc_train and z_proc_val in fit().")
            assert isinstance(self.bridge, ProcrustesAnchoredResidualBridge)
            self.bridge.set_anchor_stats(z_proc_train, seed=self.seed)

        self._fit_target_scalers(
            np.asarray(mean_pred_train, dtype=np.float32).reshape(len(h_train), -1),
            np.asarray(uq_train, dtype=np.float32).reshape(len(h_train), -1),
        )
        mp_tr_np, uq_tr_np = self._transform_targets(mean_pred_train, uq_train)
        mp_va_np, uq_va_np = self._transform_targets(mean_pred_val, uq_val)

        h_tr_t = torch.from_numpy(np.asarray(h_train))
        mp_tr_t = torch.from_numpy(mp_tr_np)
        uq_tr_t = torch.from_numpy(uq_tr_np)
        zp_tr_t = None if z_proc_train is None else torch.from_numpy(np.asarray(z_proc_train, dtype=np.float32))

        h_va_t = torch.from_numpy(np.asarray(h_val))
        mp_va_t = torch.from_numpy(mp_va_np)
        uq_va_t = torch.from_numpy(uq_va_np)
        zp_va_t = None if z_proc_val is None else torch.from_numpy(np.asarray(z_proc_val, dtype=np.float32))

        optimizer = torch.optim.AdamW(
            self.bridge.make_parameter_groups(self.config.weight_decay),
            lr=self.config.lr,
        )
        scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(self.config.epochs, 1), eta_min=self.config.lr * 0.05
            )
            if self.config.use_cosine_lr
            else None
        )

        best_val_loss = float("inf")
        best_epoch = -1
        best_state: dict[str, Tensor] = {
            name: tensor.detach().cpu().clone()
            for name, tensor in self.bridge.state_dict().items()
        }
        bad_epochs = 0
        train_history: list[dict[str, float]] = []
        val_history: list[dict[str, float]] = []
        stopped_due_to_nan = False

        n_train = h_tr_t.shape[0]
        bs = self.config.batch_size
        gen = torch.Generator().manual_seed(self.seed)
        eval_bs = bs if self.config.mask_gram_diag else max(bs * 4, 1024)

        for epoch in range(self.config.epochs):
            self.bridge.train()
            perm = torch.randperm(n_train, generator=gen)
            tr_acc = {"total": 0.0, "gram": 0.0, "pred": 0.0, "uq": 0.0, "recon": 0.0}
            seen = 0
            cur_alpha_gram = _gram_warmup_weight(
                epoch, self.config.alpha_gram, self.config.gram_warmup_epochs
            )

            for start in range(0, n_train, bs):
                idx = perm[start : start + bs]
                hb = h_tr_t[idx].to(self.device, dtype=torch.float32)
                mpb = mp_tr_t[idx].to(self.device, dtype=torch.float32)
                uqb = uq_tr_t[idx].to(self.device, dtype=torch.float32)
                zpb = None if zp_tr_t is None else zp_tr_t[idx].to(self.device, dtype=torch.float32)

                optimizer.zero_grad(set_to_none=True)
                out = self.bridge(hb, z_proc=zpb)
                loss, parts = compute_bridge_losses(
                    out,
                    hb,
                    mpb,
                    uqb,
                    self.task_type,
                    self.config,
                    alpha_gram_override=cur_alpha_gram,
                )
                if not torch.isfinite(loss):
                    stopped_due_to_nan = True
                    break

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.bridge.parameters(), max_norm=1.0)
                optimizer.step()

                cur_bs = len(idx)
                for key in tr_acc:
                    tr_acc[key] += parts[key] * cur_bs
                seen += cur_bs

            if stopped_due_to_nan:
                break

            if scheduler is not None:
                scheduler.step()

            train_metrics = {key: val / max(seen, 1) for key, val in tr_acc.items()}
            val_metrics, is_valid = self._eval_epoch(
                h_va_t, mp_va_t, uq_va_t, batch_size=eval_bs, z_proc=zp_va_t
            )
            if not is_valid:
                stopped_due_to_nan = True
                break

            train_history.append(train_metrics)
            val_history.append(val_metrics)

            if val_metrics["total"] < best_val_loss - 1e-6:
                best_val_loss = val_metrics["total"]
                best_epoch = epoch
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in self.bridge.state_dict().items()
                }
                bad_epochs = 0
            elif epoch >= self.config.gram_warmup_epochs:
                # Do not trigger early stopping before cosine warmup completes
                bad_epochs += 1
                if bad_epochs >= self.config.patience:
                    break

        self.bridge.load_state_dict(best_state)
        self.bridge.to(self.device)
        return BridgeTrainResult(
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
            train_history=train_history,
            val_history=val_history,
            elapsed_sec=time.perf_counter() - t0,
            stopped_due_to_nan=stopped_due_to_nan,
        )

    def save_checkpoint(self, path: str | Path) -> None:
        """Persist bridge weights, config, and train-fitted target scalers."""
        dst = Path(path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "bridge_class": type(self.bridge).__name__,
                "bridge_state_dict": {
                    n: p.detach().cpu() for n, p in self.bridge.state_dict().items()
                },
                "bridge_config": asdict(self.config),
                "k": self.bridge.k,
                "d_in": self.bridge.d_in,
                "d_out": self.bridge.d_out,
                "task_type": self.task_type,
                "scalers": {
                    "pred_mean": None if self.pred_mean_ is None else self.pred_mean_.tolist(),
                    "pred_std": None if self.pred_std_ is None else self.pred_std_.tolist(),
                    "uq_mean": self.uq_mean_,
                    "uq_std": self.uq_std_,
                },
            },
            dst,
        )

    @classmethod
    def load_checkpoint(cls, path: str | Path, device: str = "cuda") -> TabMBridgeTrainer:
        """Reload a saved TabMBridge / ProcrustesAnchoredResidualBridge trainer with scalers."""
        ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
        cfg = BridgeConfig(**ckpt["bridge_config"])
        b_cls = (
            ProcrustesAnchoredResidualBridge
            if ckpt.get("bridge_class") == "ProcrustesAnchoredResidualBridge"
            else TabMBridge
        )
        bridge = b_cls(k=int(ckpt["k"]), d_in=int(ckpt["d_in"]), d_out=int(ckpt["d_out"]), config=cfg)
        bridge.load_state_dict(ckpt["bridge_state_dict"], strict=True)
        trainer = cls(bridge=bridge, task_type=ckpt["task_type"], device=device)
        sc = ckpt.get("scalers") or {}
        if sc.get("pred_mean") is not None:
            trainer.pred_mean_ = np.asarray(sc["pred_mean"], dtype=np.float32)
        if sc.get("pred_std") is not None:
            trainer.pred_std_ = np.asarray(sc["pred_std"], dtype=np.float32)
        trainer.uq_mean_ = float(sc.get("uq_mean", 0.0))
        trainer.uq_std_ = float(sc.get("uq_std", 1.0))
        return trainer

    @torch.no_grad()
    def encode(
        self,
        h: np.ndarray,
        batch_size: int = 1024,
        return_aux: bool = False,
        z_proc: np.ndarray | None = None,
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Project (N, k, d_in) representations to single-vector z(x) of shape (N, d_z)."""
        if self.requires_anchor:
            if z_proc is None:
                raise ValueError("Bridge-v2 requires z_proc in encode().")
            assert isinstance(self.bridge, ProcrustesAnchoredResidualBridge)
            if not bool(self.bridge.anchor_fitted.item()):
                raise RuntimeError(
                    "Bridge-v2 anchor stats must be fitted (via fit() or set_anchor_stats()) before encode()."
                )

        self.bridge.eval()
        h_t = torch.from_numpy(np.asarray(h))
        zp_t = None if z_proc is None else torch.from_numpy(np.asarray(z_proc, dtype=np.float32))
        zs: list[np.ndarray] = []
        preds: list[np.ndarray] = []
        uqs: list[np.ndarray] = []
        attns: list[np.ndarray] = []

        for start in range(0, len(h_t), batch_size):
            end = min(start + batch_size, len(h_t))
            hb = h_t[start:end].to(self.device, dtype=torch.float32)
            zpb = None if zp_t is None else zp_t[start:end].to(self.device, dtype=torch.float32)
            out = self.bridge(hb, z_proc=zpb)
            zs.append(out.z.cpu().numpy())
            if return_aux:
                if self.task_type == "regression":
                    p_np = out.pred_logits.cpu().numpy()
                    if self.pred_mean_ is not None and self.pred_std_ is not None:
                        p_np = p_np * self.pred_std_ + self.pred_mean_
                else:
                    p_np = torch.softmax(out.pred_logits, dim=-1).cpu().numpy()
                u_scaled = out.uq_pred.cpu().numpy()
                u_orig = np.expm1(np.clip(u_scaled * self.uq_std_ + self.uq_mean_, -20.0, 20.0))
                preds.append(p_np)
                uqs.append(np.clip(u_orig, 0.0, None))
                attns.append(out.attn_weights.cpu().numpy())

        z_all = np.concatenate(zs, axis=0).astype(np.float32)
        if not return_aux:
            return z_all
        return {
            "z": z_all,
            "pred": np.concatenate(preds, axis=0).astype(np.float32),
            "uq": np.concatenate(uqs, axis=0).astype(np.float32),
            "attn_weights": np.concatenate(attns, axis=0).astype(np.float32),
        }