from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from tabm_bridge.config import BridgeConfig, TaskType


@dataclass
class BridgeOutput:
    """Forward outputs of TabM-Bridge."""

    z: Tensor                   # (B, d_z) consensus single-vector embedding
    pred_logits: Tensor         # (B, d_out) distilled prediction head output
    uq_pred: Tensor             # (B, 1) predicted standardized uncertainty proxy
    attn_weights: Tensor        # (B, k) sample-adaptive branch attention weights


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
    # (B, k, d) -> (k, B, d) @ (k, d, B) -> (k, B, B) -> mean over k -> (B, B)
    h_k = h_unit.transpose(0, 1)
    return torch.matmul(h_k, h_k.transpose(-1, -2)).mean(dim=0)


class TabMBridge(nn.Module):
    """Consensus bottleneck bridging k TabM submodel embeddings into a single vector z(x).

    Architecture:
      1. Stop-Gradient Shield (detach() when config.stop_gradient=True to prevent Learner Collusion).
      2. Parameter-efficient per-branch alignment:
           - 'low_rank' (default): W_i = W_shared + (1 / sqrt(r)) * A_i @ B_i + b_i
             (shared projection + LoRA-style zero-initialized branch adapter per Hu et al., ICLR 2022).
           - 'full': dense per-branch matrix W_i in R^{d_in x d_z} (for ablation studies).
      3. Sample-adaptive attention pooling over the k aligned branches.
      4. Residual bottleneck MLP producing single consensus vector z(x) in R^{d_z}.
      5. Auxiliary distillation heads for ensemble prediction (y_ens) and epistemic uncertainty (uq).
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
        """Exact count of trainable parameters (for transparent reporting)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def make_parameter_groups(self, weight_decay: float) -> list[dict[str, Any]]:
        """Disable weight decay for biases and LayerNorm scales (Loshchilov & Hutter, 2019)."""
        zero_wd_ids: set[int] = set()
        for _, module in self.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                if not param.requires_grad:
                    continue
                if param_name.endswith("bias") or isinstance(module, nn.LayerNorm):
                    zero_wd_ids.add(id(param))

        decay_params: list[nn.Parameter] = []
        no_decay_params: list[nn.Parameter] = []
        for param in self.parameters():
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

    def forward(self, h: Tensor) -> BridgeOutput:
        if h.ndim != 3 or h.shape[1] != self.k or h.shape[2] != self.d_in:
            raise ValueError(
                f"Expected h of shape (B, {self.k}, {self.d_in}), got {tuple(h.shape)}."
            )

        # Stop-Gradient shield (prevents Learner Collusion per TabM Appendix A.4)
        h_in = h.detach() if self.config.stop_gradient else h

        # Vectorized per-branch alignment using ONNX/cuBLAS-friendly batched matmul
        h_k = h_in.transpose(0, 1)  # (k, B, d_in)
        if self.align_mode == "low_rank":
            assert self.shared_proj is not None and self.lora_a is not None and self.lora_b is not None
            base = self.shared_proj(h_in)  # (B, k, d_z)
            delta = torch.matmul(torch.matmul(h_k, self.lora_a), self.lora_b).transpose(0, 1)
            aligned = base + delta * self.lora_scale + self.align_bias.unsqueeze(0)
        else:
            assert self.align_weight is not None
            aligned = torch.matmul(h_k, self.align_weight).transpose(0, 1) + self.align_bias.unsqueeze(0)

        aligned = self.align_dropout(F.gelu(self.align_norm(aligned)))

        # Sample-adaptive attention pooling across k branches
        attn_logits = self.attn_scorer(aligned).squeeze(-1)  # (B, k)
        attn_weights = torch.softmax(attn_logits, dim=-1)    # (B, k)
        pooled = (attn_weights.unsqueeze(-1) * aligned).sum(dim=1)  # (B, d_z)

        # Residual refinement -> single consensus embedding z(x)
        z = self.out_norm(pooled + self.refine(pooled))

        pred_logits = self.pred_head(z)
        uq_pred = self.uq_head(z)

        return BridgeOutput(
            z=z,
            pred_logits=pred_logits,
            uq_pred=uq_pred,
            attn_weights=attn_weights,
        )


def compute_bridge_losses(
    out: BridgeOutput,
    h_teacher: Tensor,
    target_mean_pred: Tensor,
    target_uq: Tensor,
    task_type: TaskType,
    config: BridgeConfig,
    eps: float = 1e-8,
) -> tuple[Tensor, dict[str, float]]:
    """Compute the three-term TabM-Bridge loss: Gram + Prediction + Uncertainty."""
    # 1. Rotation-invariant (and shift-invariant when centered) consensus Gram distillation
    if config.alpha_gram > 0.0:
        with torch.no_grad():
            g_bar = compute_consensus_gram(
                h_teacher.detach(), center=config.center_gram, eps=eps
            )
        z_unit = out.z / torch.linalg.norm(out.z, dim=-1, keepdim=True).clamp_min(eps)
        g_z = z_unit @ z_unit.T
        loss_gram = F.mse_loss(g_z, g_bar)
    else:
        loss_gram = out.z.new_zeros(())

    # 2. Ensemble consensus prediction distillation
    if config.alpha_pred > 0.0:
        if task_type == "regression":
            loss_pred = F.mse_loss(out.pred_logits, target_mean_pred)
        elif task_type in ("binclass", "multiclass"):
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

    total = (
        config.alpha_gram * loss_gram
        + config.alpha_pred * loss_pred
        + config.alpha_uq * loss_uq
    )
    parts = {
        "total": float(total.detach().item()),
        "gram": float(loss_gram.detach().item()),
        "pred": float(loss_pred.detach().item()),
        "uq": float(loss_uq.detach().item()),
    }
    return total, parts


class TabMBridgeTrainer:
    """Standalone leak-free, memory-efficient trainer for TabM-Bridge on G0 representations.

    Keeps large H arrays in their native storage dtype (e.g., float16) on CPU and casts
    to float32 strictly per minibatch on the target device.
    """

    def __init__(
        self,
        bridge: TabMBridge,
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
    ) -> tuple[dict[str, float], bool]:
        self.bridge.eval()
        n = h.shape[0]
        acc = {"total": 0.0, "gram": 0.0, "pred": 0.0, "uq": 0.0}
        count = 0
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            hb = h[start:end].to(self.device, dtype=torch.float32)
            mpb = mp[start:end].to(self.device, dtype=torch.float32)
            uqb = uq[start:end].to(self.device, dtype=torch.float32)
            out = self.bridge(hb)
            loss, parts = compute_bridge_losses(out, hb, mpb, uqb, self.task_type, self.config)
            if not torch.isfinite(loss):
                return {"total": float("inf"), "gram": float("inf"), "pred": float("inf"), "uq": float("inf")}, False
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
    ) -> BridgeTrainResult:
        t0 = time.perf_counter()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self._fit_target_scalers(
            np.asarray(mean_pred_train, dtype=np.float32).reshape(len(h_train), -1),
            np.asarray(uq_train, dtype=np.float32).reshape(len(h_train), -1),
        )
        mp_tr_np, uq_tr_np = self._transform_targets(mean_pred_train, uq_train)
        mp_va_np, uq_va_np = self._transform_targets(mean_pred_val, uq_val)

        # Keep H in native dtype (e.g., float16) on CPU; cast per minibatch to save RAM
        h_tr_t = torch.from_numpy(np.asarray(h_train))
        mp_tr_t = torch.from_numpy(mp_tr_np)
        uq_tr_t = torch.from_numpy(uq_tr_np)

        h_va_t = torch.from_numpy(np.asarray(h_val))
        mp_va_t = torch.from_numpy(mp_va_np)
        uq_va_t = torch.from_numpy(uq_va_np)

        optimizer = torch.optim.AdamW(
            self.bridge.make_parameter_groups(self.config.weight_decay),
            lr=self.config.lr,
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

        for epoch in range(self.config.epochs):
            self.bridge.train()
            perm = torch.randperm(n_train, generator=gen)
            tr_acc = {"total": 0.0, "gram": 0.0, "pred": 0.0, "uq": 0.0}
            seen = 0

            for start in range(0, n_train, bs):
                idx = perm[start : start + bs]
                hb = h_tr_t[idx].to(self.device, dtype=torch.float32)
                mpb = mp_tr_t[idx].to(self.device, dtype=torch.float32)
                uqb = uq_tr_t[idx].to(self.device, dtype=torch.float32)

                optimizer.zero_grad(set_to_none=True)
                out = self.bridge(hb)
                loss, parts = compute_bridge_losses(
                    out, hb, mpb, uqb, self.task_type, self.config
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

            train_metrics = {key: val / max(seen, 1) for key, val in tr_acc.items()}
            val_metrics, is_valid = self._eval_epoch(
                h_va_t, mp_va_t, uq_va_t, batch_size=max(bs * 4, 1024)
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
            else:
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

    @torch.no_grad()
    def encode(
        self,
        h: np.ndarray,
        batch_size: int = 1024,
        return_aux: bool = False,
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Project (N, k, d_in) representations to single-vector z(x) of shape (N, d_z)."""
        self.bridge.eval()
        h_t = torch.from_numpy(np.asarray(h))
        zs: list[np.ndarray] = []
        preds: list[np.ndarray] = []
        uqs: list[np.ndarray] = []
        attns: list[np.ndarray] = []

        for start in range(0, len(h_t), batch_size):
            hb = h_t[start : start + batch_size].to(self.device, dtype=torch.float32)
            out = self.bridge(hb)
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