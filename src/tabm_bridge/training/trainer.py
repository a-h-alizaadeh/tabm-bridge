from __future__ import annotations

import copy
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tabm_bridge.config import TrainConfig
from tabm_bridge.data.preprocessing import ProcessedDataset, ProcessedSplit, TargetStats
from tabm_bridge.evaluation.metrics import compute_metrics
from tabm_bridge.model import TabMModel, TabMOutput
from tabm_bridge.training.losses import compute_independent_loss


def configure_reproducibility(
    seed: int,
    deterministic: bool = True,
    single_thread: bool = False,
) -> None:
    """Mirror official paper/lib/util.py reproducibility settings."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    if single_thread:
        torch.set_num_threads(1)


@dataclass
class TrainResult:
    """Summary of a full training run including best checkpoint and metrics."""

    best_epoch: int
    best_val_score: float
    metrics: dict[str, dict[str, float]]
    history: list[dict[str, float]]
    elapsed_sec: float
    stopped_due_to_nan: bool = False


@dataclass
class RepresentationBundle:
    """Extracted penultimate embeddings H and predictions for G1 (Procrustes / TabM-Bridge)."""

    embeddings: np.ndarray            # (N, k, d_block)
    per_head_preds: np.ndarray        # (N, k, d_out) original scale (reg) or probs (cls)
    mean_pred: np.ndarray             # (N, d_out) or (N, 1)
    epistemic_uncertainty: np.ndarray # (N, d_out) std or (N, 1) MI
    aleatoric_proxy: np.ndarray | None
    y_raw: np.ndarray


class TabMTrainer:
    """Production training engine aligned with official paper/bin/model.py & example.ipynb.

    Features:
    - Official AdamW with `model.make_parameter_groups(weight_decay)`.
    - Safe bfloat16-only AMP (never activates untested float16 on T4 GPUs).
    - Safe torch.compile handling (no reduce-overhead mode; no_grad eval context).
    - Automatic eval_batch_size halving on CUDA OOM and immediate NaN/Inf guard.
    - Built-in representation extractor for Scenario 1 (G1).
    """

    def __init__(
        self,
        model: TabMModel,
        config: TrainConfig,
        target_stats: TargetStats | None = None,
        primary_score: str | None = None,
    ) -> None:
        self.config = config
        self.target_stats = target_stats or TargetStats()
        self.primary_score = primary_score

        use_cuda = config.device.startswith("cuda") and torch.cuda.is_available()
        self.device = torch.device(config.device if use_cuda else "cpu")
        self.model = model.to(self.device)

        # Official AMP rule: bfloat16 ONLY if hardware supports it (commented out float16 in repo)
        self.amp_enabled = (
            config.use_amp
            and self.device.type == "cuda"
            and torch.cuda.is_bf16_supported()
        )
        self.amp_dtype = torch.bfloat16

        if config.compile_model:
            # Official note: avoid mode="reduce-overhead" and use no_grad for eval
            self._exec_model: torch.nn.Module = torch.compile(self.model)
            self._eval_context = torch.no_grad
            self._eval_batch_size = 2048
        else:
            self._exec_model = self.model
            self._eval_context = torch.inference_mode
            self._eval_batch_size = 32768

        self.optimizer = torch.optim.AdamW(
            self.model.make_parameter_groups(config.weight_decay),
            lr=config.lr,
        )

    def _split_to_device(
        self, split: ProcessedSplit
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
        x_num = (
            torch.as_tensor(split.x_num, dtype=torch.float32, device=self.device)
            if split.x_num is not None
            else None
        )
        x_cat = (
            torch.as_tensor(split.x_cat, dtype=torch.long, device=self.device)
            if split.x_cat is not None
            else None
        )
        y_dtype = torch.float32 if self.config.task_type == "regression" else torch.long
        y = torch.as_tensor(split.y, dtype=y_dtype, device=self.device)
        return x_num, x_cat, y

    def _forward_batched_with_oom_guard(
        self,
        split: ProcessedSplit,
        return_embeddings: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run batched inference over a split, automatically halving eval_batch_size on CUDA OOM."""
        self.model.eval()
        x_num_all, x_cat_all, _ = self._split_to_device(split)
        n_samples = split.y.shape[0]

        while True:
            try:
                preds_chunks: list[torch.Tensor] = []
                emb_chunks: list[torch.Tensor] = []
                with self._eval_context():
                    for start in range(0, n_samples, self._eval_batch_size):
                        end = min(start + self._eval_batch_size, n_samples)
                        xb_num = None if x_num_all is None else x_num_all[start:end]
                        xb_cat = None if x_cat_all is None else x_cat_all[start:end]
                        with torch.amp.autocast(
                            device_type=self.device.type,
                            dtype=self.amp_dtype,
                            enabled=self.amp_enabled,
                        ):
                            out = self._exec_model(
                                x_num=xb_num,
                                x_cat=xb_cat,
                                return_embeddings=return_embeddings,
                            )
                        if return_embeddings:
                            assert isinstance(out, TabMOutput)
                            preds_chunks.append(out.preds.float().cpu())
                            emb_chunks.append(out.embeddings.float().cpu())
                        else:
                            assert isinstance(out, torch.Tensor)
                            preds_chunks.append(out.float().cpu())

                preds = torch.cat(preds_chunks, dim=0)
                embs = torch.cat(emb_chunks, dim=0) if return_embeddings else None
                return preds, embs
            except RuntimeError as exc:
                is_oom = isinstance(exc, torch.cuda.OutOfMemoryError) or (
                    "out of memory" in str(exc).lower()
                )
                if is_oom and self._eval_batch_size > 1:
                    self._eval_batch_size = max(1, self._eval_batch_size // 2)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                raise

    def _aggregate_predictions(
        self, raw_preds: torch.Tensor
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        """Convert raw (N, k, d_out) outputs to aggregated predictions and UQ stats."""
        eps = 1e-8
        if self.config.task_type == "regression":
            # Inverse-transform BEFORE averaging across k heads (paper/bin/model.py)
            per_head = raw_preds * self.target_stats.std + self.target_stats.mean
            mean_pred = per_head.mean(dim=1)
            epi_std = per_head.std(dim=1, unbiased=False)
            return (
                mean_pred.numpy(),
                per_head.numpy(),
                epi_std.numpy(),
                None,
            )

        probs = torch.softmax(raw_preds, dim=-1)
        if self.config.classification_agg == "probs":
            mean_probs = probs.mean(dim=1)
        else:
            mean_probs = torch.softmax(raw_preds.mean(dim=1), dim=-1)

        total_ent = -(mean_probs * mean_probs.clamp_min(eps).log()).sum(dim=-1, keepdim=True)
        head_ent = -(probs * probs.clamp_min(eps).log()).sum(dim=-1)
        exp_ent = head_ent.mean(dim=1, keepdim=True)
        mutual_info = (total_ent - exp_ent).clamp_min(0.0)

        return (
            mean_probs.numpy(),
            probs.numpy(),
            mutual_info.numpy(),
            exp_ent.numpy(),
        )

    def evaluate_split(self, split: ProcessedSplit) -> tuple[dict[str, float], np.ndarray, bool]:
        """Evaluate model on a split; returns (metrics_dict, mean_preds, is_valid)."""
        raw_preds, _ = self._forward_batched_with_oom_guard(split, return_embeddings=False)
        if not torch.isfinite(raw_preds).all():
            return {"score": -float("inf")}, np.empty(0), False

        mean_pred, _, _, _ = self._aggregate_predictions(raw_preds)
        if not np.isfinite(mean_pred).all():
            return {"score": -float("inf")}, mean_pred, False

        metrics = compute_metrics(
            y_true=split.y_raw,
            y_pred=mean_pred,
            task_type=self.config.task_type,
            primary_score=self.primary_score,
        )
        return metrics, mean_pred, True

    def extract_representations(self, split: ProcessedSplit) -> RepresentationBundle:
        """Extract penultimate embeddings H: (N, k, d_block) and UQ stats for G1."""
        raw_preds, embs = self._forward_batched_with_oom_guard(split, return_embeddings=True)
        assert embs is not None
        mean_pred, per_head, epi_uq, alea_proxy = self._aggregate_predictions(raw_preds)
        return RepresentationBundle(
            embeddings=embs.numpy(),
            per_head_preds=per_head,
            mean_pred=mean_pred,
            epistemic_uncertainty=epi_uq,
            aleatoric_proxy=alea_proxy,
            y_raw=split.y_raw.copy(),
        )

    def fit(
        self,
        dataset: ProcessedDataset,
        save_dir: str | Path | None = None,
    ) -> TrainResult:
        """Train TabM with official batching, early stopping, NaN guard, and checkpointing."""
        t0 = time.perf_counter()
        configure_reproducibility(self.config.seed)

        train_split = dataset.splits["train"]
        val_split = dataset.splits["val"]
        test_split = dataset.splits["test"]

        x_num_tr, x_cat_tr, y_tr = self._split_to_device(train_split)
        n_train = int(y_tr.shape[0])
        k = self.model.k
        batch_size = self.config.batch_size
        gen = torch.Generator(device=self.device).manual_seed(self.config.seed)

        best_val_score = -float("inf")
        best_epoch = -1
        best_state: dict[str, torch.Tensor] = {
            name: tensor.detach().cpu().clone()
            for name, tensor in self.model.state_dict().items()
        }
        epochs_no_improve = 0
        history: list[dict[str, float]] = []
        stopped_due_to_nan = False

        for epoch in range(self.config.max_epochs):
            self.model.train()
            running_loss = 0.0
            n_batches = 0

            if self.config.share_training_batches or k == 1:
                batch_indices = torch.randperm(
                    n_train, generator=gen, device=self.device
                ).split(batch_size)
            else:
                # Independent k-permutations per epoch (example.ipynb & bin/model.py)
                batch_indices = (
                    torch.rand((n_train, k), generator=gen, device=self.device)
                    .argsort(dim=0)
                    .split(batch_size, dim=0)
                )

            for idx in batch_indices:
                self.optimizer.zero_grad(set_to_none=True)
                xb_num = None if x_num_tr is None else x_num_tr[idx]
                xb_cat = None if x_cat_tr is None else x_cat_tr[idx]
                yb = y_tr[idx]

                with torch.amp.autocast(
                    device_type=self.device.type,
                    dtype=self.amp_dtype,
                    enabled=self.amp_enabled,
                ):
                    preds = self._exec_model(x_num=xb_num, x_cat=xb_cat)
                    loss = compute_independent_loss(preds, yb, self.config.task_type)

                if not torch.isfinite(loss):
                    stopped_due_to_nan = True
                    break

                loss.backward()
                if self.config.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clip_norm
                    )
                self.optimizer.step()
                running_loss += float(loss.item())
                n_batches += 1

            if stopped_due_to_nan:
                break

            val_metrics, _, is_valid = self.evaluate_split(val_split)
            if not is_valid:
                stopped_due_to_nan = True
                break

            val_score = val_metrics["score"]
            train_loss = running_loss / max(1, n_batches)
            history.append(
                {"epoch": float(epoch), "train_loss": train_loss, "val_score": val_score}
            )

            if val_score > best_val_score:
                best_val_score = val_score
                best_epoch = epoch
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in self.model.state_dict().items()
                }
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.config.patience:
                    break

        # Reload best checkpoint strictly before final train/val/test evaluation
        self.model.load_state_dict(best_state)
        self.model.to(self.device)

        final_metrics: dict[str, dict[str, float]] = {}
        for split_name, split_obj in (("train", train_split), ("val", val_split), ("test", test_split)):
            m, _, _ = self.evaluate_split(split_obj)
            final_metrics[split_name] = m

        elapsed = time.perf_counter() - t0
        result = TrainResult(
            best_epoch=best_epoch,
            best_val_score=best_val_score,
            metrics=final_metrics,
            history=history,
            elapsed_sec=elapsed,
            stopped_due_to_nan=stopped_due_to_nan,
        )

        if save_dir is not None:
            self._save_artifacts(Path(save_dir), dataset, result)

        return result

    def _save_artifacts(
        self, save_dir: Path, dataset: ProcessedDataset, result: TrainResult
    ) -> None:
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": {
                    k: v.detach().cpu() for k, v in self.model.state_dict().items()
                },
                "model_config": asdict(self.model.config),
                "train_config": asdict(self.config),
                "target_stats": asdict(self.target_stats),
            },
            save_dir / "best_checkpoint.pt",
        )
        report: dict[str, Any] = {
            "dataset": dataset.name,
            "task_type": dataset.task_type,
            "best_epoch": result.best_epoch,
            "best_val_score": result.best_val_score,
            "metrics": result.metrics,
            "elapsed_sec": result.elapsed_sec,
            "stopped_due_to_nan": result.stopped_due_to_nan,
            "model_config": asdict(self.model.config),
            "train_config": asdict(self.config),
        }
        with (save_dir / "report.json").open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)