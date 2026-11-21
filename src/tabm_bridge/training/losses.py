from __future__ import annotations

import torch
import torch.nn.functional as F

from tabm_bridge.config import TaskType


def compute_independent_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    task_type: TaskType,
) -> torch.Tensor:
    """Compute the mean of k strictly independent per-head losses.

    Never averages predictions across k before applying the loss (prevents Learner Collusion).

    Args:
        y_pred: Raw submodel outputs of shape (B, k, d_out).
        y_true: Targets of shape (B,) for shared batches (TabM♠) or (B, k) for independent batches.
        task_type: "regression", "binclass", or "multiclass".
    """
    if y_pred.ndim != 3:
        raise ValueError(f"Expected y_pred shape (B, k, d_out), got {tuple(y_pred.shape)}.")

    b, k, d_out = y_pred.shape

    if task_type == "regression":
        if y_true.ndim == 1:
            target = y_true.view(b, 1, 1).expand(b, k, d_out)
        elif y_true.ndim == 2 and y_true.shape == (b, k):
            target = y_true.unsqueeze(-1)
        elif y_true.ndim == 2 and y_true.shape == (b, d_out):
            target = y_true.unsqueeze(1).expand(b, k, d_out)
        elif y_true.ndim == 3 and y_true.shape == (b, k, d_out):
            target = y_true
        else:
            raise ValueError(
                f"Incompatible regression target shape {tuple(y_true.shape)} for y_pred {tuple(y_pred.shape)}."
            )
        return F.mse_loss(y_pred, target.to(dtype=y_pred.dtype), reduction="mean")

    # Official TabM uses d_out=2 + cross_entropy for binclass and d_out=C for multiclass
    if d_out < 2:
        raise ValueError(
            f"Classification with cross_entropy requires d_out >= 2, got d_out={d_out}."
        )

    if y_true.ndim == 1:
        target_cls = y_true.unsqueeze(1).expand(b, k)
    elif y_true.ndim == 2 and y_true.shape == (b, k):
        target_cls = y_true
    else:
        raise ValueError(
            f"Incompatible classification target shape {tuple(y_true.shape)} for y_pred {tuple(y_pred.shape)}."
        )

    return F.cross_entropy(
        y_pred.reshape(b * k, d_out),
        target_cls.reshape(b * k).long(),
        reduction="mean",
    )