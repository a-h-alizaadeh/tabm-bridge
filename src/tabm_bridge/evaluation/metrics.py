from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)

from tabm_bridge.config import TaskType

_SCORE_SIGN: dict[str, float] = {
    "accuracy": 1.0,
    "r2": 1.0,
    "roc-auc": 1.0,
    "cross-entropy": -1.0,
    "mae": -1.0,
    "rmse": -1.0,
}


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: TaskType,
    primary_score: str | None = None,
) -> dict[str, float]:
    """Compute official TabM metrics with unified higher-is-better `score` (lib/metrics.py).

    Args:
        y_true: Original-scale ground truth labels/targets of shape (N,).
        y_pred:
            - regression: predicted values of shape (N,) or (N, 1) on original scale.
            - binclass: predicted class probabilities of shape (N, 2) or positive prob (N,).
            - multiclass: predicted class probabilities of shape (N, C).
        task_type: "regression", "binclass", or "multiclass".
        primary_score: Metric key from info.json (defaults: rmse / accuracy).
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred)
    eps = 1e-7
    metrics: dict[str, float] = {}

    if task_type == "regression":
        pred_1d = y_pred.reshape(-1)
        metrics["rmse"] = float(root_mean_squared_error(y_true, pred_1d))
        metrics["mae"] = float(mean_absolute_error(y_true, pred_1d))
        metrics["r2"] = float(r2_score(y_true, pred_1d))
        key = primary_score or "rmse"
    elif task_type == "binclass":
        prob_pos = y_pred[:, 1] if (y_pred.ndim == 2 and y_pred.shape[1] == 2) else y_pred.reshape(-1)
        prob_pos = np.clip(prob_pos, eps, 1.0 - eps)
        pred_labels = np.round(prob_pos).astype(np.int64)
        metrics["accuracy"] = float(accuracy_score(y_true, pred_labels))
        metrics["cross-entropy"] = float(log_loss(y_true, prob_pos, labels=[0, 1]))
        if np.unique(y_true).size == 2:
            metrics["roc-auc"] = float(roc_auc_score(y_true, prob_pos))
        key = primary_score or "accuracy"
    elif task_type == "multiclass":
        probs = np.clip(y_pred, eps, 1.0)
        probs = probs / probs.sum(axis=-1, keepdims=True)
        pred_labels = probs.argmax(axis=-1)
        n_classes = probs.shape[1]
        metrics["accuracy"] = float(accuracy_score(y_true, pred_labels))
        metrics["cross-entropy"] = float(log_loss(y_true, probs, labels=list(range(n_classes))))
        key = primary_score or "accuracy"
    else:
        raise ValueError(f"Unsupported task_type: {task_type}")

    if key not in metrics or key not in _SCORE_SIGN:
        raise ValueError(f"Unsupported primary_score '{key}' for task '{task_type}'.")

    metrics["score"] = _SCORE_SIGN[key] * metrics[key]
    return metrics