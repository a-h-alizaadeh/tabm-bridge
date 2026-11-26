from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
    silhouette_score,
)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

from tabm_bridge.config import TaskType
from tabm_bridge.evaluation.metrics import compute_metrics

DEFAULT_REG_GRID_REGRESSION: tuple[float, ...] = (
    1e-3,
    1e-2,
    1e-1,
    1.0,
    10.0,
    100.0,
    1e3,
    1e4,
    1e5,
)
# Expanded lower bound (1e-5, 1e-4) to avoid boundary clipping on large N (e.g., Otto, Adult)
DEFAULT_REG_GRID_CLASSIFICATION: tuple[float, ...] = (
    1e-5,
    1e-4,
    1e-3,
    1e-2,
    1e-1,
    1.0,
    10.0,
    100.0,
)


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Spearman rank correlation safely, returning 0.0 for constant inputs."""
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    b = np.asarray(y, dtype=np.float64).reshape(-1)
    if a.size < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    corr = spearmanr(a, b).statistic
    return float(corr) if np.isfinite(corr) else 0.0


def _l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    return arr / np.clip(norms, eps, None)


def _resolve_torch_device(device: str) -> torch.device | None:
    """Resolve optional PyTorch backend device ('auto', 'cuda', 'cpu', or 'sklearn'/'numpy')."""
    if torch is None or device in ("sklearn", "numpy"):
        return None
    if device == "cpu":
        return torch.device("cpu")
    if device.startswith("cuda"):
        return torch.device(device) if torch.cuda.is_available() else torch.device("cpu")
    # "auto": use GPU when available, else None (stay on fast NumPy/sklearn CPU path)
    return torch.device("cuda") if torch.cuda.is_available() else None


def _isotropic_scale(
    z_train: np.ndarray,
    z_val: np.ndarray,
    z_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotation-invariant centering + global RMS scaling in float64 (strictly train-fitted).

    Unlike per-coordinate StandardScaler, isotropic scaling preserves orthogonal geometry
    and prevents near-dead embedding dimensions from causing ill-conditioned Ridge blowups.
    """
    x_tr = np.asarray(z_train, dtype=np.float64)
    x_va = np.asarray(z_val, dtype=np.float64)
    x_te = np.asarray(z_test, dtype=np.float64)

    mu = x_tr.mean(axis=0, keepdims=True)
    x_tr_c = x_tr - mu
    scale = float(np.sqrt(np.mean(x_tr_c * x_tr_c)))
    if not np.isfinite(scale) or scale < 1e-12:
        scale = 1.0
    return x_tr_c / scale, (x_va - mu) / scale, (x_te - mu) / scale


def _prepare_ridge_svd(
    x_tr_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Compute exact thin float64 SVD once for fast, warning-free multi-alpha Ridge."""
    if x_tr_s.shape[1] > 2048:
        return None
    u, s, vt = np.linalg.svd(x_tr_s, full_matrices=False)
    return u, s, vt


def _fit_ridge_1d_val_select(
    x_tr_s: np.ndarray,
    y_tr: np.ndarray,
    x_va_s: np.ndarray,
    y_va: np.ndarray,
    x_te_s: np.ndarray,
    reg_grid: tuple[float, ...],
    svd_cache: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    seed: int = 42,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Select optimal L2 alpha strictly on val RMSE using exact float64 SVD (or lsqr if d>2048)."""
    y0 = np.asarray(y_tr, dtype=np.float64).reshape(-1)
    y1 = np.asarray(y_va, dtype=np.float64).reshape(-1)
    y_mean = float(y0.mean())

    best_alpha = float(reg_grid[0])
    best_val_rmse = float("inf")
    best_pred_va: np.ndarray | None = None
    best_pred_te: np.ndarray | None = None

    if svd_cache is not None:
        u, s, vt = svd_cache
        s2 = s * s
        uty = u.T @ (y0 - y_mean)
        best_w: np.ndarray | None = None
        for alpha in reg_grid:
            a = float(alpha)
            w = vt.T @ ((s / (s2 + a)) * uty)
            pred_va = x_va_s @ w + y_mean
            val_rmse = float(root_mean_squared_error(y1, pred_va))
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_alpha = a
                best_w = w
                best_pred_va = pred_va
        assert best_w is not None and best_pred_va is not None
        best_pred_te = x_te_s @ best_w + y_mean
        return best_alpha, best_pred_va, best_pred_te

    best_model: Ridge | None = None
    for alpha in reg_grid:
        a = float(alpha)
        model = Ridge(alpha=a, solver="lsqr", random_state=seed)
        model.fit(x_tr_s, y0 - y_mean)
        pred_va = model.predict(x_va_s) + y_mean
        val_rmse = float(root_mean_squared_error(y1, pred_va))
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_alpha = a
            best_model = model
            best_pred_va = pred_va
    assert best_model is not None and best_pred_va is not None
    best_pred_te = best_model.predict(x_te_s) + y_mean
    return best_alpha, best_pred_va, best_pred_te


def _fit_torch_logistic_probe(
    x_tr_s: np.ndarray,
    y_tr_c: np.ndarray,
    x_va_s: np.ndarray,
    y_va_c: np.ndarray,
    x_te_s: np.ndarray,
    task_type: TaskType,
    primary_score: str | None,
    reg_grid: tuple[float, ...],
    torch_dev: torch.device,
    max_iter: int = 300,
) -> tuple[float, dict[str, float], np.ndarray] | None:
    """Exact float64 PyTorch L-BFGS logistic probe matching sklearn's 0.5||W||^2 + C*sum(loss).

    Warm-starts weights from smallest C to largest C on GPU to eliminate Otto's CPU bottleneck.
    Returns None if class labels are non-contiguous so caller cleanly falls back to sklearn.
    """
    assert torch is not None and nn is not None and F is not None
    classes = np.unique(y_tr_c)
    n_classes = int(classes.size)
    if n_classes < 2 or not np.array_equal(classes, np.arange(n_classes)):
        return None
    if int(y_va_c.min()) < 0 or int(y_va_c.max()) >= n_classes:
        return None

    with torch.inference_mode(False), torch.enable_grad():
        x_tr_t = torch.from_numpy(np.ascontiguousarray(x_tr_s, dtype=np.float64)).to(torch_dev)
        x_va_t = torch.from_numpy(np.ascontiguousarray(x_va_s, dtype=np.float64)).to(torch_dev)
        x_te_t = torch.from_numpy(np.ascontiguousarray(x_te_s, dtype=np.float64)).to(torch_dev)
        y_tr_t = torch.from_numpy(np.ascontiguousarray(y_tr_c, dtype=np.int64)).to(torch_dev)

        n_samples, d = x_tr_t.shape
        is_bin = n_classes == 2
        out_dim = 1 if is_bin else n_classes
        y_bin_t = y_tr_t.to(torch.float64).unsqueeze(1) if is_bin else None

        w = nn.Parameter(torch.zeros((d, out_dim), dtype=torch.float64, device=torch_dev))
        b = nn.Parameter(torch.zeros((out_dim,), dtype=torch.float64, device=torch_dev))

        best_param = float(reg_grid[0])
        best_val_score = -float("inf")
        best_val_metrics: dict[str, float] = {}
        best_w: torch.Tensor | None = None
        best_b: torch.Tensor | None = None

        for c_val in sorted(float(c) for c in reg_grid):
            reg_coeff = 1.0 / (2.0 * float(n_samples) * c_val)
            opt = torch.optim.LBFGS(
                [w, b],
                lr=1.0,
                max_iter=max_iter,
                tolerance_grad=1e-7,
                tolerance_change=1e-10,
                history_size=20,
                line_search_fn="strong_wolfe",
            )

            def closure() -> torch.Tensor:
                opt.zero_grad()
                logits = x_tr_t @ w + b
                if is_bin:
                    assert y_bin_t is not None
                    data_loss = F.binary_cross_entropy_with_logits(logits, y_bin_t, reduction="mean")
                else:
                    data_loss = F.cross_entropy(logits, y_tr_t, reduction="mean")
                loss = data_loss + reg_coeff * torch.sum(w * w)
                loss.backward()
                return loss

            opt.step(closure)
            if not (torch.isfinite(w).all() and torch.isfinite(b).all()):
                with torch.no_grad():
                    w.zero_()
                    b.zero_()
                continue

            with torch.no_grad():
                lv = x_va_t @ w + b
                if is_bin:
                    p1 = torch.sigmoid(lv).squeeze(1).cpu().numpy()
                    prob_va = np.stack([1.0 - p1, p1], axis=1)
                else:
                    prob_va = torch.softmax(lv, dim=-1).cpu().numpy()

            m_va = compute_metrics(y_va_c, prob_va, task_type, primary_score)
            if m_va["score"] > best_val_score:
                best_val_score = float(m_va["score"])
                best_val_metrics = m_va
                best_param = c_val
                best_w = w.detach().clone()
                best_b = b.detach().clone()

        if best_w is None or best_b is None:
            return None

        with torch.no_grad():
            lt = x_te_t @ best_w + best_b
            if is_bin:
                p1 = torch.sigmoid(lt).squeeze(1).cpu().numpy()
                prob_te = np.stack([1.0 - p1, p1], axis=1)
            else:
                prob_te = torch.softmax(lt, dim=-1).cpu().numpy()

        return best_param, best_val_metrics, prob_te


def _chunked_cosine_topk(
    queries: np.ndarray,
    bank: np.ndarray,
    k: int,
    chunk_size: int = 1024,
    device: str = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    """Memory-bounded exact top-k cosine similarity search (GPU when available, else BLAS CPU)."""
    q_unit = _l2_normalize(queries)
    b_unit = _l2_normalize(bank)
    n_q = q_unit.shape[0]
    n_b = b_unit.shape[0]
    k_eff = max(1, min(int(k), n_b))

    torch_dev = _resolve_torch_device(device)
    if torch_dev is not None and torch is not None:
        with torch.no_grad():
            b_t = torch.from_numpy(np.ascontiguousarray(b_unit.T)).to(torch_dev)
            top_idx = np.empty((n_q, k_eff), dtype=np.int64)
            top_sim = np.empty((n_q, k_eff), dtype=np.float32)
            for start in range(0, n_q, chunk_size):
                end = min(start + chunk_size, n_q)
                q_c = torch.from_numpy(np.ascontiguousarray(q_unit[start:end])).to(torch_dev)
                sims = q_c @ b_t
                vals, idxs = torch.topk(sims, k=k_eff, dim=1, largest=True, sorted=True)
                top_idx[start:end] = idxs.cpu().numpy()
                top_sim[start:end] = vals.cpu().numpy()
            return top_idx, top_sim

    top_idx = np.empty((n_q, k_eff), dtype=np.int64)
    top_sim = np.empty((n_q, k_eff), dtype=np.float32)

    b_t_np = np.ascontiguousarray(b_unit.T)
    for start in range(0, n_q, chunk_size):
        end = min(start + chunk_size, n_q)
        sims = q_unit[start:end] @ b_t_np  # (chunk, n_b)
        if k_eff < n_b:
            part_idx = np.argpartition(-sims, kth=k_eff - 1, axis=1)[:, :k_eff]
        else:
            part_idx = np.broadcast_to(np.arange(n_b, dtype=np.int64), (end - start, n_b)).copy()

        row_arange = np.arange(end - start)[:, None]
        part_sim = sims[row_arange, part_idx]
        order = np.argsort(-part_sim, axis=1)
        top_idx[start:end] = part_idx[row_arange, order]
        top_sim[start:end] = part_sim[row_arange, order]

    return top_idx, top_sim


def compute_gap_recovery(
    single_score: float,
    tabm_b_score: float,
    ensemble_score: float,
) -> float:
    """Fraction of the TabM[B] -> Ensemble performance gap recovered by single-vector z(x)."""
    gap = ensemble_score - tabm_b_score
    if abs(gap) < 1e-12:
        return float("nan")
    return float((single_score - tabm_b_score) / gap)


@dataclass
class LinearProbeResult:
    """Test 1: Leak-free convex linear probe on frozen single-vector embeddings z(x)."""

    best_reg_param: float
    val_metrics: dict[str, float]
    test_metrics: dict[str, float]
    test_score: float


def evaluate_linear_probe(
    z_train: np.ndarray,
    y_train_raw: np.ndarray,
    z_val: np.ndarray,
    y_val_raw: np.ndarray,
    z_test: np.ndarray,
    y_test_raw: np.ndarray,
    task_type: TaskType,
    primary_score: str | None = None,
    reg_grid: tuple[float, ...] | None = None,
    seed: int = 42,
    device: str = "auto",
) -> LinearProbeResult:
    """Fit isotropic float64 scaler + L2-regularized linear probe strictly on train; tune on val."""
    x_tr_s, x_va_s, x_te_s = _isotropic_scale(z_train, z_val, z_test)

    if task_type == "regression":
        grid = reg_grid if reg_grid is not None else DEFAULT_REG_GRID_REGRESSION
        y_tr = np.asarray(y_train_raw, dtype=np.float64).reshape(-1)
        y_va = np.asarray(y_val_raw, dtype=np.float64).reshape(-1)
        y_te = np.asarray(y_test_raw, dtype=np.float64).reshape(-1)
        y_mean = float(y_tr.mean())

        svd_cache = _prepare_ridge_svd(x_tr_s)
        best_param = float(grid[0])
        best_val_score = -float("inf")
        best_val_metrics: dict[str, float] = {}
        best_pred_te: np.ndarray | None = None

        if svd_cache is not None:
            u, s, vt = svd_cache
            s2 = s * s
            uty = u.T @ (y_tr - y_mean)
            best_w: np.ndarray | None = None
            for alpha in grid:
                a = float(alpha)
                w = vt.T @ ((s / (s2 + a)) * uty)
                pred_va = x_va_s @ w + y_mean
                m_va = compute_metrics(y_va, pred_va, task_type, primary_score)
                if m_va["score"] > best_val_score:
                    best_val_score = m_va["score"]
                    best_val_metrics = m_va
                    best_param = a
                    best_w = w
            assert best_w is not None
            best_pred_te = x_te_s @ best_w + y_mean
        else:
            best_model_r: Ridge | None = None
            for alpha in grid:
                a = float(alpha)
                model = Ridge(alpha=a, solver="lsqr", random_state=seed)
                model.fit(x_tr_s, y_tr - y_mean)
                pred_va = model.predict(x_va_s) + y_mean
                m_va = compute_metrics(y_va, pred_va, task_type, primary_score)
                if m_va["score"] > best_val_score:
                    best_val_score = m_va["score"]
                    best_val_metrics = m_va
                    best_param = a
                    best_model_r = model
            assert best_model_r is not None
            best_pred_te = best_model_r.predict(x_te_s) + y_mean

        m_te = compute_metrics(y_te, best_pred_te, task_type, primary_score)
        return LinearProbeResult(
            best_reg_param=best_param,
            val_metrics=best_val_metrics,
            test_metrics=m_te,
            test_score=float(m_te["score"]),
        )

    if task_type not in ("binclass", "multiclass"):
        raise ValueError(f"Unsupported task_type: {task_type}")

    grid = reg_grid if reg_grid is not None else DEFAULT_REG_GRID_CLASSIFICATION
    y_tr_c = np.asarray(y_train_raw, dtype=np.int64).reshape(-1)
    y_va_c = np.asarray(y_val_raw, dtype=np.int64).reshape(-1)
    y_te_c = np.asarray(y_test_raw, dtype=np.int64).reshape(-1)

    # Fast exact float64 PyTorch L-BFGS path on GPU (or explicit 'cpu' for unit parity tests)
    torch_dev = _resolve_torch_device(device)
    if torch_dev is not None and x_tr_s.shape[1] <= 4096:
        torch_out = _fit_torch_logistic_probe(
            x_tr_s=x_tr_s,
            y_tr_c=y_tr_c,
            x_va_s=x_va_s,
            y_va_c=y_va_c,
            x_te_s=x_te_s,
            task_type=task_type,
            primary_score=primary_score,
            reg_grid=grid,
            torch_dev=torch_dev,
        )
        if torch_out is not None:
            best_param, best_val_metrics, prob_te = torch_out
            m_te = compute_metrics(y_te_c, prob_te, task_type, primary_score)
            return LinearProbeResult(
                best_reg_param=best_param,
                val_metrics=best_val_metrics,
                test_metrics=m_te,
                test_score=float(m_te["score"]),
            )

    # Reference sklearn L-BFGS fallback (used on CPU 'auto', non-contiguous labels, or 'sklearn')
    n_classes = int(np.unique(y_tr_c).size)

    def _align_probs(clf: LogisticRegression, x_in: np.ndarray) -> np.ndarray:
        raw_p = clf.predict_proba(x_in)
        if raw_p.shape[1] == n_classes and np.array_equal(clf.classes_, np.arange(n_classes)):
            return raw_p
        full = np.zeros((x_in.shape[0], n_classes), dtype=np.float64)
        for col_idx, cls_label in enumerate(clf.classes_):
            full[:, int(cls_label)] = raw_p[:, col_idx]
        return full

    best_param = float(grid[0])
    best_val_score = -float("inf")
    best_val_metrics = {}
    best_model: Any = None

    for c_val in grid:
        clf = LogisticRegression(
            C=float(c_val),
            solver="lbfgs",
            max_iter=500,
            random_state=seed,
        )
        clf.fit(x_tr_s, y_tr_c)
        prob_va = _align_probs(clf, x_va_s)
        m_va = compute_metrics(y_va_c, prob_va, task_type, primary_score)
        if m_va["score"] > best_val_score:
            best_val_score = m_va["score"]
            best_val_metrics = m_va
            best_param = float(c_val)
            best_model = clf

    prob_te = _align_probs(best_model, x_te_s)
    m_te = compute_metrics(y_te_c, prob_te, task_type, primary_score)
    return LinearProbeResult(
        best_reg_param=best_param,
        val_metrics=best_val_metrics,
        test_metrics=m_te,
        test_score=float(m_te["score"]),
    )


@dataclass
class KNNGeometryResult:
    """Test 2: Zero-training cosine k-NN retrieval and clustering geometry metrics."""

    k_neighbors: int
    knn_score: float                 # Higher is better (-RMSE for reg, accuracy/primary for cls)
    nn1_score: float                 # Top-1 neighbor (-RMSE for reg, Recall@1 / 1-NN acc for cls)
    purity_at_k: float | None        # Classification only: mean fraction of top-k sharing query class
    silhouette_cosine: float | None  # Classification only: cosine silhouette on test split
    knn_r2: float | None             # Regression only


def evaluate_knn_geometry(
    z_train: np.ndarray,
    y_train_raw: np.ndarray,
    z_test: np.ndarray,
    y_test_raw: np.ndarray,
    task_type: TaskType,
    primary_score: str | None = None,
    k_neighbors: int = 15,
    chunk_size: int = 1024,
    max_silhouette_samples: int = 2000,
    seed: int = 42,
    device: str = "auto",
    precomputed_topk: tuple[np.ndarray, np.ndarray] | None = None,
) -> KNNGeometryResult:
    """Evaluate representation neighborhood geometry directly via cosine similarity."""
    if precomputed_topk is not None:
        top_idx, top_sim = precomputed_topk
        if len(top_idx) != len(z_test) or len(top_sim) != len(z_test):
            raise ValueError(
                f"precomputed_topk length mismatch: got idx={len(top_idx)}, sim={len(top_sim)}, "
                f"expected len(z_test)={len(z_test)}."
            )
    else:
        top_idx, top_sim = _chunked_cosine_topk(
            queries=z_test,
            bank=z_train,
            k=k_neighbors,
            chunk_size=chunk_size,
            device=device,
        )
    weights = np.clip(top_sim, 0.0, None) + 1e-6
    weights = weights / weights.sum(axis=1, keepdims=True)

    if task_type == "regression":
        y_tr = np.asarray(y_train_raw, dtype=np.float64).reshape(-1)
        y_te = np.asarray(y_test_raw, dtype=np.float64).reshape(-1)
        nbr_y = y_tr[top_idx]  # (N_test, k)
        pred_knn = np.sum(weights * nbr_y, axis=1)
        pred_nn1 = nbr_y[:, 0]

        m_knn = compute_metrics(y_te, pred_knn, "regression", primary_score)
        nn1_rmse = float(root_mean_squared_error(y_te, pred_nn1))
        return KNNGeometryResult(
            k_neighbors=top_idx.shape[1],
            knn_score=float(m_knn["score"]),
            nn1_score=-nn1_rmse,
            purity_at_k=None,
            silhouette_cosine=None,
            knn_r2=float(m_knn["r2"]),
        )

    if task_type not in ("binclass", "multiclass"):
        raise ValueError(f"Unsupported task_type: {task_type}")

    y_tr_c = np.asarray(y_train_raw, dtype=np.int64).reshape(-1)
    y_te_c = np.asarray(y_test_raw, dtype=np.int64).reshape(-1)
    n_classes = int(max(y_tr_c.max(), y_te_c.max()) + 1)

    nbr_labels = y_tr_c[top_idx]  # (N_test, k)
    probs = np.zeros((len(y_te_c), n_classes), dtype=np.float64)
    for c in range(n_classes):
        probs[:, c] = np.sum(weights * (nbr_labels == c), axis=1)
    probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)

    m_knn = compute_metrics(y_te_c, probs, task_type, primary_score)
    nn1_acc = float(np.mean(nbr_labels[:, 0] == y_te_c))
    purity = float(np.mean(nbr_labels == y_te_c[:, None]))

    # Cosine silhouette on deterministic test subsample
    z_te_unit = _l2_normalize(z_test)
    if len(y_te_c) > max_silhouette_samples:
        rng = np.random.default_rng(seed)
        sub_idx = rng.choice(len(y_te_c), size=max_silhouette_samples, replace=False)
        z_sil = z_te_unit[sub_idx]
        y_sil = y_te_c[sub_idx]
    else:
        z_sil = z_te_unit
        y_sil = y_te_c

    n_unique = int(np.unique(y_sil).size)
    if 2 <= n_unique < len(y_sil):
        sil = float(silhouette_score(z_sil, y_sil, metric="cosine"))
    else:
        sil = None

    return KNNGeometryResult(
        k_neighbors=top_idx.shape[1],
        knn_score=float(m_knn["score"]),
        nn1_score=nn1_acc,
        purity_at_k=purity,
        silhouette_cosine=sil,
        knn_r2=None,
    )


@dataclass
class AuxTransferResult:
    """Test 3: Auxiliary feature prediction probe (direct answer to Appendix A.4)."""

    selected_feature_indices: list[int]
    per_feature_r2: list[float]
    per_feature_rmse: list[float]
    mean_r2: float
    mean_neg_rmse: float
    per_feature_best_alpha: list[float] = field(default_factory=list)


def select_auxiliary_features(
    x_num_train: np.ndarray,
    y_train_raw: np.ndarray,
    task_type: TaskType,
    n_aux: int = 3,
    max_collinearity: float = 0.98,
) -> list[int]:
    """Select up to `n_aux` non-constant, non-collinear numerical features least correlated with y.

    Fitted STRICTLY on train split. Greedy collinearity filtering (|r| < max_collinearity)
    prevents exact complement/duplicate columns (e.g., Churn indices [5, 6]) from double-counting.
    """
    x_tr = np.asarray(x_num_train, dtype=np.float64)
    if x_tr.ndim != 2 or x_tr.shape[1] == 0:
        return []

    y_tr = np.asarray(y_train_raw, dtype=np.float64).reshape(-1)
    y_std = float(np.std(y_tr))
    candidates: list[tuple[float, int]] = []
    normed_cols: dict[int, np.ndarray] = {}

    classes = np.unique(y_tr.astype(np.int64)) if task_type == "multiclass" else None

    for j in range(x_tr.shape[1]):
        col = x_tr[:, j]
        col_std = float(np.std(col))
        if col_std < 1e-6:
            continue

        col_c = (col - col.mean()) / col_std
        normed_cols[j] = col_c
        if task_type == "multiclass" and classes is not None:
            max_corr = 0.0
            for c in classes:
                ind = (y_tr == c).astype(np.float64)
                ind_std = float(np.std(ind))
                if ind_std >= 1e-6:
                    corr = abs(float(np.mean(col_c * ((ind - ind.mean()) / ind_std))))
                    if corr > max_corr:
                        max_corr = corr
            target_corr = max_corr
        else:
            if y_std < 1e-6:
                target_corr = 0.0
            else:
                y_c = (y_tr - y_tr.mean()) / y_std
                target_corr = abs(float(np.mean(col_c * y_c)))

        candidates.append((target_corr, j))

    candidates.sort(key=lambda pair: (pair[0], pair[1]))
    selected: list[int] = []
    target_n = max(0, n_aux)
    for _, j in candidates:
        if len(selected) >= target_n:
            break
        if max_collinearity < 1.0 and selected:
            col_j = normed_cols[j]
            if any(
                abs(float(np.mean(col_j * normed_cols[prev]))) >= max_collinearity
                for prev in selected
            ):
                continue
        selected.append(j)
    return selected


def evaluate_auxiliary_transfer(
    z_train: np.ndarray,
    aux_train: np.ndarray,
    z_val: np.ndarray,
    aux_val: np.ndarray,
    z_test: np.ndarray,
    aux_test: np.ndarray,
    feature_indices: list[int] | None = None,
    reg_grid: tuple[float, ...] = DEFAULT_REG_GRID_REGRESSION,
    seed: int = 42,
) -> AuxTransferResult:
    """Evaluate linear decodability of held-out / auxiliary attributes from frozen z(x)."""
    a_tr = np.asarray(aux_train, dtype=np.float64)
    a_va = np.asarray(aux_val, dtype=np.float64)
    a_te = np.asarray(aux_test, dtype=np.float64)
    if a_tr.ndim == 1:
        a_tr, a_va, a_te = a_tr[:, None], a_va[:, None], a_te[:, None]

    n_feat = a_tr.shape[1]
    indices = feature_indices if feature_indices is not None else list(range(n_feat))

    x_tr_s, x_va_s, x_te_s = _isotropic_scale(z_train, z_val, z_test)
    svd_cache = _prepare_ridge_svd(x_tr_s)

    r2_list: list[float] = []
    rmse_list: list[float] = []
    alpha_list: list[float] = []

    for j in range(n_feat):
        best_alpha, _, pred_te = _fit_ridge_1d_val_select(
            x_tr_s=x_tr_s,
            y_tr=a_tr[:, j],
            x_va_s=x_va_s,
            y_va=a_va[:, j],
            x_te_s=x_te_s,
            reg_grid=reg_grid,
            svd_cache=svd_cache,
            seed=seed,
        )
        r2_list.append(float(r2_score(a_te[:, j], pred_te)))
        rmse_list.append(float(root_mean_squared_error(a_te[:, j], pred_te)))
        alpha_list.append(float(best_alpha))

    return AuxTransferResult(
        selected_feature_indices=list(indices),
        per_feature_r2=r2_list,
        per_feature_rmse=rmse_list,
        mean_r2=float(np.mean(r2_list)) if r2_list else float("nan"),
        mean_neg_rmse=-float(np.mean(rmse_list)) if rmse_list else float("nan"),
        per_feature_best_alpha=alpha_list,
    )


@dataclass
class UQFidelityResult:
    """Test 4a: Reproduction of ensemble epistemic uncertainty from single-vector z(x)."""

    probe_spearman: float
    probe_r2: float
    direct_head_spearman: float | None = None
    direct_head_r2: float | None = None
    best_reg_param: float | None = None


def evaluate_uq_fidelity(
    z_train: np.ndarray,
    uq_train: np.ndarray,
    z_val: np.ndarray,
    uq_val: np.ndarray,
    z_test: np.ndarray,
    uq_test: np.ndarray,
    direct_uq_test: np.ndarray | None = None,
    reg_grid: tuple[float, ...] = DEFAULT_REG_GRID_REGRESSION,
    seed: int = 42,
) -> UQFidelityResult:
    """Measure how faithfully ensemble epistemic uncertainty can be read from z(x)."""
    u_tr_raw = np.clip(np.asarray(uq_train, dtype=np.float64).reshape(-1), 0.0, None)
    u_va_raw = np.clip(np.asarray(uq_val, dtype=np.float64).reshape(-1), 0.0, None)
    u_te_raw = np.clip(np.asarray(uq_test, dtype=np.float64).reshape(-1), 0.0, None)

    u_tr = np.log1p(u_tr_raw)
    u_va = np.log1p(u_va_raw)
    u_te = np.log1p(u_te_raw)

    x_tr_s, x_va_s, x_te_s = _isotropic_scale(z_train, z_val, z_test)
    svd_cache = _prepare_ridge_svd(x_tr_s)

    best_alpha, _, pred_te_log = _fit_ridge_1d_val_select(
        x_tr_s=x_tr_s,
        y_tr=u_tr,
        x_va_s=x_va_s,
        y_va=u_va,
        x_te_s=x_te_s,
        reg_grid=reg_grid,
        svd_cache=svd_cache,
        seed=seed,
    )
    probe_rho = _safe_spearman(u_te_raw, pred_te_log)
    probe_r2 = float(r2_score(u_te, pred_te_log)) if float(np.std(u_te)) >= 1e-12 else 0.0

    direct_rho: float | None = None
    direct_r2: float | None = None
    if direct_uq_test is not None:
        d_raw = np.clip(np.asarray(direct_uq_test, dtype=np.float64).reshape(-1), 0.0, None)
        d_log = np.log1p(d_raw)
        direct_rho = _safe_spearman(u_te_raw, d_raw)
        direct_r2 = float(r2_score(u_te, d_log)) if float(np.std(u_te)) >= 1e-12 else 0.0

    return UQFidelityResult(
        probe_spearman=probe_rho,
        probe_r2=probe_r2,
        direct_head_spearman=direct_rho,
        direct_head_r2=direct_r2,
        best_reg_param=float(best_alpha),
    )


class MahalanobisScorer:
    """Shrinkage-regularized Mahalanobis distance fitted strictly on z_train."""

    def __init__(
        self,
        ridge_eps: float = 1e-4,
        max_fit_samples: int | None = 16384,
        seed: int = 42,
    ) -> None:
        self.ridge_eps = ridge_eps
        self.max_fit_samples = max_fit_samples
        self.seed = seed
        self.mean_: np.ndarray | None = None
        self.precision_: np.ndarray | None = None

    def fit(self, z_train: np.ndarray) -> MahalanobisScorer:
        arr = np.asarray(z_train, dtype=np.float64)
        n, d = arr.shape
        if self.max_fit_samples is not None and n > self.max_fit_samples:
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(n, size=self.max_fit_samples, replace=False)
            arr = arr[idx]

        self.mean_ = arr.mean(axis=0)
        centered = arr - self.mean_
        try:
            lw = LedoitWolf(assume_centered=True).fit(centered)
            cov = lw.covariance_
        except Exception:
            cov = (centered.T @ centered) / max(len(centered) - 1, 1)

        trace_scale = float(np.trace(cov)) / max(d, 1)
        jitter = max(self.ridge_eps * max(trace_scale, 1.0), 1e-6)
        cov_reg = cov + np.eye(d, dtype=np.float64) * jitter
        self.precision_ = np.linalg.pinv(cov_reg, hermitian=True)
        return self

    def score(self, z_eval: np.ndarray, batch_size: int = 2048) -> np.ndarray:
        """Return squared Mahalanobis distance (higher = more OOD)."""
        if self.mean_ is None or self.precision_ is None:
            raise RuntimeError("MahalanobisScorer must be fitted before score().")
        arr = np.asarray(z_eval, dtype=np.float64)
        n = arr.shape[0]
        scores = np.empty(n, dtype=np.float64)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            diff = arr[start:end] - self.mean_
            proj = diff @ self.precision_
            scores[start:end] = np.clip(np.sum(proj * diff, axis=1), 0.0, None)
        return scores


class CosineKNNScorer:
    """Unsupervised OOD score: 1 - mean cosine similarity to top-k training neighbors."""

    def __init__(
        self,
        k_neighbors: int = 15,
        chunk_size: int = 1024,
        device: str = "auto",
    ) -> None:
        self.k_neighbors = k_neighbors
        self.chunk_size = chunk_size
        self.device = device
        self.bank_: np.ndarray | None = None

    def fit(self, z_train: np.ndarray) -> CosineKNNScorer:
        self.bank_ = _l2_normalize(z_train)
        return self

    def score(self, z_eval: np.ndarray) -> np.ndarray:
        if self.bank_ is None:
            raise RuntimeError("CosineKNNScorer must be fitted before score().")
        _, top_sim = _chunked_cosine_topk(
            queries=z_eval,
            bank=self.bank_,
            k=self.k_neighbors,
            chunk_size=self.chunk_size,
            device=self.device,
        )
        return (1.0 - top_sim.mean(axis=1)).astype(np.float64)


@dataclass
class OODResult:
    """Test 4b: OOD / high-uncertainty detection metrics in unified z(x) space."""

    auroc: float
    auprc: float
    fpr_at_95_tpr: float


def evaluate_ood_scores(scores_id: np.ndarray, scores_ood: np.ndarray) -> OODResult:
    """Compute AUROC, AUPRC, and FPR@95%TPR where higher score indicates OOD."""
    s_id = np.asarray(scores_id, dtype=np.float64).reshape(-1)
    s_ood = np.asarray(scores_ood, dtype=np.float64).reshape(-1)
    y_true = np.concatenate([np.zeros_like(s_id, dtype=np.int64), np.ones_like(s_ood, dtype=np.int64)])
    scores = np.concatenate([s_id, s_ood])

    auroc = float(roc_auc_score(y_true, scores))
    auprc = float(average_precision_score(y_true, scores))

    # Threshold at 5th percentile of OOD scores guarantees >= 95% TPR on OOD
    threshold = float(np.quantile(s_ood, 0.05))
    fpr95 = float(np.mean(s_id >= threshold))
    return OODResult(auroc=auroc, auprc=auprc, fpr_at_95_tpr=fpr95)


@dataclass
class UnifiedSpaceOODReport:
    """Combined Mahalanobis + Cosine-kNN OOD & high-uncertainty diagnostics."""

    mahal_high_uq_auroc: float
    cosine_high_uq_auroc: float
    mahal_uq_spearman: float
    cosine_uq_spearman: float
    mahal_ood: OODResult | None = None
    cosine_ood: OODResult | None = None
    mahal_high_uq_auprc: float | None = None
    cosine_high_uq_auprc: float | None = None
    mahal_high_uq_fpr95: float | None = None
    cosine_high_uq_fpr95: float | None = None


def evaluate_unified_space_ood(
    z_train: np.ndarray,
    z_test: np.ndarray,
    uq_test: np.ndarray,
    z_ood: np.ndarray | None = None,
    top_uq_frac: float = 0.10,
    k_neighbors: int = 15,
    seed: int = 42,
    device: str = "auto",
    precomputed_test_top_sim: np.ndarray | None = None,
) -> UnifiedSpaceOODReport:
    """Evaluate distance-based high-uncertainty detection (and optional real OOD shift)."""
    mahal_scorer = MahalanobisScorer(seed=seed).fit(z_train)
    m_test = mahal_scorer.score(z_test)

    cos_scorer: CosineKNNScorer | None = None
    if precomputed_test_top_sim is not None:
        sim_arr = np.asarray(precomputed_test_top_sim, dtype=np.float64)
        if len(sim_arr) != len(z_test):
            raise ValueError(
                f"precomputed_test_top_sim length ({len(sim_arr)}) != len(z_test) ({len(z_test)})."
            )
        c_test = 1.0 - sim_arr.mean(axis=1)
    else:
        cos_scorer = CosineKNNScorer(k_neighbors=k_neighbors, device=device).fit(z_train)
        c_test = cos_scorer.score(z_test)

    uq_1d = np.asarray(uq_test, dtype=np.float64).reshape(-1)
    m_rho = _safe_spearman(m_test, uq_1d)
    c_rho = _safe_spearman(c_test, uq_1d)

    m_hard_auroc, c_hard_auroc = float("nan"), float("nan")
    m_hard_auprc: float | None = None
    c_hard_auprc: float | None = None
    m_hard_fpr95: float | None = None
    c_hard_fpr95: float | None = None

    # Top-10% highest ensemble disagreement detection on test split (imbalanced 90/10)
    if float(np.std(uq_1d)) >= 1e-12 and 0.0 < top_uq_frac < 1.0:
        cutoff = float(np.quantile(uq_1d, 1.0 - top_uq_frac))
        is_hard = (uq_1d >= cutoff).astype(np.int64)
        if 0 < int(is_hard.sum()) < len(is_hard):
            m_hard_res = evaluate_ood_scores(m_test[is_hard == 0], m_test[is_hard == 1])
            c_hard_res = evaluate_ood_scores(c_test[is_hard == 0], c_test[is_hard == 1])
            m_hard_auroc, m_hard_auprc, m_hard_fpr95 = (
                m_hard_res.auroc,
                m_hard_res.auprc,
                m_hard_res.fpr_at_95_tpr,
            )
            c_hard_auroc, c_hard_auprc, c_hard_fpr95 = (
                c_hard_res.auroc,
                c_hard_res.auprc,
                c_hard_res.fpr_at_95_tpr,
            )

    m_ood_res: OODResult | None = None
    c_ood_res: OODResult | None = None
    if z_ood is not None and len(z_ood) > 0:
        if cos_scorer is None:
            cos_scorer = CosineKNNScorer(k_neighbors=k_neighbors, device=device).fit(z_train)
        m_ood_res = evaluate_ood_scores(m_test, mahal_scorer.score(z_ood))
        c_ood_res = evaluate_ood_scores(c_test, cos_scorer.score(z_ood))

    return UnifiedSpaceOODReport(
        mahal_high_uq_auroc=m_hard_auroc,
        cosine_high_uq_auroc=c_hard_auroc,
        mahal_uq_spearman=m_rho,
        cosine_uq_spearman=c_rho,
        mahal_ood=m_ood_res,
        cosine_ood=c_ood_res,
        mahal_high_uq_auprc=m_hard_auprc,
        cosine_high_uq_auprc=c_hard_auprc,
        mahal_high_uq_fpr95=m_hard_fpr95,
        cosine_high_uq_fpr95=c_hard_fpr95,
    )