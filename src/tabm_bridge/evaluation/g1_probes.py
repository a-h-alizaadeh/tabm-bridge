from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, r2_score, roc_auc_score, root_mean_squared_error, silhouette_score
from sklearn.preprocessing import StandardScaler

from tabm_bridge.config import TaskType
from tabm_bridge.evaluation.metrics import compute_metrics


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


def _chunked_cosine_topk(
    queries: np.ndarray,
    bank: np.ndarray,
    k: int,
    chunk_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """Memory-bounded exact top-k cosine similarity search using BLAS matmul + argpartition."""
    q_unit = _l2_normalize(queries)
    b_unit = _l2_normalize(bank)
    n_q = q_unit.shape[0]
    n_b = b_unit.shape[0]
    k_eff = max(1, min(int(k), n_b))

    top_idx = np.empty((n_q, k_eff), dtype=np.int64)
    top_sim = np.empty((n_q, k_eff), dtype=np.float32)

    b_t = np.ascontiguousarray(b_unit.T)
    for start in range(0, n_q, chunk_size):
        end = min(start + chunk_size, n_q)
        sims = q_unit[start:end] @ b_t  # (chunk, n_b)
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
    reg_grid: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0),
    seed: int = 42,
) -> LinearProbeResult:
    """Fit StandardScaler + L2-regularized linear probe strictly on train; tune on val."""
    x_tr = np.asarray(z_train, dtype=np.float32)
    x_va = np.asarray(z_val, dtype=np.float32)
    x_te = np.asarray(z_test, dtype=np.float32)

    scaler = StandardScaler()
    x_tr_s = scaler.fit_transform(x_tr)
    x_va_s = scaler.transform(x_va)
    x_te_s = scaler.transform(x_te)

    best_param = reg_grid[0]
    best_val_score = -float("inf")
    best_val_metrics: dict[str, float] = {}
    best_model: Any = None

    if task_type == "regression":
        y_tr = np.asarray(y_train_raw, dtype=np.float64).reshape(-1)
        y_va = np.asarray(y_val_raw, dtype=np.float64).reshape(-1)
        y_te = np.asarray(y_test_raw, dtype=np.float64).reshape(-1)

        y_mean = float(y_tr.mean())
        y_std = float(y_tr.std(ddof=0))
        if y_std < 1e-12:
            y_std = 1.0
        y_tr_std = (y_tr - y_mean) / y_std

        solver = "lsqr" if x_tr_s.shape[1] > 2048 else "auto"
        for alpha in reg_grid:
            model = Ridge(alpha=float(alpha), solver=solver, random_state=seed)
            model.fit(x_tr_s, y_tr_std)
            pred_va = model.predict(x_va_s) * y_std + y_mean
            m_va = compute_metrics(y_va, pred_va, task_type, primary_score)
            if m_va["score"] > best_val_score:
                best_val_score = m_va["score"]
                best_val_metrics = m_va
                best_param = float(alpha)
                best_model = model

        pred_te = best_model.predict(x_te_s) * y_std + y_mean
        m_te = compute_metrics(y_te, pred_te, task_type, primary_score)
        return LinearProbeResult(
            best_reg_param=best_param,
            val_metrics=best_val_metrics,
            test_metrics=m_te,
            test_score=float(m_te["score"]),
        )

    if task_type not in ("binclass", "multiclass"):
        raise ValueError(f"Unsupported task_type: {task_type}")

    y_tr_c = np.asarray(y_train_raw, dtype=np.int64).reshape(-1)
    y_va_c = np.asarray(y_val_raw, dtype=np.int64).reshape(-1)
    y_te_c = np.asarray(y_test_raw, dtype=np.int64).reshape(-1)
    n_classes = int(np.unique(y_tr_c).size)

    def _align_probs(clf: LogisticRegression, x_in: np.ndarray) -> np.ndarray:
        raw_p = clf.predict_proba(x_in)
        if raw_p.shape[1] == n_classes and np.array_equal(clf.classes_, np.arange(n_classes)):
            return raw_p
        full = np.zeros((x_in.shape[0], n_classes), dtype=np.float64)
        for col_idx, cls_label in enumerate(clf.classes_):
            full[:, int(cls_label)] = raw_p[:, col_idx]
        return full

    for c_val in reg_grid:
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
    knn_score: float             # Higher is better (-RMSE for reg, accuracy/primary for cls)
    nn1_score: float             # Top-1 neighbor (-RMSE for reg, Recall@1 / 1-NN acc for cls)
    purity_at_k: float | None    # Classification only: mean fraction of top-k sharing query class
    silhouette_cosine: float | None  # Classification only: cosine silhouette on test split
    knn_r2: float | None         # Regression only


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
) -> KNNGeometryResult:
    """Evaluate representation neighborhood geometry directly via cosine similarity."""
    top_idx, top_sim = _chunked_cosine_topk(
        queries=z_test,
        bank=z_train,
        k=k_neighbors,
        chunk_size=chunk_size,
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


def select_auxiliary_features(
    x_num_train: np.ndarray,
    y_train_raw: np.ndarray,
    task_type: TaskType,
    n_aux: int = 3,
) -> list[int]:
    """Select up to `n_aux` non-constant numerical features least correlated with primary target y.

    Fitted STRICTLY on train split. Choosing low-target-correlation features ensures a
    representation cannot cheat by collapsing solely to y_hat.
    """
    x_tr = np.asarray(x_num_train, dtype=np.float64)
    if x_tr.ndim != 2 or x_tr.shape[1] == 0:
        return []

    y_tr = np.asarray(y_train_raw, dtype=np.float64).reshape(-1)
    y_std = float(np.std(y_tr))
    candidates: list[tuple[float, int]] = []

    classes = np.unique(y_tr.astype(np.int64)) if task_type == "multiclass" else None

    for j in range(x_tr.shape[1]):
        col = x_tr[:, j]
        col_std = float(np.std(col))
        if col_std < 1e-6:
            continue

        col_c = (col - col.mean()) / col_std
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
    return [j for _, j in candidates[: max(0, n_aux)]]


def evaluate_auxiliary_transfer(
    z_train: np.ndarray,
    aux_train: np.ndarray,
    z_val: np.ndarray,
    aux_val: np.ndarray,
    z_test: np.ndarray,
    aux_test: np.ndarray,
    feature_indices: list[int] | None = None,
    reg_grid: tuple[float, ...] = (1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0),
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

    scaler = StandardScaler()
    x_tr_s = scaler.fit_transform(np.asarray(z_train, dtype=np.float32))
    x_va_s = scaler.transform(np.asarray(z_val, dtype=np.float32))
    x_te_s = scaler.transform(np.asarray(z_test, dtype=np.float32))
    solver = "lsqr" if x_tr_s.shape[1] > 2048 else "auto"

    r2_list: list[float] = []
    rmse_list: list[float] = []

    for j in range(n_feat):
        best_rmse = float("inf")
        best_model: Ridge | None = None
        for alpha in reg_grid:
            model = Ridge(alpha=float(alpha), solver=solver, random_state=seed)
            model.fit(x_tr_s, a_tr[:, j])
            pred_va = model.predict(x_va_s)
            val_rmse = float(root_mean_squared_error(a_va[:, j], pred_va))
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                best_model = model
        assert best_model is not None
        pred_te = best_model.predict(x_te_s)
        r2_list.append(float(r2_score(a_te[:, j], pred_te)))
        rmse_list.append(float(root_mean_squared_error(a_te[:, j], pred_te)))

    return AuxTransferResult(
        selected_feature_indices=list(indices),
        per_feature_r2=r2_list,
        per_feature_rmse=rmse_list,
        mean_r2=float(np.mean(r2_list)) if r2_list else float("nan"),
        mean_neg_rmse=-float(np.mean(rmse_list)) if rmse_list else float("nan"),
    )


@dataclass
class UQFidelityResult:
    """Test 4a: Reproduction of ensemble epistemic uncertainty from single-vector z(x)."""

    probe_spearman: float
    probe_r2: float
    direct_head_spearman: float | None = None
    direct_head_r2: float | None = None


def evaluate_uq_fidelity(
    z_train: np.ndarray,
    uq_train: np.ndarray,
    z_val: np.ndarray,
    uq_val: np.ndarray,
    z_test: np.ndarray,
    uq_test: np.ndarray,
    direct_uq_test: np.ndarray | None = None,
    reg_grid: tuple[float, ...] = (1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0),
    seed: int = 42,
) -> UQFidelityResult:
    """Measure how faithfully ensemble epistemic uncertainty can be read from z(x)."""
    u_tr_raw = np.clip(np.asarray(uq_train, dtype=np.float64).reshape(-1), 0.0, None)
    u_va_raw = np.clip(np.asarray(uq_val, dtype=np.float64).reshape(-1), 0.0, None)
    u_te_raw = np.clip(np.asarray(uq_test, dtype=np.float64).reshape(-1), 0.0, None)

    u_tr = np.log1p(u_tr_raw)
    u_va = np.log1p(u_va_raw)
    u_te = np.log1p(u_te_raw)

    scaler = StandardScaler()
    x_tr_s = scaler.fit_transform(np.asarray(z_train, dtype=np.float32))
    x_va_s = scaler.transform(np.asarray(z_val, dtype=np.float32))
    x_te_s = scaler.transform(np.asarray(z_test, dtype=np.float32))
    solver = "lsqr" if x_tr_s.shape[1] > 2048 else "auto"

    best_rmse = float("inf")
    best_model: Ridge | None = None
    for alpha in reg_grid:
        model = Ridge(alpha=float(alpha), solver=solver, random_state=seed)
        model.fit(x_tr_s, u_tr)
        pred_va = model.predict(x_va_s)
        val_rmse = float(root_mean_squared_error(u_va, pred_va))
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_model = model

    assert best_model is not None
    pred_te_log = best_model.predict(x_te_s)
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
    )


class MahalanobisScorer:
    """Shrinkage-regularized Mahalanobis distance fitted strictly on z_train."""

    def __init__(self, ridge_eps: float = 1e-4, max_fit_samples: int | None = 16384, seed: int = 42) -> None:
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

    def __init__(self, k_neighbors: int = 15, chunk_size: int = 1024) -> None:
        self.k_neighbors = k_neighbors
        self.chunk_size = chunk_size
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


def evaluate_unified_space_ood(
    z_train: np.ndarray,
    z_test: np.ndarray,
    uq_test: np.ndarray,
    z_ood: np.ndarray | None = None,
    top_uq_frac: float = 0.10,
    k_neighbors: int = 15,
    seed: int = 42,
) -> UnifiedSpaceOODReport:
    """Evaluate distance-based high-uncertainty detection (and optional real OOD shift)."""
    mahal_scorer = MahalanobisScorer(seed=seed).fit(z_train)
    cos_scorer = CosineKNNScorer(k_neighbors=k_neighbors).fit(z_train)

    m_test = mahal_scorer.score(z_test)
    c_test = cos_scorer.score(z_test)
    uq_1d = np.asarray(uq_test, dtype=np.float64).reshape(-1)

    m_rho = _safe_spearman(m_test, uq_1d)
    c_rho = _safe_spearman(c_test, uq_1d)

    # Top-10% highest ensemble disagreement detection on test split
    if float(np.std(uq_1d)) >= 1e-12 and 0.0 < top_uq_frac < 1.0:
        cutoff = float(np.quantile(uq_1d, 1.0 - top_uq_frac))
        is_hard = (uq_1d >= cutoff).astype(np.int64)
        if 0 < int(is_hard.sum()) < len(is_hard):
            m_hard_auroc = float(roc_auc_score(is_hard, m_test))
            c_hard_auroc = float(roc_auc_score(is_hard, c_test))
        else:
            m_hard_auroc, c_hard_auroc = float("nan"), float("nan")
    else:
        m_hard_auroc, c_hard_auroc = float("nan"), float("nan")

    m_ood_res: OODResult | None = None
    c_ood_res: OODResult | None = None
    if z_ood is not None and len(z_ood) > 0:
        m_ood_res = evaluate_ood_scores(m_test, mahal_scorer.score(z_ood))
        c_ood_res = evaluate_ood_scores(c_test, cos_scorer.score(z_ood))

    return UnifiedSpaceOODReport(
        mahal_high_uq_auroc=m_hard_auroc,
        cosine_high_uq_auroc=c_hard_auroc,
        mahal_uq_spearman=m_rho,
        cosine_uq_spearman=c_rho,
        mahal_ood=m_ood_res,
        cosine_ood=c_ood_res,
    )