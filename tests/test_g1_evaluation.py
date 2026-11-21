from __future__ import annotations

import numpy as np
import pytest

from tabm_bridge.evaluation import (
    CosineKNNScorer,
    MahalanobisScorer,
    compute_gap_recovery,
    evaluate_auxiliary_transfer,
    evaluate_knn_geometry,
    evaluate_linear_probe,
    evaluate_ood_scores,
    evaluate_unified_space_ood,
    evaluate_uq_fidelity,
    select_auxiliary_features,
)


def _random_orthogonal(d: int, rng: np.random.Generator) -> np.ndarray:
    q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    return q.astype(np.float32)


@pytest.mark.parametrize("task_type", ["regression", "binclass", "multiclass"])
def test_linear_probe_all_tasks_and_gap_recovery(task_type: str) -> None:
    rng = np.random.default_rng(0)
    n_tr, n_va, n_te, d = 180, 60, 60, 12
    z_tr = rng.standard_normal((n_tr, d)).astype(np.float32)
    z_va = rng.standard_normal((n_va, d)).astype(np.float32)
    z_te = rng.standard_normal((n_te, d)).astype(np.float32)

    w = rng.standard_normal(d).astype(np.float32)

    def _make_y(z: np.ndarray) -> np.ndarray:
        s = z @ w
        if task_type == "regression":
            return (s * 50.0 + 200.0).astype(np.float32)
        if task_type == "binclass":
            return (s > 0.0).astype(np.int64)
        bins = np.quantile(z_tr @ w, [0.33, 0.66])
        return np.digitize(s, bins).astype(np.int64)

    y_tr, y_va, y_te = _make_y(z_tr), _make_y(z_va), _make_y(z_te)
    res = evaluate_linear_probe(z_tr, y_tr, z_va, y_va, z_te, y_te, task_type=task_type, seed=0)

    assert np.isfinite(res.test_score)
    if task_type == "regression":
        assert res.test_metrics["r2"] > 0.98
    else:
        assert res.test_metrics["accuracy"] > 0.85

    assert compute_gap_recovery(0.85, 0.80, 0.90) == pytest.approx(0.5)
    assert np.isnan(compute_gap_recovery(0.85, 0.90, 0.90))


def test_knn_geometry_rotation_invariance_and_chunk_parity() -> None:
    rng = np.random.default_rng(7)
    d = 16
    centers = rng.standard_normal((3, d)).astype(np.float32) * 3.0

    def _sample(n_per_class: int) -> tuple[np.ndarray, np.ndarray]:
        zs, ys = [], []
        for c in range(3):
            zs.append(centers[c] + rng.standard_normal((n_per_class, d)).astype(np.float32) * 0.4)
            ys.append(np.full(n_per_class, c, dtype=np.int64))
        return np.vstack(zs), np.concatenate(ys)

    z_tr, y_tr = _sample(40)
    z_te, y_te = _sample(20)

    res_full = evaluate_knn_geometry(z_tr, y_tr, z_te, y_te, "multiclass", k_neighbors=7, chunk_size=1000)
    res_chunked = evaluate_knn_geometry(z_tr, y_tr, z_te, y_te, "multiclass", k_neighbors=7, chunk_size=11)
    assert res_full.knn_score == pytest.approx(res_chunked.knn_score, abs=1e-6)
    assert res_full.purity_at_k == pytest.approx(res_chunked.purity_at_k, abs=1e-6)
    assert res_full.nn1_score > 0.95
    assert res_full.purity_at_k is not None and res_full.purity_at_k > 0.95
    assert res_full.silhouette_cosine is not None and res_full.silhouette_cosine > 0.4

    # Strict orthogonal rotation invariance
    q = _random_orthogonal(d, rng)
    res_rot = evaluate_knn_geometry(z_tr @ q, y_tr, z_te @ q, y_te, "multiclass", k_neighbors=7)
    assert res_rot.knn_score == pytest.approx(res_full.knn_score, abs=1e-6)
    assert res_rot.purity_at_k == pytest.approx(res_full.purity_at_k, abs=1e-6)

    # Regression k-NN path
    y_tr_reg = (z_tr[:, 0] * 2.0 - z_tr[:, 1]).astype(np.float32)
    y_te_reg = (z_te[:, 0] * 2.0 - z_te[:, 1]).astype(np.float32)
    res_reg = evaluate_knn_geometry(z_tr, y_tr_reg, z_te, y_te_reg, "regression", k_neighbors=7)
    assert res_reg.knn_score < 0.0 and res_reg.knn_r2 is not None and res_reg.knn_r2 > 0.5


def test_auxiliary_feature_selection_and_transfer_probe() -> None:
    rng = np.random.default_rng(11)
    n_tr, n_va, n_te = 160, 50, 50

    def _make_block(n: int) -> tuple[np.ndarray, np.ndarray]:
        x = rng.standard_normal((n, 5)).astype(np.float32)
        x[:, 4] = 3.0  # Constant -> must be ignored
        # Primary target depends strongly on col 0 and col 1, independent of col 2 and col 3
        y = (4.0 * x[:, 0] - 3.0 * x[:, 1] + 0.1 * rng.standard_normal(n)).astype(np.float32)
        return x, y

    x_tr, y_tr = _make_block(n_tr)
    x_va, _ = _make_block(n_va)
    x_te, _ = _make_block(n_te)

    selected = select_auxiliary_features(x_tr, y_tr, task_type="regression", n_aux=2)
    assert len(selected) == 2
    assert set(selected) == {2, 3}  # Least target-correlated non-constant columns

    # Embed all features linearly into z -> transfer probe must recover aux features accurately
    proj = rng.standard_normal((5, 16)).astype(np.float32)
    z_tr, z_va, z_te = x_tr @ proj, x_va @ proj, x_te @ proj

    res = evaluate_auxiliary_transfer(
        z_tr, x_tr[:, selected], z_va, x_va[:, selected], z_te, x_te[:, selected], feature_indices=selected
    )
    assert res.selected_feature_indices == selected
    assert res.mean_r2 > 0.98
    assert res.mean_neg_rmse < 0.0


def test_uq_fidelity_and_unified_space_ood_stability() -> None:
    rng = np.random.default_rng(21)
    n_tr, n_va, n_te, d = 120, 40, 80, 12
    z_tr = rng.standard_normal((n_tr, d)).astype(np.float32)
    z_va = rng.standard_normal((n_va, d)).astype(np.float32)
    z_te = rng.standard_normal((n_te, d)).astype(np.float32)

    # True UQ correlates with norm/direction of z
    uq_tr = np.exp(0.6 * z_tr[:, 0] + 0.3 * z_tr[:, 1]).astype(np.float32)
    uq_va = np.exp(0.6 * z_va[:, 0] + 0.3 * z_va[:, 1]).astype(np.float32)
    uq_te = np.exp(0.6 * z_te[:, 0] + 0.3 * z_te[:, 1]).astype(np.float32)

    uq_res = evaluate_uq_fidelity(
        z_tr, uq_tr, z_va, uq_va, z_te, uq_te, direct_uq_test=uq_te * 1.05
    )
    assert uq_res.probe_spearman > 0.95
    assert uq_res.probe_r2 > 0.90
    assert uq_res.direct_head_spearman is not None and uq_res.direct_head_spearman > 0.99

    # Constant UQ edge case (e.g. k=1 MLP) must never produce NaN
    zero_uq = np.zeros(n_te, dtype=np.float32)
    uq_const = evaluate_uq_fidelity(z_tr, np.zeros(n_tr), z_va, np.zeros(n_va), z_te, zero_uq)
    assert uq_const.probe_spearman == 0.0 and np.isfinite(uq_const.probe_r2)

    # OOD detection on structured directional clusters (both radial and angular OOD shift)
    c0 = np.zeros(d, dtype=np.float32)
    c1 = np.zeros(d, dtype=np.float32)
    c_ood = np.zeros(d, dtype=np.float32)
    c0[0] = 3.0
    c1[1] = 3.0
    c_ood[0], c_ood[1], c_ood[2] = -3.0, -3.0, 3.0

    def _sample_id(n: int) -> np.ndarray:
        half = n // 2
        a = c0 + rng.standard_normal((half, d)).astype(np.float32) * 0.5
        b = c1 + rng.standard_normal((n - half, d)).astype(np.float32) * 0.5
        return np.vstack([a, b])

    z_id_tr = _sample_id(n_tr)
    z_id_te = _sample_id(n_te)
    z_ood = (c_ood + rng.standard_normal((n_te, d)).astype(np.float32) * 0.5)
    ood_report = evaluate_unified_space_ood(z_id_tr, z_id_te, uq_te, z_ood=z_ood, k_neighbors=7, seed=0)
    assert ood_report.mahal_ood is not None and ood_report.mahal_ood.auroc > 0.99
    assert ood_report.cosine_ood is not None and ood_report.cosine_ood.auroc > 0.99
    assert np.isfinite(ood_report.mahal_high_uq_auroc)

    # High-dimensional singular covariance stability (N < d)
    id_dir = np.zeros(32, dtype=np.float32)
    ood_dir = np.zeros(32, dtype=np.float32)
    id_dir[0] = 3.0
    ood_dir[0], ood_dir[1] = -3.0, 3.0
    z_small_tr = id_dir + rng.standard_normal((10, 32)).astype(np.float32) * 0.5
    z_small_id = id_dir + rng.standard_normal((15, 32)).astype(np.float32) * 0.5
    z_small_ood = ood_dir + rng.standard_normal((15, 32)).astype(np.float32) * 0.5
    ms = MahalanobisScorer().fit(z_small_tr)
    cs = CosineKNNScorer(k_neighbors=5).fit(z_small_tr)
    m_res = evaluate_ood_scores(ms.score(z_small_id), ms.score(z_small_ood))
    c_res = evaluate_ood_scores(cs.score(z_small_id), cs.score(z_small_ood))
    assert m_res.auroc > 0.95 and c_res.auroc > 0.95