from __future__ import annotations

import numpy as np
import pytest
import torch

from tabm_bridge.bridge import (
    ConcatBaseline,
    GeneralizedOrthogonalProcrustes,
    PCAConcatBaseline,
    SimpleMeanBaseline,
    TabMBestSubmodelBaseline,
    TabMBridge,
    TabMBridgeTrainer,
    compute_bridge_losses,
    compute_consensus_gram,
    compute_misalignment_diagnostics,
)
from tabm_bridge.config import BridgeConfig


def _random_orthogonal_matrix(d: int, rng: np.random.Generator) -> np.ndarray:
    mat = rng.standard_normal((d, d))
    q, _ = np.linalg.qr(mat)
    return q.astype(np.float32)


def test_representation_baselines_shapes_float16_and_tabm_b() -> None:
    rng = np.random.default_rng(0)
    n_tr, n_va, k, d = 80, 40, 6, 16
    h_tr = rng.standard_normal((n_tr, k, d)).astype(np.float16)
    h_va = rng.standard_normal((n_va, k, d)).astype(np.float16)

    # Head 4 is exact on validation regression -> TabM[B] must pick index 4
    y_va = rng.standard_normal(n_va).astype(np.float32)
    val_preds = rng.standard_normal((n_va, k, 1)).astype(np.float32) * 5.0
    val_preds[:, 4, 0] = y_va

    tabm_b = TabMBestSubmodelBaseline().fit(val_preds, y_va, task_type="regression")
    assert tabm_b.best_index_ == 4
    z_b = tabm_b.transform(h_va)
    assert z_b.shape == (n_va, d) and z_b.dtype == np.float32
    assert np.allclose(z_b, h_va[:, 4, :].astype(np.float32))

    z_mean = SimpleMeanBaseline().fit(h_tr).transform(h_va)
    assert z_mean.shape == (n_va, d) and z_mean.dtype == np.float32
    assert np.allclose(z_mean, h_va.astype(np.float32).mean(axis=1), atol=1e-4)

    z_concat = ConcatBaseline().fit(h_tr).transform(h_va)
    assert z_concat.shape == (n_va, k * d)

    # Subsampled fit + chunked transform
    pca = PCAConcatBaseline(n_components=12, max_fit_samples=50, batch_size=13, random_state=0)
    z_pca = pca.fit(h_tr).transform(h_va)
    assert z_pca.shape == (n_va, 12) and z_pca.dtype == np.float32
    assert np.isfinite(z_pca).all()


def test_procrustes_orthogonality_monotone_convergence_and_recovery() -> None:
    rng = np.random.default_rng(123)
    n_tr, n_te, k, d = 200, 60, 8, 16
    z_tr_true = rng.standard_normal((n_tr, d)).astype(np.float32)
    z_te_true = rng.standard_normal((n_te, d)).astype(np.float32)

    h_tr = np.empty((n_tr, k, d), dtype=np.float32)
    h_te = np.empty((n_te, k, d), dtype=np.float32)
    for i in range(k):
        q = _random_orthogonal_matrix(d, rng)
        shift = rng.standard_normal((1, d)).astype(np.float32) * 2.0
        h_tr[:, i, :] = z_tr_true @ q + shift
        h_te[:, i, :] = z_te_true @ q + shift

    proc = GeneralizedOrthogonalProcrustes(center=True, max_iter=20, batch_size=17).fit(h_tr)
    assert proc.rotations_ is not None
    assert len(proc.obj_history_) >= 1

    # Gower (1975) alternating minimization must be monotonically non-increasing
    for prev, curr in zip(proc.obj_history_, proc.obj_history_[1:]):
        assert curr <= prev + 1e-10

    # Exact orthogonality Q_i^T Q_i = I_d
    eye = np.eye(d, dtype=np.float32)
    for i in range(k):
        q_i = proc.rotations_[i]
        assert np.allclose(q_i.T @ q_i, eye, atol=1e-4)

    # Diagnostics on unseen test split must show near-perfect recovery
    diag = compute_misalignment_diagnostics(proc, h_te, max_samples=40)
    assert diag.mean_pairwise_cosine_after > 0.999
    assert diag.mean_pairwise_cosine_after > diag.mean_pairwise_cosine_before + 0.5
    assert diag.rel_frob_error_after < 1e-3
    assert 0.5 < diag.mean_ortho_departure <= 2.0 + 1e-5

    z_te_aligned = proc.transform(h_te)
    assert z_te_aligned.shape == (n_te, d)
    assert np.isfinite(z_te_aligned).all()


def test_consensus_gram_rotation_and_shift_invariance() -> None:
    torch.manual_seed(7)
    rng = np.random.default_rng(7)
    b, k, d = 24, 8, 16
    h = torch.randn(b, k, d)

    # Apply independent random orthogonal rotations Q_i AND branch-specific shifts c_i
    h_rot = torch.empty_like(h)
    h_rot_shift = torch.empty_like(h)
    for i in range(k):
        q = torch.from_numpy(_random_orthogonal_matrix(d, rng))
        shift = torch.randn(1, d) * 3.0
        h_rot[:, i, :] = h[:, i, :] @ q
        h_rot_shift[:, i, :] = h[:, i, :] @ q + shift

    # Uncentered Gram: strictly invariant to rotations
    g_uncentered = compute_consensus_gram(h, center=False)
    g_rot_uncentered = compute_consensus_gram(h_rot, center=False)
    assert g_uncentered.shape == (b, b)
    assert torch.allclose(g_uncentered, g_rot_uncentered, atol=1e-5)

    # Centered Gram: strictly invariant to BOTH rotations and arbitrary shifts
    g_centered = compute_consensus_gram(h, center=True)
    g_rot_shift_centered = compute_consensus_gram(h_rot_shift, center=True)
    assert torch.allclose(g_centered, g_rot_shift_centered, atol=1e-5)

    # Edge case: batch size B=1 must remain finite
    g_single = compute_consensus_gram(h[:1], center=True)
    assert g_single.shape == (1, 1) and torch.isfinite(g_single).all()


def test_tabm_bridge_low_rank_efficiency_weight_decay_and_stop_gradient() -> None:
    torch.manual_seed(0)
    b, k, d_in, d_out = 16, 32, 128, 3

    cfg_lr = BridgeConfig(d_z=64, d_hidden=64, align_mode="low_rank", align_rank=8, stop_gradient=True)
    cfg_full = BridgeConfig(d_z=64, d_hidden=64, align_mode="full", stop_gradient=True)
    bridge_lr = TabMBridge(k=k, d_in=d_in, d_out=d_out, config=cfg_lr)
    bridge_full = TabMBridge(k=k, d_in=d_in, d_out=d_out, config=cfg_full)

    # Low-rank mode must be substantially more parameter-efficient than full per-branch matrices
    assert bridge_lr.parameter_count() < bridge_full.parameter_count() * 0.35

    # Weight decay groups must exempt biases and LayerNorm scales
    groups = bridge_lr.make_parameter_groups(weight_decay=1e-4)
    assert len(groups) == 2
    assert groups[0]["weight_decay"] == 1e-4
    assert groups[1]["weight_decay"] == 0.0
    no_decay_ids = {id(p) for p in groups[1]["params"]}
    assert id(bridge_lr.align_bias) in no_decay_ids
    assert id(bridge_lr.out_norm.weight) in no_decay_ids
    assert id(bridge_lr.lora_a) not in no_decay_ids

    # 1. Stop-Gradient shield ON -> zero gradient flows into h
    h_shielded = torch.randn(b, k, d_in, requires_grad=True)
    out = bridge_lr(h_shielded)
    assert out.z.shape == (b, 64)
    assert out.pred_logits.shape == (b, d_out)
    assert out.uq_pred.shape == (b, 1)
    assert torch.allclose(out.attn_weights.sum(dim=-1), torch.ones(b), atol=1e-5)

    target_probs = torch.softmax(torch.randn(b, d_out), dim=-1)
    target_uq = torch.rand(b, 1)
    loss, parts = compute_bridge_losses(
        out, h_shielded, target_probs, target_uq, task_type="multiclass", config=cfg_lr
    )
    loss.backward()
    assert h_shielded.grad is None
    assert all(np.isfinite(v) for v in parts.values())

    # 2. Ablation stop_gradient=False -> gradients DO flow back into h
    h_open = torch.randn(b, k, d_in, requires_grad=True)
    cfg_open = BridgeConfig(d_z=64, d_hidden=64, align_mode="low_rank", align_rank=8, stop_gradient=False)
    bridge_open = TabMBridge(k=k, d_in=d_in, d_out=d_out, config=cfg_open)
    out_open = bridge_open(h_open)
    loss_open, _ = compute_bridge_losses(
        out_open, h_open, target_probs, target_uq, task_type="multiclass", config=cfg_open
    )
    loss_open.backward()
    assert h_open.grad is not None
    assert float(h_open.grad.abs().sum()) > 0.0


@pytest.mark.parametrize(
    "task_type,d_out,alpha_gram",
    [
        ("regression", 1, 1.0),
        ("binclass", 2, 1.0),
        ("multiclass", 4, 1.0),
        ("regression", 1, 0.0),  # Ablation: no Gram loss
    ],
)
def test_tabm_bridge_trainer_end_to_end_and_ablations(
    task_type: str, d_out: int, alpha_gram: float
) -> None:
    rng = np.random.default_rng(42)
    n_tr, n_va, k, d_in = 128, 32, 6, 24
    h_tr = rng.standard_normal((n_tr, k, d_in)).astype(np.float16)
    h_va = rng.standard_normal((n_va, k, d_in)).astype(np.float16)

    if task_type == "regression":
        mp_tr = (h_tr.astype(np.float32).mean(axis=(1, 2)) * 100.0)[:, None]
        mp_va = (h_va.astype(np.float32).mean(axis=(1, 2)) * 100.0)[:, None]
    else:
        logits_tr = rng.standard_normal((n_tr, d_out)).astype(np.float32)
        logits_va = rng.standard_normal((n_va, d_out)).astype(np.float32)
        mp_tr = np.exp(logits_tr) / np.exp(logits_tr).sum(axis=-1, keepdims=True)
        mp_va = np.exp(logits_va) / np.exp(logits_va).sum(axis=-1, keepdims=True)

    uq_tr = rng.uniform(0.0, 2.0, size=(n_tr, 1)).astype(np.float32)
    uq_va = rng.uniform(0.0, 2.0, size=(n_va, 1)).astype(np.float32)

    cfg = BridgeConfig(
        d_z=16,
        d_hidden=32,
        align_mode="low_rank",
        align_rank=4,
        alpha_gram=alpha_gram,
        epochs=10,
        patience=5,
        batch_size=32,
        lr=5e-3,
    )
    bridge = TabMBridge(k=k, d_in=d_in, d_out=d_out, config=cfg)
    trainer = TabMBridgeTrainer(bridge, task_type=task_type, device="cpu", seed=0)  # type: ignore[arg-type]

    res = trainer.fit(h_tr, mp_tr, uq_tr, h_va, mp_va, uq_va)
    assert not res.stopped_due_to_nan
    assert np.isfinite(res.best_val_loss)
    assert res.train_history[-1]["total"] < res.train_history[0]["total"]
    if alpha_gram == 0.0:
        assert res.val_history[-1]["gram"] == 0.0

    encoded = trainer.encode(h_va, return_aux=True)
    assert isinstance(encoded, dict)
    assert encoded["z"].shape == (n_va, 16)
    assert encoded["pred"].shape == (n_va, d_out)
    assert encoded["uq"].shape == (n_va, 1)
    assert np.isfinite(encoded["z"]).all()
    assert (encoded["uq"] >= 0.0).all()


def test_tabm_bridge_trainer_nan_guard_breaks_cleanly() -> None:
    rng = np.random.default_rng(0)
    h_tr = rng.standard_normal((32, 4, 12)).astype(np.float32)
    h_tr[0, 0, 0] = np.nan
    h_va = rng.standard_normal((16, 4, 12)).astype(np.float32)
    mp_tr = rng.standard_normal((32, 1)).astype(np.float32)
    mp_va = rng.standard_normal((16, 1)).astype(np.float32)
    uq_tr = np.ones((32, 1), dtype=np.float32)
    uq_va = np.ones((16, 1), dtype=np.float32)

    cfg = BridgeConfig(d_z=8, d_hidden=16, align_rank=4, epochs=5, batch_size=16)
    bridge = TabMBridge(k=4, d_in=12, d_out=1, config=cfg)
    trainer = TabMBridgeTrainer(bridge, task_type="regression", device="cpu", seed=0)

    res = trainer.fit(h_tr, mp_tr, uq_tr, h_va, mp_va, uq_va)
    assert res.stopped_due_to_nan is True
    # Restored weights must remain finite and usable
    z_va = trainer.encode(h_va)
    assert isinstance(z_va, np.ndarray) and np.isfinite(z_va).all()


def test_bridge_full_align_mode_config_validation_and_tabm_b_multiclass() -> None:
    with pytest.raises(ValueError):
        BridgeConfig(align_mode="low_rank", align_rank=0)
    with pytest.raises(ValueError):
        BridgeConfig(dropout=1.0)

    # Exercise align_mode='full' forward & backward path (used in architecture ablation)
    torch.manual_seed(0)
    cfg_full = BridgeConfig(d_z=16, d_hidden=16, align_mode="full", alpha_uq=0.0, stop_gradient=True)
    bridge_full = TabMBridge(k=4, d_in=12, d_out=3, config=cfg_full)
    h = torch.randn(8, 4, 12, requires_grad=True)
    out = bridge_full(h)
    assert out.z.shape == (8, 16)
    target_probs = torch.softmax(torch.randn(8, 3), dim=-1)
    loss, parts = compute_bridge_losses(
        out, h, target_probs, torch.zeros(8, 1), task_type="multiclass", config=cfg_full
    )
    loss.backward()
    assert h.grad is None
    assert parts["uq"] == 0.0 and np.isfinite(parts["total"])

    # Verify TabM[B] selection on multiclass probabilities
    rng = np.random.default_rng(1)
    y_va = rng.integers(0, 3, size=30)
    probs = np.full((30, 5, 3), 1.0 / 3.0, dtype=np.float32)
    # Make head 2 100% accurate
    probs[:, 2, :] = 0.05
    probs[np.arange(30), 2, y_va] = 0.90
    tabm_b = TabMBestSubmodelBaseline().fit(probs, y_va, task_type="multiclass")
    assert tabm_b.best_index_ == 2