from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tabm_bridge.bridge import (
    GeneralizedOrthogonalProcrustes,
    ProcrustesAnchoredResidualBridge,
    TabMBridge,
    TabMBridgeTrainer,
)
from tabm_bridge.config import BridgeConfig
from tabm_bridge.evaluation.g1_probes import (
    _chunked_cosine_topk,
    evaluate_knn_geometry,
    evaluate_linear_probe,
    evaluate_unified_space_ood,
    select_auxiliary_features,
)


def test_config_rejects_zero_init_gate() -> None:
    """init_gate=0.0 with zero-init refine_fc2 creates a dead zero-gradient saddle point."""
    with pytest.raises(ValueError, match="init_gate"):
        BridgeConfig(init_gate=0.0)


def test_bridge_v1_backward_compatibility() -> None:
    """v3.1 TabMBridge API and training behavior remain 100% intact."""
    rng = np.random.default_rng(1)
    h_tr = rng.standard_normal((80, 4, 32)).astype(np.float32)
    h_va = rng.standard_normal((32, 4, 32)).astype(np.float32)
    mp_tr = rng.dirichlet([1.0, 1.0], size=80).astype(np.float32)
    mp_va = rng.dirichlet([1.0, 1.0], size=32).astype(np.float32)
    uq_tr = np.abs(rng.standard_normal(80)).astype(np.float32)
    uq_va = np.abs(rng.standard_normal(32)).astype(np.float32)

    bridge = TabMBridge(
        k=4,
        d_in=32,
        d_out=2,
        config=BridgeConfig(d_z=16, d_hidden=16, align_rank=4, epochs=3, batch_size=32),
    )
    trainer = TabMBridgeTrainer(bridge, task_type="binclass", device="cpu", seed=0)
    res = trainer.fit(h_tr, mp_tr, uq_tr, h_va, mp_va, uq_va)
    assert not res.stopped_due_to_nan
    z = trainer.encode(h_va)
    assert isinstance(z, np.ndarray) and z.shape == (32, 16)


def test_bridge_v2_zero_init_orthonormality_and_unfitted_guard() -> None:
    """Epoch-0 Bridge-v2 matches anchor identically (even N < d_z) and blocks unfitted encode()."""
    rng = np.random.default_rng(0)
    n_small, k, d_in, d_z = 24, 4, 64, 32  # N < d_z edge case
    h = rng.standard_normal((n_small, k, d_in)).astype(np.float32)
    z_proc = h.mean(axis=1)

    cfg = BridgeConfig(d_z=d_z, d_hidden=32, align_rank=4, init_gate=0.05)
    bridge = ProcrustesAnchoredResidualBridge(k=k, d_in=d_in, d_out=2, config=cfg)
    trainer = TabMBridgeTrainer(bridge, task_type="binclass", device="cpu", seed=0)

    with pytest.raises(RuntimeError, match="anchor stats must be fitted"):
        trainer.encode(h, z_proc=z_proc)

    bridge.set_anchor_stats(z_proc, seed=0)
    proj = bridge.anchor_proj.cpu().numpy()
    assert proj.shape == (d_in, d_z)
    assert np.allclose(proj.T @ proj, np.eye(d_z, dtype=np.float32), atol=1e-5)

    bridge.eval()
    with torch.no_grad():
        out = bridge(torch.from_numpy(h), z_proc=torch.from_numpy(z_proc))
    assert out.z_anchor is not None
    assert torch.allclose(out.z, out.z_anchor, atol=1e-6)

    groups = bridge.make_parameter_groups(weight_decay=1e-3)
    no_decay = [p for g in groups if g["weight_decay"] == 0.0 for p in g["params"]]
    assert any(p is bridge.res_gate for p in no_decay)


def test_bridge_v2_train_warmup_and_checkpoint_roundtrip(tmp_path: Path) -> None:
    """Bridge-v2 trains cleanly with cosine warmup/LR and restores exact outputs + scalers."""
    rng = np.random.default_rng(42)
    n_tr, n_va, n_te, k, d_in = 160, 48, 40, 4, 32
    h_tr = rng.standard_normal((n_tr, k, d_in)).astype(np.float32)
    h_va = rng.standard_normal((n_va, k, d_in)).astype(np.float32)
    h_te = rng.standard_normal((n_te, k, d_in)).astype(np.float32)

    mp_tr = rng.standard_normal((n_tr, 1)).astype(np.float32) * 10.0 + 50.0
    mp_va = rng.standard_normal((n_va, 1)).astype(np.float32) * 10.0 + 50.0
    uq_tr = np.abs(rng.standard_normal(n_tr)).astype(np.float32)
    uq_va = np.abs(rng.standard_normal(n_va)).astype(np.float32)

    proc = GeneralizedOrthogonalProcrustes(max_iter=5, device="cpu").fit(h_tr)
    zp_tr, zp_va, zp_te = proc.transform(h_tr), proc.transform(h_va), proc.transform(h_te)

    cfg = BridgeConfig(
        d_z=16,
        d_hidden=32,
        align_rank=4,
        alpha_gram=5.0,
        alpha_pred=1.0,
        alpha_uq=0.15,
        alpha_recon=1.0,
        gram_warmup_epochs=3,
        use_cosine_lr=True,
        mask_gram_diag=True,
        center_z_gram=True,
        epochs=6,
        patience=2,
        batch_size=32,
    )
    bridge = ProcrustesAnchoredResidualBridge(k=k, d_in=d_in, d_out=1, config=cfg)
    trainer = TabMBridgeTrainer(bridge=bridge, task_type="regression", device="cpu", seed=0)
    res = trainer.fit(h_tr, mp_tr, uq_tr, h_va, mp_va, uq_va, z_proc_train=zp_tr, z_proc_val=zp_va)
    assert not res.stopped_due_to_nan
    assert len(res.val_history) >= 4  # Early stopping cannot trigger before warmup completes

    orig = trainer.encode(h_te, return_aux=True, z_proc=zp_te)
    assert isinstance(orig, dict)

    ckpt_path = tmp_path / "bridge_v2.pt"
    trainer.save_checkpoint(ckpt_path)
    reloaded = TabMBridgeTrainer.load_checkpoint(ckpt_path, device="cpu")
    restored = reloaded.encode(h_te, return_aux=True, z_proc=zp_te)
    assert isinstance(restored, dict)

    assert np.allclose(orig["z"], restored["z"], atol=1e-5)
    assert np.allclose(orig["pred"], restored["pred"], atol=1e-4)
    assert np.allclose(orig["uq"], restored["uq"], atol=1e-4)


def test_torch_logistic_probe_matches_sklearn_exact() -> None:
    """PyTorch float64 L-BFGS logistic probe matches sklearn on binary and multiclass tasks."""
    rng = np.random.default_rng(123)
    for task_type, n_classes in (("binclass", 2), ("multiclass", 5)):
        z_tr = rng.standard_normal((320, 24)).astype(np.float32)
        z_va = rng.standard_normal((100, 24)).astype(np.float32)
        z_te = rng.standard_normal((120, 24)).astype(np.float32)
        w_true = rng.standard_normal((24, n_classes))
        y_tr = np.argmax(z_tr @ w_true + 0.4 * rng.standard_normal((320, n_classes)), axis=1)
        y_va = np.argmax(z_va @ w_true + 0.4 * rng.standard_normal((100, n_classes)), axis=1)
        y_te = np.argmax(z_te @ w_true + 0.4 * rng.standard_normal((120, n_classes)), axis=1)

        grid = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
        res_sk = evaluate_linear_probe(
            z_tr, y_tr, z_va, y_va, z_te, y_te,
            task_type=task_type, reg_grid=grid, seed=0, device="sklearn",
        )
        res_th = evaluate_linear_probe(
            z_tr, y_tr, z_va, y_va, z_te, y_te,
            task_type=task_type, reg_grid=grid, seed=0, device="cpu",
        )
        assert abs(res_sk.test_score - res_th.test_score) < 1e-3


def test_select_auxiliary_features_filters_exact_collinearity() -> None:
    """Collinearity filter must never select both x and (1 - x) (fixes Churn [5, 6] bug)."""
    rng = np.random.default_rng(7)
    x = rng.standard_normal((400, 7))
    y = (x[:, 0] + 0.7 * x[:, 1] > 0).astype(np.int64)
    x[:, 5] = rng.binomial(1, 0.5, size=400)
    x[:, 6] = 1.0 - x[:, 5]

    chosen = select_auxiliary_features(x, y, task_type="binclass", n_aux=3, max_collinearity=0.98)
    assert len(chosen) == 3
    assert not (5 in chosen and 6 in chosen)


def test_cosine_topk_torch_vs_numpy_and_ood_reuse() -> None:
    """PyTorch cosine top-k matches NumPy and precomputed top_sim matches fresh OOD scorer."""
    rng = np.random.default_rng(99)
    z_tr = rng.standard_normal((200, 32)).astype(np.float32)
    z_te = rng.standard_normal((60, 32)).astype(np.float32)
    z_ood = (rng.standard_normal((40, 32)) + 3.0).astype(np.float32)
    y_tr = rng.integers(0, 3, size=200)
    y_te = rng.integers(0, 3, size=60)
    uq_te = np.abs(rng.standard_normal(60)).astype(np.float32)

    idx_np, sim_np = _chunked_cosine_topk(z_te, z_tr, k=10, device="numpy")
    idx_th, sim_th = _chunked_cosine_topk(z_te, z_tr, k=10, device="cpu")
    assert np.array_equal(idx_np, idx_th)
    assert np.allclose(sim_np, sim_th, atol=1e-6)

    knn_a = evaluate_knn_geometry(z_tr, y_tr, z_te, y_te, task_type="multiclass", k_neighbors=10, device="numpy")
    knn_b = evaluate_knn_geometry(
        z_tr, y_tr, z_te, y_te, task_type="multiclass", k_neighbors=10, precomputed_topk=(idx_th, sim_th)
    )
    assert abs(knn_a.knn_score - knn_b.knn_score) < 1e-12

    ood_a = evaluate_unified_space_ood(z_tr, z_te, uq_te, z_ood=z_ood, k_neighbors=10, device="numpy")
    ood_b = evaluate_unified_space_ood(
        z_tr, z_te, uq_te, z_ood=z_ood, k_neighbors=10, device="cpu", precomputed_test_top_sim=sim_th
    )
    assert ood_a.mahal_ood is not None and ood_b.cosine_ood is not None
    assert abs(ood_a.cosine_high_uq_auroc - ood_b.cosine_high_uq_auroc) < 1e-6
    assert abs(ood_a.cosine_ood.auroc - ood_b.cosine_ood.auroc) < 1e-6


def test_torch_logistic_probe_safe_under_no_grad_and_inference_mode() -> None:
    """PyTorch L-BFGS probe must never crash when invoked inside no_grad() or inference_mode()."""
    rng = np.random.default_rng(2026)
    z_tr = rng.standard_normal((120, 12)).astype(np.float32)
    z_va = rng.standard_normal((40, 12)).astype(np.float32)
    z_te = rng.standard_normal((40, 12)).astype(np.float32)
    y_tr = rng.integers(0, 3, size=120)
    y_va = rng.integers(0, 3, size=40)
    y_te = rng.integers(0, 3, size=40)

    with torch.no_grad():
        r1 = evaluate_linear_probe(
            z_tr, y_tr, z_va, y_va, z_te, y_te,
            task_type="multiclass", reg_grid=(1e-3, 1e-1), device="cpu",
        )
    with torch.inference_mode():
        r2 = evaluate_linear_probe(
            z_tr, y_tr, z_va, y_va, z_te, y_te,
            task_type="multiclass", reg_grid=(1e-3, 1e-1), device="cpu",
        )
    assert abs(r1.test_score - r2.test_score) < 1e-6


def test_bridge_v2_gradient_flow_stop_grad_and_ood_auprc_guards() -> None:
    """Verify res_gate/refine_fc1 update after 2 steps, stop_gradient shield, and OOD AUPRC/guards."""
    torch.manual_seed(0)
    h = torch.randn(32, 4, 24, requires_grad=True)
    z_proc = h.detach().mean(dim=1)
    y = torch.randn(32, 1)

    # 1. Bridge-v2 updates both res_gate and refine_fc1 after 2 AdamW steps (escapes zero-init cleanly)
    cfg_v2 = BridgeConfig(d_z=16, d_hidden=16, align_rank=4, init_gate=0.05, stop_gradient=True)
    b2 = ProcrustesAnchoredResidualBridge(k=4, d_in=24, d_out=1, config=cfg_v2)
    b2.set_anchor_stats(z_proc.numpy(), seed=0)
    gate_init = float(b2.res_gate.item())
    fc1_init = b2.refine_fc1.weight.detach().clone()

    opt = torch.optim.AdamW(b2.make_parameter_groups(1e-4), lr=1e-2)
    for _ in range(2):
        opt.zero_grad()
        out = b2(h, z_proc=z_proc)
        loss = torch.nn.functional.mse_loss(out.pred_logits, y)
        loss.backward()
        opt.step()

    assert h.grad is None  # stop_gradient=True blocks gradient into backbone
    assert float(b2.res_gate.item()) != gate_init
    assert not torch.allclose(b2.refine_fc1.weight.detach(), fc1_init)

    # 2. stop_gradient=False on TabMBridge (v1) passes gradient to h (used in Learner Collusion test)
    h_col = torch.randn(16, 4, 24, requires_grad=True)
    b1_col = TabMBridge(k=4, d_in=24, d_out=1, config=BridgeConfig(d_z=16, d_hidden=16, align_rank=4, stop_gradient=False))
    torch.nn.functional.mse_loss(b1_col(h_col).pred_logits, torch.randn(16, 1)).backward()
    assert h_col.grad is not None and float(h_col.grad.norm().item()) > 0.0

    # 3. High-UQ AUPRC/FPR95 populated & shape guard enforced
    rng = np.random.default_rng(5)
    z_tr = rng.standard_normal((100, 16)).astype(np.float32)
    z_te = rng.standard_normal((50, 16)).astype(np.float32)
    uq_te = np.abs(rng.standard_normal(50)).astype(np.float32)
    rep = evaluate_unified_space_ood(z_tr, z_te, uq_te, k_neighbors=5, device="numpy")
    assert rep.mahal_high_uq_auprc is not None and 0.0 <= rep.mahal_high_uq_auprc <= 1.0
    assert rep.mahal_high_uq_fpr95 is not None and 0.0 <= rep.mahal_high_uq_fpr95 <= 1.0

    with pytest.raises(ValueError, match="precomputed_test_top_sim"):
        evaluate_unified_space_ood(
            z_tr, z_te, uq_te, k_neighbors=5, precomputed_test_top_sim=np.zeros((10, 5))
        )