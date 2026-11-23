from __future__ import annotations

import argparse
import gc
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from tabm_bridge.bridge import (
    ConcatBaseline,
    GeneralizedOrthogonalProcrustes,
    PCAConcatBaseline,
    SimpleMeanBaseline,
    TabMBestSubmodelBaseline,
    TabMBridge,
    TabMBridgeTrainer,
    compute_misalignment_diagnostics,
)
from tabm_bridge.config import BridgeAlignMode, BridgeConfig, TaskType
from tabm_bridge.data import (
    load_official_dataset,
    load_sklearn_california_housing,
    preprocess_dataset,
)
from tabm_bridge.evaluation import (
    compute_gap_recovery,
    compute_metrics,
    evaluate_auxiliary_transfer,
    evaluate_knn_geometry,
    evaluate_linear_probe,
    evaluate_unified_space_ood,
    evaluate_uq_fidelity,
    select_auxiliary_features,
)
from tabm_bridge.training import configure_reproducibility

_SLUG_TO_OFFICIAL_NAME: dict[str, str] = {
    "california": "California Housing",
    "house": "House 16H",
    "adult": "Adult",
    "churn": "Churn Modelling",
    "otto": "Otto Group Products",
    "sberbank-housing": "Sberbank Housing",
}


def _sanitize_json(obj: Any) -> Any:
    """Convert NaN/Inf floats and numpy scalars into strict RFC 8259 JSON values."""
    if isinstance(obj, dict):
        return {str(k): _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_json(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return val if math.isfinite(val) else None
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _sanitize_json(obj.tolist())
    return obj


def _resolve_dataset_paths(
    dataset_arg: str,
    reps_dir: Path,
    data_dir: Path | None,
) -> tuple[Path | None, Path | None]:
    """Resolve both G0 Drive folder (e.g. 'California Housing') and raw data folder ('california')."""
    raw_ds_dir: Path | None = None
    official_name: str | None = _SLUG_TO_OFFICIAL_NAME.get(dataset_arg)

    if data_dir is not None and dataset_arg != "california_sklearn":
        direct_raw = data_dir / dataset_arg
        if (direct_raw / "info.json").exists():
            raw_ds_dir = direct_raw
            with (direct_raw / "info.json").open("r", encoding="utf-8") as f:
                official_name = str(json.load(f).get("name", dataset_arg))
        elif data_dir.exists():
            # Reverse lookup if user passed the Drive display name directly
            for info_path in data_dir.glob("*/info.json"):
                with info_path.open("r", encoding="utf-8") as f:
                    name_in_info = str(json.load(f).get("name", ""))
                if name_in_info == dataset_arg:
                    raw_ds_dir = info_path.parent
                    official_name = name_in_info
                    break

    reps_ds_dir: Path | None = None
    candidates = [dataset_arg]
    if official_name and official_name not in candidates:
        candidates.append(official_name)
    for cand in candidates:
        cand_dir = reps_dir / cand
        if cand_dir.exists():
            reps_ds_dir = cand_dir
            break

    return reps_ds_dir, raw_ds_dir


def _load_split_npz(run_dir: Path, split: str) -> dict[str, np.ndarray]:
    path = run_dir / f"reps_{split}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing representation bundle: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def _infer_task_and_primary(
    run_dir: Path,
    raw_ds_dir: Path | None,
) -> tuple[TaskType, str | None]:
    task_type: TaskType | None = None
    primary_score: str | None = None

    report_path = run_dir / "report.json"
    if report_path.exists():
        with report_path.open("r", encoding="utf-8") as f:
            rep = json.load(f)
        task_type = rep.get("task_type")

    if raw_ds_dir is not None and (raw_ds_dir / "info.json").exists():
        with (raw_ds_dir / "info.json").open("r", encoding="utf-8") as f:
            info = json.load(f)
        if task_type is None:
            task_type = info.get("task_type")
        primary_score = info.get("score")

    if task_type not in ("regression", "binclass", "multiclass"):
        raise ValueError(
            f"Could not infer valid task_type for {run_dir}. Ensure report.json or info.json exists."
        )
    return task_type, primary_score


def _load_aux_features(
    dataset_name: str,
    raw_ds_dir: Path | None,
    y_train_ref: np.ndarray,
    y_val_ref: np.ndarray,
    y_test_ref: np.ndarray,
    task_type: TaskType,
    n_aux: int = 3,
) -> tuple[list[int], dict[str, np.ndarray]] | None:
    """Load preprocessed numerical features strictly aligned with G0 splits for Test 3."""
    if n_aux <= 0:
        return None
    try:
        if dataset_name == "california_sklearn":
            raw = load_sklearn_california_housing(seed=0)
        elif raw_ds_dir is not None and (raw_ds_dir / "info.json").exists():
            raw = load_official_dataset(raw_ds_dir)
        else:
            return None
    except Exception:
        return None

    proc = preprocess_dataset(raw, seed=0)
    if proc.n_num_features == 0 or proc.splits["train"].x_num is None:
        return None

    # Strict alignment check against saved reps_*.npz splits
    if (
        len(proc.splits["train"].y_raw) != len(y_train_ref)
        or len(proc.splits["val"].y_raw) != len(y_val_ref)
        or len(proc.splits["test"].y_raw) != len(y_test_ref)
    ):
        return None

    selected_cols = select_auxiliary_features(
        proc.splits["train"].x_num,
        proc.splits["train"].y_raw,
        task_type=task_type,
        n_aux=n_aux,
    )
    if not selected_cols:
        return None

    aux_splits = {
        s: proc.splits[s].x_num[:, selected_cols].astype(np.float32, copy=True)  # type: ignore[index]
        for s in ("train", "val", "test")
    }
    return selected_cols, aux_splits


def _evaluate_single_representation(
    method_name: str,
    z_tr: np.ndarray,
    z_va: np.ndarray,
    z_te: np.ndarray,
    y_tr: np.ndarray,
    y_va: np.ndarray,
    y_te: np.ndarray,
    uq_tr: np.ndarray,
    uq_va: np.ndarray,
    uq_te: np.ndarray,
    task_type: TaskType,
    primary_score: str | None,
    tabm_b_ref_score: float,
    ensemble_ref_score: float,
    aux_bundle: tuple[list[int], dict[str, np.ndarray]] | None,
    k_neighbors: int,
    seed: int,
    direct_uq_te: np.ndarray | None = None,
    direct_pred_te: np.ndarray | None = None,
    fit_time_sec: float = 0.0,
) -> dict[str, Any]:
    """Run all 4 official G1 tests on a single representation method."""
    # Test 1: Single-Vector Parity (Convex Linear Probe)
    t1 = evaluate_linear_probe(
        z_train=z_tr,
        y_train_raw=y_tr,
        z_val=z_va,
        y_val_raw=y_va,
        z_test=z_te,
        y_test_raw=y_te,
        task_type=task_type,
        primary_score=primary_score,
        seed=seed,
    )
    gap_rec = compute_gap_recovery(t1.test_score, tabm_b_ref_score, ensemble_ref_score)

    direct_pred_score: float | None = None
    if direct_pred_te is not None:
        direct_pred_score = float(
            compute_metrics(y_te, direct_pred_te, task_type, primary_score)["score"]
        )

    # Test 2: Zero-Training Cosine k-NN Geometry
    t2 = evaluate_knn_geometry(
        z_train=z_tr,
        y_train_raw=y_tr,
        z_test=z_te,
        y_test_raw=y_te,
        task_type=task_type,
        primary_score=primary_score,
        k_neighbors=k_neighbors,
        seed=seed,
    )

    # Test 3: Auxiliary Feature Transfer Probe (Appendix A.4)
    t3_dict: dict[str, Any] | None = None
    if aux_bundle is not None:
        feat_idx, aux_splits = aux_bundle
        t3 = evaluate_auxiliary_transfer(
            z_train=z_tr,
            aux_train=aux_splits["train"],
            z_val=z_va,
            aux_val=aux_splits["val"],
            z_test=z_te,
            aux_test=aux_splits["test"],
            feature_indices=feat_idx,
            seed=seed,
        )
        t3_dict = asdict(t3)

    # Test 4a: UQ Fidelity
    t4_uq = evaluate_uq_fidelity(
        z_train=z_tr,
        uq_train=uq_tr,
        z_val=z_va,
        uq_val=uq_va,
        z_test=z_te,
        uq_test=uq_te,
        direct_uq_test=direct_uq_te,
        seed=seed,
    )

    # Test 4b: Unified-Space Distance OOD / High-Uncertainty Detection
    t4_ood = evaluate_unified_space_ood(
        z_train=z_tr,
        z_test=z_te,
        uq_test=uq_te,
        z_ood=None,
        k_neighbors=k_neighbors,
        seed=seed,
    )

    return {
        "method": method_name,
        "dim": int(z_tr.shape[1]),
        "fit_time_sec": float(fit_time_sec),
        "test1_parity": {
            "linear_test_score": t1.test_score,
            "linear_val_score": float(t1.val_metrics["score"]),
            "best_reg_param": t1.best_reg_param,
            "gap_recovery": gap_rec,
            "direct_distilled_score": direct_pred_score,
            "test_metrics": t1.test_metrics,
        },
        "test2_geometry": asdict(t2),
        "test3_transfer": t3_dict,
        "test4_uq": asdict(t4_uq),
        "test4_ood": asdict(t4_ood),
    }


def _evaluate_run(
    dataset_name: str,
    reps_ds_dir: Path,
    raw_ds_dir: Path | None,
    variant: str,
    seed: int,
    args: argparse.Namespace,
    aux_cache: dict[str, tuple[list[int], dict[str, np.ndarray]] | None],
) -> dict[str, Any]:
    run_dir = reps_ds_dir / variant / f"seed_{seed}"
    task_type, primary_score = _infer_task_and_primary(run_dir, raw_ds_dir)

    tr = _load_split_npz(run_dir, "train")
    va = _load_split_npz(run_dir, "val")
    te = _load_split_npz(run_dir, "test")

    h_tr, h_va, h_te = tr["embeddings"], va["embeddings"], te["embeddings"]
    y_tr, y_va, y_te = tr["y_raw"], va["y_raw"], te["y_raw"]
    mp_tr, mp_va, mp_te = tr["mean_pred"], va["mean_pred"], te["mean_pred"]
    uq_tr, uq_va, uq_te = (
        tr["epistemic_uncertainty"],
        va["epistemic_uncertainty"],
        te["epistemic_uncertainty"],
    )

    # Normalize shapes for Bridge training
    mp_tr_2d = mp_tr[:, None] if mp_tr.ndim == 1 else mp_tr
    mp_va_2d = mp_va[:, None] if mp_va.ndim == 1 else mp_va
    _, k, d_in = h_tr.shape
    d_out = mp_tr_2d.shape[1]

    # Cache auxiliary transfer targets per dataset
    if dataset_name not in aux_cache:
        aux_cache[dataset_name] = _load_aux_features(
            dataset_name=dataset_name,
            raw_ds_dir=raw_ds_dir,
            y_train_ref=y_tr,
            y_val_ref=y_va,
            y_test_ref=y_te,
            task_type=task_type,
            n_aux=args.n_aux_features,
        )
    aux_bundle = aux_cache[dataset_name]

    # Reference G0 Ensemble and TabM[B] prediction scores
    ensemble_test_score = float(compute_metrics(y_te, mp_te, task_type, primary_score)["score"])

    t0 = time.perf_counter()
    tabm_b = TabMBestSubmodelBaseline().fit(
        va["per_head_preds"], y_va, task_type=task_type, primary_score=primary_score
    )
    t_tabm_b = time.perf_counter() - t0
    assert tabm_b.best_index_ is not None
    tabm_b_pred_score = float(
        compute_metrics(
            y_te, te["per_head_preds"][:, tabm_b.best_index_, :], task_type, primary_score
        )["score"]
    )

    methods_out: dict[str, dict[str, Any]] = {}

    def _run_eval(
        name: str,
        z_tr_m: np.ndarray,
        z_va_m: np.ndarray,
        z_te_m: np.ndarray,
        fit_sec: float,
        direct_uq_te: np.ndarray | None = None,
        direct_pred_te: np.ndarray | None = None,
    ) -> None:
        methods_out[name] = _evaluate_single_representation(
            method_name=name,
            z_tr=z_tr_m,
            z_va=z_va_m,
            z_te=z_te_m,
            y_tr=y_tr,
            y_va=y_va,
            y_te=y_te,
            uq_tr=uq_tr,
            uq_va=uq_va,
            uq_te=uq_te,
            task_type=task_type,
            primary_score=primary_score,
            tabm_b_ref_score=tabm_b_pred_score,
            ensemble_ref_score=ensemble_test_score,
            aux_bundle=aux_bundle,
            k_neighbors=args.k_neighbors,
            seed=seed,
            direct_uq_te=direct_uq_te,
            direct_pred_te=direct_pred_te,
            fit_time_sec=fit_sec,
        )

    # 1. TabM[B]
    _run_eval("tabm_b", tabm_b.transform(h_tr), tabm_b.transform(h_va), tabm_b.transform(h_te), t_tabm_b)

    # 2. Simple Mean (unaligned)
    t0 = time.perf_counter()
    sm = SimpleMeanBaseline().fit(h_tr)
    z_tr_m, z_va_m, z_te_m = sm.transform(h_tr), sm.transform(h_va), sm.transform(h_te)
    _run_eval("simple_mean", z_tr_m, z_va_m, z_te_m, time.perf_counter() - t0)
    del z_tr_m, z_va_m, z_te_m

    # 3. PCA(Concat)
    t0 = time.perf_counter()
    pca = PCAConcatBaseline(
        n_components=args.d_z,
        max_fit_samples=args.pca_max_fit_samples,
        random_state=seed,
    ).fit(h_tr)
    t_pca = time.perf_counter() - t0
    _run_eval("pca_concat", pca.transform(h_tr), pca.transform(h_va), pca.transform(h_te), t_pca)
    del pca

    # 4. Optional full Concat (high-dim upper bound; opt-in for 12 GB RAM safety)
    if args.include_concat:
        t0 = time.perf_counter()
        cb = ConcatBaseline().fit(h_tr)
        z_tr_c, z_va_c, z_te_c = cb.transform(h_tr), cb.transform(h_va), cb.transform(h_te)
        _run_eval("concat", z_tr_c, z_va_c, z_te_c, time.perf_counter() - t0)
        del z_tr_c, z_va_c, z_te_c

    # 5. Generalized Orthogonal Procrustes + Misalignment Diagnostics
    t0 = time.perf_counter()
    proc = GeneralizedOrthogonalProcrustes(
        center=True,
        max_iter=args.procrustes_max_iter,
        max_fit_samples=args.procrustes_max_fit_samples,
        random_state=seed,
    ).fit(h_tr)
    t_proc = time.perf_counter() - t0
    misalignment = compute_misalignment_diagnostics(proc, h_te, max_samples=2048, seed=seed)
    _run_eval("procrustes", proc.transform(h_tr), proc.transform(h_va), proc.transform(h_te), t_proc)
    del proc

    # 6. TabM-Bridge (Main Method)
    configure_reproducibility(seed)
    align_mode: BridgeAlignMode = args.align_mode
    bridge_cfg = BridgeConfig(
        d_z=args.d_z,
        d_hidden=args.d_hidden,
        align_mode=align_mode,
        align_rank=args.align_rank,
        center_gram=not args.uncentered_gram,
        alpha_gram=args.alpha_gram,
        alpha_pred=args.alpha_pred,
        alpha_uq=args.alpha_uq,
        stop_gradient=True,
        lr=args.bridge_lr,
        weight_decay=args.bridge_wd,
        epochs=args.bridge_epochs,
        patience=args.bridge_patience,
        batch_size=args.bridge_batch_size,
    )
    bridge = TabMBridge(k=k, d_in=d_in, d_out=d_out, config=bridge_cfg)
    bridge_trainer = TabMBridgeTrainer(
        bridge=bridge,
        task_type=task_type,
        device=args.device,
        seed=seed,
    )
    bridge_res = bridge_trainer.fit(h_tr, mp_tr_2d, uq_tr, h_va, mp_va_2d, uq_va)
    z_tr_b = bridge_trainer.encode(h_tr, return_aux=False)
    z_va_b = bridge_trainer.encode(h_va, return_aux=False)
    te_b = bridge_trainer.encode(h_te, return_aux=True)
    assert isinstance(z_tr_b, np.ndarray) and isinstance(z_va_b, np.ndarray) and isinstance(te_b, dict)

    _run_eval(
        "tabm_bridge",
        z_tr_b,
        z_va_b,
        te_b["z"],
        bridge_res.elapsed_sec,
        direct_uq_te=te_b["uq"],
        direct_pred_te=te_b["pred"],
    )
    methods_out["tabm_bridge"]["bridge_train"] = {
        "best_epoch": bridge_res.best_epoch,
        "best_val_loss": bridge_res.best_val_loss,
        "stopped_due_to_nan": bridge_res.stopped_due_to_nan,
        "parameter_count": bridge.parameter_count(),
    }

    if args.save_bridge_dir is not None:
        ckpt_dir = args.save_bridge_dir / dataset_name / variant / f"seed_{seed}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "bridge_state_dict": {
                    n: p.detach().cpu() for n, p in bridge.state_dict().items()
                },
                "bridge_config": asdict(bridge_cfg),
                "k": k,
                "d_in": d_in,
                "d_out": d_out,
                "task_type": task_type,
            },
            ckpt_dir / "bridge_checkpoint.pt",
        )

    del bridge, bridge_trainer, z_tr_b, z_va_b, te_b

    # 7. Optional MLP Embedding Baseline (if matching mlp/seed_<seed> exists)
    mlp_dir = reps_ds_dir / "mlp" / f"seed_{seed}"
    if (mlp_dir / "reps_train.npz").exists() and (mlp_dir / "reps_test.npz").exists():
        m_tr = _load_split_npz(mlp_dir, "train")
        m_va = _load_split_npz(mlp_dir, "val")
        m_te = _load_split_npz(mlp_dir, "test")
        z_tr_mlp = np.asarray(m_tr["embeddings"][:, 0, :], dtype=np.float32)
        z_va_mlp = np.asarray(m_va["embeddings"][:, 0, :], dtype=np.float32)
        z_te_mlp = np.asarray(m_te["embeddings"][:, 0, :], dtype=np.float32)
        _run_eval("mlp_emb", z_tr_mlp, z_va_mlp, z_te_mlp, 0.0)
        del m_tr, m_va, m_te, z_tr_mlp, z_va_mlp, z_te_mlp

    run_record = {
        "dataset": dataset_name,
        "variant": variant,
        "seed": seed,
        "task_type": task_type,
        "primary_score": primary_score,
        "k": int(k),
        "d_in": int(d_in),
        "g0_references": {
            "ensemble_test_score": ensemble_test_score,
            "tabm_b_pred_score": tabm_b_pred_score,
            "tabm_b_head_index": int(tabm_b.best_index_),
        },
        "misalignment_diagnostics": asdict(misalignment),
        "methods": methods_out,
    }

    del tr, va, te, h_tr, h_va, h_te
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return run_record


def _export_summary_tables(records: list[dict[str, Any]], out_dir: Path) -> None:
    """Aggregate per-seed records into clean lightweight CSV tables for paper/report."""
    rows_main: list[dict[str, Any]] = []
    rows_mis: list[dict[str, Any]] = []

    for rec in records:
        mis = rec["misalignment_diagnostics"]
        rows_mis.append(
            {
                "dataset": rec["dataset"],
                "variant": rec["variant"],
                "seed": rec["seed"],
                "cos_before": mis["mean_pairwise_cosine_before"],
                "cos_after": mis["mean_pairwise_cosine_after"],
                "frob_before": mis["rel_frob_error_before"],
                "frob_after": mis["rel_frob_error_after"],
                "mean_ortho_departure": mis["mean_ortho_departure"],
                "max_ortho_departure": mis["max_ortho_departure"],
                "n_iterations": mis["n_iterations"],
            }
        )
        for m_name, m_data in rec["methods"].items():
            t1 = m_data["test1_parity"]
            t2 = m_data["test2_geometry"]
            t3 = m_data["test3_transfer"] or {}
            t4u = m_data["test4_uq"]
            t4o = m_data["test4_ood"]
            rows_main.append(
                {
                    "dataset": rec["dataset"],
                    "variant": rec["variant"],
                    "seed": rec["seed"],
                    "method": m_name,
                    "dim": m_data["dim"],
                    "g0_ensemble_score": rec["g0_references"]["ensemble_test_score"],
                    "g0_tabm_b_score": rec["g0_references"]["tabm_b_pred_score"],
                    "linear_test_score": t1["linear_test_score"],
                    "gap_recovery": t1["gap_recovery"],
                    "direct_distilled_score": t1["direct_distilled_score"],
                    "knn_score": t2["knn_score"],
                    "nn1_score": t2["nn1_score"],
                    "purity_at_k": t2["purity_at_k"],
                    "silhouette_cosine": t2["silhouette_cosine"],
                    "aux_mean_r2": t3.get("mean_r2"),
                    "uq_probe_spearman": t4u["probe_spearman"],
                    "uq_probe_r2": t4u["probe_r2"],
                    "uq_direct_spearman": t4u["direct_head_spearman"],
                    "mahal_high_uq_auroc": t4o["mahal_high_uq_auroc"],
                    "cosine_high_uq_auroc": t4o["cosine_high_uq_auroc"],
                    "fit_time_sec": m_data["fit_time_sec"],
                }
            )

    if rows_mis:
        df_mis = pd.DataFrame(rows_mis)
        df_mis.to_csv(out_dir / "per_seed_misalignment.csv", index=False)
        agg_mis = (
            df_mis.drop(columns=["seed"])
            .groupby(["dataset", "variant"], as_index=False)
            .agg(["mean", "std"])
        )
        agg_mis.columns = [
            f"{c[0]}_{c[1]}" if c[1] else c[0] for c in agg_mis.columns.to_flat_index()
        ]
        agg_mis.to_csv(out_dir / "summary_misalignment.csv", index=False)

    if rows_main:
        df_main = pd.DataFrame(rows_main)
        df_main.to_csv(out_dir / "per_seed_g1_metrics.csv", index=False)

        numeric_cols = [
            "g0_ensemble_score",
            "g0_tabm_b_score",
            "linear_test_score",
            "gap_recovery",
            "direct_distilled_score",
            "knn_score",
            "nn1_score",
            "purity_at_k",
            "silhouette_cosine",
            "aux_mean_r2",
            "uq_probe_spearman",
            "uq_direct_spearman",
            "mahal_high_uq_auroc",
            "cosine_high_uq_auroc",
            "fit_time_sec",
        ]
        agg = (
            df_main.groupby(["dataset", "variant", "method"], as_index=False)[numeric_cols]
            .agg(["mean", "std"])
        )
        agg.columns = [
            f"{c[0]}_{c[1]}" if c[1] else c[0] for c in agg.columns.to_flat_index()
        ]
        agg.to_csv(out_dir / "summary_g1_metrics.csv", index=False)

        print("\n=== G1 Representation Benchmark Summary (Mean across seeds) ===")
        for (ds, var), grp in df_main.groupby(["dataset", "variant"]):
            ens_s = grp["g0_ensemble_score"].mean()
            tb_s = grp["g0_tabm_b_score"].mean()
            print(f"\n[{ds} | {var}] G0 Ensemble={ens_s:.6f} | TabM[B]={tb_s:.6f}")
            print(
                f"  {'Method':<13} | {'Linear':>10} | {'GapRec':>7} | {'kNN':>10} "
                f"| {'AuxR2':>7} | {'UQ_rho':>7} | {'MahalUQ':>7}"
            )
            print("  " + "-" * 74)
            for m_name, m_grp in grp.groupby("method"):
                lin = m_grp["linear_test_score"].mean()
                gr = m_grp["gap_recovery"].mean()
                knn = m_grp["knn_score"].mean()
                aux = m_grp["aux_mean_r2"].mean()
                uq_r = m_grp["uq_probe_spearman"].mean()
                mh = m_grp["mahal_high_uq_auroc"].mean()
                aux_s = f"{aux:>7.4f}" if pd.notna(aux) else "    N/A"
                gr_s = f"{gr:>7.3f}" if pd.notna(gr) else "    N/A"
                mh_s = f"{mh:>7.4f}" if pd.notna(mh) else "    N/A"
                print(
                    f"  {m_name:<13} | {lin:>10.6f} | {gr_s} | {knn:>10.6f} "
                    f"| {aux_s} | {uq_r:>7.4f} | {mh_s}"
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Step 4 G1 official 4-test evaluation directly on saved G0 .npz bundles."
    )
    parser.add_argument("--reps-dir", type=Path, required=True, help="Root G0 artifacts dir (e.g., Drive g0 folder).")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Official TabM data dir for Test 3.")
    parser.add_argument("--datasets", nargs="+", default=["california", "house", "adult", "churn", "otto"])
    parser.add_argument("--variants", nargs="+", default=["tabm", "tabm_mini"], choices=["tabm", "tabm_mini"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--out-dir", type=Path, default=Path("results/g1"))
    parser.add_argument("--save-bridge-dir", type=Path, default=None, help="Optional dir on Drive to save bridge checkpoints.")
    parser.add_argument("--resume", action="store_true", help="Skip (dataset, variant, seed) already saved in per_run_results.json.")
    parser.add_argument("--include-concat", action="store_true", help="Also run high-dim full Concat baseline (higher RAM).")

    # Probe & Baseline hyperparameters
    parser.add_argument("--k-neighbors", type=int, default=15)
    parser.add_argument("--n-aux-features", type=int, default=3)
    parser.add_argument("--pca-max-fit-samples", type=int, default=16384)
    parser.add_argument("--procrustes-max-iter", type=int, default=15)
    parser.add_argument("--procrustes-max-fit-samples", type=int, default=8192)

    # TabM-Bridge hyperparameters
    parser.add_argument("--d-z", type=int, default=256)
    parser.add_argument("--d-hidden", type=int, default=256)
    parser.add_argument("--align-mode", type=str, default="low_rank", choices=["low_rank", "full"])
    parser.add_argument("--align-rank", type=int, default=16)
    parser.add_argument("--uncentered-gram", action="store_true")
    parser.add_argument("--alpha-gram", type=float, default=25.0)
    parser.add_argument("--alpha-pred", type=float, default=1.0)
    parser.add_argument("--alpha-uq", type=float, default=0.15)
    parser.add_argument("--bridge-lr", type=float, default=1e-3)
    parser.add_argument("--bridge-wd", type=float, default=1e-4)
    parser.add_argument("--bridge-epochs", type=int, default=60)
    parser.add_argument("--bridge-patience", type=int, default=12)
    parser.add_argument("--bridge-batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "per_run_results.json"

    records: list[dict[str, Any]] = []
    done_keys: set[tuple[str, str, int]] = set()
    if args.resume and json_path.exists():
        with json_path.open("r", encoding="utf-8") as f:
            records = json.load(f)
        done_keys = {(r["dataset"], r["variant"], int(r["seed"])) for r in records}
        print(f"Resuming with {len(done_keys)} completed runs from {json_path}.")

    aux_cache: dict[str, tuple[list[int], dict[str, np.ndarray]] | None] = {}

    for dataset_name in args.datasets:
        reps_ds_dir, raw_ds_dir = _resolve_dataset_paths(
            dataset_name, args.reps_dir, args.data_dir
        )
        if reps_ds_dir is None:
            print(f"[MISS] No G0 representation folder found in {args.reps_dir} for '{dataset_name}'")
            continue

        for variant in args.variants:
            for seed in args.seeds:
                key = (dataset_name, variant, int(seed))
                if key in done_keys:
                    print(f"[SKIP] {dataset_name} | {variant} | seed={seed} (already done)")
                    continue

                run_dir = reps_ds_dir / variant / f"seed_{seed}"
                if not (run_dir / "reps_train.npz").exists():
                    print(f"[MISS] Skipping {run_dir} (reps_train.npz not found)")
                    continue

                t0 = time.perf_counter()
                print(f"\n[RUN ] {dataset_name} ({reps_ds_dir.name}) | {variant} | seed={seed} ...")
                rec = _evaluate_run(
                    dataset_name=dataset_name,
                    reps_ds_dir=reps_ds_dir,
                    raw_ds_dir=raw_ds_dir,
                    variant=variant,
                    seed=seed,
                    args=args,
                    aux_cache=aux_cache,
                )
                records.append(rec)
                done_keys.add(key)

                # Incremental checkpoint save after every single run
                with json_path.open("w", encoding="utf-8") as f:
                    json.dump(_sanitize_json(records), f, indent=2)

                mis = rec["misalignment_diagnostics"]
                b_lin = rec["methods"]["tabm_bridge"]["test1_parity"]["linear_test_score"]
                p_lin = rec["methods"]["procrustes"]["test1_parity"]["linear_test_score"]
                m_lin = rec["methods"]["simple_mean"]["test1_parity"]["linear_test_score"]
                print(
                    f"[DONE] {dataset_name} | {variant} | seed={seed} ({time.perf_counter() - t0:.1f}s) "
                    f"| cos: {mis['mean_pairwise_cosine_before']:.3f}->{mis['mean_pairwise_cosine_after']:.3f} "
                    f"| linear: mean={m_lin:.5f}, proc={p_lin:.5f}, bridge={b_lin:.5f}"
                )

    _export_summary_tables(records, args.out_dir)


if __name__ == "__main__":
    main()