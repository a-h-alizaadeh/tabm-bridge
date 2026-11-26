from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch

from tabm_bridge.bridge import (
    GeneralizedOrthogonalProcrustes,
    PCAConcatBaseline,
    ProcrustesAnchoredResidualBridge,
    SimpleMeanBaseline,
    TabMBestSubmodelBaseline,
    TabMBridge,
    TabMBridgeTrainer,
    compute_misalignment_diagnostics,
)
from tabm_bridge.config import (
    BridgeAlignMode,
    BridgeConfig,
    ModelConfig,
    PLEConfig,
    TaskType,
    TrainConfig,
)
from tabm_bridge.data import (
    ProcessedDataset,
    ProcessedSplit,
    compute_ple_bins,
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
from tabm_bridge.evaluation.g1_probes import _chunked_cosine_topk
from tabm_bridge.model import TabMModel
from tabm_bridge.training import TabMTrainer, configure_reproducibility

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


def _config_fingerprint(args: argparse.Namespace) -> str:
    """Deterministic hash of Phase 3 evaluation hyperparameters for stale-proof Resume."""
    keys = (
        "mode",
        "enable_ood",
        "include_bridge_v1",
        "skip_v2_512",
        "k_neighbors",
        "n_aux_features",
        "max_collinearity",
        "d_z_v2",
        "d_hidden_v2",
        "align_mode",
        "align_rank",
        "alpha_gram",
        "alpha_pred",
        "alpha_uq",
        "alpha_recon",
        "init_gate",
        "gram_warmup_epochs",
        "bridge_lr",
        "bridge_wd",
        "bridge_epochs",
        "bridge_patience",
        "bridge_batch_size",
    )
    payload: dict[str, Any] = {k: getattr(args, k) for k in keys}
    payload["has_v1_dir"] = args.v1_bridge_dir is not None
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


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


def _infer_task_and_primary(run_dir: Path, raw_ds_dir: Path | None) -> tuple[TaskType, str | None]:
    task_type: TaskType | None = None
    primary_score: str | None = None

    report_path = run_dir / "report.json"
    if report_path.exists():
        with report_path.open("r", encoding="utf-8") as f:
            task_type = json.load(f).get("task_type")

    if raw_ds_dir is not None and (raw_ds_dir / "info.json").exists():
        with (raw_ds_dir / "info.json").open("r", encoding="utf-8") as f:
            info = json.load(f)
        if task_type is None:
            task_type = info.get("task_type")
        primary_score = info.get("score")

    if task_type not in ("regression", "binclass", "multiclass"):
        raise ValueError(f"Could not infer valid task_type for {run_dir}.")
    return task_type, primary_score


def _load_processed_and_aux(
    dataset_name: str,
    raw_ds_dir: Path | None,
    y_train_ref: np.ndarray,
    y_val_ref: np.ndarray,
    y_test_ref: np.ndarray,
    task_type: TaskType,
    n_aux: int = 3,
    max_collinearity: float = 0.98,
) -> tuple[ProcessedDataset | None, tuple[list[int], dict[str, np.ndarray]] | None]:
    """Load preprocessed dataset + collinearity-filtered Test 3 targets with strict row alignment."""
    try:
        if dataset_name == "california_sklearn":
            raw = load_sklearn_california_housing(seed=0)
        elif raw_ds_dir is not None and (raw_ds_dir / "info.json").exists():
            raw = load_official_dataset(raw_ds_dir)
        else:
            return None, None
    except Exception:
        return None, None

    proc = preprocess_dataset(raw, seed=0)
    for split_name, y_ref in (("train", y_train_ref), ("val", y_val_ref), ("test", y_test_ref)):
        y_cur = np.asarray(proc.splits[split_name].y_raw, dtype=np.float64).reshape(-1)
        y_exp = np.asarray(y_ref, dtype=np.float64).reshape(-1)
        if len(y_cur) != len(y_exp) or not np.allclose(y_cur, y_exp, rtol=1e-4, atol=1e-4):
            raise RuntimeError(
                f"Split alignment mismatch on {dataset_name}/{split_name}: "
                f"raw rows do not match saved G0 reps_{split_name}.npz!"
            )

    aux_bundle: tuple[list[int], dict[str, np.ndarray]] | None = None
    x_num_tr = proc.splits["train"].x_num
    if n_aux > 0 and proc.n_num_features > 0 and x_num_tr is not None:
        selected_cols = select_auxiliary_features(
            x_num_tr,
            proc.splits["train"].y_raw,
            task_type=task_type,
            n_aux=n_aux,
            max_collinearity=max_collinearity,
        )
        if selected_cols:
            aux_splits: dict[str, np.ndarray] = {}
            for s in ("train", "val", "test"):
                x_s = proc.splits[s].x_num
                assert x_s is not None
                aux_splits[s] = x_s[:, selected_cols].astype(np.float32, copy=True)
            aux_bundle = (selected_cols, aux_splits)

    return proc, aux_bundle


def _slice_split(split: ProcessedSplit, n_rows: int) -> ProcessedSplit:
    updates: dict[str, Any] = {}
    for f in dataclasses.fields(split):
        val = getattr(split, f.name)
        if isinstance(val, np.ndarray) and val.ndim >= 1:
            updates[f.name] = val[:n_rows]
    return dataclasses.replace(split, **updates)


def _make_ood_splits(
    test_split: ProcessedSplit,
    cat_cardinalities: list[int],
    seed: int,
) -> dict[str, ProcessedSplit]:
    """Create two leak-free tabular OOD shifts strictly from the test split.

    1. 'swap': Marginal-preserving column permutation on 50% of features (breaks joint correlations).
    2. 'shift': Covariate scale/shift (+2.0 sigma on 50% of numerical cols + 30% categorical shift).
    """
    rng = np.random.default_rng(seed + 1000)
    n = len(test_split.y_raw)

    # 1. Feature-Swap OOD (marginal-preserving)
    x_num_swap = None if test_split.x_num is None else test_split.x_num.copy()
    if x_num_swap is not None and x_num_swap.shape[1] > 0:
        cols = rng.choice(
            x_num_swap.shape[1],
            size=max(1, (x_num_swap.shape[1] + 1) // 2),
            replace=False,
        )
        for j in cols:
            x_num_swap[:, j] = x_num_swap[rng.permutation(n), j]

    x_cat_swap = None if test_split.x_cat is None else test_split.x_cat.copy()
    if x_cat_swap is not None and x_cat_swap.shape[1] > 0:
        cols = rng.choice(
            x_cat_swap.shape[1],
            size=max(1, (x_cat_swap.shape[1] + 1) // 2),
            replace=False,
        )
        for j in cols:
            x_cat_swap[:, j] = x_cat_swap[rng.permutation(n), j]

    split_swap = dataclasses.replace(test_split, x_num=x_num_swap, x_cat=x_cat_swap)

    # 2. Covariate Scale+Shift OOD
    x_num_shift = None if test_split.x_num is None else test_split.x_num.copy()
    if x_num_shift is not None and x_num_shift.shape[1] > 0:
        k_sh = max(1, (x_num_shift.shape[1] + 1) // 2)
        cols = rng.choice(x_num_shift.shape[1], size=k_sh, replace=False)
        signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=k_sh)
        for idx_c, j in enumerate(cols):
            col_std = float(np.std(x_num_shift[:, j]))
            step = 2.0 * (col_std if col_std >= 1e-4 else 1.0)
            x_num_shift[:, j] = x_num_shift[:, j] * 1.5 + np.float32(signs[idx_c] * step)

    x_cat_shift = None if test_split.x_cat is None else test_split.x_cat.copy()
    if x_cat_shift is not None and x_cat_shift.shape[1] > 0:
        for j in range(x_cat_shift.shape[1]):
            card = (
                int(cat_cardinalities[j])
                if j < len(cat_cardinalities)
                else int(x_cat_shift[:, j].max() + 1)
            )
            mask = rng.random(n) < 0.30
            if mask.any() and card >= 2:
                x_cat_shift[mask, j] = rng.integers(
                    0, card, size=int(mask.sum()), dtype=x_cat_shift.dtype
                )

    split_shift = dataclasses.replace(test_split, x_num=x_num_shift, x_cat=x_cat_shift)
    return {"swap": split_swap, "shift": split_shift}


def _reconstruct_model_config(raw_cfg: Any) -> ModelConfig:
    if isinstance(raw_cfg, ModelConfig):
        return raw_cfg
    valid = {f.name for f in dataclasses.fields(ModelConfig)}
    d: dict[str, Any] = {k: v for k, v in dict(raw_cfg).items() if k in valid}
    if isinstance(d.get("ple"), dict):
        ple_valid = {f.name for f in dataclasses.fields(PLEConfig)}
        d["ple"] = PLEConfig(**{k: v for k, v in d["ple"].items() if k in ple_valid})
    return ModelConfig(**d)


def _reconstruct_train_config(raw_cfg: Any, device: str) -> TrainConfig:
    if isinstance(raw_cfg, TrainConfig):
        return dataclasses.replace(raw_cfg, device=device)
    valid = {f.name for f in dataclasses.fields(TrainConfig)}
    d: dict[str, Any] = {k: v for k, v in dict(raw_cfg).items() if k in valid}
    d["device"] = device
    return TrainConfig(**d)


def _extract_real_ood_from_checkpoint(
    run_dir: Path,
    proc: ProcessedDataset | None,
    h_te_ref: np.ndarray,
    seed: int,
    device: str,
) -> dict[str, Any] | None:
    """Reload frozen Phase 1 best_checkpoint.pt, verify 64-sample parity, and extract H_ood."""
    ckpt_path = run_dir / "best_checkpoint.pt"
    if proc is None or not ckpt_path.exists():
        return None

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict") or ckpt.get("state_dict")
    if state_dict is None or "model_config" not in ckpt:
        return None

    model_cfg = _reconstruct_model_config(ckpt["model_config"])
    train_cfg = (
        _reconstruct_train_config(ckpt["train_config"], device=device)
        if "train_config" in ckpt
        else TrainConfig(task_type=proc.task_type, device=device, seed=seed)
    )
    use_ple = model_cfg.ple.enabled and (proc.n_num_features > 0)
    ple_bins = compute_ple_bins(proc, n_bins=model_cfg.ple.n_bins) if use_ple else None

    model = TabMModel(model_cfg, ple_bins=ple_bins)
    model.load_state_dict(state_dict, strict=True)
    trainer = TabMTrainer(
        model=model,
        config=train_cfg,
        target_stats=proc.target_stats,
        primary_score=proc.primary_score,
    )

    # 64-sample parity verification proving reloaded .pt matches saved .npz embeddings
    n_chk = min(64, len(proc.splits["test"].y_raw))
    chk_bundle = trainer.extract_representations(_slice_split(proc.splits["test"], n_chk))
    h_chk = np.asarray(chk_bundle.embeddings, dtype=np.float32)
    h_ref = np.asarray(h_te_ref[:n_chk], dtype=np.float32)
    dot = np.sum(h_chk * h_ref, axis=-1)
    denom = np.clip(np.linalg.norm(h_chk, axis=-1) * np.linalg.norm(h_ref, axis=-1), 1e-8, None)
    parity_cos = float(np.mean(dot / denom))
    parity_max_diff = float(np.max(np.abs(h_chk - h_ref)))

    if parity_cos < 0.995:
        raise RuntimeError(
            f"Checkpoint parity FAILED for {ckpt_path}: "
            f"cos={parity_cos:.6f}, max_diff={parity_max_diff:.6f}"
        )

    ood_splits = _make_ood_splits(proc.splits["test"], proc.cat_cardinalities, seed=seed)
    b_swap = trainer.extract_representations(ood_splits["swap"])
    b_shift = trainer.extract_representations(ood_splits["shift"])

    del model, trainer, ckpt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "swap": b_swap.embeddings.astype(np.float16),
        "shift": b_shift.embeddings.astype(np.float16),
        "swap_uq": b_swap.epistemic_uncertainty.astype(np.float32),
        "shift_uq": b_shift.epistemic_uncertainty.astype(np.float32),
        "parity_cosine": parity_cos,
        "parity_max_diff": parity_max_diff,
    }


def _find_v1_bridge_checkpoint(
    v1_bridge_dir: Path | None,
    dataset_name: str,
    variant: str,
    seed: int,
) -> Path | None:
    if v1_bridge_dir is None or not v1_bridge_dir.exists():
        return None
    direct_candidates = (
        v1_bridge_dir / dataset_name / variant / f"seed_{seed}" / "bridge_checkpoint.pt",
        v1_bridge_dir / f"{variant}_seed{seed}" / dataset_name / variant / f"seed_{seed}" / "bridge_checkpoint.pt",
        v1_bridge_dir / variant / f"seed_{seed}" / "bridge_checkpoint.pt",
    )
    for p in direct_candidates:
        if p.exists():
            return p
    official_name = _SLUG_TO_OFFICIAL_NAME.get(dataset_name, "")
    for p in v1_bridge_dir.rglob("bridge_checkpoint.pt"):
        ps = p.as_posix()
        if f"/{variant}/" in ps and f"seed_{seed}" in ps and (
            dataset_name in ps or (official_name and official_name in ps)
        ):
            return p
    return None


def _compute_top_uq_slice_scores(
    y_te: np.ndarray,
    pred_te: np.ndarray,
    uq_te: np.ndarray,
    task_type: TaskType,
    primary_score: str | None,
    top_frac: float = 0.10,
) -> dict[str, float | None]:
    """Compare prediction quality on Top-10% hardest (highest UQ) vs Bottom-90% test samples."""
    uq_1d = np.asarray(uq_te, dtype=np.float64).reshape(-1)
    if float(np.std(uq_1d)) < 1e-12:
        return {"top10_score": None, "bottom90_score": None}
    cutoff = float(np.quantile(uq_1d, 1.0 - top_frac))
    hard = uq_1d >= cutoff
    easy = ~hard
    if int(hard.sum()) < 2 or int(easy.sum()) < 2:
        return {"top10_score": None, "bottom90_score": None}
    try:
        s_hard: float | None = float(
            compute_metrics(y_te[hard], pred_te[hard], task_type, primary_score)["score"]
        )
    except Exception:
        s_hard = None
    try:
        s_easy: float | None = float(
            compute_metrics(y_te[easy], pred_te[easy], task_type, primary_score)["score"]
        )
    except Exception:
        s_easy = None
    return {"top10_score": s_hard, "bottom90_score": s_easy}


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
    probe_device: str,
    z_ood_swap: np.ndarray | None = None,
    z_ood_shift: np.ndarray | None = None,
    direct_uq_te: np.ndarray | None = None,
    direct_pred_te: np.ndarray | None = None,
    fit_time_sec: float = 0.0,
) -> dict[str, Any]:
    """Run all 4 official G1 tests + real OOD shifts with GPU probes and zero duplicate kNN."""
    t_probe0 = time.perf_counter()

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
        device=probe_device,
    )
    gap_rec = compute_gap_recovery(t1.test_score, tabm_b_ref_score, ensemble_ref_score)

    direct_pred_score: float | None = None
    direct_top10: dict[str, float | None] | None = None
    if direct_pred_te is not None:
        direct_pred_score = float(
            compute_metrics(y_te, direct_pred_te, task_type, primary_score)["score"]
        )
        direct_top10 = _compute_top_uq_slice_scores(
            y_te, direct_pred_te, uq_te, task_type, primary_score
        )

    # Compute Cosine Top-K ONCE on GPU and share across Test 2 and Test 4b
    t_knn0 = time.perf_counter()
    top_idx, top_sim = _chunked_cosine_topk(
        queries=z_te,
        bank=z_tr,
        k=k_neighbors,
        device=probe_device,
    )
    knn_search_time_sec = float(time.perf_counter() - t_knn0)

    t2 = evaluate_knn_geometry(
        z_train=z_tr,
        y_train_raw=y_tr,
        z_test=z_te,
        y_test_raw=y_te,
        task_type=task_type,
        primary_score=primary_score,
        k_neighbors=k_neighbors,
        seed=seed,
        device=probe_device,
        precomputed_topk=(top_idx, top_sim),
    )

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

    t4_ood = evaluate_unified_space_ood(
        z_train=z_tr,
        z_test=z_te,
        uq_test=uq_te,
        z_ood=z_ood_swap,
        k_neighbors=k_neighbors,
        seed=seed,
        device=probe_device,
        precomputed_test_top_sim=top_sim,
    )

    t4_ood_shift_dict: dict[str, Any] | None = None
    if z_ood_shift is not None and len(z_ood_shift) > 0:
        t4_sh = evaluate_unified_space_ood(
            z_train=z_tr,
            z_test=z_te,
            uq_test=uq_te,
            z_ood=z_ood_shift,
            k_neighbors=k_neighbors,
            seed=seed,
            device=probe_device,
            precomputed_test_top_sim=top_sim,
        )
        t4_ood_shift_dict = {
            "mahal_ood": None if t4_sh.mahal_ood is None else asdict(t4_sh.mahal_ood),
            "cosine_ood": None if t4_sh.cosine_ood is None else asdict(t4_sh.cosine_ood),
        }

    return {
        "method": method_name,
        "dim": int(z_tr.shape[1]),
        "fit_time_sec": float(fit_time_sec),
        "probe_time_sec": float(time.perf_counter() - t_probe0),
        "knn_search_time_sec": knn_search_time_sec,
        "test1_parity": {
            "linear_test_score": t1.test_score,
            "linear_val_score": float(t1.val_metrics["score"]),
            "best_reg_param": t1.best_reg_param,
            "gap_recovery": gap_rec,
            "delta_vs_tabm_b": float(t1.test_score - tabm_b_ref_score),
            "direct_distilled_score": direct_pred_score,
            "direct_top10_uq": direct_top10,
            "test_metrics": t1.test_metrics,
        },
        "test2_geometry": asdict(t2),
        "test3_transfer": t3_dict,
        "test4_uq": asdict(t4_uq),
        "test4_ood": asdict(t4_ood),
        "test4_ood_shift": t4_ood_shift_dict,
    }


def _train_and_eval_bridge(
    label: str,
    bridge_cls: type[TabMBridge] | type[ProcrustesAnchoredResidualBridge],
    cfg: BridgeConfig,
    *,
    k: int,
    d_in: int,
    d_out: int,
    task_type: TaskType,
    seed: int,
    device: str,
    h_tr: np.ndarray,
    mp_tr_2d: np.ndarray,
    uq_tr: np.ndarray,
    h_va: np.ndarray,
    mp_va_2d: np.ndarray,
    uq_va: np.ndarray,
    h_te: np.ndarray,
    z_tr_proc: np.ndarray | None,
    z_va_proc: np.ndarray | None,
    z_te_proc: np.ndarray | None,
    ood_bundle: dict[str, Any] | None,
    z_ood_swap_proc: np.ndarray | None,
    z_ood_shift_proc: np.ndarray | None,
    eval_fn: Callable[..., dict[str, Any]],
    save_ckpt_path: Path | None = None,
) -> dict[str, Any]:
    configure_reproducibility(seed)
    bridge = bridge_cls(k=k, d_in=d_in, d_out=d_out, config=cfg)
    trainer = TabMBridgeTrainer(bridge=bridge, task_type=task_type, device=device, seed=seed)
    use_anchor = isinstance(bridge, ProcrustesAnchoredResidualBridge)

    train_res = trainer.fit(
        h_tr,
        mp_tr_2d,
        uq_tr,
        h_va,
        mp_va_2d,
        uq_va,
        z_proc_train=z_tr_proc if use_anchor else None,
        z_proc_val=z_va_proc if use_anchor else None,
    )
    z_tr_b = trainer.encode(h_tr, return_aux=False, z_proc=z_tr_proc if use_anchor else None)
    z_va_b = trainer.encode(h_va, return_aux=False, z_proc=z_va_proc if use_anchor else None)
    te_b = trainer.encode(h_te, return_aux=True, z_proc=z_te_proc if use_anchor else None)
    assert isinstance(z_tr_b, np.ndarray) and isinstance(z_va_b, np.ndarray) and isinstance(te_b, dict)

    z_ood_sw = (
        None
        if ood_bundle is None
        else trainer.encode(
            ood_bundle["swap"],
            return_aux=False,
            z_proc=z_ood_swap_proc if use_anchor else None,
        )
    )
    z_ood_sh = (
        None
        if ood_bundle is None
        else trainer.encode(
            ood_bundle["shift"],
            return_aux=False,
            z_proc=z_ood_shift_proc if use_anchor else None,
        )
    )
    assert z_ood_sw is None or isinstance(z_ood_sw, np.ndarray)
    assert z_ood_sh is None or isinstance(z_ood_sh, np.ndarray)

    out = eval_fn(
        label,
        z_tr_b,
        z_va_b,
        te_b["z"],
        train_res.elapsed_sec,
        z_ood_swap=z_ood_sw,
        z_ood_shift=z_ood_sh,
        direct_uq_te=te_b["uq"],
        direct_pred_te=te_b["pred"],
    )
    meta: dict[str, Any] = {
        "bridge_class": bridge_cls.__name__,
        "best_epoch": train_res.best_epoch,
        "best_val_loss": train_res.best_val_loss,
        "stopped_due_to_nan": train_res.stopped_due_to_nan,
        "parameter_count": bridge.parameter_count(),
        "config": asdict(cfg),
    }
    if isinstance(bridge, ProcrustesAnchoredResidualBridge):
        meta["final_res_gate"] = float(bridge.res_gate.detach().cpu().item())
    out["bridge_train"] = meta

    if save_ckpt_path is not None:
        trainer.save_checkpoint(save_ckpt_path)

    del bridge, trainer, z_tr_b, z_va_b, te_b, z_ood_sw, z_ood_sh
    return out


def _evaluate_run_v4(
    dataset_name: str,
    reps_ds_dir: Path,
    raw_ds_dir: Path | None,
    variant: str,
    seed: int,
    args: argparse.Namespace,
    ds_cache: dict[str, tuple[ProcessedDataset | None, tuple[list[int], dict[str, np.ndarray]] | None]],
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

    mp_tr_2d = mp_tr[:, None] if mp_tr.ndim == 1 else mp_tr
    mp_va_2d = mp_va[:, None] if mp_va.ndim == 1 else mp_va
    _, k, d_in = h_tr.shape
    d_out = mp_tr_2d.shape[1]

    if dataset_name not in ds_cache:
        ds_cache[dataset_name] = _load_processed_and_aux(
            dataset_name=dataset_name,
            raw_ds_dir=raw_ds_dir,
            y_train_ref=y_tr,
            y_val_ref=y_va,
            y_test_ref=y_te,
            task_type=task_type,
            n_aux=args.n_aux_features,
            max_collinearity=args.max_collinearity,
        )
    proc_ds, aux_bundle = ds_cache[dataset_name]

    ensemble_test_score = float(compute_metrics(y_te, mp_te, task_type, primary_score)["score"])
    ens_top10 = _compute_top_uq_slice_scores(y_te, mp_te, uq_te, task_type, primary_score)

    t0 = time.perf_counter()
    tabm_b = TabMBestSubmodelBaseline().fit(
        va["per_head_preds"], y_va, task_type=task_type, primary_score=primary_score
    )
    t_tabm_b = time.perf_counter() - t0
    assert tabm_b.best_index_ is not None
    tabm_b_preds_te = te["per_head_preds"][:, tabm_b.best_index_, :]
    tabm_b_pred_score = float(compute_metrics(y_te, tabm_b_preds_te, task_type, primary_score)["score"])
    tabm_b_top10 = _compute_top_uq_slice_scores(y_te, tabm_b_preds_te, uq_te, task_type, primary_score)

    # Real OOD extraction from frozen Phase 1 best_checkpoint.pt (with 64-sample parity check)
    ood_bundle: dict[str, Any] | None = None
    ood_diag: dict[str, Any] | None = None
    if args.enable_ood:
        ood_bundle = _extract_real_ood_from_checkpoint(
            run_dir=run_dir,
            proc=proc_ds,
            h_te_ref=h_te,
            seed=seed,
            device=args.device,
        )
        if ood_bundle is not None:
            ood_diag = {
                "parity_cosine": ood_bundle["parity_cosine"],
                "parity_max_diff": ood_bundle["parity_max_diff"],
                "n_ood_swap": int(len(ood_bundle["swap"])),
                "n_ood_shift": int(len(ood_bundle["shift"])),
                "mean_uq_id": float(np.mean(uq_te)),
                "mean_uq_ood_swap": float(np.mean(ood_bundle["swap_uq"])),
                "mean_uq_ood_shift": float(np.mean(ood_bundle["shift_uq"])),
            }

    probe_device = args.device if args.device in ("cuda", "cpu") else "auto"
    methods_out: dict[str, dict[str, Any]] = {}

    def _run_eval(
        name: str,
        z_tr_m: np.ndarray,
        z_va_m: np.ndarray,
        z_te_m: np.ndarray,
        fit_sec: float,
        z_ood_swap: np.ndarray | None = None,
        z_ood_shift: np.ndarray | None = None,
        direct_uq_te: np.ndarray | None = None,
        direct_pred_te: np.ndarray | None = None,
    ) -> dict[str, Any]:
        res = _evaluate_single_representation(
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
            probe_device=probe_device,
            z_ood_swap=z_ood_swap,
            z_ood_shift=z_ood_shift,
            direct_uq_te=direct_uq_te,
            direct_pred_te=direct_pred_te,
            fit_time_sec=fit_sec,
        )
        dim_m = int(z_tr_m.shape[1])
        total_n = len(z_tr_m) + len(z_va_m) + len(z_te_m)
        res["efficiency"] = {
            "storage_mb_fp32_all_splits": float(total_n * dim_m * 4 / (1024 ** 2)),
            "compression_ratio_vs_raw_k": float((k * d_in) / max(dim_m, 1)),
            "knn_search_time_sec": float(res.get("knn_search_time_sec", 0.0)),
        }
        methods_out[name] = res
        return res

    # Compute Procrustes once (serves as both baseline & Bridge-v2 anchor)
    t0 = time.perf_counter()
    proc_model = GeneralizedOrthogonalProcrustes(
        center=True,
        max_iter=args.procrustes_max_iter,
        max_fit_samples=args.procrustes_max_fit_samples,
        random_state=seed,
    ).fit(h_tr)
    t_proc = time.perf_counter() - t0
    misalignment = compute_misalignment_diagnostics(proc_model, h_te, max_samples=2048, seed=seed)
    z_tr_proc = proc_model.transform(h_tr)
    z_va_proc = proc_model.transform(h_va)
    z_te_proc = proc_model.transform(h_te)
    z_ood_swap_proc = None if ood_bundle is None else proc_model.transform(ood_bundle["swap"])
    z_ood_shift_proc = None if ood_bundle is None else proc_model.transform(ood_bundle["shift"])
    del proc_model

    bridge_common: dict[str, Any] = {
        "k": k,
        "d_in": d_in,
        "d_out": d_out,
        "task_type": task_type,
        "seed": seed,
        "device": args.device,
        "h_tr": h_tr,
        "mp_tr_2d": mp_tr_2d,
        "uq_tr": uq_tr,
        "h_va": h_va,
        "mp_va_2d": mp_va_2d,
        "uq_va": uq_va,
        "h_te": h_te,
        "z_tr_proc": z_tr_proc,
        "z_va_proc": z_va_proc,
        "z_te_proc": z_te_proc,
        "ood_bundle": ood_bundle,
        "z_ood_swap_proc": z_ood_swap_proc,
        "z_ood_shift_proc": z_ood_shift_proc,
        "eval_fn": _run_eval,
    }

    # Evaluate Procrustes under identical v4 probes + real OOD
    _run_eval(
        "procrustes",
        z_tr_proc,
        z_va_proc,
        z_te_proc,
        t_proc,
        z_ood_swap=z_ood_swap_proc,
        z_ood_shift=z_ood_shift_proc,
    )

    if args.mode == "main":
        # 1. TabM[B] under identical v4 probes + real OOD
        _run_eval(
            "tabm_b",
            tabm_b.transform(h_tr),
            tabm_b.transform(h_va),
            tabm_b.transform(h_te),
            t_tabm_b,
            z_ood_swap=None if ood_bundle is None else tabm_b.transform(ood_bundle["swap"]),
            z_ood_shift=None if ood_bundle is None else tabm_b.transform(ood_bundle["shift"]),
        )

        # 2. PCA(Concat)
        t0 = time.perf_counter()
        pca = PCAConcatBaseline(
            n_components=args.d_z_v2,
            max_fit_samples=args.pca_max_fit_samples,
            random_state=seed,
        ).fit(h_tr)
        _run_eval(
            "pca_concat",
            pca.transform(h_tr),
            pca.transform(h_va),
            pca.transform(h_te),
            time.perf_counter() - t0,
            z_ood_swap=None if ood_bundle is None else pca.transform(ood_bundle["swap"]),
            z_ood_shift=None if ood_bundle is None else pca.transform(ood_bundle["shift"]),
        )
        del pca

        # 3. Simple Mean
        t0 = time.perf_counter()
        sm = SimpleMeanBaseline().fit(h_tr)
        _run_eval(
            "simple_mean",
            sm.transform(h_tr),
            sm.transform(h_va),
            sm.transform(h_te),
            time.perf_counter() - t0,
            z_ood_swap=None if ood_bundle is None else sm.transform(ood_bundle["swap"]),
            z_ood_shift=None if ood_bundle is None else sm.transform(ood_bundle["shift"]),
        )

        # 4. Zero-retrain reload of Phase 2 Bridge-v1 (or train if --include-bridge-v1)
        v1_ckpt = _find_v1_bridge_checkpoint(args.v1_bridge_dir, dataset_name, variant, seed)
        if v1_ckpt is not None:
            v1_tr = TabMBridgeTrainer.load_checkpoint(v1_ckpt, device=args.device)
            if (task_type == "regression" and v1_tr.pred_mean_ is None) or (
                v1_tr.uq_mean_ == 0.0 and v1_tr.uq_std_ == 1.0
            ):
                v1_tr._fit_target_scalers(mp_tr_2d, uq_tr)
            z_tr_v1 = v1_tr.encode(h_tr, return_aux=False)
            z_va_v1 = v1_tr.encode(h_va, return_aux=False)
            te_v1 = v1_tr.encode(h_te, return_aux=True)
            assert isinstance(z_tr_v1, np.ndarray) and isinstance(z_va_v1, np.ndarray) and isinstance(te_v1, dict)
            z_sw_v1 = None if ood_bundle is None else v1_tr.encode(ood_bundle["swap"], return_aux=False)
            z_sh_v1 = None if ood_bundle is None else v1_tr.encode(ood_bundle["shift"], return_aux=False)
            assert z_sw_v1 is None or isinstance(z_sw_v1, np.ndarray)
            assert z_sh_v1 is None or isinstance(z_sh_v1, np.ndarray)
            _run_eval(
                "bridge_v1",
                z_tr_v1,
                z_va_v1,
                te_v1["z"],
                0.0,
                z_ood_swap=z_sw_v1,
                z_ood_shift=z_sh_v1,
                direct_uq_te=te_v1["uq"],
                direct_pred_te=te_v1["pred"],
            )
            methods_out["bridge_v1"]["reloaded_from"] = str(v1_ckpt)
            del v1_tr, z_tr_v1, z_va_v1, te_v1, z_sw_v1, z_sh_v1
        elif args.include_bridge_v1:
            align_mode_v1: BridgeAlignMode = args.align_mode
            cfg_v1 = BridgeConfig(
                d_z=256,
                d_hidden=256,
                align_mode=align_mode_v1,
                align_rank=args.align_rank,
                alpha_gram=25.0,
                alpha_pred=1.0,
                alpha_uq=0.15,
                epochs=60,
                patience=12,
                batch_size=256,
            )
            _train_and_eval_bridge("bridge_v1", TabMBridge, cfg_v1, **bridge_common)

        # 5. Bridge-v2 (d_z=256): Procrustes-Anchored Residual Bridge (64x compression vs k*512)
        align_mode: BridgeAlignMode = args.align_mode
        cfg_v2_256 = BridgeConfig(
            d_z=args.d_z_v2,
            d_hidden=args.d_hidden_v2,
            align_mode=align_mode,
            align_rank=args.align_rank,
            center_gram=True,
            alpha_gram=args.alpha_gram,
            alpha_pred=args.alpha_pred,
            alpha_uq=args.alpha_uq,
            alpha_recon=args.alpha_recon,
            init_gate=args.init_gate,
            gram_warmup_epochs=args.gram_warmup_epochs,
            use_cosine_lr=True,
            mask_gram_diag=True,
            center_z_gram=True,
            stop_gradient=True,
            lr=args.bridge_lr,
            weight_decay=args.bridge_wd,
            epochs=args.bridge_epochs,
            patience=args.bridge_patience,
            batch_size=args.bridge_batch_size,
        )
        ckpt_256 = (
            args.save_bridge_dir / dataset_name / variant / f"seed_{seed}" / "bridge_v2_256.pt"
            if args.save_bridge_dir is not None
            else None
        )
        _train_and_eval_bridge(
            "bridge_v2_256",
            ProcrustesAnchoredResidualBridge,
            cfg_v2_256,
            save_ckpt_path=ckpt_256,
            **bridge_common,
        )

        # 6. Bridge-v2 (d_z=512): Full-dimension isometric anchor (32x compression vs k*512)
        if not args.skip_v2_512:
            cfg_v2_512 = dataclasses.replace(cfg_v2_256, d_z=d_in)
            ckpt_512 = (
                args.save_bridge_dir / dataset_name / variant / f"seed_{seed}" / "bridge_v2_512.pt"
                if args.save_bridge_dir is not None
                else None
            )
            _train_and_eval_bridge(
                "bridge_v2_512",
                ProcrustesAnchoredResidualBridge,
                cfg_v2_512,
                save_ckpt_path=ckpt_512,
                **bridge_common,
            )

    elif args.mode == "ablation":
        base_kw: dict[str, Any] = {
            "d_hidden": args.d_hidden_v2,
            "align_mode": args.align_mode,
            "align_rank": args.align_rank,
            "center_gram": True,
            "gram_warmup_epochs": args.gram_warmup_epochs,
            "use_cosine_lr": True,
            "mask_gram_diag": True,
            "center_z_gram": True,
            "stop_gradient": True,
            "lr": args.bridge_lr,
            "weight_decay": args.bridge_wd,
            "epochs": args.bridge_epochs,
            "patience": args.bridge_patience,
            "batch_size": args.bridge_batch_size,
        }
        abl_specs: list[
            tuple[str, type[TabMBridge] | type[ProcrustesAnchoredResidualBridge], BridgeConfig]
        ] = [
            (
                "abl_pred_only",
                TabMBridge,
                BridgeConfig(d_z=256, alpha_gram=0.0, alpha_pred=1.0, alpha_uq=0.0, alpha_recon=0.0, **base_kw),
            ),
            (
                "abl_pred_gram",
                TabMBridge,
                BridgeConfig(d_z=256, alpha_gram=args.alpha_gram, alpha_pred=1.0, alpha_uq=0.0, alpha_recon=0.0, **base_kw),
            ),
            (
                "abl_pred_gram_uq",
                TabMBridge,
                BridgeConfig(d_z=256, alpha_gram=args.alpha_gram, alpha_pred=1.0, alpha_uq=args.alpha_uq, alpha_recon=0.0, **base_kw),
            ),
            (
                "abl_v2_no_recon",
                ProcrustesAnchoredResidualBridge,
                BridgeConfig(
                    d_z=256,
                    alpha_gram=args.alpha_gram,
                    alpha_pred=1.0,
                    alpha_uq=args.alpha_uq,
                    alpha_recon=0.0,
                    init_gate=args.init_gate,
                    **base_kw,
                ),
            ),
            (
                "abl_v2_full_256",
                ProcrustesAnchoredResidualBridge,
                BridgeConfig(
                    d_z=256,
                    alpha_gram=args.alpha_gram,
                    alpha_pred=1.0,
                    alpha_uq=args.alpha_uq,
                    alpha_recon=args.alpha_recon,
                    init_gate=args.init_gate,
                    **base_kw,
                ),
            ),
            (
                "abl_v2_full_512",
                ProcrustesAnchoredResidualBridge,
                BridgeConfig(
                    d_z=d_in,
                    alpha_gram=args.alpha_gram,
                    alpha_pred=1.0,
                    alpha_uq=args.alpha_uq,
                    alpha_recon=args.alpha_recon,
                    init_gate=args.init_gate,
                    **base_kw,
                ),
            ),
        ]
        for abl_name, abl_cls, abl_cfg in abl_specs:
            _train_and_eval_bridge(abl_name, abl_cls, abl_cfg, **bridge_common)

    run_record: dict[str, Any] = {
        "phase": "phase3_v4",
        "mode": args.mode,
        "config_hash": _config_fingerprint(args),
        "dataset": dataset_name,
        "variant": variant,
        "seed": int(seed),
        "task_type": task_type,
        "primary_score": primary_score,
        "k": int(k),
        "d_in": int(d_in),
        "g0_references": {
            "ensemble_test_score": ensemble_test_score,
            "ensemble_top10_uq": ens_top10,
            "tabm_b_pred_score": tabm_b_pred_score,
            "tabm_b_top10_uq": tabm_b_top10,
            "tabm_b_head_index": int(tabm_b.best_index_),
            "tabm_b_fit_time_sec": float(t_tabm_b),
        },
        "misalignment_diagnostics": asdict(misalignment),
        "ood_diagnostics": ood_diag,
        "methods": methods_out,
    }

    del bridge_common
    del tr, va, te, h_tr, h_va, h_te, z_tr_proc, z_va_proc, z_te_proc
    del z_ood_swap_proc, z_ood_shift_proc, ood_bundle
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return run_record


def _export_phase3_summary_tables(records: list[dict[str, Any]], out_dir: Path) -> None:
    """Export clean Phase 3 CSVs with Ratio-of-Means GapRec, AUPRC/FPR95, Efficiency, and Real OOD."""
    rows: list[dict[str, Any]] = []
    for rec in records:
        ens = rec["g0_references"]["ensemble_test_score"]
        tb = rec["g0_references"]["tabm_b_pred_score"]
        for m_name, m_data in rec["methods"].items():
            t1 = m_data["test1_parity"]
            t2 = m_data["test2_geometry"]
            t3 = m_data["test3_transfer"] or {}
            t4u = m_data["test4_uq"]
            t4o = m_data["test4_ood"]
            t4s = m_data.get("test4_ood_shift") or {}
            eff = m_data.get("efficiency") or {}
            rows.append(
                {
                    "dataset": rec["dataset"],
                    "variant": rec["variant"],
                    "seed": rec["seed"],
                    "mode": rec.get("mode", "main"),
                    "method": m_name,
                    "dim": m_data["dim"],
                    "g0_ens": ens,
                    "tabm_b": tb,
                    "linear_score": t1["linear_test_score"],
                    "delta_vs_tabm_b": t1["delta_vs_tabm_b"],
                    "direct_pred_score": t1["direct_distilled_score"],
                    "best_reg_param": t1["best_reg_param"],
                    "knn_score": t2["knn_score"],
                    "purity_at_k": t2["purity_at_k"],
                    "aux_r2": t3.get("mean_r2"),
                    "uq_rho": t4u["probe_spearman"],
                    "uq_r2": t4u["probe_r2"],
                    "uq_direct_rho": t4u["direct_head_spearman"],
                    "mahal_high_uq_auroc": t4o["mahal_high_uq_auroc"],
                    "mahal_high_uq_auprc": t4o.get("mahal_high_uq_auprc"),
                    "mahal_high_uq_fpr95": t4o.get("mahal_high_uq_fpr95"),
                    "cosine_high_uq_auroc": t4o["cosine_high_uq_auroc"],
                    "mahal_ood_swap_auroc": (t4o.get("mahal_ood") or {}).get("auroc"),
                    "mahal_ood_swap_auprc": (t4o.get("mahal_ood") or {}).get("auprc"),
                    "mahal_ood_swap_fpr95": (t4o.get("mahal_ood") or {}).get("fpr_at_95_tpr"),
                    "cosine_ood_swap_auroc": (t4o.get("cosine_ood") or {}).get("auroc"),
                    "mahal_ood_shift_auroc": (t4s.get("mahal_ood") or {}).get("auroc"),
                    "mahal_ood_shift_auprc": (t4s.get("mahal_ood") or {}).get("auprc"),
                    "mahal_ood_shift_fpr95": (t4s.get("mahal_ood") or {}).get("fpr_at_95_tpr"),
                    "cosine_ood_shift_auroc": (t4s.get("cosine_ood") or {}).get("auroc"),
                    "storage_mb_fp32": eff.get("storage_mb_fp32_all_splits"),
                    "compression_ratio": eff.get("compression_ratio_vs_raw_k"),
                    "knn_search_time_sec": m_data.get("knn_search_time_sec"),
                    "fit_time_sec": m_data["fit_time_sec"],
                    "probe_time_sec": m_data.get("probe_time_sec"),
                }
            )

    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "per_seed_phase3_metrics.csv", index=False)

    summary_rows: list[dict[str, Any]] = []
    for (ds, var, meth), grp in df.groupby(["dataset", "variant", "method"]):
        ens_m = float(grp["g0_ens"].mean())
        tb_m = float(grp["tabm_b"].mean())
        lin_m = float(grp["linear_score"].mean())
        denom = ens_m - tb_m

        def _col_mean(col: str) -> float:
            return float(grp[col].mean()) if grp[col].notna().any() else float("nan")

        def _col_median(col: str) -> float:
            return float(grp[col].median()) if grp[col].notna().any() else float("nan")

        summary_rows.append(
            {
                "dataset": ds,
                "variant": var,
                "method": meth,
                "n_seeds": int(len(grp)),
                "dim": int(grp["dim"].iloc[0]),
                "g0_ens": ens_m,
                "tabm_b": tb_m,
                "linear_mean": lin_m,
                "linear_std": float(grp["linear_score"].std(ddof=0)),
                "delta_vs_tabm_b": float(lin_m - tb_m),
                "clean_gap_rec": float((lin_m - tb_m) / denom) if abs(denom) >= 1e-6 else float("nan"),
                "direct_pred_mean": _col_mean("direct_pred_score"),
                "knn_mean": _col_mean("knn_score"),
                "aux_r2_mean": _col_mean("aux_r2"),
                "aux_r2_median": _col_median("aux_r2"),
                "uq_rho_mean": _col_mean("uq_rho"),
                "uq_r2_median": _col_median("uq_r2"),
                "uq_direct_rho_mean": _col_mean("uq_direct_rho"),
                "mahal_high_uq_auroc": _col_mean("mahal_high_uq_auroc"),
                "mahal_high_uq_auprc": _col_mean("mahal_high_uq_auprc"),
                "mahal_high_uq_fpr95": _col_mean("mahal_high_uq_fpr95"),
                "mahal_ood_swap_auroc": _col_mean("mahal_ood_swap_auroc"),
                "mahal_ood_swap_auprc": _col_mean("mahal_ood_swap_auprc"),
                "mahal_ood_swap_fpr95": _col_mean("mahal_ood_swap_fpr95"),
                "cosine_ood_swap_auroc": _col_mean("cosine_ood_swap_auroc"),
                "mahal_ood_shift_auroc": _col_mean("mahal_ood_shift_auroc"),
                "mahal_ood_shift_auprc": _col_mean("mahal_ood_shift_auprc"),
                "mahal_ood_shift_fpr95": _col_mean("mahal_ood_shift_fpr95"),
                "storage_mb_fp32": _col_mean("storage_mb_fp32"),
                "compression_ratio": _col_mean("compression_ratio"),
                "knn_search_time_sec": _col_mean("knn_search_time_sec"),
                "fit_time_sec": _col_mean("fit_time_sec"),
                "probe_time_sec": _col_mean("probe_time_sec"),
            }
        )

    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(out_dir / "summary_phase3_metrics.csv", index=False)

    print("\n=== Phase 3 (v4.0) Evaluation Summary ===")
    for (ds, var), sub in df_sum.groupby(["dataset", "variant"]):
        print(
            f"\n[{ds} | {var}] G0_Ens={sub['g0_ens'].iloc[0]:.5f} | "
            f"TabM[B]={sub['tabm_b'].iloc[0]:.5f}"
        )
        print(
            f"  {'Method':<16} | {'Linear':>10} | {'kNN':>10} | {'AuxR2':>7} "
            f"| {'UQ_rho':>7} | {'MahalUQ':>7} | {'OOD_Swp':>7} | {'OOD_Shf':>7}"
        )
        print("  " + "-" * 88)
        for _, r in sub.iterrows():
            def _fmt(v: Any, w: int = 7, d: int = 4) -> str:
                if isinstance(v, (int, float, np.floating)) and np.isfinite(float(v)):
                    return f"{float(v):>{w}.{d}f}"
                return f"{'N/A':>{w}}"

            print(
                f"  {r['method']:<16} | {_fmt(r['linear_mean'], 10, 5)} | {_fmt(r['knn_mean'], 10, 5)} "
                f"| {_fmt(r['aux_r2_median'])} | {_fmt(r['uq_rho_mean'])} | {_fmt(r['mahal_high_uq_auroc'])} "
                f"| {_fmt(r['mahal_ood_swap_auroc'])} | {_fmt(r['mahal_ood_shift_auroc'])}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 3 (v4.0): Bridge-v2, Real OOD from G0 .pt, and Ablations."
    )
    parser.add_argument("--reps-dir", type=Path, required=True, help="Read-only Phase 1 G0 folder (.npz + .pt).")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Official TabM raw data dir.")
    parser.add_argument(
        "--v1-bridge-dir",
        type=Path,
        default=None,
        help="Optional Phase 2 bridge dir for zero-retrain v1 reload.",
    )
    parser.add_argument("--mode", type=str, default="main", choices=["main", "ablation"])
    parser.add_argument("--datasets", nargs="+", default=["california", "house", "adult", "churn", "otto"])
    parser.add_argument("--variants", nargs="+", default=["tabm", "tabm_mini"], choices=["tabm", "tabm_mini"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--out-dir", type=Path, default=Path("results/phase3_v4"))
    parser.add_argument(
        "--save-bridge-dir",
        type=Path,
        default=None,
        help="Optional dir to save Bridge-v2 checkpoints.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume only runs matching the exact config_hash.")
    parser.add_argument(
        "--include-bridge-v1",
        action="store_true",
        help="Train Bridge-v1 if --v1-bridge-dir is unavailable.",
    )
    parser.add_argument("--skip-v2-512", action="store_true", help="Skip 512-d Bridge-v2 (run only 256-d).")
    parser.add_argument(
        "--no-ood",
        dest="enable_ood",
        action="store_false",
        help="Disable real OOD extraction from .pt.",
    )
    parser.set_defaults(enable_ood=True)

    # Probe & baseline parameters
    parser.add_argument("--k-neighbors", type=int, default=15)
    parser.add_argument("--n-aux-features", type=int, default=3)
    parser.add_argument("--max-collinearity", type=float, default=0.98)
    parser.add_argument("--pca-max-fit-samples", type=int, default=16384)
    parser.add_argument("--procrustes-max-iter", type=int, default=15)
    parser.add_argument("--procrustes-max-fit-samples", type=int, default=8192)

    # Bridge-v2 calibrated hyperparameters
    parser.add_argument("--d-z-v2", type=int, default=256)
    parser.add_argument("--d-hidden-v2", type=int, default=256)
    parser.add_argument("--align-mode", type=str, default="low_rank", choices=["low_rank", "full"])
    parser.add_argument("--align-rank", type=int, default=16)
    parser.add_argument("--alpha-gram", type=float, default=10.0)
    parser.add_argument("--alpha-pred", type=float, default=1.0)
    parser.add_argument("--alpha-uq", type=float, default=0.15)
    parser.add_argument("--alpha-recon", type=float, default=1.0)
    parser.add_argument("--init-gate", type=float, default=0.05)
    parser.add_argument("--gram-warmup-epochs", type=int, default=10)
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
    cur_hash = _config_fingerprint(args)

    records: list[dict[str, Any]] = []
    done_keys: set[tuple[str, str, int]] = set()
    if args.resume and json_path.exists():
        with json_path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        for r in loaded:
            if r.get("config_hash") == cur_hash and r.get("mode") == args.mode:
                records.append(r)
                done_keys.add((r["dataset"], r["variant"], int(r["seed"])))
        print(
            f"Resuming {len(done_keys)} matching runs "
            f"(mode={args.mode}, hash={cur_hash}) from {json_path}."
        )

    ds_cache: dict[
        str, tuple[ProcessedDataset | None, tuple[list[int], dict[str, np.ndarray]] | None]
    ] = {}
    for dataset_name in args.datasets:
        reps_ds_dir, raw_ds_dir = _resolve_dataset_paths(dataset_name, args.reps_dir, args.data_dir)
        if reps_ds_dir is None:
            print(f"[MISS] No G0 folder found in {args.reps_dir} for '{dataset_name}'")
            continue

        for variant in args.variants:
            for seed in args.seeds:
                key = (dataset_name, variant, int(seed))
                if key in done_keys:
                    print(f"[SKIP] {dataset_name} | {variant} | seed={seed} (verified hash={cur_hash})")
                    continue

                run_dir = reps_ds_dir / variant / f"seed_{seed}"
                if not (run_dir / "reps_train.npz").exists():
                    print(f"[MISS] Skipping {run_dir} (reps_train.npz not found)")
                    continue

                t0 = time.perf_counter()
                print(f"\n[RUN ] mode={args.mode} | {dataset_name} | {variant} | seed={seed} ...")
                rec = _evaluate_run_v4(
                    dataset_name=dataset_name,
                    reps_ds_dir=reps_ds_dir,
                    raw_ds_dir=raw_ds_dir,
                    variant=variant,
                    seed=seed,
                    args=args,
                    ds_cache=ds_cache,
                )
                records.append(rec)
                done_keys.add(key)

                with json_path.open("w", encoding="utf-8") as f:
                    json.dump(_sanitize_json(records), f, indent=2)

                ood_str = (
                    f" | pt_cos={rec['ood_diagnostics']['parity_cosine']:.5f}"
                    if rec.get("ood_diagnostics")
                    else ""
                )
                p_lin = rec["methods"]["procrustes"]["test1_parity"]["linear_test_score"]
                v2_key = "bridge_v2_256" if "bridge_v2_256" in rec["methods"] else "abl_v2_full_256"
                v2_lin = rec["methods"][v2_key]["test1_parity"]["linear_test_score"]
                v2_aux = (rec["methods"][v2_key]["test3_transfer"] or {}).get("mean_r2", float("nan"))
                v2_mh = rec["methods"][v2_key]["test4_ood"]["mahal_high_uq_auroc"]
                print(
                    f"[DONE] {dataset_name} | {variant} | seed={seed} ({time.perf_counter() - t0:.1f}s)"
                    f"{ood_str} | proc={p_lin:.5f}, {v2_key}={v2_lin:.5f}, AuxR2={v2_aux:.4f}, MahalUQ={v2_mh:.4f}"
                )

    _export_phase3_summary_tables(records, args.out_dir)


if __name__ == "__main__":
    main()