from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tabm_bridge.config import ModelConfig, PLEConfig, TrainConfig, VariantType
from tabm_bridge.data import (
    ProcessedDataset,
    ProcessedSplit,
    compute_ple_bins,
    load_official_dataset,
    load_sklearn_california_housing,
    preprocess_dataset,
)
from tabm_bridge.evaluation import compute_metrics
from tabm_bridge.model import TabMModel
from tabm_bridge.training import TabMTrainer

# Official exp/tabm paper reference test scores (higher is better)
OFFICIAL_PAPER_REFERENCE: dict[str, tuple[float, float, str]] = {
    "california": (-0.450932, 0.003154, "RMSE (negated)"),
    "house": (-30002.387181, 181.962989, "RMSE (negated)"),
    "adult": (0.858158, 0.001100, "Accuracy"),
    "churn": (0.861300, 0.002463, "Accuracy"),
    "otto": (0.826756, 0.001436, "Accuracy"),
    "sberbank-housing": (-0.246943, 0.003539, "RMSE (negated)"),
}


def _build_gbdt_frame(split: ProcessedSplit) -> tuple[pd.DataFrame, list[str]]:
    cols: dict[str, Any] = {}
    cat_cols: list[str] = []
    if split.x_num is not None:
        for j in range(split.x_num.shape[1]):
            cols[f"num_{j}"] = split.x_num[:, j].astype(np.float32)
    if split.x_cat is not None:
        for j in range(split.x_cat.shape[1]):
            name = f"cat_{j}"
            cols[name] = split.x_cat[:, j].astype(np.int64)
            cat_cols.append(name)
    return pd.DataFrame(cols), cat_cols


def _run_gbdt_baselines(
    dataset: ProcessedDataset,
    seeds: list[int],
) -> dict[str, list[dict[str, float]]]:
    """Run LightGBM and CatBoost (CPU, thread_count >= 4 per official paper/bin/evaluate.py)."""
    from catboost import CatBoostClassifier, CatBoostRegressor
    from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping

    x_tr, cat_cols = _build_gbdt_frame(dataset.splits["train"])
    x_va, _ = _build_gbdt_frame(dataset.splits["val"])
    x_te, _ = _build_gbdt_frame(dataset.splits["test"])
    y_tr = dataset.splits["train"].y_raw
    y_va = dataset.splits["val"].y_raw
    y_te = dataset.splits["test"].y_raw

    n_threads = max(4, os.cpu_count() or 4)
    results: dict[str, list[dict[str, float]]] = {"lightgbm": [], "catboost": []}

    for seed in seeds:
        # 1. LightGBM
        t0 = time.perf_counter()
        if dataset.task_type == "regression":
            lgbm = LGBMRegressor(
                n_estimators=2000,
                learning_rate=0.05,
                random_state=seed,
                n_jobs=n_threads,
                verbosity=-1,
            )
            lgbm.fit(
                x_tr,
                y_tr,
                eval_set=[(x_va, y_va)],
                categorical_feature=cat_cols or "auto",
                callbacks=[early_stopping(50, verbose=False)],
            )
            pred_te = lgbm.predict(x_te)
        else:
            lgbm = LGBMClassifier(
                n_estimators=2000,
                learning_rate=0.05,
                random_state=seed,
                n_jobs=n_threads,
                verbosity=-1,
            )
            lgbm.fit(
                x_tr,
                y_tr,
                eval_set=[(x_va, y_va)],
                categorical_feature=cat_cols or "auto",
                callbacks=[early_stopping(50, verbose=False)],
            )
            pred_te = lgbm.predict_proba(x_te)
        m_lgb = compute_metrics(y_te, pred_te, dataset.task_type, dataset.primary_score)
        m_lgb["elapsed_sec"] = time.perf_counter() - t0
        results["lightgbm"].append(m_lgb)

        # 2. CatBoost (strictly CPU with thread_count >= 4 per official repo)
        t0 = time.perf_counter()
        cb_common: dict[str, Any] = {
            "iterations": 2000,
            "learning_rate": 0.05,
            "random_seed": seed,
            "task_type": "CPU",
            "thread_count": n_threads,
            "verbose": 0,
        }
        if dataset.task_type == "regression":
            cb = CatBoostRegressor(**cb_common)
            cb.fit(
                x_tr,
                y_tr,
                cat_features=cat_cols,
                eval_set=(x_va, y_va),
                early_stopping_rounds=50,
                verbose=False,
            )
            pred_te = cb.predict(x_te)
        else:
            cb = CatBoostClassifier(**cb_common)
            cb.fit(
                x_tr,
                y_tr,
                cat_features=cat_cols,
                eval_set=(x_va, y_va),
                early_stopping_rounds=50,
                verbose=False,
            )
            pred_te = cb.predict_proba(x_te)
        m_cb = compute_metrics(y_te, pred_te, dataset.task_type, dataset.primary_score)
        m_cb["elapsed_sec"] = time.perf_counter() - t0
        results["catboost"].append(m_cb)

    return results


def _save_representation_bundles(
    trainer: TabMTrainer,
    dataset: ProcessedDataset,
    run_dir: Path,
) -> None:
    """Save penultimate embeddings H (float16) and predictions/UQ (float32) for G1."""
    for split_name in ("train", "val", "test"):
        bundle = trainer.extract_representations(dataset.splits[split_name])
        payload: dict[str, np.ndarray] = {
            "embeddings": bundle.embeddings.astype(np.float16),
            "per_head_preds": bundle.per_head_preds.astype(np.float32),
            "mean_pred": bundle.mean_pred.astype(np.float32),
            "epistemic_uncertainty": bundle.epistemic_uncertainty.astype(np.float32),
            "y_raw": bundle.y_raw,
        }
        if bundle.aleatoric_proxy is not None:
            payload["aleatoric_proxy"] = bundle.aleatoric_proxy.astype(np.float32)
        np.savez(run_dir / f"reps_{split_name}.npz", **payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run G0 baseline training, numerical parity check, and H extraction."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="california_sklearn",
        help="Either 'california_sklearn' (zero-download sanity check) or official dataset folder name under --data-dir.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["tabm_mini", "tabm", "mlp"],
        choices=["mlp", "tabm_mini", "tabm"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--n-blocks", type=int, default=None, help="None = official smart default (2 with PLE, 3 without).")
    parser.add_argument("--d-block", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--n-bins", type=int, default=48)
    parser.add_argument("--d-embedding", type=int, default=16)
    parser.add_argument("--no-ple", action="store_true")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--independent-batches", action="store_true", help="Disable TabM♠ shared batches.")
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--save-dir", type=Path, default=Path("artifacts/g0"))
    parser.add_argument(
        "--save-reps",
        action="store_true",
        help="Extract and save penultimate H tensors (N, k, d_block) for G1 Procrustes & TabM-Bridge.",
    )
    parser.add_argument("--run-gbdt", action="store_true", help="Also run LightGBM and CatBoost baselines.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Load & preprocess strictly without leakage
    if args.dataset == "california_sklearn":
        raw = load_sklearn_california_housing(seed=0)
    else:
        raw = load_official_dataset(args.data_dir / args.dataset)

    dataset = preprocess_dataset(raw, seed=0)
    use_ple = (not args.no_ple) and (dataset.n_num_features > 0)
    ple_bins = compute_ple_bins(dataset, n_bins=args.n_bins) if use_ple else None

    print(
        f"\nDataset: {dataset.name} | Task: {dataset.task_type} | Primary: {dataset.primary_score} "
        f"| n_num={dataset.n_num_features}, n_cat={len(dataset.cat_cardinalities)}, d_out={dataset.d_out}"
    )
    print(
        f"Splits: train={len(dataset.splits['train'].y)}, "
        f"val={len(dataset.splits['val'].y)}, test={len(dataset.splits['test'].y)}"
    )

    summary: dict[str, dict[str, Any]] = {}
    out_root = args.save_dir / dataset.name
    out_root.mkdir(parents=True, exist_ok=True)

    # 2. Train neural variants across seeds
    for variant in args.variants:
        variant_t: VariantType = variant
        seed_scores: list[float] = []
        seed_times: list[float] = []
        seed_epochs: list[int] = []

        for seed in args.seeds:
            model_cfg = ModelConfig.make(
                n_num_features=dataset.n_num_features,
                cat_cardinalities=dataset.cat_cardinalities,
                d_out=dataset.d_out,
                variant=variant_t,
                k=args.k,
                n_blocks=args.n_blocks,
                d_block=args.d_block,
                dropout=args.dropout,
                ple=PLEConfig(
                    enabled=use_ple,
                    n_bins=args.n_bins,
                    d_embedding=args.d_embedding,
                    version="B",
                ),
            )
            train_cfg = TrainConfig(
                task_type=dataset.task_type,
                batch_size=args.batch_size,
                share_training_batches=not args.independent_batches,
                lr=args.lr,
                weight_decay=args.weight_decay,
                patience=args.patience,
                max_epochs=args.max_epochs,
                compile_model=args.compile,
                seed=seed,
                device=args.device,
            )

            model = TabMModel(model_cfg, ple_bins=ple_bins)
            trainer = TabMTrainer(
                model=model,
                config=train_cfg,
                target_stats=dataset.target_stats,
                primary_score=dataset.primary_score,
            )
            run_dir = out_root / variant / f"seed_{seed}"
            res = trainer.fit(dataset, save_dir=run_dir)

            if args.save_reps:
                _save_representation_bundles(trainer, dataset, run_dir)

            test_score = res.metrics["test"]["score"]
            val_score = res.metrics["val"]["score"]
            seed_scores.append(test_score)
            seed_times.append(res.elapsed_sec)
            seed_epochs.append(res.best_epoch)

            print(
                f"  [{variant:<9} | seed={seed}] best_epoch={res.best_epoch:3d} "
                f"| val_score={val_score:.6f} | test_score={test_score:.6f} "
                f"| time={res.elapsed_sec:.1f}s | device={trainer.device}"
            )

        summary[variant] = {
            "test_score_mean": float(np.mean(seed_scores)),
            "test_score_std": float(np.std(seed_scores)),
            "avg_best_epoch": float(np.mean(seed_epochs)),
            "avg_time_sec": float(np.mean(seed_times)),
            "per_seed_scores": seed_scores,
        }

    # 3. Optional GBDT baselines
    if args.run_gbdt:
        gbdt_runs = _run_gbdt_baselines(dataset, args.seeds)
        for name, runs in gbdt_runs.items():
            scores = [r["score"] for r in runs]
            times = [r["elapsed_sec"] for r in runs]
            summary[name] = {
                "test_score_mean": float(np.mean(scores)),
                "test_score_std": float(np.std(scores)),
                "avg_time_sec": float(np.mean(times)),
                "per_seed_scores": scores,
            }
            print(
                f"  [{name:<9}] test_score={np.mean(scores):.6f} ± {np.std(scores):.6f} "
                f"| avg_time={np.mean(times):.1f}s"
            )

    # 4. Print parity table
    print("\n=== G0 Baseline Summary (Higher Score is Better) ===")
    print(f"{'Model':<12} | {'Test Score (mean ± std)':<26} | {'Avg Time (s)':<12}")
    print("-" * 58)
    for name, stats in summary.items():
        print(
            f"{name:<12} | {stats['test_score_mean']:>11.6f} ± {stats['test_score_std']:<10.6f} "
            f"| {stats['avg_time_sec']:<12.2f}"
        )

    ref_key = "california" if args.dataset == "california_sklearn" else args.dataset
    if ref_key in OFFICIAL_PAPER_REFERENCE:
        ref_mean, ref_std, metric_label = OFFICIAL_PAPER_REFERENCE[ref_key]
        print("-" * 58)
        print(
            f"Paper exp/tabm ref ({ref_key}, {metric_label}): "
            f"{ref_mean:.6f} ± {ref_std:.6f}"
        )

    with (out_root / "g0_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()