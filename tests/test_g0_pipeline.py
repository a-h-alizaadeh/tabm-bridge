from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tabm_bridge.config import ModelConfig, PLEConfig, TrainConfig
from tabm_bridge.data import (
    RawTabularDataset,
    compute_ple_bins,
    load_official_dataset,
    preprocess_dataset,
)
from tabm_bridge.evaluation import compute_metrics
from tabm_bridge.model import TabMModel
from tabm_bridge.training import TabMTrainer, compute_independent_loss


def _make_synthetic_raw_dataset(task_type: str = "regression") -> RawTabularDataset:
    rng = np.random.default_rng(123)
    n_tr, n_va, n_te = 160, 40, 40

    def _make_x_num(n: int) -> np.ndarray:
        x = rng.normal(0.0, 1.0, size=(n, 4)).astype(np.float32)
        x[:, 2] = 5.0  # Constant column in train -> must be dropped
        x[0, 1] = np.nan  # Missing value -> must become 0.0 after quantile norm
        return x

    def _make_x_bin(n: int) -> np.ndarray:
        b = rng.integers(0, 2, size=(n, 2)).astype(np.float32)
        b[:, 1] = 1.0  # Constant binary column -> must be dropped
        return b

    x_cat_tr = np.column_stack(
        [rng.choice(["a", "b", "c"], size=n_tr), rng.choice(["x", "y"], size=n_tr)]
    )
    x_cat_va = np.column_stack(
        [rng.choice(["a", "b", "OOV_VAL"], size=n_va), rng.choice(["x", "y"], size=n_va)]
    )
    x_cat_te = np.column_stack(
        [rng.choice(["a", "OOV_TEST"], size=n_te), rng.choice(["x", "OOV_Y"], size=n_te)]
    )

    x_num = {"train": _make_x_num(n_tr), "val": _make_x_num(n_va), "test": _make_x_num(n_te)}
    x_bin = {"train": _make_x_bin(n_tr), "val": _make_x_bin(n_va), "test": _make_x_bin(n_te)}
    x_cat = {"train": x_cat_tr, "val": x_cat_va, "test": x_cat_te}

    y = {}
    for split, arr in x_num.items():
        signal = np.nan_to_num(arr[:, 0]) * 2.0 - np.nan_to_num(arr[:, 3])
        if task_type == "regression":
            y[split] = (signal * 100.0 + 500.0).astype(np.float32)
        elif task_type == "binclass":
            y[split] = (signal > 0.0).astype(np.int64)
        else:
            bins = np.quantile(signal, [0.33, 0.66])
            y[split] = np.digitize(signal, bins).astype(np.int64)

    return RawTabularDataset(
        name=f"synth_{task_type}",
        task_type=task_type,  # type: ignore[arg-type]
        x_num=x_num,
        x_bin=x_bin,
        x_cat=x_cat,
        y=y,
    )


def test_preprocessing_anti_leakage_nan_constant_bin_and_oov() -> None:
    raw = _make_synthetic_raw_dataset("regression")
    proc = preprocess_dataset(raw, seed=0)

    # 1 constant num col dropped -> 3 kept; NaNs replaced cleanly with 0.0
    assert proc.n_num_features == 3
    for split in ("train", "val", "test"):
        arr = proc.splits[split].x_num
        assert arr is not None
        assert arr.dtype == np.float32
        assert np.isfinite(arr).all()
        assert arr[0, 1] == pytest.approx(0.0)

    # 2 cat cols (cards 3, 2) + 1 kept bin col (card 2); constant bin col dropped
    assert proc.cat_cardinalities == [3, 2, 2]
    assert proc.splits["val"].x_cat is not None
    assert proc.splits["test"].x_cat is not None
    # OOV values must map strictly to column cardinality (3 for col 0, 2 for col 1)
    assert int(proc.splits["val"].x_cat[:, 0].max()) == 3
    assert int(proc.splits["test"].x_cat[:, 1].max()) == 2

    # Regression targets standardized on train only
    assert abs(float(proc.splits["train"].y.mean())) < 1e-5
    assert abs(float(proc.splits["train"].y.std(ddof=0)) - 1.0) < 1e-5
    assert proc.target_stats.mean == pytest.approx(float(raw.y["train"].mean()), rel=1e-5)


def test_independent_loss_differs_from_loss_of_mean() -> None:
    # Two heads predicting -2 and +2 for target 0:
    # loss(mean) == 0, whereas mean(loss_i) == 4.0
    y_pred = torch.tensor([[[-2.0], [2.0]]])  # (B=1, k=2, d_out=1)
    y_true = torch.tensor([0.0])
    loss = compute_independent_loss(y_pred, y_true, task_type="regression")
    assert loss.item() == pytest.approx(4.0, abs=1e-6)


def test_official_directory_loader(tmp_path: Path) -> None:
    raw = _make_synthetic_raw_dataset("binclass")
    ds_dir = tmp_path / "toy_bin"
    ds_dir.mkdir()
    with (ds_dir / "info.json").open("w", encoding="utf-8") as f:
        json.dump({"name": "toy_bin", "task_type": "binclass"}, f)
    for s in ("train", "val", "test"):
        assert raw.x_num is not None and raw.x_cat is not None and raw.x_bin is not None
        np.save(ds_dir / f"X_num_{s}.npy", raw.x_num[s])
        np.save(ds_dir / f"X_cat_{s}.npy", raw.x_cat[s])
        np.save(ds_dir / f"X_bin_{s}.npy", raw.x_bin[s])
        np.save(ds_dir / f"Y_{s}.npy", raw.y[s])

    loaded = load_official_dataset(ds_dir)
    proc = preprocess_dataset(loaded, seed=0)
    assert proc.task_type == "binclass"
    assert proc.d_out == 2  # Official d_out=2 for binclass
    assert proc.primary_score == "accuracy"


@pytest.mark.parametrize("task_type", ["regression", "binclass", "multiclass"])
@pytest.mark.parametrize("share_batches", [True, False])
def test_trainer_end_to_end_and_representation_extraction(
    tmp_path: Path, task_type: str, share_batches: bool
) -> None:
    raw = _make_synthetic_raw_dataset(task_type)
    proc = preprocess_dataset(raw, seed=0)
    ple_bins = compute_ple_bins(proc, n_bins=8)

    model_cfg = ModelConfig.make(
        n_num_features=proc.n_num_features,
        cat_cardinalities=proc.cat_cardinalities,
        d_out=proc.d_out,
        variant="tabm_mini",
        k=4,
        d_block=32,
        dropout=0.1,
        ple=PLEConfig(enabled=True, n_bins=8, d_embedding=4, version="B"),
    )
    model = TabMModel(model_cfg, ple_bins=ple_bins)
    train_cfg = TrainConfig(
        task_type=proc.task_type,
        batch_size=32,
        share_training_batches=share_batches,
        lr=3e-3,
        weight_decay=3e-4,
        patience=3,
        max_epochs=6,
        use_amp=False,
        device="cpu",
        seed=42,
    )

    trainer = TabMTrainer(
        model=model,
        config=train_cfg,
        target_stats=proc.target_stats,
        primary_score=proc.primary_score,
    )
    save_dir = tmp_path / f"run_{task_type}_{share_batches}"
    result = trainer.fit(proc, save_dir=save_dir)

    assert not result.stopped_due_to_nan
    assert result.best_epoch >= 0
    assert "score" in result.metrics["test"]
    assert (save_dir / "best_checkpoint.pt").exists()
    assert (save_dir / "report.json").exists()

    # Verify G1 representation bundle extraction
    bundle = trainer.extract_representations(proc.splits["test"])
    n_test = proc.splits["test"].y.shape[0]
    assert bundle.embeddings.shape == (n_test, 4, 32)
    assert bundle.per_head_preds.shape == (n_test, 4, proc.d_out)
    assert np.isfinite(bundle.embeddings).all()
    assert np.isfinite(bundle.epistemic_uncertainty).all()


def test_metrics_orientation() -> None:
    m_reg = compute_metrics(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 5.0]), "regression")
    assert m_reg["rmse"] > 0
    assert m_reg["score"] == pytest.approx(-m_reg["rmse"])

    m_bin = compute_metrics(
        np.array([0, 1, 0, 1]),
        np.array([[0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.1, 0.9]]),
        "binclass",
    )
    assert m_bin["accuracy"] == pytest.approx(1.0)
    assert m_bin["roc-auc"] == pytest.approx(1.0)
    assert m_bin["score"] == pytest.approx(1.0)