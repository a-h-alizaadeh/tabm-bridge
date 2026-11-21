from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import rtdl_num_embeddings
import torch
from sklearn.preprocessing import OrdinalEncoder, QuantileTransformer

from tabm_bridge.config import TaskType


@dataclass
class RawTabularDataset:
    """Unprocessed train/val/test arrays matching the official TabM disk layout."""

    name: str
    task_type: TaskType
    y: dict[str, np.ndarray]
    x_num: dict[str, np.ndarray] | None = None
    x_bin: dict[str, np.ndarray] | None = None
    x_cat: dict[str, np.ndarray] | None = None
    primary_score: str | None = None


@dataclass
class TargetStats:
    """Regression target standardization statistics fitted strictly on train."""

    mean: float = 0.0
    std: float = 1.0


@dataclass
class ProcessedSplit:
    """Single preprocessed data split ready for tensor conversion."""

    x_num: np.ndarray | None
    x_cat: np.ndarray | None
    y: np.ndarray       # Standardized float32 for regression, int64 for classification
    y_raw: np.ndarray   # Original-scale targets for exact metric evaluation


@dataclass
class ProcessedDataset:
    """Leakage-free preprocessed dataset with metadata for TabMModel construction."""

    name: str
    task_type: TaskType
    primary_score: str
    splits: dict[str, ProcessedSplit]
    n_num_features: int
    cat_cardinalities: list[int] = field(default_factory=list)
    d_out: int = 1
    target_stats: TargetStats = field(default_factory=TargetStats)


def _transform_num_noisy_quantile(
    x_num: dict[str, np.ndarray],
    seed: int = 0,
) -> dict[str, np.ndarray] | None:
    """Official 5-step noisy-quantile numerical preprocessing (lib/data.py & example.ipynb).

    1. Add N(0, 1e-5) noise ONLY to a copy of train before QuantileTransformer.fit().
    2. Transform raw (un-noised) train, val, and test splits.
    3. Replace NaNs with 0.0 (mean of standard normal) AFTER normalization.
    4. Drop numerical columns that have only 1 unique value in train.
    5. Cast to float32 and assert finite values (fixing the official repo's NaN-check bug).
    """
    x_train_raw = np.asarray(x_num["train"], dtype=np.float32)
    n_train = x_train_raw.shape[0]
    n_quantiles = max(min(n_train // 30, 1000), 10)

    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1e-5, size=x_train_raw.shape).astype(np.float32)
    x_train_noisy = x_train_raw + noise

    qt = QuantileTransformer(
        n_quantiles=n_quantiles,
        output_distribution="normal",
        subsample=10**9,
        random_state=seed,
    )
    qt.fit(x_train_noisy)

    transformed: dict[str, np.ndarray] = {}
    for split_name in ("train", "val", "test"):
        arr = np.asarray(x_num[split_name], dtype=np.float32)
        arr = qt.transform(arr)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        transformed[split_name] = arr

    # Drop constant columns based strictly on train split
    keep_cols = [
        j
        for j in range(transformed["train"].shape[1])
        if np.unique(transformed["train"][:, j]).size > 1
    ]
    if not keep_cols:
        return None

    for split_name in ("train", "val", "test"):
        transformed[split_name] = np.ascontiguousarray(
            transformed[split_name][:, keep_cols], dtype=np.float32
        )
        if not np.isfinite(transformed[split_name]).all():
            raise ValueError(f"Non-finite values remain in x_num['{split_name}'] after preprocessing.")

    return transformed


def _sanitize_cat_array(arr: np.ndarray) -> np.ndarray:
    """Convert categorical array to clean string representation with explicit missing marker."""
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr[:, None]
    if np.issubdtype(arr.dtype, np.floating):
        out = np.where(np.isnan(arr), "__MISSING__", arr.astype(str))
        return out.astype(str)
    if arr.dtype == object:
        flat = [
            "__MISSING__" if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v)
            for v in arr.ravel()
        ]
        return np.asarray(flat, dtype=str).reshape(arr.shape)
    return arr.astype(str)


def _transform_cat_oov_safe(
    x_cat: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], list[int]]:
    """Official OOV-safe OrdinalEncoder fitted strictly on train (lib/data.py).

    Unknown categories in val/test are mapped to `cardinality[j]`, which pairs with
    TabMModel's `F.one_hot(..., card + 1)[..., :-1]` to emit all-zero vectors.
    """
    unknown_sentinel = np.iinfo(np.int64).max - 3
    train_clean = _sanitize_cat_array(x_cat["train"])

    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=unknown_sentinel,
        dtype=np.int64,
    )
    x_train_enc = encoder.fit_transform(train_clean).astype(np.int64)
    cardinalities = [int(np.unique(x_train_enc[:, j]).size) for j in range(x_train_enc.shape[1])]

    encoded: dict[str, np.ndarray] = {"train": np.ascontiguousarray(x_train_enc, dtype=np.int64)}
    for split_name in ("val", "test"):
        clean = _sanitize_cat_array(x_cat[split_name])
        enc = encoder.transform(clean).astype(np.int64)
        for j, card in enumerate(cardinalities):
            mask = enc[:, j] == unknown_sentinel
            if np.any(mask):
                enc[mask, j] = card
        encoded[split_name] = np.ascontiguousarray(enc, dtype=np.int64)

    return encoded, cardinalities


def _convert_bin_to_cat(
    x_bin: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray] | None, list[int]]:
    """Official binary feature handling (paper/bin/model.py).

    Drops binary columns with only 1 unique value in train, converts remaining
    columns to int64, and prepares them to be appended to x_cat (cardinality 2).
    """
    train_bin = np.asarray(x_bin["train"])
    if train_bin.ndim == 1:
        train_bin = train_bin[:, None]

    keep_cols = []
    for j in range(train_bin.shape[1]):
        col = train_bin[:, j]
        valid = col[~np.isnan(col)] if np.issubdtype(col.dtype, np.floating) else col
        if np.unique(valid).size > 1:
            keep_cols.append(j)

    if not keep_cols:
        return None, []

    # Use OOV-safe ordinal encoding on kept binary columns so non-{0,1} or NaN values
    # in val/test never crash int64 indexing while preserving cardinality=2 for standard {0,1}.
    sub_bin = {s: np.asarray(x_bin[s])[:, keep_cols] for s in ("train", "val", "test")}
    train_vals = sub_bin["train"]
    is_clean_binary = (
        np.all(np.isfinite(train_vals))
        and set(np.unique(train_vals).tolist()).issubset({0, 1, 0.0, 1.0})
        and all(
            np.all(np.isfinite(sub_bin[s]))
            and set(np.unique(sub_bin[s]).tolist()).issubset({0, 1, 0.0, 1.0})
            for s in ("val", "test")
        )
    )
    if is_clean_binary:
        converted = {
            s: np.ascontiguousarray(sub_bin[s].astype(np.int64), dtype=np.int64)
            for s in ("train", "val", "test")
        }
        return converted, [2] * len(keep_cols)

    return _transform_cat_oov_safe(sub_bin)


def preprocess_dataset(
    raw: RawTabularDataset,
    seed: int = 0,
) -> ProcessedDataset:
    """Run the full leakage-free official TabM preprocessing pipeline."""
    x_num_proc: dict[str, np.ndarray] | None = None
    if raw.x_num is not None and raw.x_num["train"].shape[1] > 0:
        x_num_proc = _transform_num_noisy_quantile(raw.x_num, seed=seed)

    cat_parts: list[dict[str, np.ndarray]] = []
    cat_cardinalities: list[int] = []

    if raw.x_cat is not None and raw.x_cat["train"].shape[1] > 0:
        cat_enc, cat_cards = _transform_cat_oov_safe(raw.x_cat)
        cat_parts.append(cat_enc)
        cat_cardinalities.extend(cat_cards)

    if raw.x_bin is not None and raw.x_bin["train"].shape[1] > 0:
        bin_enc, bin_cards = _convert_bin_to_cat(raw.x_bin)
        if bin_enc is not None:
            cat_parts.append(bin_enc)
            cat_cardinalities.extend(bin_cards)

    x_cat_proc: dict[str, np.ndarray] | None = None
    if cat_parts:
        x_cat_proc = {
            s: np.ascontiguousarray(
                np.concatenate([part[s] for part in cat_parts], axis=1), dtype=np.int64
            )
            for s in ("train", "val", "test")
        }

    # Target processing
    target_stats = TargetStats()
    y_proc: dict[str, np.ndarray] = {}
    y_raw_out: dict[str, np.ndarray] = {}

    if raw.task_type == "regression":
        y_train_f = np.asarray(raw.y["train"], dtype=np.float32).reshape(-1)
        mean = float(y_train_f.mean())
        std = float(y_train_f.std(ddof=0))
        if std == 0.0:
            std = 1.0
        target_stats = TargetStats(mean=mean, std=std)
        for s in ("train", "val", "test"):
            arr = np.asarray(raw.y[s], dtype=np.float32).reshape(-1)
            y_raw_out[s] = arr
            y_proc[s] = ((arr - mean) / std).astype(np.float32)
        d_out = 1
        default_primary = "rmse"
    else:
        for s in ("train", "val", "test"):
            arr = np.asarray(raw.y[s], dtype=np.int64).reshape(-1)
            y_raw_out[s] = arr
            y_proc[s] = arr
        n_classes = int(np.unique(y_proc["train"]).size)
        if raw.task_type == "binclass":
            if n_classes != 2:
                raise ValueError(f"binclass requires 2 unique classes in train, got {n_classes}.")
            d_out = 2  # Official TabM uses d_out=2 + cross_entropy for binclass
        else:
            d_out = n_classes
        default_primary = "accuracy"

    primary_score = raw.primary_score or default_primary
    n_num_features = 0 if x_num_proc is None else int(x_num_proc["train"].shape[1])

    splits = {
        s: ProcessedSplit(
            x_num=None if x_num_proc is None else x_num_proc[s],
            x_cat=None if x_cat_proc is None else x_cat_proc[s],
            y=y_proc[s],
            y_raw=y_raw_out[s],
        )
        for s in ("train", "val", "test")
    }

    return ProcessedDataset(
        name=raw.name,
        task_type=raw.task_type,
        primary_score=primary_score,
        splits=splits,
        n_num_features=n_num_features,
        cat_cardinalities=cat_cardinalities,
        d_out=d_out,
        target_stats=target_stats,
    )


def compute_ple_bins(
    dataset: ProcessedDataset,
    n_bins: int = 48,
    tree_kwargs: dict[str, Any] | None = None,
) -> list[torch.Tensor] | None:
    """Compute PLE bin edges STRICTLY on the training split (paper/bin/model.py)."""
    if dataset.n_num_features == 0 or dataset.splits["train"].x_num is None:
        return None

    x_train = torch.as_tensor(dataset.splits["train"].x_num, dtype=torch.float32)
    safe_n_bins = min(n_bins, max(2, x_train.shape[0] - 1))

    if tree_kwargs is not None:
        y_dtype = torch.float32 if dataset.task_type == "regression" else torch.long
        y_train = torch.as_tensor(dataset.splits["train"].y, dtype=y_dtype)
        return rtdl_num_embeddings.compute_bins(
            x_train,
            n_bins=safe_n_bins,
            tree_kwargs=tree_kwargs,
            y=y_train,
            regression=(dataset.task_type == "regression"),
        )

    return rtdl_num_embeddings.compute_bins(x_train, n_bins=safe_n_bins)