from __future__ import annotations

import json
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from tabm_bridge.config import TaskType
from tabm_bridge.data.preprocessing import RawTabularDataset

OFFICIAL_TABM_DATA_URL = (
    "https://huggingface.co/datasets/rototoHF/tabm-data/resolve/main/data.tar"
)


def _load_optional_splits(dataset_dir: Path, prefix: str) -> dict[str, np.ndarray] | None:
    train_path = dataset_dir / f"{prefix}_train.npy"
    if not train_path.exists():
        return None
    return {
        split: np.load(dataset_dir / f"{prefix}_{split}.npy", allow_pickle=True)
        for split in ("train", "val", "test")
    }


def load_official_dataset(dataset_dir: str | Path) -> RawTabularDataset:
    """Load a dataset stored in the official TabM .npy + info.json directory layout."""
    path = Path(dataset_dir)
    info_path = path / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing info.json in dataset directory: {path}")

    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)

    task_type: TaskType = info["task_type"]
    if task_type not in ("regression", "binclass", "multiclass"):
        raise ValueError(f"Unsupported task_type in {info_path}: {task_type}")

    y = {
        split: np.load(path / f"Y_{split}.npy", allow_pickle=False)
        for split in ("train", "val", "test")
    }

    return RawTabularDataset(
        name=str(info.get("name", path.name)),
        task_type=task_type,
        y=y,
        x_num=_load_optional_splits(path, "X_num"),
        x_bin=_load_optional_splits(path, "X_bin"),
        x_cat=_load_optional_splits(path, "X_cat"),
        primary_score=info.get("score"),
    )


def download_official_tabm_data(
    target_dir: str | Path = "data",
    archive_path: str | Path = "local/tabm-data.tar",
    force: bool = False,
) -> Path:
    """Download and extract the official TabM preprocessed datasets archive."""
    out_dir = Path(target_dir)
    tar_path = Path(archive_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    tar_path.parent.mkdir(parents=True, exist_ok=True)

    if force or not tar_path.exists():
        urllib.request.urlretrieve(OFFICIAL_TABM_DATA_URL, tar_path)

    with tarfile.open(tar_path, "r:*") as tar:
        try:
            tar.extractall(path=out_dir, filter="data")
        except TypeError:
            tar.extractall(path=out_dir)

    return out_dir


def load_sklearn_california_housing(seed: int = 0) -> RawTabularDataset:
    """Load California Housing using the exact split protocol from official example.ipynb.

    Useful for fast local/Colab G0 numerical parity checks before downloading data.tar.
    """
    dataset = fetch_california_housing()
    x_all = dataset["data"].astype(np.float32)
    y_all = dataset["target"].astype(np.float32)

    all_idx = np.arange(len(y_all))
    trainval_idx, test_idx = train_test_split(all_idx, train_size=0.8, random_state=seed)
    train_idx, val_idx = train_test_split(trainval_idx, train_size=0.8, random_state=seed)

    return RawTabularDataset(
        name="california_sklearn",
        task_type="regression",
        x_num={
            "train": x_all[train_idx],
            "val": x_all[val_idx],
            "test": x_all[test_idx],
        },
        y={
            "train": y_all[train_idx],
            "val": y_all[val_idx],
            "test": y_all[test_idx],
        },
        primary_score="rmse",
    )