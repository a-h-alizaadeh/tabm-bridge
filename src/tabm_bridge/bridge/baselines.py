from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from tabm_bridge.config import TaskType
from tabm_bridge.evaluation.metrics import compute_metrics


class TabMBestSubmodelBaseline:
    """TabM[B]: selects the single best submodel index strictly on validation split.

    Follows official paper/bin/model.py protocol: argmax over per-head validation score.
    """

    def __init__(self) -> None:
        self.best_index_: int | None = None
        self.val_head_scores_: np.ndarray | None = None

    def fit(
        self,
        val_per_head_preds: np.ndarray,
        y_val_raw: np.ndarray,
        task_type: TaskType,
        primary_score: str | None = None,
    ) -> TabMBestSubmodelBaseline:
        if val_per_head_preds.ndim != 3:
            raise ValueError(
                f"Expected val_per_head_preds of shape (N, k, d_out), got {val_per_head_preds.shape}."
            )
        k = val_per_head_preds.shape[1]
        scores = np.empty(k, dtype=np.float64)
        for i in range(k):
            m = compute_metrics(
                y_true=y_val_raw,
                y_pred=val_per_head_preds[:, i, :],
                task_type=task_type,
                primary_score=primary_score,
            )
            scores[i] = m["score"]
        self.val_head_scores_ = scores
        self.best_index_ = int(np.argmax(scores))
        return self

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        if self.best_index_ is None:
            raise RuntimeError("TabMBestSubmodelBaseline must be fitted before transform().")
        return np.asarray(embeddings[:, self.best_index_, :], dtype=np.float32)


class SimpleMeanBaseline:
    """Unaligned mean across all k submodels: (1/k) * sum_i h_i.

    Accumulates directly in float32 without materializing a full (N, k, d) float32 copy.
    """

    def fit(self, embeddings_train: np.ndarray) -> SimpleMeanBaseline:
        return self

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        arr = np.asarray(embeddings)
        if arr.ndim != 3:
            raise ValueError(f"Expected embeddings of shape (N, k, d), got {arr.shape}.")
        return arr.mean(axis=1, dtype=np.float32)


class ConcatBaseline:
    """Full concatenation [h_1; ...; h_k] of shape (N, k * d) as high-dim upper bound."""

    def fit(self, embeddings_train: np.ndarray) -> ConcatBaseline:
        return self

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        arr = np.asarray(embeddings, dtype=np.float32)
        if arr.ndim != 3:
            raise ValueError(f"Expected embeddings of shape (N, k, d), got {arr.shape}.")
        return arr.reshape(arr.shape[0], -1)


class PCAConcatBaseline:
    """Leak-free, memory-bounded linear compression of concatenated embeddings.

    Fits StandardScaler + randomized PCA strictly on train (subsampling when N exceeds
    `max_fit_samples` to stay well inside 12 GB Colab CPU RAM) and transforms in chunks.
    """

    def __init__(
        self,
        n_components: int = 256,
        max_fit_samples: int | None = 16384,
        batch_size: int = 2048,
        random_state: int = 42,
    ) -> None:
        self.n_components = n_components
        self.max_fit_samples = max_fit_samples
        self.batch_size = batch_size
        self.random_state = random_state
        self.scaler_: StandardScaler | None = None
        self.pca_: PCA | None = None

    def fit(self, embeddings_train: np.ndarray) -> PCAConcatBaseline:
        arr = np.asarray(embeddings_train)
        if arr.ndim != 3:
            raise ValueError(f"Expected embeddings of shape (N, k, d), got {arr.shape}.")

        n = arr.shape[0]
        if self.max_fit_samples is not None and n > self.max_fit_samples:
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(n, size=self.max_fit_samples, replace=False)
            flat = arr[idx].reshape(self.max_fit_samples, -1).astype(np.float32, copy=True)
        else:
            flat = arr.reshape(n, -1).astype(np.float32, copy=True)

        self.scaler_ = StandardScaler(copy=False)
        flat = self.scaler_.fit_transform(flat)

        max_comp = min(self.n_components, flat.shape[0], flat.shape[1])
        self.pca_ = PCA(
            n_components=max_comp,
            svd_solver="randomized",
            random_state=self.random_state,
        )
        self.pca_.fit(flat)
        return self

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        if self.scaler_ is None or self.pca_ is None:
            raise RuntimeError("PCAConcatBaseline must be fitted before transform().")
        arr = np.asarray(embeddings)
        if arr.ndim != 3:
            raise ValueError(f"Expected embeddings of shape (N, k, d), got {arr.shape}.")

        n = arr.shape[0]
        out = np.empty((n, self.pca_.n_components_), dtype=np.float32)
        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            chunk = arr[start:end].reshape(end - start, -1).astype(np.float32, copy=True)
            chunk = self.scaler_.transform(chunk)
            out[start:end] = self.pca_.transform(chunk).astype(np.float32, copy=False)
        return out