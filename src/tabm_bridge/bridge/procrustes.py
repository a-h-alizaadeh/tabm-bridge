from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


@dataclass
class MisalignmentDiagnostics:
    """Geometric alignment diagnostics comparing branches before vs. after Procrustes."""

    mean_pairwise_cosine_before: float
    mean_pairwise_cosine_after: float
    rel_frob_error_before: float
    rel_frob_error_after: float
    mean_ortho_departure: float
    max_ortho_departure: float
    n_iterations: int


def _pairwise_cosine_mean(embeddings: np.ndarray) -> float:
    """Average pairwise cosine similarity across the k branches for the same sample."""
    arr = np.asarray(embeddings, dtype=np.float32)
    _, k, _ = arr.shape
    if k < 2:
        return 1.0

    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    unit = arr / np.clip(norms, 1e-8, None)
    # Gram matrix across k branches per sample: (N, k, k) via BLAS batched matmul
    sim = np.matmul(unit, unit.transpose(0, 2, 1))
    triu_idx = np.triu_indices(k, k=1)
    return float(sim[:, triu_idx[0], triu_idx[1]].mean())


def _relative_frobenius_to_mean(embeddings: np.ndarray) -> float:
    """Relative Frobenius dispersion of k branches around their per-sample consensus mean."""
    arr = np.asarray(embeddings, dtype=np.float32)
    mean_vec = arr.mean(axis=1, keepdims=True)
    num = np.linalg.norm(arr - mean_vec)
    den = np.linalg.norm(arr)
    return float(num / max(den, 1e-8))


class GeneralizedOrthogonalProcrustes:
    """Analytical Generalized Orthogonal Procrustes alignment across k TabM branches.

    References:
      - Schonemann (1966), Psychometrika 31(1): closed-form orthogonal Procrustes via SVD.
      - Gower (1975), Psychometrika 40(1): Generalized Procrustes Analysis (GPA).

    Finds orthogonal matrices Q_i in O(d) strictly on the training split to minimize:
        sum_{i=1}^k || (H_i - mu_i) Q_i - M ||_F^2
    where M = (1/k) sum_{i=1}^k (H_i - mu_i) Q_i is the consensus representation.
    """

    def __init__(
        self,
        center: bool = True,
        max_iter: int = 15,
        tol: float = 1e-5,
        max_fit_samples: int | None = 8192,
        batch_size: int = 2048,
        random_state: int = 42,
        device: str = "auto",
    ) -> None:
        self.center = center
        self.max_iter = max_iter
        self.tol = tol
        self.max_fit_samples = max_fit_samples
        self.batch_size = batch_size
        self.random_state = random_state
        self.device = device

        self.means_: np.ndarray | None = None        # (k, d)
        self.rotations_: np.ndarray | None = None    # (k, d, d)
        self.n_iter_: int = 0
        self.obj_history_: list[float] = []

    def _should_use_cuda(self, k: int, d: int) -> bool:
        if torch is None or not torch.cuda.is_available():
            return False
        if self.device == "cpu":
            return False
        if self.device.startswith("cuda"):
            return True
        # Auto mode: use GPU for large TabM tensors (e.g. k=32, d=512); keep tiny mats on CPU f64
        return (k * d * d) >= 1_000_000

    def _fit_cuda(self, sub_arr: np.ndarray) -> None:
        assert torch is not None
        cuda_dev = torch.device("cuda" if self.device == "auto" else self.device)
        n_sub, k, d = sub_arr.shape

        with torch.no_grad():
            hc = torch.from_numpy(np.ascontiguousarray(sub_arr, dtype=np.float32)).to(cuda_dev)
            if self.center:
                means_t = hc.mean(dim=0)  # (k, d)
                hc.sub_(means_t.unsqueeze(0))
            else:
                means_t = torch.zeros((k, d), dtype=torch.float32, device=cuda_dev)

            total_norm_sq = float((hc * hc).sum().item())
            consensus = hc.mean(dim=1)  # (N_sub, d)
            if float(torch.linalg.norm(consensus).item()) < 1e-8:
                consensus = hc[:, 0, :].clone()

            # (k, N_sub, d) for batched GEMM; free hc to keep peak VRAM ~1 GB
            h_knd = hc.permute(1, 0, 2).contiguous()
            del hc

            rotations_t = torch.eye(d, dtype=torch.float32, device=cuda_dev).unsqueeze(0).expand(k, d, d).clone()
            prev_obj: float | None = None
            denom = float(k * n_sub)

            for it in range(1, self.max_iter + 1):
                # Batched cross-covariance: (k, d, N_sub) @ (1, N_sub, d) -> (k, d, d)
                cross = torch.matmul(h_knd.transpose(1, 2), consensus.unsqueeze(0))
                u, _, vh = torch.linalg.svd(cross, full_matrices=False)
                rotations_t = torch.bmm(u, vh)
                consensus = torch.bmm(h_knd, rotations_t).mean(dim=0)

                consensus_norm_sq = float((consensus * consensus).sum().item())
                obj = max(0.0, (total_norm_sq - float(k) * consensus_norm_sq) / denom)
                self.obj_history_.append(obj)
                self.n_iter_ = it

                if prev_obj is not None:
                    rel_improv = abs(prev_obj - obj) / max(abs(prev_obj), 1e-12)
                    if rel_improv < self.tol:
                        break
                prev_obj = obj

            self.means_ = means_t.cpu().numpy().astype(np.float32, copy=False)
            self.rotations_ = rotations_t.cpu().numpy().astype(np.float32, copy=False)

    def fit(self, embeddings_train: np.ndarray) -> GeneralizedOrthogonalProcrustes:
        arr = np.asarray(embeddings_train)
        if arr.ndim != 3:
            raise ValueError(f"Expected embeddings of shape (N, k, d), got {arr.shape}.")

        n, k, d = arr.shape
        if self.max_fit_samples is not None and n > self.max_fit_samples:
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(n, size=self.max_fit_samples, replace=False)
            sub_view = arr[idx]
        else:
            sub_view = arr

        self.obj_history_ = []
        if k == 1:
            means = (
                sub_view.astype(np.float64, copy=False).mean(axis=0)
                if self.center
                else np.zeros((1, d), dtype=np.float64)
            )
            self.means_ = means.astype(np.float32)
            self.rotations_ = np.eye(d, dtype=np.float32)[None, :, :]
            self.n_iter_ = 0
            return self

        if self._should_use_cuda(k, d):
            self._fit_cuda(sub_view)
            return self

        # Exact float64 CPU path (used when CUDA is unavailable or for small matrices)
        centered = sub_view.astype(np.float64, copy=True)

        # In-place per-branch centering strictly on train (halves peak RAM)
        if self.center:
            means = centered.mean(axis=0)  # (k, d)
            centered -= means[None, :, :]
        else:
            means = np.zeros((k, d), dtype=np.float64)

        rotations = np.broadcast_to(np.eye(d, dtype=np.float64), (k, d, d)).copy()

        # Exact orthogonal invariance identity:
        #   sum_i ||H_i Q_i - M||_F^2 = sum_i ||H_i||_F^2 - k * ||M||_F^2
        total_norm_sq = float(np.sum(centered * centered))

        # Symmetric centroid initialization with fallback if branches cancel exactly
        consensus = centered.mean(axis=1)  # (N_sub, d)
        if float(np.linalg.norm(consensus)) < 1e-12:
            consensus = centered[:, 0, :].copy()

        prev_obj: float | None = None
        denom = float(k * centered.shape[0])

        for it in range(1, self.max_iter + 1):
            aligned_sum = np.zeros_like(consensus)
            for i in range(k):
                # Cross-covariance C_i = H_i^T M -> (d, d); optimal Q_i = U V^T in O(d)
                h_i = centered[:, i, :]
                cross = h_i.T @ consensus
                u, _, vt = np.linalg.svd(cross, full_matrices=False)
                q_i = u @ vt
                rotations[i] = q_i
                aligned_sum += h_i @ q_i

            consensus = aligned_sum / float(k)

            # Mean squared Frobenius distance to consensus without repeating k matmuls
            consensus_norm_sq = float(np.sum(consensus * consensus))
            obj = max(0.0, (total_norm_sq - float(k) * consensus_norm_sq) / denom)

            self.obj_history_.append(obj)
            self.n_iter_ = it

            if prev_obj is not None:
                rel_improv = abs(prev_obj - obj) / max(abs(prev_obj), 1e-12)
                if rel_improv < self.tol:
                    break
            prev_obj = obj

        self.means_ = means.astype(np.float32)
        self.rotations_ = rotations.astype(np.float32)
        return self

    def transform_branches(self, embeddings: np.ndarray) -> np.ndarray:
        """Align all k branches into the consensus frame -> shape (N, k, d) via BLAS matmul."""
        if self.means_ is None or self.rotations_ is None:
            raise RuntimeError("GeneralizedOrthogonalProcrustes must be fitted before transform().")
        arr = np.asarray(embeddings, dtype=np.float32)
        shifted = arr - self.means_[None, :, :] if self.center else arr
        # (N, k, d) -> (k, N, d) @ (k, d, d) -> (k, N, d) -> (N, k, d)
        return np.matmul(shifted.transpose(1, 0, 2), self.rotations_).transpose(1, 0, 2).astype(
            np.float32, copy=False
        )

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Return single-vector consensus embedding z_proc(x) of shape (N, d) in chunks."""
        if self.means_ is None or self.rotations_ is None:
            raise RuntimeError("GeneralizedOrthogonalProcrustes must be fitted before transform().")
        arr = np.asarray(embeddings)
        if arr.ndim != 3:
            raise ValueError(f"Expected embeddings of shape (N, k, d), got {arr.shape}.")

        n, _, d = arr.shape
        out = np.empty((n, d), dtype=np.float32)
        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            chunk_aligned = self.transform_branches(arr[start:end])
            out[start:end] = chunk_aligned.mean(axis=1)
        return out


def compute_misalignment_diagnostics(
    procrustes: GeneralizedOrthogonalProcrustes,
    embeddings_eval: np.ndarray,
    max_samples: int = 2048,
    seed: int = 0,
) -> MisalignmentDiagnostics:
    """Quantify geometric misalignment across TabM submodels before vs. after Procrustes.

    Subsamples `embeddings_eval` once before transforming so before/after statistics are
    computed on identical rows with minimal CPU memory footprint.
    """
    if procrustes.rotations_ is None or procrustes.means_ is None:
        raise RuntimeError("Procrustes model must be fitted before running diagnostics.")

    raw = np.asarray(embeddings_eval)
    if raw.ndim != 3:
        raise ValueError(f"Expected embeddings of shape (N, k, d), got {raw.shape}.")

    if raw.shape[0] > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(raw.shape[0], size=max_samples, replace=False)
        raw_eval = raw[idx].astype(np.float32, copy=False)
    else:
        raw_eval = raw.astype(np.float32, copy=False)

    centered_raw = raw_eval - procrustes.means_[None, :, :] if procrustes.center else raw_eval
    aligned = procrustes.transform_branches(raw_eval)

    d = procrustes.rotations_.shape[-1]
    eye = np.eye(d, dtype=np.float32)[None, :, :]
    # Bounded in [0, 2] since ||Q - I||_F = sqrt(2d - 2 tr(Q)) <= 2 sqrt(d)
    departures = np.linalg.norm(procrustes.rotations_ - eye, axis=(1, 2)) / np.sqrt(float(d))

    return MisalignmentDiagnostics(
        mean_pairwise_cosine_before=_pairwise_cosine_mean(centered_raw),
        mean_pairwise_cosine_after=_pairwise_cosine_mean(aligned),
        rel_frob_error_before=_relative_frobenius_to_mean(centered_raw),
        rel_frob_error_after=_relative_frobenius_to_mean(aligned),
        mean_ortho_departure=float(departures.mean()),
        max_ortho_departure=float(departures.max()),
        n_iterations=procrustes.n_iter_,
    )