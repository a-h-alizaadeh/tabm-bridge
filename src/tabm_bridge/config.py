from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

VariantType = Literal["mlp", "tabm_mini", "tabm"]
TaskType = Literal["regression", "binclass", "multiclass"]
ScalingInitType = Literal["normal", "random-signs"]
ClassificationAggType = Literal["probs", "logits"]
BridgeAlignMode = Literal["low_rank", "full"]


@dataclass
class PLEConfig:
    """Configuration for Piecewise Linear Embeddings (version='B' by default)."""

    enabled: bool = True
    n_bins: int = 48
    d_embedding: int = 16
    activation: bool = False
    version: Literal["A", "B"] = "B"


@dataclass
class ModelConfig:
    """Architecture specification for MLP, TabM_mini, and TabM."""

    n_num_features: int = 0
    n_bin_features: int = 0
    cat_cardinalities: list[int] = field(default_factory=list)
    d_out: int = 1
    variant: VariantType = "tabm_mini"
    k: int = 32
    n_blocks: int = 3
    d_block: int = 512
    dropout: float = 0.1
    scaling_init: ScalingInitType = "normal"
    tabm_init: bool = True
    ple: PLEConfig = field(default_factory=PLEConfig)

    def __post_init__(self) -> None:
        if self.variant == "mlp":
            self.k = 1
        if self.k < 1:
            raise ValueError(f"k must be >= 1, got {self.k}.")
        if self.n_blocks < 1:
            raise ValueError(f"n_blocks must be >= 1, got {self.n_blocks}.")
        if (
            self.n_num_features == 0
            and self.n_bin_features == 0
            and len(self.cat_cardinalities) == 0
        ):
            raise ValueError("At least one input feature (num, bin, or cat) is required.")

    @classmethod
    def make(
        cls,
        *,
        n_num_features: int = 0,
        n_bin_features: int = 0,
        cat_cardinalities: list[int] | None = None,
        d_out: int = 1,
        variant: VariantType = "tabm_mini",
        k: int = 32,
        n_blocks: int | None = None,
        d_block: int = 512,
        dropout: float = 0.1,
        scaling_init: ScalingInitType | None = None,
        tabm_init: bool = True,
        ple: PLEConfig | None = None,
    ) -> ModelConfig:
        """Factory mirroring official TabM.make smart defaults (tabm.py)."""
        ple_cfg = ple if ple is not None else PLEConfig(enabled=(n_num_features > 0))
        has_num_embeddings = (n_num_features > 0) and ple_cfg.enabled

        resolved_blocks = n_blocks if n_blocks is not None else (2 if has_num_embeddings else 3)
        resolved_init: ScalingInitType = (
            scaling_init
            if scaling_init is not None
            else ("normal" if has_num_embeddings else "random-signs")
        )

        return cls(
            n_num_features=n_num_features,
            n_bin_features=n_bin_features,
            cat_cardinalities=list(cat_cardinalities) if cat_cardinalities is not None else [],
            d_out=d_out,
            variant=variant,
            k=k,
            n_blocks=resolved_blocks,
            d_block=d_block,
            dropout=dropout,
            scaling_init=resolved_init,
            tabm_init=tabm_init,
            ple=ple_cfg,
        )


@dataclass
class TrainConfig:
    """Training protocol aligned with official TabM defaults (paper/bin/model.py)."""

    task_type: TaskType = "regression"
    batch_size: int = 256
    share_training_batches: bool = True  # TabM♠ default per official README
    lr: float = 2e-3                     # Official AdamW default
    weight_decay: float = 3e-4           # Official AdamW default
    grad_clip_norm: float = 1.0
    patience: int = 16
    max_epochs: int = 300
    classification_agg: ClassificationAggType = "probs"
    use_amp: bool = True                 # Activated in trainer ONLY when CUDA supports bfloat16
    amp_dtype: Literal["bfloat16"] = "bfloat16"
    compile_model: bool = False
    seed: int = 42
    device: str = "cuda"


@dataclass
class BridgeConfig:
    """Configuration for Scenario 1 (G1) TabM-Bridge (v1 & v2 Procrustes-Anchored Residual)."""

    d_z: int = 256
    d_hidden: int = 256
    align_mode: BridgeAlignMode = "low_rank"
    align_rank: int = 16
    center_gram: bool = True             # Batch-centered Gram: invariant to O(d) rotations + shifts
    dropout: float = 0.1
    alpha_gram: float = 25.0
    alpha_pred: float = 1.0
    alpha_uq: float = 0.15
    stop_gradient: bool = True           # Set False ONLY in Learner Collusion ablation
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 60
    patience: int = 12
    batch_size: int = 256
    # v4.0 extensions (backward-compatible defaults preserve exact v3.1 behavior)
    alpha_recon: float = 0.0             # Trust-region anchor reconstruction weight (Bridge-v2)
    init_gate: float = 0.05              # Non-zero learnable residual gate init (Bridge-v2)
    gram_warmup_epochs: int = 0          # Cosine warmup epochs for alpha_gram (>0 in Bridge-v2)
    use_cosine_lr: bool = False          # CosineAnnealingLR schedule
    mask_gram_diag: bool = False         # Exclude trivial 1.0 diagonal entries in Gram MSE
    center_z_gram: bool = False          # Symmetric batch-centering of z inside G_z

    def __post_init__(self) -> None:
        if self.d_z < 1 or self.d_hidden < 1:
            raise ValueError(f"d_z and d_hidden must be >= 1, got {self.d_z}, {self.d_hidden}.")
        if self.align_mode == "low_rank" and self.align_rank < 1:
            raise ValueError(
                f"align_rank must be >= 1 when align_mode='low_rank', got {self.align_rank}."
            )
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}.")
        if (
            self.alpha_recon < 0.0
            or self.alpha_gram < 0.0
            or self.alpha_pred < 0.0
            or self.alpha_uq < 0.0
        ):
            raise ValueError("All loss weights must be non-negative.")
        if self.gram_warmup_epochs < 0:
            raise ValueError(f"gram_warmup_epochs must be >= 0, got {self.gram_warmup_epochs}.")
        if self.init_gate == 0.0:
            raise ValueError(
                "init_gate must be non-zero to avoid a zero-gradient saddle point with zero-init refine_fc2."
            )