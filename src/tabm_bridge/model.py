from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import rtdl_num_embeddings
import torch
import torch.nn as nn
import torch.nn.functional as F
from rtdl_num_embeddings import PiecewiseLinearEmbeddings

from tabm_bridge.config import ClassificationAggType, ModelConfig, TaskType
from tabm_bridge.layers import LinearBatchEnsemble, MultiHeadLinear, ScaleEnsemble

_ZERO_WD_MODULE_TYPES: tuple[type[nn.Module], ...] = (
    nn.BatchNorm1d,
    nn.LayerNorm,
    nn.InstanceNorm1d,
    rtdl_num_embeddings.LinearEmbeddings,
    rtdl_num_embeddings.LinearReLUEmbeddings,
)
if hasattr(rtdl_num_embeddings, "_Periodic"):
    _ZERO_WD_MODULE_TYPES = (*_ZERO_WD_MODULE_TYPES, rtdl_num_embeddings._Periodic)


def default_zero_weight_decay_condition(
    module_name: str,
    module: nn.Module,
    parameter_name: str,
    parameter: nn.Parameter,
) -> bool:
    """Official TabM zero-weight-decay rule (paper/lib/deep.py & tabm_reference.py).

    Disables weight decay ONLY for:
        - parameters ending with 'bias'
        - normalization layers (BatchNorm1d, LayerNorm, InstanceNorm1d)
        - LinearEmbeddings (including PLE version='B' linear0), LinearReLUEmbeddings, _Periodic
    Rank-1 adapters (r, s) and PLE piecewise-linear weights DO receive weight decay.
    """
    del module_name, parameter
    return parameter_name.endswith("bias") or isinstance(module, _ZERO_WD_MODULE_TYPES)


@dataclass
class TabMOutput:
    """Container returned when `return_embeddings=True`."""

    preds: torch.Tensor       # (B, k, d_out)
    embeddings: torch.Tensor  # (B, k, d_block) penultimate representations H


@dataclass
class PredictionOutput:
    """Aggregated inference predictions with zero-overhead epistemic uncertainty."""

    mean: torch.Tensor                  # (B, d_out) mean prediction or mean probability
    per_head: torch.Tensor              # (B, k, d_out) raw predictions or probabilities
    epistemic_uncertainty: torch.Tensor # (B, d_out) std for reg/bin, or (B, 1) MI for multiclass
    aleatoric_proxy: torch.Tensor | None = None  # (B, 1) expected entropy for multiclass


class TabMModel(nn.Module):
    """Unified implementation of MLP, TabM_mini, and TabM with PLE version='B'.

    Key design properties:
    1. Strictly separates input encoding (PLE + OOV-safe OHE) from k-branch backbone so
       TabM♠ computes PLE only once per sample and Adaptive-k can reuse it.
    2. Automatically constructs feature chunk sizes (`d_features`) for
       `scaling_init_chunks` in the first Rank-1 adapter.
    3. Exposes `return_embeddings=True` to return penultimate (B, k, d_block)
       representations for Generalized Orthogonal Procrustes and TabM-Bridge.
    """

    def __init__(
        self,
        config: ModelConfig,
        ple_bins: list[torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.k = config.k
        self.variant = config.variant
        self.cat_cardinalities = list(config.cat_cardinalities)

        # 1. Numerical feature embeddings (PLE version='B' from rtdl_num_embeddings)
        self.num_embeddings: nn.Module | None = None
        num_chunks: list[int] = []
        if config.n_num_features > 0:
            if config.ple.enabled:
                if ple_bins is None:
                    raise ValueError(
                        "ple_bins must be provided (computed strictly on Train split) "
                        "when config.ple.enabled is True."
                    )
                if len(ple_bins) != config.n_num_features:
                    raise ValueError(
                        f"Expected {config.n_num_features} bin tensors, got {len(ple_bins)}."
                    )
                self.num_embeddings = PiecewiseLinearEmbeddings(
                    bins=ple_bins,
                    d_embedding=config.ple.d_embedding,
                    activation=config.ple.activation,
                    version=config.ple.version,
                )
                num_chunks = [config.ple.d_embedding] * config.n_num_features
            else:
                num_chunks = [1] * config.n_num_features

        bin_chunks = [1] * config.n_bin_features
        cat_chunks = list(self.cat_cardinalities)

        # Feature chunk layout for scaling_init_chunks
        self.d_features: list[int] = num_chunks + bin_chunks + cat_chunks
        self.d_in: int = sum(self.d_features)

        # 2. Input adapter (TabM_mini only)
        self.input_adapter: ScaleEnsemble | None = None
        if self.variant == "tabm_mini":
            self.input_adapter = ScaleEnsemble(
                k=self.k,
                d_in=self.d_in,
                scaling_init=config.scaling_init,
                init_chunks=self.d_features,
            )

        # 3. Modular backbone blocks: (Linear/BatchEnsemble -> ReLU -> Dropout) x n_blocks
        self.blocks = nn.ModuleList()
        in_dim = self.d_in
        for idx in range(config.n_blocks):
            if self.variant == "tabm":
                linear: nn.Module = LinearBatchEnsemble(
                    in_features=in_dim,
                    out_features=config.d_block,
                    k=self.k,
                    bias=True,
                    is_first_layer=(idx == 0),
                    scaling_init=config.scaling_init,
                    init_chunks=self.d_features if idx == 0 else None,
                    tabm_init=config.tabm_init,
                )
            else:
                linear = nn.Linear(in_dim, config.d_block)

            block = nn.ModuleDict(
                {
                    "linear": linear,
                    "act": nn.ReLU(),
                    "drop": nn.Dropout(config.dropout),
                }
            )
            self.blocks.append(block)
            in_dim = config.d_block

        # 4. k independent prediction heads
        self.head = MultiHeadLinear(
            in_features=config.d_block,
            out_features=config.d_out,
            k=self.k,
            bias=True,
        )

    def encode_inputs(
        self,
        x_num: torch.Tensor | None = None,
        x_bin: torch.Tensor | None = None,
        x_cat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode raw numerical, binary, and categorical inputs into a single feature tensor.

        Supports both standard 2D shared-batch inputs (B, ...) and 3D independent-batch
        inputs (B, k, ...). Out-of-vocabulary categorical indices (>= card or < 0)
        safely map to an all-zero one-hot vector, matching official OneHotEncoding0d.
        """
        parts: list[torch.Tensor] = []
        ref_dtype = torch.float32

        if self.config.n_num_features > 0 and x_num is None:
            raise ValueError(
                f"Expected x_num with {self.config.n_num_features} features, got None."
            )
        if self.config.n_bin_features > 0 and x_bin is None:
            raise ValueError(
                f"Expected x_bin with {self.config.n_bin_features} features, got None."
            )
        if len(self.cat_cardinalities) > 0 and x_cat is None:
            raise ValueError(
                f"Expected x_cat with {len(self.cat_cardinalities)} features, got None."
            )

        if x_num is not None and self.config.n_num_features > 0:
            if x_num.shape[-1] != self.config.n_num_features:
                raise ValueError(
                    f"Expected {self.config.n_num_features} numerical cols, got {x_num.shape[-1]}."
                )
            ref_dtype = x_num.dtype
            if self.num_embeddings is not None:
                if x_num.ndim == 3:
                    b, k, m = x_num.shape
                    emb = self.num_embeddings(x_num.reshape(b * k, m))
                    emb = emb.reshape(b, k, -1)
                else:
                    emb = self.num_embeddings(x_num).flatten(1)
                parts.append(emb)
            else:
                parts.append(x_num)

        if x_bin is not None and self.config.n_bin_features > 0:
            if x_bin.shape[-1] != self.config.n_bin_features:
                raise ValueError(
                    f"Expected {self.config.n_bin_features} binary cols, got {x_bin.shape[-1]}."
                )
            ref_dtype = x_bin.dtype if x_num is None else ref_dtype
            parts.append(x_bin.to(dtype=ref_dtype))

        if x_cat is not None and len(self.cat_cardinalities) > 0:
            if x_cat.shape[-1] != len(self.cat_cardinalities):
                raise ValueError(
                    f"Expected {len(self.cat_cardinalities)} categorical cols, got {x_cat.shape[-1]}."
                )
            ohe_cols = []
            for j, card in enumerate(self.cat_cardinalities):
                col = x_cat[..., j].long()
                # Map negative or >= card OOV values to sentinel index `card`, then drop it
                oov_safe = torch.where((col >= 0) & (col < card), col, torch.full_like(col, card))
                ohe = F.one_hot(oov_safe, num_classes=card + 1)[..., :-1].to(dtype=ref_dtype)
                ohe_cols.append(ohe)
            parts.append(torch.cat(ohe_cols, dim=-1))

        if not parts:
            raise ValueError("At least one of x_num, x_bin, or x_cat must be provided.")

        return torch.cat(parts, dim=-1)

    def forward_backbone(
        self,
        x_encoded: torch.Tensor,
        submodel_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the k-branch backbone on pre-encoded inputs.

        Args:
            x_encoded: Either (B, D_in) [shared batch TabM♠] or (B, k_active, D_in).
            submodel_indices: Optional 1D index tensor for executing a subset of submodels.

        Returns:
            Tuple of (preds: (B, k_active, d_out), embeddings: (B, k_active, d_block)).
        """
        k_active = self.k if submodel_indices is None else int(submodel_indices.numel())

        if x_encoded.ndim == 2:
            # Zero-copy broadcast expansion along ensemble dimension k
            h = x_encoded.unsqueeze(1).expand(-1, k_active, -1)
        elif x_encoded.ndim == 3:
            if x_encoded.shape[1] != k_active:
                raise ValueError(
                    f"Input ensemble dim {x_encoded.shape[1]} != active k {k_active}."
                )
            h = x_encoded
        else:
            raise ValueError(f"Expected 2D or 3D encoded input, got shape {tuple(x_encoded.shape)}.")

        if self.input_adapter is not None:
            h = self.input_adapter(h, submodel_indices=submodel_indices)

        for block in self.blocks:
            linear = block["linear"]
            if isinstance(linear, LinearBatchEnsemble):
                h = linear(h, submodel_indices=submodel_indices)
            else:
                h = linear(h)
            h = block["act"](h)
            h = block["drop"](h)

        embeddings = h  # Penultimate representations (B, k_active, d_block)
        preds = self.head(embeddings, submodel_indices=submodel_indices)
        return preds, embeddings

    def forward(
        self,
        x_num: torch.Tensor | None = None,
        x_bin: torch.Tensor | None = None,
        x_cat: torch.Tensor | None = None,
        return_embeddings: bool = False,
        submodel_indices: torch.Tensor | None = None,
    ) -> torch.Tensor | TabMOutput:
        """Full forward pass returning (B, k, d_out) or TabMOutput(preds, embeddings)."""
        x_encoded = self.encode_inputs(x_num=x_num, x_bin=x_bin, x_cat=x_cat)
        preds, embeddings = self.forward_backbone(
            x_encoded, submodel_indices=submodel_indices
        )
        if return_embeddings:
            return TabMOutput(preds=preds, embeddings=embeddings)
        return preds

    @torch.no_grad()
    def predict(
        self,
        x_num: torch.Tensor | None = None,
        x_bin: torch.Tensor | None = None,
        x_cat: torch.Tensor | None = None,
        task_type: TaskType = "regression",
        classification_agg: ClassificationAggType = "probs",
        submodel_indices: torch.Tensor | None = None,
    ) -> PredictionOutput:
        """Inference aggregation with zero-overhead epistemic uncertainty estimates."""
        was_training = self.training
        self.eval()
        try:
            raw_preds = self.forward(
                x_num=x_num,
                x_bin=x_bin,
                x_cat=x_cat,
                return_embeddings=False,
                submodel_indices=submodel_indices,
            )
            assert isinstance(raw_preds, torch.Tensor)
            eps = 1e-8

            if task_type == "regression":
                mean = raw_preds.mean(dim=1)
                epistemic_std = raw_preds.std(dim=1, unbiased=False)
                return PredictionOutput(
                    mean=mean,
                    per_head=raw_preds,
                    epistemic_uncertainty=epistemic_std,
                )

            if task_type not in ("binclass", "multiclass"):
                raise ValueError(f"Unsupported task_type: {task_type}")

            if classification_agg not in ("probs", "logits"):
                raise ValueError(f"Unsupported classification_agg: {classification_agg}")

            if task_type == "binclass" and raw_preds.shape[-1] == 1:
                probs = torch.sigmoid(raw_preds)
                mean = (
                    probs.mean(dim=1)
                    if classification_agg == "probs"
                    else torch.sigmoid(raw_preds.mean(dim=1))
                )
                epistemic_std = probs.std(dim=1, unbiased=False)
                return PredictionOutput(
                    mean=mean,
                    per_head=probs,
                    epistemic_uncertainty=epistemic_std,
                )

            # Multiclass (or official 2-logit binclass)
            probs = torch.softmax(raw_preds, dim=-1)
            prob_mean = probs.mean(dim=1)
            mean_probs = (
                prob_mean
                if classification_agg == "probs"
                else torch.softmax(raw_preds.mean(dim=1), dim=-1)
            )

            # BALD / Jensen-Shannon Mutual Information strictly uses arithmetic mean of probs
            total_entropy = -(prob_mean * prob_mean.clamp_min(eps).log()).sum(dim=-1, keepdim=True)
            head_entropy = -(probs * probs.clamp_min(eps).log()).sum(dim=-1)
            expected_entropy = head_entropy.mean(dim=1, keepdim=True)
            mutual_info = (total_entropy - expected_entropy).clamp_min(0.0)

            return PredictionOutput(
                mean=mean_probs,
                per_head=probs,
                epistemic_uncertainty=mutual_info,
                aleatoric_proxy=expected_entropy,
            )
        finally:
            if was_training:
                self.train()

    def make_parameter_groups(self, weight_decay: float) -> list[dict[str, Any]]:
        """Separate parameters into weight-decayed and zero-weight-decay groups.

        Follows `default_zero_weight_decay_condition` from the official TabM repo.
        """
        zero_wd_ids: set[int] = set()
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                if not param.requires_grad:
                    continue
                if default_zero_weight_decay_condition(module_name, module, param_name, param):
                    zero_wd_ids.add(id(param))

        decay_params: list[nn.Parameter] = []
        no_decay_params: list[nn.Parameter] = []
        for param in self.parameters():
            if not param.requires_grad:
                continue
            if id(param) in zero_wd_ids:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        groups: list[dict[str, Any]] = []
        if decay_params:
            groups.append({"params": decay_params, "weight_decay": weight_decay})
        if no_decay_params:
            groups.append({"params": no_decay_params, "weight_decay": 0.0})
        return groups