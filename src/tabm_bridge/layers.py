from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from tabm_bridge.config import ScalingInitType


def _init_scaling_adapter(
    param: nn.Parameter,
    scaling_init: ScalingInitType,
    init_chunks: Sequence[int] | None = None,
) -> None:
    """Initialize a (k, d) Rank-1 scaling adapter with optional feature-chunk tying.

    When `init_chunks` is provided (e.g. d_features where sum(init_chunks) == d),
    a single random scalar is sampled per feature chunk per submodel and repeated
    across that feature's representation dimensions.
    """
    k, d = param.shape
    with torch.no_grad():
        if init_chunks is None:
            if scaling_init == "normal":
                param.normal_(mean=0.0, std=1.0)
            elif scaling_init == "random-signs":
                param.copy_(torch.randint(0, 2, (k, d), dtype=param.dtype) * 2.0 - 1.0)
            else:
                raise ValueError(f"Unknown scaling_init: {scaling_init}")
            return

        if sum(init_chunks) != d:
            raise ValueError(
                f"sum(init_chunks)={sum(init_chunks)} must equal adapter dim d={d}."
            )

        offset = 0
        for chunk_size in init_chunks:
            if chunk_size <= 0:
                raise ValueError(f"All chunk sizes must be > 0, got {chunk_size}.")
            if scaling_init == "normal":
                scalars = torch.empty((k, 1), dtype=param.dtype, device=param.device).normal_(
                    mean=0.0, std=1.0
                )
            elif scaling_init == "random-signs":
                scalars = (
                    torch.randint(0, 2, (k, 1), dtype=param.dtype, device=param.device)
                    * 2.0
                    - 1.0
                )
            else:
                raise ValueError(f"Unknown scaling_init: {scaling_init}")

            param[:, offset : offset + chunk_size] = scalars
            offset += chunk_size


class ScaleEnsemble(nn.Module):
    """Input Rank-1 multiplicative adapter R of shape (k, d_in) for TabM_mini."""

    def __init__(
        self,
        k: int,
        d_in: int,
        scaling_init: ScalingInitType = "normal",
        init_chunks: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        self.k = k
        self.d_in = d_in
        self.scaling_init = scaling_init
        self.init_chunks = tuple(init_chunks) if init_chunks is not None else None
        self.r = nn.Parameter(torch.empty(k, d_in))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _init_scaling_adapter(self.r, self.scaling_init, self.init_chunks)

    def forward(
        self,
        x: torch.Tensor,
        submodel_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply R scaling to x of shape (B, k_active, d_in)."""
        r = self.r if submodel_indices is None else self.r.index_select(0, submodel_indices)
        return x * r


class LinearBatchEnsemble(nn.Module):
    """Vectorized Rank-1 BatchEnsemble linear layer for full TabM.

    Computes for each submodel i in {0, ..., k-1}:
        y_i = ((x_i * r_i) @ W^T) * s_i + b_i
    Shapes: (B, k, d_in) -> (B, k, d_out) with zero Python loops.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        k: int,
        bias: bool = True,
        is_first_layer: bool = False,
        scaling_init: ScalingInitType = "normal",
        init_chunks: Sequence[int] | None = None,
        tabm_init: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k = k
        self.is_first_layer = is_first_layer
        self.scaling_init = scaling_init
        self.init_chunks = tuple(init_chunks) if init_chunks is not None else None
        self.tabm_init = tabm_init

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.r = nn.Parameter(torch.empty(k, in_features))
        self.s = nn.Parameter(torch.empty(k, out_features))
        self.bias = nn.Parameter(torch.empty(k, out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.tabm_init:
            # Per official README: under tabm_init=True, submodels differ ONLY
            # in the first ensembled layer's R adapter and are identical in all
            # other adapters and deeper layers at initialization.
            if self.is_first_layer:
                _init_scaling_adapter(self.r, self.scaling_init, self.init_chunks)
            else:
                nn.init.ones_(self.r)
            nn.init.ones_(self.s)
        else:
            _init_scaling_adapter(self.r, self.scaling_init, self.init_chunks if self.is_first_layer else None)
            _init_scaling_adapter(self.s, self.scaling_init, None)

        if self.bias is not None:
            fan_in = self.in_features
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            with torch.no_grad():
                if self.tabm_init:
                    # Identical bias across all k submodels at initialization
                    shared_b = torch.empty(1, self.out_features, dtype=self.bias.dtype).uniform_(
                        -bound, bound
                    )
                    self.bias.copy_(shared_b.expand(self.k, self.out_features))
                else:
                    self.bias.uniform_(-bound, bound)

    def forward(
        self,
        x: torch.Tensor,
        submodel_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if submodel_indices is None:
            r, s, b = self.r, self.s, self.bias
        else:
            r = self.r.index_select(0, submodel_indices)
            s = self.s.index_select(0, submodel_indices)
            b = self.bias.index_select(0, submodel_indices) if self.bias is not None else None

        out = x * r
        out = F.linear(out, self.weight, bias=None)
        out = out * s
        if b is not None:
            out = out + b
        return out


class MultiHeadLinear(nn.Module):
    """k independent non-shared linear prediction heads.

    Uses ONNX/TorchScript/AMP-friendly batched matmul instead of einsum.
    Shapes: (B, k, d_in) -> (B, k, d_out).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        k: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k = k

        self.weight = nn.Parameter(torch.empty(k, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(k, out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.in_features) if self.in_features > 0 else 0.0
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)
            if self.bias is not None:
                self.bias.uniform_(-bound, bound)

    def forward(
        self,
        x: torch.Tensor,
        submodel_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if submodel_indices is None:
            w, b = self.weight, self.bias
        else:
            w = self.weight.index_select(0, submodel_indices)
            b = self.bias.index_select(0, submodel_indices) if self.bias is not None else None

        # (B, k, d_in) -> (k, B, d_in) @ (k, d_in, d_out) -> (k, B, d_out) -> (B, k, d_out)
        out = torch.matmul(x.transpose(0, 1), w).transpose(0, 1)
        if b is not None:
            out = out + b
        return out
