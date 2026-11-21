from __future__ import annotations

import pytest
import torch
from rtdl_num_embeddings import compute_bins

from tabm_bridge.config import BridgeConfig, ModelConfig, PLEConfig, TrainConfig
from tabm_bridge.layers import LinearBatchEnsemble, MultiHeadLinear, ScaleEnsemble
from tabm_bridge.model import PredictionOutput, TabMModel, TabMOutput


def _make_sample_data(batch_size: int = 16, n_num: int = 4, n_bin: int = 2, cat_cards: list[int] | None = None):
    torch.manual_seed(0)
    cat_cards = cat_cards if cat_cards is not None else [3, 5]
    x_num_train = torch.randn(128, n_num)
    ple_bins = compute_bins(x_num_train, n_bins=8)

    x_num = torch.randn(batch_size, n_num)
    x_bin = torch.randint(0, 2, (batch_size, n_bin)).float()
    x_cat = torch.stack([torch.randint(0, c, (batch_size,)) for c in cat_cards], dim=-1)
    return x_num, x_bin, x_cat, ple_bins, cat_cards


def test_config_defaults_and_validation() -> None:
    mlp_cfg = ModelConfig(n_num_features=4, variant="mlp", k=32)
    assert mlp_cfg.k == 1

    with pytest.raises(ValueError):
        ModelConfig(n_num_features=0, n_bin_features=0, cat_cardinalities=[])

    train_cfg = TrainConfig()
    bridge_cfg = BridgeConfig()
    assert train_cfg.share_training_batches is True
    assert train_cfg.amp_dtype == "bfloat16"
    assert bridge_cfg.stop_gradient is True


def test_model_config_make_smart_defaults() -> None:
    cfg_with_ple = ModelConfig.make(n_num_features=4, ple=PLEConfig(enabled=True))
    assert cfg_with_ple.n_blocks == 2
    assert cfg_with_ple.scaling_init == "normal"

    cfg_without_ple = ModelConfig.make(n_num_features=4, ple=PLEConfig(enabled=False))
    assert cfg_without_ple.n_blocks == 3
    assert cfg_without_ple.scaling_init == "random-signs"

    cfg_cat_only = ModelConfig.make(cat_cardinalities=[3, 5])
    assert cfg_cat_only.n_blocks == 3
    assert cfg_cat_only.scaling_init == "random-signs"


def test_scaling_init_chunks_tying() -> None:
    torch.manual_seed(42)
    chunks = [4, 4, 1, 3]
    d_in = sum(chunks)
    large_adapter = ScaleEnsemble(k=256, d_in=64, scaling_init="normal")
    assert abs(large_adapter.r.mean().item()) < 0.1
    assert 0.9 < large_adapter.r.std().item() < 1.1

    adapter_norm = ScaleEnsemble(k=8, d_in=d_in, scaling_init="normal", init_chunks=chunks)
    offset = 0
    for c in chunks:
        chunk_slice = adapter_norm.r[:, offset : offset + c]
        assert torch.allclose(chunk_slice, chunk_slice[:, :1].expand_as(chunk_slice))
        offset += c
    assert not torch.allclose(adapter_norm.r[:, 0], adapter_norm.r[:, 4])

    adapter_signs = ScaleEnsemble(k=8, d_in=d_in, scaling_init="random-signs", init_chunks=chunks)
    unique_vals = torch.unique(adapter_signs.r)
    assert set(unique_vals.tolist()).issubset({-1.0, 1.0})


def test_tabm_init_submodel_parity_in_deep_layers() -> None:
    torch.manual_seed(123)
    first = LinearBatchEnsemble(
        in_features=12,
        out_features=16,
        k=8,
        is_first_layer=True,
        scaling_init="normal",
        init_chunks=[4, 4, 4],
        tabm_init=True,
    )
    deep = LinearBatchEnsemble(
        in_features=16,
        out_features=16,
        k=8,
        is_first_layer=False,
        scaling_init="normal",
        tabm_init=True,
    )

    assert not torch.allclose(first.r, first.r[:1].expand_as(first.r))
    assert torch.allclose(first.s, torch.ones_like(first.s))
    assert first.bias is not None
    assert torch.allclose(first.bias, first.bias[:1].expand_as(first.bias))

    assert torch.allclose(deep.r, torch.ones_like(deep.r))
    assert torch.allclose(deep.s, torch.ones_like(deep.s))
    assert deep.bias is not None
    assert torch.allclose(deep.bias, deep.bias[:1].expand_as(deep.bias))


def test_linear_batch_ensemble_math_equivalence() -> None:
    torch.manual_seed(7)
    b, k, d_in, d_out = 10, 6, 12, 8
    layer = LinearBatchEnsemble(d_in, d_out, k=k, is_first_layer=True, tabm_init=False)
    x = torch.randn(b, k, d_in)

    y_vec = layer(x)
    assert y_vec.shape == (b, k, d_out)

    y_loop = []
    for i in range(k):
        xi = x[:, i, :] * layer.r[i]
        yi = (xi @ layer.weight.T) * layer.s[i] + layer.bias[i]
        y_loop.append(yi)
    y_ref = torch.stack(y_loop, dim=1)

    assert torch.allclose(y_vec, y_ref, atol=1e-5, rtol=1e-4)


def test_multihead_linear_math_equivalence_and_subset() -> None:
    torch.manual_seed(9)
    b, k, d_in, d_out = 10, 6, 16, 3
    head = MultiHeadLinear(d_in, d_out, k=k)
    x = torch.randn(b, k, d_in)

    y_vec = head(x)
    y_ref = torch.stack([x[:, i] @ head.weight[i] + head.bias[i] for i in range(k)], dim=1)
    assert torch.allclose(y_vec, y_ref, atol=1e-5, rtol=1e-4)

    idx = torch.tensor([1, 4, 5])
    y_sub = head(x[:, idx, :], submodel_indices=idx)
    assert torch.allclose(y_sub, y_vec[:, idx, :], atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("variant", ["mlp", "tabm_mini", "tabm"])
def test_variants_forward_and_embeddings(variant: str) -> None:
    x_num, x_bin, x_cat, ple_bins, cat_cards = _make_sample_data(batch_size=12)
    cfg = ModelConfig(
        n_num_features=x_num.shape[1],
        n_bin_features=x_bin.shape[1],
        cat_cardinalities=cat_cards,
        d_out=3,
        variant=variant,
        k=8,
        n_blocks=2,
        d_block=32,
        dropout=0.1,
        ple=PLEConfig(enabled=True, n_bins=8, d_embedding=4, version="B"),
    )
    model = TabMModel(cfg, ple_bins=ple_bins)
    expected_k = 1 if variant == "mlp" else 8
    expected_d_in = x_num.shape[1] * 4 + x_bin.shape[1] + sum(cat_cards)
    assert model.d_in == expected_d_in
    assert sum(model.d_features) == expected_d_in

    preds = model(x_num=x_num, x_bin=x_bin, x_cat=x_cat)
    assert isinstance(preds, torch.Tensor)
    assert preds.shape == (12, expected_k, 3)

    out = model(x_num=x_num, x_bin=x_bin, x_cat=x_cat, return_embeddings=True)
    assert isinstance(out, TabMOutput)
    assert out.preds.shape == (12, expected_k, 3)
    assert out.embeddings.shape == (12, expected_k, 32)


def test_oov_categorical_handling() -> None:
    cfg = ModelConfig.make(cat_cardinalities=[3, 4], d_out=1, variant="tabm_mini", k=4)
    model = TabMModel(cfg)

    # Row 0: valid categories; Row 1: OOV == cardinality; Row 2: OOV > cardinality & negative
    x_cat = torch.tensor([[1, 2], [3, 4], [99, -1]], dtype=torch.long)
    encoded = model.encode_inputs(x_cat=x_cat)
    assert encoded.shape == (3, 7)

    # Valid row has one 1.0 in each slice
    assert encoded[0, :3].tolist() == [0.0, 1.0, 0.0]
    assert encoded[0, 3:].tolist() == [0.0, 0.0, 1.0, 0.0]
    # OOV rows map safely to all-zero vectors
    assert torch.all(encoded[1] == 0.0)
    assert torch.all(encoded[2] == 0.0)

    preds = model(x_cat=x_cat)
    assert preds.shape == (3, 4, 1)
    assert torch.isfinite(preds).all()


def test_shared_vs_independent_batches_and_submodel_indices() -> None:
    x_num, x_bin, x_cat, ple_bins, cat_cards = _make_sample_data(batch_size=10)
    k = 8
    cfg = ModelConfig(
        n_num_features=x_num.shape[1],
        n_bin_features=x_bin.shape[1],
        cat_cardinalities=cat_cards,
        d_out=1,
        variant="tabm",
        k=k,
        n_blocks=2,
        d_block=24,
        dropout=0.0,
        ple=PLEConfig(enabled=True, n_bins=8, d_embedding=4, version="B"),
    )
    model = TabMModel(cfg, ple_bins=ple_bins).eval()

    out_2d = model(x_num=x_num, x_bin=x_bin, x_cat=x_cat)
    out_3d = model(
        x_num=x_num.unsqueeze(1).expand(-1, k, -1).contiguous(),
        x_bin=x_bin.unsqueeze(1).expand(-1, k, -1).contiguous(),
        x_cat=x_cat.unsqueeze(1).expand(-1, k, -1).contiguous(),
    )
    assert torch.allclose(out_2d, out_3d, atol=1e-5, rtol=1e-4)

    idx = torch.tensor([0, 3, 6])
    encoded = model.encode_inputs(x_num=x_num, x_bin=x_bin, x_cat=x_cat)
    preds_sub, _ = model.forward_backbone(encoded, submodel_indices=idx)
    assert torch.allclose(preds_sub, out_2d[:, idx, :], atol=1e-5, rtol=1e-4)


def test_zero_cross_sample_and_cross_head_gradient_leakage() -> None:
    torch.manual_seed(0)
    x_num = torch.randn(8, 5, requires_grad=True)
    cfg = ModelConfig(
        n_num_features=5,
        d_out=1,
        variant="tabm",
        k=6,
        n_blocks=2,
        d_block=16,
        dropout=0.0,
        ple=PLEConfig(enabled=False),
    )
    model = TabMModel(cfg)
    preds = model(x_num=x_num)

    loss = preds[2, 3, 0]
    loss.backward()

    assert x_num.grad is not None
    assert torch.allclose(x_num.grad[:2], torch.zeros_like(x_num.grad[:2]))
    assert torch.allclose(x_num.grad[3:], torch.zeros_like(x_num.grad[3:]))

    assert model.head.weight.grad is not None
    for h in range(cfg.k):
        if h != 3:
            assert torch.allclose(model.head.weight.grad[h], torch.zeros_like(model.head.weight.grad[h]))


def test_predict_inference_and_uncertainty() -> None:
    x_num, x_bin, x_cat, ple_bins, cat_cards = _make_sample_data(batch_size=8)

    reg_cfg = ModelConfig(
        n_num_features=4,
        n_bin_features=2,
        cat_cardinalities=cat_cards,
        d_out=1,
        variant="tabm_mini",
        k=6,
        n_blocks=2,
        d_block=16,
        ple=PLEConfig(enabled=True, n_bins=8, d_embedding=4, version="B"),
    )
    reg_model = TabMModel(reg_cfg, ple_bins=ple_bins)
    reg_model.train()
    reg_out = reg_model.predict(x_num=x_num, x_bin=x_bin, x_cat=x_cat, task_type="regression")
    assert isinstance(reg_out, PredictionOutput)
    assert reg_out.mean.shape == (8, 1)
    assert reg_out.per_head.shape == (8, 6, 1)
    assert reg_out.epistemic_uncertainty.shape == (8, 1)
    assert (reg_out.epistemic_uncertainty >= 0).all()
    assert reg_model.training is True

    cls_cfg = ModelConfig(
        n_num_features=4,
        n_bin_features=2,
        cat_cardinalities=cat_cards,
        d_out=4,
        variant="tabm",
        k=6,
        n_blocks=2,
        d_block=16,
        ple=PLEConfig(enabled=True, n_bins=8, d_embedding=4, version="B"),
    )
    cls_model = TabMModel(cls_cfg, ple_bins=ple_bins)
    cls_out = cls_model.predict(x_num=x_num, x_bin=x_bin, x_cat=x_cat, task_type="multiclass")
    assert cls_out.mean.shape == (8, 4)
    assert torch.allclose(cls_out.mean.sum(dim=-1), torch.ones(8), atol=1e-5)
    assert cls_out.epistemic_uncertainty.shape == (8, 1)
    assert (cls_out.epistemic_uncertainty >= 0).all()
    assert cls_out.aleatoric_proxy is not None
    assert cls_out.aleatoric_proxy.shape == (8, 1)


def test_make_parameter_groups_official_rules() -> None:
    _, _, _, ple_bins, cat_cards = _make_sample_data()
    cfg = ModelConfig.make(
        n_num_features=4,
        n_bin_features=2,
        cat_cardinalities=cat_cards,
        d_out=2,
        variant="tabm",
        k=4,
        d_block=16,
        ple=PLEConfig(enabled=True, n_bins=8, d_embedding=4, version="B"),
    )
    model = TabMModel(cfg, ple_bins=ple_bins)
    wd = 3e-4
    groups = model.make_parameter_groups(weight_decay=wd)

    assert len(groups) == 2
    decay_ids = {id(p) for g in groups if g["weight_decay"] == wd for p in g["params"]}
    no_decay_ids = {id(p) for g in groups if g["weight_decay"] == 0.0 for p in g["params"]}
    all_trainable_ids = {id(p) for p in model.parameters() if p.requires_grad}

    assert decay_ids.isdisjoint(no_decay_ids)
    assert decay_ids | no_decay_ids == all_trainable_ids

    for name, param in model.named_parameters():
        if name.endswith("bias") or name.startswith("num_embeddings.linear0."):
            assert id(param) in no_decay_ids, f"{name} should have weight_decay=0.0"
        elif (
            name.endswith("linear.weight")
            or name == "head.weight"
            or name.endswith(".r")
            or name.endswith(".s")
        ):
            assert id(param) in decay_ids, f"{name} should receive weight_decay"