"""Frozen feature protocol, strict checkpoint completeness, and source isolation."""

import numpy as np
import pytest
import torch

from fusionvc.io import write_json
from fusionvc.models import FrozenFeaturePredictor, apply_residual
from fusionvc.scgpt import FrozenScGPT, bin_expression, load_frozen_encoder


def small_encoder():
    args = {
        "embsize": 8,
        "nheads": 2,
        "d_hid": 8,
        "nlayers": 2,
        "dropout": 0.2,
        "input_emb_style": "continuous",
        "input_style": "binned",
        "pad_token": "<pad>",
    }
    vocab = {"<pad>": 0, "A": 1, "B": 2}
    return FrozenScGPT(args, vocab), args, vocab


def test_encoder_strict_load_and_permanent_freeze(tmp_path):
    model, args, vocab = small_encoder()
    write_json(tmp_path / "args.json", args)
    write_json(tmp_path / "vocab.json", vocab)
    source = {key.replace("in_proj_", "Wqkv."): value for key, value in model.state_dict().items()}
    torch.save(source, tmp_path / "best_model.pt")
    loaded, _, _, info = load_frozen_encoder(tmp_path, torch.device("cpu"))
    assert info["loaded_encoder_tensors"] == len(model.state_dict())
    loaded.train()
    assert all(not layer.training for layer in loaded.modules())
    assert all(not p.requires_grad for p in loaded.parameters())
    tokens = torch.tensor([[1, 2]])
    values = torch.tensor([[1.0, 50.0]])
    torch.testing.assert_close(loaded(tokens, values), model(tokens, values), rtol=0, atol=0)
    source.pop("transformer_encoder.layers.1.self_attn.Wqkv.weight")
    torch.save(source, tmp_path / "best_model.pt")
    with pytest.raises(RuntimeError, match="Missing key"):
        load_frozen_encoder(tmp_path, torch.device("cpu"))


def test_binning_zeros_ties_and_rng_isolation():
    rows = np.array([[0, 1, 1, 5, 10], [0, 0, 0, 0, 0]], dtype=np.float32)
    np.random.seed(72)
    expected_next = np.random.random()
    np.random.seed(72)
    binned = bin_expression(rows, 51, np.random.default_rng(2025))
    assert np.random.random() == expected_next
    np.testing.assert_array_equal(binned, bin_expression(rows, 51, np.random.default_rng(2025)))
    assert (binned[rows == 0] == 0).all()
    assert ((binned[rows > 0] >= 1) & (binned[rows > 0] <= 50)).all()
    with pytest.raises(ValueError):
        bin_expression(np.array([[np.nan]]), 51, np.random.default_rng(0))


def test_cached_features_stay_frozen_while_head_learns():
    model = FrozenFeaturePredictor({"width": 4, "dropout": 0.0}, torch.randn(8), 10, 17)
    context = model.context.clone()
    targets = torch.randn(3, 10)
    initial = model(targets)
    assert initial.shape == (3, 17)
    assert torch.count_nonzero(initial) == 0
    assert not model.context.requires_grad
    controls = torch.ones(2, 17)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    prediction = apply_residual(controls, model(targets), 17.0)
    loss = (prediction.mean(1) - torch.randn(3, 17)).square().mean()
    loss.backward()
    optimizer.step()
    torch.testing.assert_close(model.context, context, rtol=0, atol=0)
    assert torch.count_nonzero(model(targets)) > 0
