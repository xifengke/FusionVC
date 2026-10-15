"""Identity-aware aggregation, padding isolation, and supervised-only fusion."""

import numpy as np
import pytest
import torch

from fusionvc.fusion import FusionPredictor
from fusionvc.fusion_features import accumulate_tokens
from fusionvc.distribution import energy_mmd
from fusionvc.models import apply_residual


def assets():
    return {
        "scgpt_context": torch.randn(8),
        "geneformer_context": torch.randn(6),
        "scgpt_tokens": torch.randn(3, 8),
        "geneformer_tokens": torch.randn(4, 6),
        "scgpt_gene_indices": [4, 1, 8],
        "geneformer_gene_indices": [8, 2, 1, -1],
        "scgpt_valid": [True, True, True],
        "geneformer_valid": [True, True, True, False],
    }


def test_native_rank_outputs_aggregate_by_gene_id_not_position():
    sums, counts = np.zeros((4, 2)), np.zeros(4, dtype=int)
    hidden = np.array(
        [[[1.0, 2.0], [3.0, 4.0], [999.0, 999.0]], [[5.0, 6.0], [7.0, 8.0], [999.0, 999.0]]]
    )
    ids = np.array([[2, 1, -1], [1, 2, -1]])
    accumulate_tokens(sums, counts, hidden, ids, ids >= 0)
    np.testing.assert_array_equal(counts, [0, 2, 2, 0])
    np.testing.assert_array_equal(sums[1], [8, 10])
    np.testing.assert_array_equal(sums[2], [8, 10])


@pytest.mark.parametrize("kind", ["concat", "gated", "cross_attention"])
def test_fusion_learns_without_updating_frozen_assets(kind):
    model = FusionPredictor(
        {"model": kind, "width": 4, "heads": 2, "dropout": 0.0}, assets(), [7, 5], 11
    )
    targets = torch.randn(3, 12)
    before = {k: v.clone() for k, v in model.named_buffers()}
    assert torch.count_nonzero(model(targets)) == 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    truth = torch.randn(3, 11)
    for _ in range(2):
        optimizer.zero_grad()
        (model(targets) - truth).square().mean().backward()
        optimizer.step()
    assert torch.count_nonzero(model(targets)) > 0
    for key, value in model.named_buffers():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        assert not value.requires_grad
    for projection in model.target_projections:
        assert projection.weight.grad.abs().sum() > 0
    if kind == "gated":
        weights = model.gate_weights(targets)
        assert weights.shape == (3, 1) and ((weights > 0) & (weights < 1)).all()
        assert model.gate.weight.grad.abs().sum() > 0
    if kind == "cross_attention":
        assert all(layer.in_proj_weight.grad.abs().sum() > 0 for layer in model.cross)


def test_cross_attention_padding_and_token_permutations_preserve_prediction():
    model = FusionPredictor(
        {"model": "cross_attention", "width": 4, "heads": 2, "dropout": 0.0}, assets(), [7, 5], 11
    ).eval()
    torch.nn.init.normal_(model.decoder.weight)
    targets = torch.randn(2, 12)
    expected = model(targets).detach()
    model.geneformer_tokens[-1] = 1e20
    torch.testing.assert_close(model(targets), expected, rtol=0, atol=0)
    # Model token outputs carry identity already; no cross-model positional pairing.
    for name in ["scgpt", "geneformer"]:
        permutation = torch.arange(len(getattr(model, f"{name}_tokens")) - 1, -1, -1)
        for suffix in ["tokens", "gene_indices", "valid"]:
            value = getattr(model, f"{name}_{suffix}")
            value.copy_(value[permutation])
    torch.testing.assert_close(model(targets), expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("kind", ["concat", "gated", "cross_attention"])
@pytest.mark.parametrize("variant", ["energy", "hybrid"])
def test_distribution_losses_reach_fusion_head_without_updating_frozen_inputs(kind, variant):
    torch.manual_seed(2026)
    model = FusionPredictor(
        {"model": kind, "width": 4, "heads": 2, "dropout": 0.0}, assets(), [7, 5], 11
    )
    frozen = {key: value.clone() for key, value in model.named_buffers()}
    targets = torch.randn(3, 12)
    controls = torch.rand(4, 11) + 0.5
    observed = torch.rand(3, 5, 11) + 1.0
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    for _ in range(3):
        optimizer.zero_grad()
        prediction = apply_residual(controls, model(targets), 30.0)
        energy = energy_mmd(prediction, observed)
        mean = (prediction.mean(1) - observed.mean(1)).square().mean()
        objective = energy if variant == "energy" else mean + 0.1 * energy
        assert torch.isfinite(objective)
        objective.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step()
    assert all(projection.weight.grad.abs().sum() > 0 for projection in model.target_projections)
    if kind == "gated":
        assert model.gate.weight.grad.abs().sum() > 0
    if kind == "cross_attention":
        assert all(layer.in_proj_weight.grad.abs().sum() > 0 for layer in model.cross)
    for name, value in model.named_buffers():
        torch.testing.assert_close(value, frozen[name], rtol=0, atol=0)
        assert not value.requires_grad
