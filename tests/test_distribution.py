"""Distribution matching must preserve scientific, unpaired-set invariants."""

import numpy as np
import pytest
import torch

from fusionvc.distribution import TrainingCellSampler, energy_mmd, prepare_training_cell_cache
from fusionvc.io import CountsReader
from fusionvc.preprocess import prepare
from test_feature_isolation import write_fixture


def reference_energy(p, o):
    cross = np.linalg.norm(p[:, None] - o[None], axis=-1).mean()
    within_p = np.linalg.norm(p[:, None] - p[None], axis=-1).mean()
    within_o = np.linalg.norm(o[:, None] - o[None], axis=-1).mean()
    return (2 * cross - within_p - within_o) / np.sqrt(p.shape[-1])


def test_energy_matches_independent_distance_formula_with_unequal_set_sizes():
    p = torch.tensor([[0.0, 2.0], [1.0, 4.0], [2.0, 1.0]], dtype=torch.float64)
    o = torch.tensor([[3.0, 0.0], [4.0, 2.0]], dtype=torch.float64)
    assert energy_mmd(p, o).item() == pytest.approx(reference_energy(p.numpy(), o.numpy()))


def test_matching_is_unpaired_and_distinguishes_equal_means():
    observation = torch.tensor([[0.0], [10.0]], dtype=torch.float64)
    same_mean = torch.tensor([[5.0], [5.0]], dtype=torch.float64)
    assert energy_mmd(same_mean, observation).item() == pytest.approx(5.0)
    assert energy_mmd(observation.flip(0), observation).item() == pytest.approx(0.0, abs=1e-8)
    assert energy_mmd(same_mean.flip(0), observation.flip(0)) == energy_mmd(same_mean, observation)


def test_conditions_cannot_be_pooled_into_one_distribution():
    p = torch.tensor([[[0.0], [0.0]], [[10.0], [10.0]]], dtype=torch.float64)
    o = p.flip(0)
    assert energy_mmd(p, o).item() == pytest.approx(20.0)
    assert energy_mmd(p.flatten(0, 1), o.flatten(0, 1)).item() == pytest.approx(0.0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_energy_has_finite_gradients_including_duplicate_cells(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    p = torch.tensor([[[1.0, 2.0], [1.0, 2.0], [3.0, 4.0]]], device=device, requires_grad=True)
    o = torch.tensor([[[0.0, 3.0], [2.0, 5.0], [4.0, 1.0]]], device=device)
    loss = energy_mmd(p, o)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(p.grad).all()
    assert p.grad.abs().sum() > 0


def test_selected_rows_preserve_order_duplicates_and_counts(tmp_path):
    write_fixture(tmp_path, 0)
    reader = CountsReader(tmp_path / "data/raw/vcc2025/train/adata_Training.h5ad")
    try:
        selected = np.array([30, 2, 30, 49, 1])
        expected = reader.block(0, reader.shape[0]).toarray()[selected]
        np.testing.assert_array_equal(reader.rows(selected).toarray(), expected)
        with pytest.raises(ValueError):
            reader.rows(np.array([-1]))
    finally:
        reader.close()


def test_training_sampler_rejects_held_out_targets_and_preserves_scale(tmp_path):
    write_fixture(tmp_path, 100)
    genes = [f"G{i}" for i in range(40)]
    with pytest.raises(ValueError, match="training perturbations"):
        TrainingCellSampler(tmp_path, genes, ["G2"], 10000.0)
    sampler = TrainingCellSampler(tmp_path, genes, ["G0", "G1"], 10000.0)
    try:
        sampled = sampler.sample(["G1", "G0"], 16, np.random.default_rng(42))
        assert sampled.shape == (2, 16, 40)
        np.testing.assert_allclose(np.expm1(sampled).sum(-1), 10000.0, rtol=1e-5)
        repeated = sampler.sample(["G1", "G0"], 16, np.random.default_rng(42))
        np.testing.assert_array_equal(sampled, repeated)
    finally:
        sampler.close()


def test_cell_cache_exactly_preserves_raw_training_samples(tmp_path):
    write_fixture(tmp_path, 100)
    prepare(tmp_path, tokens=8, components=3, feature_cells=16)
    from fusionvc.io import read_json

    scale = read_json(tmp_path / "data/processed/baselines-v1/manifest.json")["target_sum"]
    genes = [f"G{i}" for i in range(40)]
    sampler = TrainingCellSampler(tmp_path, genes, ["G0", "G1"], scale)
    original = sampler.sample(["G1", "G0"], 16, np.random.default_rng(42))
    sampler.close()
    prepare_training_cell_cache(tmp_path, block_size=9)
    prepare_training_cell_cache(tmp_path, block_size=9)
    sampler = TrainingCellSampler(tmp_path, genes, ["G0", "G1"], scale)
    try:
        repeated = sampler.sample(["G1", "G0"], 16, np.random.default_rng(42))
        np.testing.assert_array_equal(original, repeated)
    finally:
        sampler.close()
    with pytest.raises(ValueError, match="scale mismatch"):
        TrainingCellSampler(tmp_path, genes, ["G0", "G1"], scale + 1)
