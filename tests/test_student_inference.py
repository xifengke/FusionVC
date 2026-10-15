"""A student package predicts using only its own local files."""

import numpy as np
import pytest
import torch

from fusionvc.io import file_sha256, write_json
from fusionvc.models import ResidualPredictor, apply_residual
from fusionvc.student_inference import (
    export_student_bundle,
    load_student_bundle,
    predict_student_bundle,
)


def synthetic_student(root):
    torch.manual_seed(17)
    base = root / "data/processed/baselines-v1"
    base.mkdir(parents=True)
    features = np.random.default_rng(17).normal(size=(7, 3)).astype(np.float32)
    selected = np.array([1, 5])
    genes = list("ABCDEFG")
    controls = np.log1p(np.random.default_rng(18).uniform(1, 8, size=(5, 7)).astype(np.float32))
    write_json(base / "genes.json", genes)
    np.save(base / "gene_descriptors.npy", features)
    np.save(base / "input_gene_indices.npy", selected)
    np.save(base / "controls.npy", controls)
    config = {"model": "mlp", "protocol": "distillation-v1", "width": 4, "dropout": 0.0}
    model = ResidualPredictor(
        config, torch.from_numpy(features[selected]), torch.tensor([0.2, 0.3]), 7
    ).eval()
    torch.nn.init.normal_(model.decoder.weight, std=0.1)
    run = root / "outputs/08_distillation/seed_42"
    run.mkdir(parents=True)
    signature = {
        f"data/processed/baselines-v1/{name}": file_sha256(base / name)
        for name in ["genes.json", "gene_descriptors.npy"]
    }
    torch.save(
        {
            "config": config,
            "state_dict": model.state_dict(),
            "preprocessing": {"target_sum": 20.0},
            "distillation_cache": {"signature": {"source_sha256": signature}},
        },
        run / "best.pt",
    )
    return run, model, features


def test_exported_student_matches_direct_model_without_teacher_or_original_data(tmp_path):
    run, original, features = synthetic_student(tmp_path)
    package = tmp_path / "standalone"
    manifest = export_student_bundle(tmp_path, run, package, control_cells=3)
    assert manifest["genes"] == 7 and len(manifest["files"]) == 5
    # Remove the original query table after exporting; package-only loading works.
    (tmp_path / "data/processed/baselines-v1/gene_descriptors.npy").unlink()
    output, genes, metadata = predict_student_bundle(package, "C", cells=2, seed=41)
    bank = np.load(package / "controls.npy")
    picked = np.random.default_rng(41).choice(3, 2, False)
    with torch.no_grad():
        expected = apply_residual(
            torch.from_numpy(bank[picked]), original(torch.from_numpy(features[[2]])), 20.0
        )[0]
    np.testing.assert_array_equal(output, expected.numpy())
    assert genes == list("ABCDEFG") and metadata["target"] == "C"
    assert not (tmp_path / "models").exists()
    assert not (tmp_path / "data/processed/distillation-v1").exists()


def test_student_bundle_detects_tampering_and_invalid_targets(tmp_path):
    run, _, _ = synthetic_student(tmp_path)
    package = tmp_path / "standalone"
    export_student_bundle(tmp_path, run, package, control_cells=3)
    with pytest.raises(ValueError, match="Unknown official gene"):
        predict_student_bundle(package, "unknown")
    with pytest.raises(ValueError, match="positive"):
        predict_student_bundle(package, "A", cells=0)
    np.save(package / "target_features.npy", np.zeros((7, 3), np.float32))
    with pytest.raises(ValueError, match="package changed"):
        load_student_bundle(package)
