"""Changing held-out labels must not change fitted model inputs."""

import anndata as ad
import numpy as np
import pandas as pd

from fusionvc.preprocess import prepare


def write_fixture(root, validation_shift):
    rng = np.random.default_rng(2025)
    genes = [f"G{i}" for i in range(40)]
    for split, title, targets in [
        ("train", "Training", ["G0", "G1"]),
        ("validation", "Validation", ["G2", "G3"]),
    ]:
        labels = ["non-targeting"] * 25 + [targets[0]] * 12 + [targets[1]] * 13
        counts = rng.poisson(3, (len(labels), len(genes))).astype(np.float32)
        if split == "validation":
            counts[25:, 0] += validation_shift
        from scipy.sparse import csr_matrix

        data = ad.AnnData(
            csr_matrix(counts),
            obs=pd.DataFrame(
                {"target_gene": pd.Categorical(labels)},
                index=[f"{split}_{i}" for i in range(len(labels))],
            ),
            var=pd.DataFrame(index=genes),
        )
        path = root / f"data/raw/vcc2025/{split}/adata_{title}.h5ad"
        path.parent.mkdir(parents=True, exist_ok=True)
        data.write_h5ad(path)


def test_validation_labels_cannot_fit_inputs(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    write_fixture(a, 0)
    write_fixture(b, 100)
    for root in [a, b]:
        prepare(root, tokens=8, components=3, feature_cells=16)
    for name in ["gene_descriptors.npy", "input_gene_indices.npy", "controls.npy"]:
        np.testing.assert_array_equal(
            np.load(a / "data/processed/baselines-v1" / name),
            np.load(b / "data/processed/baselines-v1" / name),
        )
