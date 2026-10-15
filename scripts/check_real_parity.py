"""Check the DE adapter against native pdex using all official validation controls."""
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from pdex import parallel_differential_expression

from fusionvc.fast_de import differential_expression
from fusionvc.io import CountsReader, write_json


def main():
    root = Path(__file__).resolve().parents[1]
    reader = CountsReader(root / "data/raw/vcc2025/validation/adata_Validation.h5ad")
    labels = np.array(reader.categories)[reader.codes]
    genes = reader.genes
    reader.close()
    matrix = np.load(root / "outputs/evaluation-cache/validation_log.npy", mmap_mode="r")
    targets = [p for p in pd.unique(labels) if p != "non-targeting"][:3]
    rows = np.flatnonzero(np.isin(labels, targets + ["non-targeting"]))
    indices = np.array([0, 1, 10, 30, 100, 255, 256, 512, 1024, 2048, 4096,
                        8192, 12000, 15000, 17000, 18079])
    values = np.array(matrix[rows[:, None], indices[None, :]])
    selected_genes = [genes[i] for i in indices]
    data = ad.AnnData(values, obs=pd.DataFrame({"target_gene": labels[rows]},
                       index=[str(i) for i in rows]), var=pd.DataFrame(index=selected_genes))
    reference = parallel_differential_expression(data, num_workers=2, is_log1p=True,
                  tie_correct=True, as_polars=True).sort(["target", "feature"])
    blocked = differential_expression(values, labels[rows], selected_genes,
                                       block_size=5).sort(["target", "feature"])
    errors = {}
    for column in ["p_value", "statistic", "fdr", "fold_change", "target_mean", "reference_mean"]:
        a, b = reference[column].to_numpy(), blocked[column].to_numpy()
        np.testing.assert_allclose(a, b, atol=1e-10, rtol=2e-6)
        errors[column] = float(np.max(np.abs(a - b)))
    write_json(root / "results/baselines/real-data-parity.json", {
        "status": "passed", "comparison": "native pdex 0.1.27 vs blocked adapter",
        "cells": len(rows), "control_cells": int(sum(labels[rows] == "non-targeting")),
        "targets": targets, "genes": selected_genes, "maximum_absolute_errors": errors})
    print("FULL-CONTROL REAL-DATA PARITY PASSED", errors, flush=True)


if __name__ == "__main__":
    main()
