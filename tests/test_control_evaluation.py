"""A zero shift preserves heterogeneous cell sets and full-cell DES semantics."""

import anndata as ad
import numpy as np
import pandas as pd
import torch
from cell_eval import MetricsEvaluator
from pdex import parallel_differential_expression

from fusionvc.control_evaluation import predict_zero_shift
from fusionvc.evaluate import grouped_means, mean_adata
from fusionvc.fast_de import differential_expression
from fusionvc.models import apply_residual


def test_reference_sampling_counts_scales_and_cell_heterogeneity(tmp_path):
    rng = np.random.default_rng(14)
    counts = rng.integers(1, 8, (19, 5)).astype(np.float32)
    controls = np.log1p(counts / counts.sum(1, keepdims=True) * 100).astype(np.float32)
    labels = np.array(["non-targeting"] * 23 + ["G0"] * 17 + ["G1"] * 11)
    names = sorted(set(labels))
    manifest = {"target_sum": 100.0, "validation_eval_target_sum": 120.0}
    pred = predict_zero_shift(tmp_path, controls, labels, names, manifest, torch.device("cpu"))
    sampling = np.load(tmp_path / "control_sampling.npz")
    sample_rng = np.random.default_rng(2025)
    for name in names:
        rows = np.flatnonzero(labels == name)
        chosen = sample_rng.choice(len(controls), len(rows), replace=len(rows) > len(controls))
        np.testing.assert_array_equal(sampling[name], chosen)
        expected = apply_residual(torch.from_numpy(controls[chosen]), torch.zeros(1, 5), 100.0)[0]
        expected = torch.log1p(expected.expm1() * 1.2).numpy()
        np.testing.assert_allclose(pred[rows], expected, rtol=0, atol=0)
    np.testing.assert_allclose(np.expm1(pred).sum(1), 120.0, rtol=1e-6)
    assert np.unique(pred, axis=0).shape[0] > 1
    assert pred.shape == (51, 5)
    sampling.close()


def test_zero_shift_des_matches_native_full_cell_evaluator(tmp_path):
    rng = np.random.default_rng(33)
    genes = [f"G{i}" for i in range(8)]
    counts = rng.poisson(4, (24, 8)).astype(np.float32)
    controls = np.log1p(counts / counts.sum(1, keepdims=True) * 100).astype(np.float32)
    labels = np.array(["non-targeting"] * 24 + ["G0"] * 17 + ["G1"] * 19)
    names = sorted(set(labels))
    pred = predict_zero_shift(
        tmp_path,
        controls,
        labels,
        names,
        {"target_sum": 100.0, "validation_eval_target_sum": 100.0},
        torch.device("cpu"),
    )
    real = np.array(pred)
    real[labels == "G0", 3] += 1.0
    real[labels == "G1", 4] += 1.0
    obs = pd.DataFrame({"target_gene": labels}, index=[f"C{i}" for i in range(len(labels))])
    var = pd.DataFrame(index=genes)
    pred_adata = ad.AnnData(np.array(pred), obs=obs, var=var)
    real_adata = ad.AnnData(real, obs=obs.copy(), var=var.copy())
    native_de_pred = parallel_differential_expression(
        pred_adata, num_workers=1, is_log1p=True, as_polars=True, tie_correct=True
    )
    native_de_real = parallel_differential_expression(
        real_adata, num_workers=1, is_log1p=True, as_polars=True, tie_correct=True
    )
    native = MetricsEvaluator(
        pred_adata,
        real_adata,
        de_pred=native_de_pred,
        de_real=native_de_real,
        pert_col="target_gene",
        outdir=str(tmp_path / "native"),
    )
    adapted = MetricsEvaluator(
        mean_adata(grouped_means(pred, labels, names), names, genes),
        mean_adata(grouped_means(real, labels, names), names, genes),
        de_pred=differential_expression(pred, labels, genes, block_size=3),
        de_real=differential_expression(real, labels, genes, block_size=3),
        pert_col="target_gene",
        outdir=str(tmp_path / "adapted"),
    )
    a, _ = native.compute(profile="vcc", break_on_error=True, write_csv=False)
    b, _ = adapted.compute(profile="vcc", break_on_error=True, write_csv=False)
    a, b = a.sort("perturbation"), b.sort("perturbation")
    for metric in ["mae", "discrimination_score_l1", "overlap_at_N"]:
        np.testing.assert_allclose(a[metric].to_numpy(), b[metric].to_numpy(), rtol=1e-6, atol=1e-7)
