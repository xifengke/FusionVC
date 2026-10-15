"""Independent reference checks for the official-evaluation adapter."""

import anndata as ad
import numpy as np
import pandas as pd
from cell_eval import MetricsEvaluator
from pdex import parallel_differential_expression
from scipy.stats import mannwhitneyu

from fusionvc.evaluate import grouped_means, mean_adata
from fusionvc.fast_de import differential_expression, ranksum_reusing_reference, reference_ties


def test_fast_u_matches_scipy_with_ties_and_constant_genes():
    rng = np.random.default_rng(8)
    ref = rng.integers(0, 5, (11, 64)).astype(np.float32)
    tgt = rng.integers(0, 5, (11, 21)).astype(np.float32)
    ref[0] = 0
    tgt[0] = 0
    sr, st = np.sort(ref, axis=1), np.sort(tgt, axis=1)
    p, u = ranksum_reusing_reference(st, sr, reference_ties(sr))
    official = mannwhitneyu(tgt, ref, axis=1, method="asymptotic", use_continuity=True)
    np.testing.assert_allclose(p, official.pvalue, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(u, official.statistic, atol=1e-10)


def test_blocked_de_matches_pdex_global_fdr():
    rng = np.random.default_rng(21)
    labels = np.array(["non-targeting"] * 40 + ["G0"] * 19 + ["G1"] * 23)
    x = np.log1p(rng.poisson(2, (len(labels), 9))).astype(np.float32)
    x[labels == "G0", 1] += 0.7
    genes = [f"G{i}" for i in range(x.shape[1])]
    truth = ad.AnnData(x, obs=pd.DataFrame({"target_gene": labels}), var=pd.DataFrame(index=genes))
    full = parallel_differential_expression(
        truth, num_workers=1, is_log1p=True, as_polars=True, tie_correct=True
    )
    blocked = differential_expression(x, labels, genes, block_size=4)
    full, blocked = full.sort(["target", "feature"]), blocked.sort(["target", "feature"])
    for column in ["p_value", "statistic", "fdr", "fold_change", "target_mean", "reference_mean"]:
        np.testing.assert_allclose(
            full[column].to_numpy(), blocked[column].to_numpy(), rtol=2e-6, atol=1e-8
        )


def test_mean_adapter_preserves_official_mae_and_pds(tmp_path):
    rng = np.random.default_rng(42)
    labels = np.array(["non-targeting"] * 12 + ["G0"] * 11 + ["G1"] * 15)
    genes = [f"G{i}" for i in range(8)]
    real = rng.uniform(0, 3, (len(labels), len(genes))).astype(np.float32)
    pred = np.maximum(real + rng.normal(0, 0.3, real.shape), 0).astype(np.float32)
    names = sorted(set(labels))
    full_real = ad.AnnData(
        real, obs=pd.DataFrame({"target_gene": labels}), var=pd.DataFrame(index=genes)
    )
    full_pred = ad.AnnData(pred, obs=full_real.obs.copy(), var=full_real.var.copy())
    full = MetricsEvaluator(
        full_pred, full_real, pert_col="target_gene", skip_de=True, outdir=str(tmp_path / "full")
    )
    compact = MetricsEvaluator(
        mean_adata(grouped_means(pred, labels, names), names, genes),
        mean_adata(grouped_means(real, labels, names), names, genes),
        pert_col="target_gene",
        skip_de=True,
        outdir=str(tmp_path / "compact"),
    )
    a, _ = full.compute(profile="vcc", break_on_error=True, write_csv=False)
    b, _ = compact.compute(profile="vcc", break_on_error=True, write_csv=False)
    a, b = a.sort("perturbation"), b.sort("perturbation")
    for column in ["mae", "discrimination_score_l1"]:
        np.testing.assert_allclose(a[column].to_numpy(), b[column].to_numpy(), rtol=1e-6, atol=1e-7)
