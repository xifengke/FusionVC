"""Cheap auxiliary references for interpreting mean-response baselines.

Full prescribed control sampling; cheap MAE/PDS and any separately verified DES.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from cell_eval import MetricsEvaluator

from .evaluate import mean_adata
from .io import CountsReader, read_json
from .models import apply_residual
from .preprocess import CONTROL


@torch.no_grad()
def auxiliary_controls(root: Path):
    cache = root / "data/processed/baselines-v1"
    manifest = read_json(cache / "manifest.json")
    stats = np.load(cache / "train_stats.npz")
    controls = np.load(cache / "controls.npy", mmap_mode="r")
    reader = CountsReader(root / "data/raw/vcc2025/validation/adata_Validation.h5ad")
    labels = np.array(reader.categories)[reader.codes]
    names, genes = sorted(reader.categories), reader.genes
    reader.close()
    real_means = np.load(root / "outputs/evaluation-cache/real_means.npy")
    ctrl_idx = list(stats["names"]).index(CONTROL)
    train_idx = [i for i, p in enumerate(stats["names"]) if p != CONTROL]
    average_delta = stats["mean"][train_idx].mean(0) - stats["mean"][ctrl_idx]
    results = []
    for method, shift in [
        ("no_perturbation", np.zeros(len(genes), np.float32)),
        ("average_training_shift", average_delta),
    ]:
        rng = np.random.default_rng(2025)
        means = np.zeros((len(names), len(genes)), np.float32)
        for i, target in enumerate(names):
            count = int(sum(labels == target))
            chosen = rng.choice(len(controls), count, replace=count > len(controls))
            delta = torch.from_numpy(shift[None]).cuda()
            if target == CONTROL:
                delta = torch.zeros_like(delta)
            summed = np.zeros(len(genes), np.float64)
            for start in range(0, count, 64):
                ctrl = torch.from_numpy(np.array(controls[chosen[start : start + 64]])).cuda()
                values = apply_residual(ctrl, delta, manifest["target_sum"])[0]
                values = torch.log1p(
                    values.expm1()
                    * (manifest["validation_eval_target_sum"] / manifest["target_sum"])
                )
                summed += values.cpu().numpy().sum(0, dtype=np.float64)
            means[i] = summed / count
        evaluator = MetricsEvaluator(
            mean_adata(means, names, genes),
            mean_adata(real_means, names, genes),
            pert_col="target_gene",
            skip_de=True,
            outdir=str(root / f"outputs/auxiliary/{method}"),
        )
        score, _ = evaluator.compute(profile="vcc", break_on_error=True)
        score.write_csv(root / f"results/baselines/{method}_per_perturbation.csv")
        results.append(
            {
                "reference": method,
                "mae": float(score["mae"].mean()),
                "discrimination_score_l1": float(score["discrimination_score_l1"].mean()),
                "DES": "not evaluated",
                "split": "validation",
            }
        )
        full_reference = root / "results/controls/no_perturbation/metrics.json"
        if method == "no_perturbation" and full_reference.exists():
            from .control_evaluation import load_full_reference

            full, _ = load_full_reference(root)
            for metric in ["mae", "discrimination_score_l1"]:
                if not np.isclose(results[-1][metric], full["metrics"][metric], rtol=0, atol=1e-7):
                    raise ValueError("Cheap reference changed after full DES evaluation")
            results[-1]["overlap_at_N"] = full["metrics"]["overlap_at_N"]
            results[-1]["DES"] = full["metrics"]["overlap_at_N"]
    pd.DataFrame(results).to_csv(root / "results/baselines/auxiliary.csv", index=False)
    print(results, flush=True)
