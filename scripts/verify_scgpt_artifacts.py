"""Audit complete validation runs, frozen assets, and the shared control plan."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    from fusionvc.io import file_sha256, read_json, write_json
    from fusionvc.scgpt import load_scgpt_features

    _, _, feature_manifest = load_scgpt_features(root)
    baseline_plan = np.load(root / "outputs/01_mlp/seed_42/evaluation/control_sampling.npz")
    checks = []
    for experiment in ["03_scgpt", "03_scgpt_energy", "03_scgpt_hybrid"]:
        for seed in [42, 43, 44]:
            run = root / f"outputs/{experiment}/seed_{seed}"
            small = root / f"results/scgpt/{experiment}/seed_{seed}"
            summary = read_json(run / "summary.json")
            metrics = read_json(run / "evaluation/metrics.json")
            info = read_json(run / "evaluation/prediction-info.json")
            target_scores = pd.read_csv(small / "per_perturbation.csv")
            distribution = pd.read_csv(small / "distribution_per_perturbation.csv")
            checkpoint_hash = file_sha256(run / "best.pt")
            plan = np.load(run / "evaluation/control_sampling.npz")
            same_plan = set(plan.files) == set(baseline_plan.files)
            same_plan = same_plan and all(
                np.array_equal(plan[key], baseline_plan[key]) for key in plan.files
            )
            check = {
                "experiment": experiment,
                "seed": seed,
                "checkpoint_hash_matches": checkpoint_hash
                == summary["checkpoint_sha256"]
                == metrics["checkpoint_sha256"],
                "full_genes_and_cells": info["shape"] == [98927, 18080],
                "targets_scored": len(target_scores),
                "distribution_targets_scored": len(distribution),
                "metrics_finite": bool(np.isfinite(list(metrics["metrics"].values())).all()),
                "aggregates_match_target_scores": all(
                    np.isclose(target_scores[name].mean(), value, rtol=1e-12, atol=1e-12)
                    for name, value in metrics["metrics"].items()
                ),
                "distribution_aggregate_matches": bool(
                    np.isclose(
                        distribution["energy_mmd_rms"].mean(),
                        metrics["distribution"]["energy_mmd_rms"],
                        rtol=1e-12,
                        atol=1e-12,
                    )
                ),
                "frozen_feature_manifest_matches": summary["frozen_features"] == feature_manifest,
                "control_plan_matches_original_mlp": same_plan,
                "test_unused": metrics["test_used"] is False,
                "temporary_prediction_removed": not (run / "evaluation/predictions.npy").exists(),
                "published_metrics_match": read_json(small / "metrics.json") == metrics,
            }
            valid = all(value for key, value in check.items() if isinstance(value, bool))
            valid = valid and check["targets_scored"] == check["distribution_targets_scored"] == 50
            if not valid:
                raise RuntimeError(f"Artifact verification failed: {check}")
            checks.append(check)
            plan.close()
    original_checks = []
    for group, pattern in [
        ("baselines", "0*/seed_*/training.json"),
        ("mmd", "mmd_*/seed_*/training.json"),
    ]:
        for path in sorted((root / "results" / group).glob(pattern)):
            summary = read_json(path)
            run = root / f"outputs/{summary['experiment']}/seed_{summary['seed']}"
            unchanged = file_sha256(run / "best.pt") == summary["checkpoint_sha256"]
            if not unchanged:
                raise RuntimeError(f"Original checkpoint changed: {run}")
            original_checks.append(
                {
                    "experiment": summary["experiment"],
                    "seed": summary["seed"],
                    "checkpoint_unchanged": unchanged,
                }
            )
    baseline_plan.close()
    write_json(
        root / "results/scgpt/artifact-verification.json",
        {
            "scgpt_runs": checks,
            "original_runs": original_checks,
            "all_checks_passed": True,
            "test_used": False,
        },
    )
    print(
        f"Verified 9 scGPT runs and {len(original_checks)} unchanged original checkpoints",
        flush=True,
    )


if __name__ == "__main__":
    main()
