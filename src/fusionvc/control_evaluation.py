"""Full-cell zero-shift reference, including official differential expression."""

from __future__ import annotations

import gc
import logging
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import torch
from cell_eval import MetricsEvaluator

from .evaluate import grouped_means, mean_adata, real_cache
from .fast_de import differential_expression
from .io import file_sha256, read_json, seed_all, write_json
from .models import apply_residual
from .preprocess import CONTROL

METRICS = ["mae", "discrimination_score_l1", "overlap_at_N"]


@torch.no_grad()
def predict_zero_shift(folder, controls, labels, names, manifest, device):
    """Preserve the neural runs' cell counts, sampling, clipping and scales."""
    folder.mkdir(parents=True, exist_ok=True)
    pred = np.lib.format.open_memmap(
        folder / "predictions.npy",
        mode="w+",
        dtype="float32",
        shape=(len(labels), controls.shape[1]),
    )
    rng, plan = np.random.default_rng(2025), {}
    delta = torch.zeros((1, controls.shape[1]), device=device)
    for name in names:
        rows = np.flatnonzero(labels == name)
        chosen = rng.choice(len(controls), len(rows), replace=len(rows) > len(controls))
        plan[name] = chosen
        for start in range(0, len(rows), 64):
            stop = min(start + 64, len(rows))
            ctrl = torch.from_numpy(np.array(controls[chosen[start:stop]])).to(device)
            values = apply_residual(ctrl, delta, manifest["target_sum"])[0]
            values = torch.log1p(
                values.expm1() * (manifest["validation_eval_target_sum"] / manifest["target_sum"])
            )
            pred[rows[start:stop]] = values.cpu().numpy()
        print(f"NO-PERT PRED {name}: {len(rows)} cells", flush=True)
    pred.flush()
    np.savez_compressed(folder / "control_sampling.npz", **plan)
    return pred


def reference_signature(root: Path):
    base = root / "data/processed/baselines-v1"
    controls = base / "controls.npy"
    real = root / "outputs/evaluation-cache"
    return {
        "protocol": "no-perturbation-full-v1",
        "predictor": "delta=0 for every perturbation and control; same apply_residual and scales",
        "source_sha256": {
            name: file_sha256(root / name)
            for name in [
                "src/fusionvc/control_evaluation.py",
                "src/fusionvc/models.py",
                "src/fusionvc/fast_de.py",
                "src/fusionvc/evaluate.py",
                "data/processed/baselines-v1/manifest.json",
                "data/processed/baselines-v1/genes.json",
                "outputs/evaluation-cache/real-ready.json",
                "outputs/evaluation-cache/real_means.npy",
                "outputs/evaluation-cache/real_de.parquet",
            ]
        },
        "training_controls": {
            "bytes": controls.stat().st_size,
            "mtime_ns": controls.stat().st_mtime_ns,
        },
        "validation_matrix": {
            "bytes": (real / "validation_log.npy").stat().st_size,
            "mtime_ns": (real / "validation_log.npy").stat().st_mtime_ns,
        },
        "control_sampling_seed": 2025,
        "test_used": False,
    }


def evaluate_no_perturbation(root: Path):
    logging.getLogger().setLevel(logging.WARNING)
    folder = root / "outputs/auxiliary/no_perturbation_full"
    small = root / "results/controls/no_perturbation"
    started = time.perf_counter()
    real, labels, names, genes, real_means, real_de = real_cache(root)
    signature = reference_signature(root)
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "source.json").exists() and read_json(folder / "source.json") != signature:
        raise ValueError("Reference sources changed; use a new evaluation version")
    write_json(folder / "source.json", signature)
    if (folder / "metrics.json").exists():
        metrics = read_json(folder / "metrics.json")
        if metrics["signature"] != signature:
            raise ValueError("Completed reference provenance mismatch")
        checks = read_json(small / "verification.json")
        for name, digest in checks["small_artifact_hashes"].items():
            if file_sha256(small / name) != digest:
                raise ValueError(f"Corrupted reference artifact: {name}")
        update_auxiliary(root, metrics)
        print("NO-PERT full evaluation already complete; sources verified", flush=True)
        return metrics
    if shutil.disk_usage(root).free < 18 * 10**9:
        raise RuntimeError("Full reference evaluation requires 18GB free")
    if not torch.cuda.is_available():
        raise RuntimeError("Use the existing experiment GPU environment")
    seed_all(2025)
    controls = np.load(root / "data/processed/baselines-v1/controls.npy", mmap_mode="r")
    manifest = read_json(root / "data/processed/baselines-v1/manifest.json")
    pred = predict_zero_shift(folder, controls, labels, names, manifest, torch.device("cuda"))
    if pred.shape != (98927, 18080):
        raise ValueError("Reference must include all official validation cells and genes")
    pred_means = grouped_means(pred, labels, names)
    np.save(folder / "pred_means.npy", pred_means)
    pred_de = differential_expression(pred, labels, genes, folder / "de-blocks")
    pred_de.write_parquet(folder / "pred_de.parquet")
    evaluator = MetricsEvaluator(
        adata_pred=mean_adata(pred_means, names, genes),
        adata_real=mean_adata(real_means, names, genes),
        de_real=real_de,
        de_pred=pred_de,
        control_pert=CONTROL,
        pert_col="target_gene",
        outdir=str(folder),
        num_threads=1,
        pdex_kwargs={"tie_correct": True, "is_log1p": True},
    )
    scores, _ = evaluator.compute(profile="vcc", break_on_error=True)
    if scores.height != 50 or not set(METRICS) <= set(scores.columns):
        raise ValueError("Incomplete official reference scores")
    values = {metric: float(scores[metric].mean()) for metric in METRICS}
    if not np.isfinite(list(values.values())).all():
        raise FloatingPointError("Non-finite reference metrics")
    # Match the already published cheap reference on every validation target.
    legacy_path = root / "results/baselines/no_perturbation_per_perturbation.csv"
    legacy = pd.read_csv(legacy_path).set_index("perturbation")
    current = scores.to_pandas().set_index("perturbation").reindex(legacy.index)
    differences = {}
    for metric in METRICS[:2]:
        difference = float(np.max(np.abs(current[metric] - legacy[metric])))
        differences[metric] = difference
        if difference > 1e-7:
            raise ValueError(f"Full reference differs from legacy {metric}: {difference}")
    plan = np.load(folder / "control_sampling.npz")
    baseline_plan = np.load(root / "outputs/01_mlp/seed_42/evaluation/control_sampling.npz")
    same_plan = set(plan.files) == set(baseline_plan.files) and all(
        np.array_equal(plan[name], baseline_plan[name]) for name in plan.files
    )
    plan.close()
    baseline_plan.close()
    if not same_plan:
        raise ValueError("Reference control sampling differs from model evaluation")
    # Diagnostics explain any finite-sample significant genes without assuming DES=0.
    real_sig = real_de.filter(pl.col("fdr") < 0.05).group_by("target").len()
    pred_sig = pred_de.filter(pl.col("fdr") < 0.05).group_by("target").len()
    real_counts = dict(real_sig.iter_rows())
    pred_counts = dict(pred_sig.iter_rows())
    diagnostic = pd.DataFrame(
        {
            "perturbation": [name for name in names if name != CONTROL],
            "real_significant_genes": [
                real_counts.get(name, 0) for name in names if name != CONTROL
            ],
            "predicted_significant_genes": [
                pred_counts.get(name, 0) for name in names if name != CONTROL
            ],
        }
    )
    small.mkdir(parents=True, exist_ok=True)
    scores.write_csv(small / "per_perturbation.csv")
    diagnostic.to_csv(small / "de_gene_counts.csv", index=False)
    metrics = {
        "reference": "no_perturbation",
        "signature": signature,
        "metrics": values,
        "split": "official validation (50 held-out perturbations)",
        "gene_count": len(genes),
        "predicted_cell_count": len(labels),
        "cell_eval_version": "0.6.6",
        "pdex_version": "0.1.27",
        "profile": "vcc",
        "de": "same validated asymptotic Mann-Whitney as all models; full cells",
        "global_fdr": "BH over all genes and all perturbations; FDR < 0.05",
        "evaluation_seconds": time.perf_counter() - started,
        "test_used": False,
    }
    write_json(small / "metrics.json", metrics)
    write_json(
        small / "verification.json",
        {
            "all_checks_passed": True,
            "control_plan_matches_original_mlp": same_plan,
            "legacy_per_target_max_absolute_difference": differences,
            "legacy_per_target_sha256": file_sha256(legacy_path),
            "full_genes_and_cells": pred.shape == (98927, 18080),
            "targets_scored": scores.height,
            "predicted_significant_genes_total": int(diagnostic.predicted_significant_genes.sum()),
            "small_artifact_hashes": {
                name: file_sha256(small / name)
                for name in ["metrics.json", "per_perturbation.csv", "de_gene_counts.csv"]
            },
            "test_used": False,
        },
    )
    del pred, real
    gc.collect()
    (folder / "predictions.npy").unlink()
    update_auxiliary(root, metrics)
    # Completion marker is written after the small artifacts and cleanup.
    write_json(folder / "metrics.json", metrics)
    print(f"NO-PERT FULL COMPLETE {values}", flush=True)
    return metrics


def update_auxiliary(root: Path, metrics: dict):
    path = root / "results/baselines/auxiliary.csv"
    frame = pd.read_csv(path)
    index = frame.index[frame.reference == "no_perturbation"]
    if len(index) != 1:
        raise ValueError("Expected exactly one no-perturbation reference")
    for name, value in metrics["metrics"].items():
        frame.loc[index, name] = value
    # Preserve the legacy column for old consumers; overlap_at_N is canonical.
    frame.loc[index, "DES"] = str(metrics["metrics"]["overlap_at_N"])
    frame.to_csv(path, index=False)


def load_full_reference(root: Path):
    small = root / "results/controls/no_perturbation"
    metrics = read_json(small / "metrics.json")
    checks = read_json(small / "verification.json")
    if not checks["all_checks_passed"] or metrics["test_used"]:
        raise ValueError("Reference has not passed full validation checks")
    if metrics["signature"] != reference_signature(root):
        raise ValueError("Full reference sources changed")
    for name, digest in checks["small_artifact_hashes"].items():
        if file_sha256(small / name) != digest:
            raise ValueError(f"Reference artifact changed: {name}")
    return metrics, pd.read_csv(small / "per_perturbation.csv")
