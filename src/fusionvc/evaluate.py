"""Full validation cell counts and official VCC metrics in bounded memory."""

from __future__ import annotations

import gc
import logging
import shutil
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
import torch
from cell_eval import MetricsEvaluator

from .fast_de import differential_expression
from .io import CountsReader, file_sha256, normalize_log, read_json, write_json
from .models import apply_residual
from .preprocess import CONTROL
from .train import load_inputs


def grouped_means(matrix, labels, names, block_size=256):
    result = np.zeros((len(names), matrix.shape[1]), dtype=np.float32)
    masks = {name: np.flatnonzero(labels == name) for name in names}
    for start in range(0, matrix.shape[1], block_size):
        stop = min(start + block_size, matrix.shape[1])
        block = np.array(matrix[:, start:stop])
        for i, name in enumerate(names):
            result[i, start:stop] = block[masks[name]].mean(0, dtype=np.float64)
    return result


def mean_adata(means, names, genes):
    return ad.AnnData(
        means, obs=pd.DataFrame({"target_gene": names}, index=names), var=pd.DataFrame(index=genes)
    )


def real_cache(root: Path):
    cache = root / "outputs/evaluation-cache"
    cache.mkdir(parents=True, exist_ok=True)
    manifest = read_json(root / "data/processed/baselines-v1/manifest.json")
    reader = CountsReader(root / "data/raw/vcc2025/validation/adata_Validation.h5ad")
    labels = np.array(reader.categories)[reader.codes]
    names, genes = sorted(reader.categories), reader.genes
    matrix_path = cache / "validation_log.npy"
    if not (cache / "real-ready.json").exists():
        print("EVAL preparing reusable full validation matrix", flush=True)
        matrix = np.lib.format.open_memmap(
            matrix_path, mode="w+", dtype=np.float32, shape=reader.shape
        )
        for start in range(0, reader.shape[0], 512):
            stop = min(start + 512, reader.shape[0])
            matrix[start:stop] = normalize_log(
                reader.block(start, stop), manifest["validation_eval_target_sum"]
            ).toarray()
        matrix.flush()
        np.save(cache / "real_means.npy", grouped_means(matrix, labels, names))
        del matrix
        write_json(
            cache / "real-ready.json",
            {
                "shape": reader.shape,
                "names": names,
                "target_sum": manifest["validation_eval_target_sum"],
                "source": "official validation labels; evaluation only",
            },
        )
    reader.close()
    matrix = np.load(matrix_path, mmap_mode="r")
    de_path = cache / "real_de.parquet"
    if not de_path.exists():
        print("EVAL computing real DE once", flush=True)
        differential_expression(matrix, labels, genes, cache / "real-de-blocks").write_parquet(
            de_path
        )
    return matrix, labels, names, genes, np.load(cache / "real_means.npy"), pl.read_parquet(de_path)


@torch.no_grad()
def predict(root: Path, run: Path, real_labels, names, genes):
    device = torch.device("cuda")
    checkpoint = torch.load(run / "best.pt", map_location=device, weights_only=True)
    model, manifest, _, features, controls = load_inputs(root, checkpoint["config"], device)
    if hasattr(model, "feature_manifest"):
        if checkpoint.get("frozen_features") != model.feature_manifest:
            raise ValueError("Checkpoint and frozen feature provenance differ")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    out = run / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    pred = np.lib.format.open_memmap(
        out / "predictions.npy", mode="w+", dtype="float32", shape=(len(real_labels), len(genes))
    )
    target_index = {g: i for i, g in enumerate(genes)}
    rng = np.random.default_rng(2025)
    plan = {}
    started = time.perf_counter()
    for target in names:
        rows = np.flatnonzero(real_labels == target)
        chosen = rng.choice(len(controls), len(rows), replace=len(rows) > len(controls))
        plan[target] = chosen
        if target == CONTROL:
            delta = torch.zeros((1, len(genes)), device=device)
        else:
            q = torch.from_numpy(features[target_index[target]][None]).to(device)
            delta = model(q)
        for start in range(0, len(rows), 64):
            stop = min(start + 64, len(rows))
            ctrl = torch.from_numpy(np.array(controls[chosen[start:stop]])).to(device)
            values = apply_residual(ctrl, delta, manifest["target_sum"])[0]
            values = torch.log1p(
                torch.expm1(values)
                * (manifest["validation_eval_target_sum"] / manifest["target_sum"])
            )
            pred[rows[start:stop]] = values.cpu().numpy()
        print(f"PRED {run.parent.name}/{run.name} {target}: {len(rows)} cells", flush=True)
    pred.flush()
    np.savez_compressed(out / "control_sampling.npz", **plan)
    write_json(
        out / "prediction-info.json",
        {
            "shape": list(pred.shape),
            "seconds": time.perf_counter() - started,
            "control_sampling_seed": 2025,
            "target_sum": manifest["validation_eval_target_sum"],
            "control_counts_match_official": True,
            "genes": len(genes),
        },
    )
    del model
    torch.cuda.empty_cache()
    return pred


def publish_evaluation(root: Path, run: Path) -> None:
    """Finish publication/cleanup even if a prior evaluation stopped after scoring."""
    out = run / "evaluation"
    config = read_json(run / "config.json")
    evaluation = read_json(out / "metrics.json")
    result_group = config.get("results_group", "baselines")
    if result_group not in {"baselines", "mmd", "scgpt", "geneformer", "fusion", "distillation"}:
        raise ValueError("Unsupported result group")
    small = root / "results" / result_group / run.parent.name / run.name
    small.mkdir(parents=True, exist_ok=True)
    sources = [
        (out / "metrics.json", "metrics.json"),
        (run / "summary.json", "training.json"),
        (run / "history.csv", "history.csv"),
        (run / "config.json", "config.json"),
        (run / "environment.json", "environment.json"),
        (out / "results.csv", "per_perturbation.csv"),
    ]
    if "distribution" in evaluation:
        sources.append(
            (out / "distribution_per_perturbation.csv", "distribution_per_perturbation.csv")
        )
    # Retain the temporary prediction if a missing source prevents publication.
    for source, _ in sources:
        if not source.is_file():
            raise FileNotFoundError(f"Incomplete scored evaluation: {source}")
    for source, dest in sources:
        shutil.copy2(source, small / dest)
    (out / "predictions.npy").unlink(missing_ok=True)


def evaluate(root: Path, run: Path) -> None:
    logging.getLogger().setLevel(logging.WARNING)
    out = run / "evaluation"
    if (out / "metrics.json").exists():
        saved_hash = read_json(out / "metrics.json").get("checkpoint_sha256")
        if saved_hash is not None and saved_hash != file_sha256(run / "best.pt"):
            raise ValueError("Checkpoint changed after evaluation; use a new run directory")
        publish_evaluation(root, run)
        print(f"EVAL already complete: {run}", flush=True)
        return
    if shutil.disk_usage(root).free < 18 * 10**9:
        raise RuntimeError("Evaluation requires 18GB free for real and temporary predictions")
    started = time.perf_counter()
    checkpoint_hash = file_sha256(run / "best.pt")
    if (out / "checkpoint.json").exists():
        if read_json(out / "checkpoint.json")["sha256"] != checkpoint_hash:
            raise ValueError("Checkpoint changed; previous DE blocks cannot be reused")
    write_json(out / "checkpoint.json", {"sha256": checkpoint_hash})
    real, labels, names, genes, real_means, real_de = real_cache(root)
    pred = predict(root, run, labels, names, genes)
    config = read_json(run / "config.json")
    distribution = None
    if config.get("protocol") in {
        "mmd-v1",
        "scgpt-v1",
        "geneformer-v1",
        "fusion-v1",
        "distillation-v1",
    }:
        from .distribution import distribution_scores

        distribution_rows = distribution_scores(
            pred, real, labels, names, config["distribution_eval_cells"]
        )
        pd.DataFrame(distribution_rows).to_csv(
            out / "distribution_per_perturbation.csv", index=False
        )
        distribution = {
            "energy_mmd_rms": float(np.mean([r["energy_mmd_rms"] for r in distribution_rows])),
            "cells_per_set": config["distribution_eval_cells"],
            "prediction_seed": 2026,
            "observation_seed": 2027,
            "kernel": "negative Euclidean distance divided by sqrt(gene_count)",
            "scope": "supplementary sampled distribution metric; not an official VCC score",
        }
    pred_de = differential_expression(pred, labels, genes, out / "de-blocks")
    pred_de.write_parquet(out / "pred_de.parquet")
    pred_means = grouped_means(pred, labels, names)
    np.save(out / "pred_means.npy", pred_means)
    evaluator = MetricsEvaluator(
        adata_pred=mean_adata(pred_means, names, genes),
        adata_real=mean_adata(real_means, names, genes),
        de_real=real_de,
        de_pred=pred_de,
        control_pert=CONTROL,
        pert_col="target_gene",
        outdir=str(out),
        num_threads=1,
        pdex_kwargs={"tie_correct": True, "is_log1p": True},
    )
    scores, _ = evaluator.compute(profile="vcc", break_on_error=True)
    expected = {"mae", "discrimination_score_l1", "overlap_at_N"}
    if not expected <= set(scores.columns) or scores.height != 50:
        raise ValueError("Incomplete official metric results")
    values = {name: float(scores[name].mean()) for name in sorted(expected)}
    if not all(np.isfinite(x) for x in values.values()):
        raise FloatingPointError("Non-finite official metric")
    write_json(
        out / "metrics.json",
        {
            "split": "official validation (50 held-out perturbations)",
            "metrics": values,
            "evaluation_seconds": time.perf_counter() - started,
            "cell_eval_version": "0.6.6",
            "pdex_version": "0.1.27",
            "gene_count": len(genes),
            "predicted_cell_count": len(labels),
            "profile": "vcc",
            "de": "validated asymptotic Mann-Whitney acceleration",
            "global_fdr": "BH over all genes and all perturbations",
            "test_used": False,
            "checkpoint_sha256": checkpoint_hash,
            **({"distribution": distribution} if distribution is not None else {}),
        },
    )
    del pred, real
    gc.collect()
    publish_evaluation(root, run)
    print(f"EVAL COMPLETE {run}: {values}", flush=True)
