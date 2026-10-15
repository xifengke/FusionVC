"""Frozen-selection, full-cell official test evaluation in independent directories.

Importing this module never opens test data. Every runtime entry verifies the
pre-test selection before accessing either test counts or their cached values.
"""

from __future__ import annotations

import csv
import gc
import importlib.metadata
import json
import math
import re
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import torch
from cell_eval import MetricsEvaluator

from .evaluate import grouped_means, mean_adata
from .fast_de import differential_expression
from .io import CountsReader, file_sha256, normalize_log, read_json, seed_all
from .models import apply_residual
from .preprocess import CONTROL, library_sizes
from .train import load_inputs

PROTOCOL = "final-test-v1"
SELECTION_BASIS = "official_validation_mae_mean_over_seeds"
DOWNLOAD_RECEIPT = "docs/test-download-status.json"
RESERVE_BYTES = 18 * 10**9
METRICS = ["mae", "discrimination_score_l1", "overlap_at_N"]
MAIN_MODELS = {
    "mlp",
    "transformer",
    "scgpt",
    "geneformer",
    "concat",
    "gated",
    "cross_attention",
    "distillation",
}


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), "utf-8")
    temporary.replace(path)


def project_path(root: Path, relative: str) -> Path:
    path = Path(relative)
    resolved = (root / path).resolve()
    if path.is_absolute() or not resolved.is_relative_to(root.resolve()):
        raise ValueError("Expected a project-relative path inside the workspace")
    return resolved


def _validate_validation_record(root: Path, row: dict, candidate: bool = False) -> None:
    from .fusion_mmd_report import run_spec
    from .fusion_report import validate_evaluation

    if row.get("model") == "distillation":
        group, experiment = "distillation", "08_distillation"
    else:
        group, experiment, _ = run_spec(row["model"], row["variant"])
    relative = f"results/{group}/{experiment}/seed_{row['seed']}/metrics.json"
    required = {
        "validation_metrics_path",
        "validation_metrics_sha256",
        "mae" if candidate else "validation_mae",
    }
    if not required <= set(row) or row["validation_metrics_path"].replace("\\", "/") != relative:
        raise ValueError(
            "Every frozen candidate/selected checkpoint needs its canonical official validation source"
        )
    if row.get("experiment") != experiment:
        raise ValueError("Frozen validation architecture/loss and experiment disagree")
    path = root / relative
    if file_sha256(path) != row["validation_metrics_sha256"]:
        raise ValueError("Frozen validation metric source changed")
    saved = read_json(path)
    validate_evaluation(saved)
    checkpoint = root / f"outputs/{experiment}/seed_{row['seed']}/best.pt"
    if (
        saved.get("checkpoint_sha256") != row["checkpoint_sha256"]
        or file_sha256(checkpoint) != row["checkpoint_sha256"]
    ):
        raise ValueError("Frozen validation score does not bind its current checkpoint")
    if (
        not candidate
        and row["checkpoint_path"].replace("\\", "/") != checkpoint.relative_to(root).as_posix()
    ):
        raise ValueError("Selected checkpoint path differs from the validated original run")
    recorded_mae = row["mae" if candidate else "validation_mae"]
    if not math.isclose(saved["metrics"]["mae"], recorded_mae, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("Selection must bind the original official validation MAE")
    scale = read_json(root / "data/processed/baselines-v1/manifest.json")[
        "validation_eval_target_sum"
    ]
    info = read_json(checkpoint.parent / "evaluation/prediction-info.json")
    if info.get("shape") != [98927, 18080] or info.get("target_sum") != scale:
        raise ValueError(
            "Selected validation score did not use the original complete validation scale"
        )


def _validate_pretest_evidence(root: Path, selection: dict) -> None:
    from .distillation import choose_teacher
    from .distillation_report import verification_passport
    from .final_selection import MODELS, PRETEST_RECEIPTS, rank_validation_groups
    from .fusion_mmd_report import FUSION_MODELS, material_passport

    candidates = selection.get("candidate_runs", [])
    if (
        len(candidates) != 63
        or len(selection.get("candidate_groups", [])) != 21
        or len(selection.get("selected_groups", [])) != 7
    ):
        raise ValueError(
            "Frozen selection must include all 63 candidates, 21 groups and seven choices"
        )
    ranked = rank_validation_groups(candidates)
    chosen = [next(row for row in ranked if row["model"] == model) for model in MODELS]
    if selection["candidate_groups"] != ranked or selection["selected_groups"] != chosen:
        raise ValueError("Frozen groups do not follow the three-seed MAE mean and exact tie rule")
    for row in candidates:
        _validate_validation_record(root, row, candidate=True)
    for row in selection["runs"]:
        if row["kind"] == "checkpoint":
            _validate_validation_record(root, row)
            if row["role"] == "main" and row["model"] != "distillation":
                winner = next(choice for choice in chosen if choice["model"] == row["model"])
                if row["variant"] != winner["variant"] or row["experiment"] != winner["experiment"]:
                    raise ValueError(
                        "Selected main seeds differ from the validation-winning whole group"
                    )
    receipts = selection.get("validation_receipts", [])
    if len(receipts) != len(PRETEST_RECEIPTS) or {row["path"] for row in receipts} != set(
        PRETEST_RECEIPTS
    ):
        raise ValueError(
            "Frozen selection requires the complete fixed set of pre-test verification receipts"
        )
    for receipt in receipts:
        path = project_path(root, receipt["path"])
        if file_sha256(path) != receipt["sha256"] or read_json(path).get("test_used") is not False:
            raise ValueError("A frozen validation verification receipt changed or used test data")
    for relative in [
        "results/fusion-mmd/report-provenance.json",
        "results/distillation/report-provenance.json",
    ]:
        report = read_json(root / relative)
        if (
            report.get("complete") is not True
            or report.get("material_passport", {}).get("verification_status") != "VERIFIED"
        ):
            raise ValueError("Both complete validation-stage reports must already be VERIFIED")
    if (
        material_passport(root)["verification_status"] != "VERIFIED"
        or verification_passport(root)["verification_status"] != "VERIFIED"
    ):
        raise ValueError(
            "Current MMD and distillation verification gates must both pass before test access"
        )
    expected_teacher = choose_teacher([row for row in candidates if row["model"] in FUSION_MODELS])
    if (
        selection.get("teacher_selection") != expected_teacher
        or read_json(root / "results/distillation/teacher-selection.json") != expected_teacher
    ):
        raise ValueError("Final frozen teacher must follow the nine-group rule and fixed seed 42")
    if not any(
        row.get("checkpoint_sha256") == expected_teacher["checkpoint_sha256"]
        for row in selection["runs"]
        if row["role"] == "main"
    ):
        raise ValueError("The actual fixed teacher must be included in the frozen main evaluation")
    if selection.get("student") != {
        "model": "distillation",
        "variant": "mean_kd",
        "experiment": "08_distillation",
        "kd_weight": 1.0,
    }:
        raise ValueError("Final student must retain the fixed supervised MLP plus KD protocol")


def validate_selection(root: Path, path: Path) -> tuple[dict, str]:
    """Require the complete pre-test selection and current checkpoint hashes."""
    selection = read_json(path)
    if (
        selection.get("protocol") != PROTOCOL
        or selection.get("frozen") is not True
        or selection.get("producer") != "fusionvc.final_selection-v1"
        or selection.get("test_used") is not False
        or selection.get("selection_basis") != SELECTION_BASIS
    ):
        raise ValueError("A frozen final-test-v1 selection based only on validation is required")
    runs = selection.get("runs", [])
    labels = [row.get("label", "") for row in runs]
    if (
        not runs
        or len(set(labels)) != len(labels)
        or any(
            not isinstance(label, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", label)
            for label in labels
        )
    ):
        raise ValueError("Final selection labels must be nonempty, unique and safe ASCII names")
    main, controls, references = [], [], []
    for row in runs:
        if row.get("kind") == "no_perturbation":
            if row.get("model") != "no_perturbation" or row.get("role") != "reference":
                raise ValueError("The zero-shift run must be the no_perturbation reference")
            references.append(row)
            continue
        if row.get("kind") != "checkpoint" or row.get("role") not in {"main", "kd_control"}:
            raise ValueError("Unknown final evaluation run kind or role")
        checkpoint = project_path(root, row["checkpoint_path"])
        if (
            not checkpoint.is_relative_to((root / "outputs").resolve())
            or checkpoint.name != "best.pt"
        ):
            raise ValueError("Selected checkpoints must be original outputs/.../best.pt files")
        if file_sha256(checkpoint) != row.get("checkpoint_sha256"):
            raise ValueError(f"Frozen selected checkpoint changed: {checkpoint}")
        if row.get("seed") not in {42, 43, 44}:
            raise ValueError("Final checkpoint selection requires seeds 42/43/44")
        (main if row["role"] == "main" else controls).append(row)
    if len(references) != 1 or len(main) != 24:
        raise ValueError(
            "Select eight architectures times three seeds and one zero-shift reference"
        )
    if {(row.get("model"), row["seed"]) for row in main} != {
        (model, seed) for model in MAIN_MODELS for seed in [42, 43, 44]
    }:
        raise ValueError("Final main selection has incomplete architecture/seed coverage")
    for model in MAIN_MODELS:
        if len({row.get("variant") for row in main if row["model"] == model}) != 1:
            raise ValueError("A model's three seeds must use one validation-selected loss")
    mlp_mean = all(row.get("variant") == "mean" for row in main if row["model"] == "mlp")
    if controls and (
        mlp_mean
        or len(controls) != 3
        or {(row.get("model"), row.get("variant"), row["seed"]) for row in controls}
        != {("mlp", "mean", seed) for seed in [42, 43, 44]}
    ):
        raise ValueError("Additional KD controls must be exactly the three original MLP mean runs")
    if not mlp_mean and len(controls) != 3:
        raise ValueError("A non-mean main MLP requires the three MLP mean KD controls")
    _validate_pretest_evidence(root, selection)
    return selection, file_sha256(path)


def test_asset_specs() -> list[dict]:
    base = "https://storage.googleapis.com/arc-institute-virtual-cell-atlas/"
    specifications = [
        ("adata_Test.h5ad", "1765905295402700", 11950739168, "JMP5RQ=="),
        ("pert_counts_Test.csv", "1765906557881304", 1894, "dCaUQw=="),
    ]
    result = []
    for name, generation, size, crc in specifications:
        object_name = f"virtual-cell-challenge/2025/test/{name}"
        result.append(
            {
                "name": object_name,
                "path": f"data/raw/vcc2025/test/{name}",
                "generation": generation,
                "size_bytes": size,
                "crc32c": crc,
                "url": f"{base}{object_name}?generation={generation}",
            }
        )
    return result


def ensure_space(root: Path, additional_bytes: int, reserve: int = RESERVE_BYTES) -> None:
    free = shutil.disk_usage(root).free
    if free < additional_bytes + reserve:
        raise RuntimeError(
            f"Final test needs {additional_bytes / 1e9:.2f} GB additional plus "
            f"{reserve / 1e9:.2f} GB reserve; only {free / 1e9:.2f} GB free"
        )


def verified_test_download(root: Path, selection_sha: str) -> dict:
    """Check a completed CRC/SHA receipt without rehashing the 12 GB matrix per run."""
    receipt = read_json(root / DOWNLOAD_RECEIPT)
    files = receipt.get("files", [])
    if (
        receipt.get("status") != "complete"
        or receipt.get("selection_sha256") != selection_sha
        or len(files) != 2
        or {row["path"] for row in files} != {s["path"] for s in test_asset_specs()}
    ):
        raise ValueError("Test downloads must bind the frozen final selection")
    by_path = {row["path"]: row for row in files}
    for spec in test_asset_specs():
        row = by_path[spec["path"]]
        path = root / spec["path"]
        stat = path.stat()
        if (
            any(row.get(key) != spec[key] for key in ["url", "generation", "size_bytes", "crc32c"])
            or row.get("verified_against_source_checksum") is not True
            or stat.st_size != spec["size_bytes"]
            or stat.st_mtime_ns != row.get("mtime_ns")
            or not re.fullmatch(r"[0-9a-f]{64}", row.get("sha256", ""))
        ):
            raise ValueError(f"Test file or verified download provenance changed: {path}")
    return receipt


def validate_test_cache_sources(root: Path, selection_sha: str) -> dict:
    """Audit current cache lineage without opening test expression values."""
    download = verified_test_download(root, selection_sha)
    folder = root / "outputs/final-test-cache"
    source_path, ready_path = folder / "source.json", folder / "real-ready.json"
    source, ready = read_json(source_path), read_json(ready_path)
    baseline_names = {
        "manifest.json",
        "genes.json",
        "gene_descriptors.npy",
        "input_gene_indices.npy",
    }
    scoring_names = {
        "final_evaluation.py",
        "evaluate.py",
        "fast_de.py",
        "models.py",
        "train.py",
        "io.py",
        "distribution.py",
    }
    if (
        source.get("protocol") != PROTOCOL
        or source.get("test_used") is not True
        or ready.get("test_used") is not True
        or source.get("selection_sha256") != selection_sha
        or source.get("download_receipt_sha256") != file_sha256(root / DOWNLOAD_RECEIPT)
        or source.get("raw_files") != download["files"]
        or ready.get("source_sha256") != file_sha256(source_path)
        or set(source.get("baseline_sources", {})) != baseline_names
        or set(source.get("scoring_sources", {})) != scoring_names
    ):
        raise ValueError("Final test cache lineage no longer binds its frozen sources")
    for key, directory in [
        ("baseline_sources", "data/processed/baselines-v1"),
        ("scoring_sources", "src/fusionvc"),
    ]:
        for name, expected in source[key].items():
            if file_sha256(root / directory / name) != expected:
                raise ValueError(f"Final test source changed after evaluation: {directory}/{name}")
    controls = (root / "data/processed/baselines-v1/controls.npy").stat()
    matrix = (folder / "test_log.npy").stat()
    if (
        source["training_controls"] != {"bytes": controls.st_size, "mtime_ns": controls.st_mtime_ns}
        or ready["matrix_bytes"] != matrix.st_size
        or ready["matrix_mtime_ns"] != matrix.st_mtime_ns
        or ready["shape"][1] != 18080
        or len(ready["perturbations"]) != 100
        or ready["control_cell_count"] <= 8
        or ready["shape"][0] != ready["perturbed_cell_count"] + ready["control_cell_count"]
    ):
        raise ValueError("Final test cache/control matrix metadata changed")
    return {
        "selection_sha256": selection_sha,
        "source_sha256": file_sha256(source_path),
        "real_cache_manifest_sha256": file_sha256(ready_path),
        "download_receipt_sha256": file_sha256(root / DOWNLOAD_RECEIPT),
        "all_checks_passed": True,
        "test_used": True,
        "scope": "current raw-download receipt and matrix stats, baseline and scoring source hashes; no test-expression reread",
    }


def inspect_test_metadata(root: Path, reader: CountsReader) -> dict:
    """Audit actual full shape and controls against the official perturbation CSV."""
    genes = read_json(root / "data/processed/baselines-v1/genes.json")
    if len(genes) != 18080 or len(set(genes)) != 18080 or reader.genes != genes:
        raise ValueError("Test must retain all 18,080 genes in the original output order")
    labels = np.asarray(reader.categories)[reader.codes]
    names, counts = np.unique(labels, return_counts=True)
    observed = dict(zip(names.tolist(), counts.tolist()))
    with (root / "data/raw/vcc2025/test/pert_counts_Test.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    expected = {row["target_gene"]: int(row["n_cells"]) for row in rows}
    if (
        len(rows) != 100
        or len(expected) != 100
        or CONTROL in expected
        or not set(expected) <= set(genes)
    ):
        raise ValueError("Expected the 100 official test perturbation counts")
    if set(observed) != set(expected) | {CONTROL} or any(
        observed[name] != count for name, count in expected.items()
    ):
        raise ValueError("Actual test labels/counts differ from the official CSV or lack controls")
    split = read_json(root / "configs/splits/vcc2025.json")
    if set(expected) & (set(split["train"]) | set(split["validation"])):
        raise ValueError("Test perturbations must be disjoint from training and validation")
    if min(observed.values()) <= 8 or reader.shape != (len(labels), len(genes)):
        raise ValueError("Actual test shape/groups cannot use the validated asymptotic DE path")
    return {
        "shape": list(reader.shape),
        "names": sorted(observed),
        "target_counts": observed,
        "perturbations": sorted(expected),
        "perturbed_cell_count": sum(expected.values()),
        "control_cell_count": observed[CONTROL],
        "test_used": True,
    }


def close_memmap(matrix) -> None:
    mapping = getattr(matrix, "_mmap", None)
    if mapping is not None:
        mapping.close()


def prepare_test_cache(root: Path, selection_path: Path):
    """Create/reuse the real test matrix and DE, isolated from validation artifacts."""
    _, selection_sha = validate_selection(root, selection_path)
    download = verified_test_download(root, selection_sha)
    folder = root / "outputs/final-test-cache"
    folder.mkdir(parents=True, exist_ok=True)
    source = {
        "protocol": PROTOCOL,
        "selection_sha256": selection_sha,
        "download_receipt_sha256": file_sha256(root / DOWNLOAD_RECEIPT),
        "raw_files": download["files"],
        "baseline_sources": {
            name: file_sha256(root / "data/processed/baselines-v1" / name)
            for name in [
                "manifest.json",
                "genes.json",
                "gene_descriptors.npy",
                "input_gene_indices.npy",
            ]
        },
        "scoring_sources": {
            name: file_sha256(root / "src/fusionvc" / name)
            for name in [
                "final_evaluation.py",
                "evaluate.py",
                "fast_de.py",
                "models.py",
                "train.py",
                "io.py",
                "distribution.py",
            ]
        },
        "training_controls": {
            "bytes": (root / "data/processed/baselines-v1/controls.npy").stat().st_size,
            "mtime_ns": (root / "data/processed/baselines-v1/controls.npy").stat().st_mtime_ns,
        },
        "normalization": "median positive library size over the complete test matrix; log1p",
        "test_used": True,
    }
    signature_path = folder / "source.json"
    if signature_path.exists() and read_json(signature_path) != source:
        raise ValueError(
            "Final test cache sources changed; preserve results and use a new protocol/cache"
        )
    atomic_json(signature_path, source)
    reader = CountsReader(root / "data/raw/vcc2025/test/adata_Test.h5ad")
    try:
        metadata = inspect_test_metadata(root, reader)
        labels = np.asarray(reader.categories)[reader.codes]
        names, genes = metadata["names"], reader.genes
        matrix_path = folder / "test_log.npy"
        ready_path = folder / "real-ready.json"
        dense_bytes = math.prod(reader.shape) * 4 + 1024
        if not ready_path.exists():
            temporary = folder / "test_log.partial.npy"
            temporary.unlink(missing_ok=True)
            # An interrupted rename without its ready marker is not a complete cache.
            matrix_path.unlink(missing_ok=True)
            # Both the reusable matrix and one temporary prediction must fit.
            ensure_space(root, dense_bytes * 2)
            libraries = library_sizes(reader, 512)
            positive = libraries[libraries > 0]
            if not len(positive) or not np.isfinite(libraries).all():
                raise ValueError("Invalid test library sizes")
            scale = float(np.median(positive))
            matrix = np.lib.format.open_memmap(
                temporary, mode="w+", dtype=np.float32, shape=reader.shape
            )
            try:
                for start in range(0, reader.shape[0], 512):
                    stop = min(start + 512, reader.shape[0])
                    block = reader.block(start, stop)
                    if np.any(~np.isfinite(block.data)) or np.any(block.data < 0):
                        raise ValueError("Test expression contains nonfinite or negative counts")
                    matrix[start:stop] = normalize_log(block, scale).toarray()
                matrix.flush()
            finally:
                close_memmap(matrix)
            temporary.replace(matrix_path)
            atomic_json(
                ready_path,
                {
                    **metadata,
                    "target_sum": scale,
                    "source_sha256": file_sha256(signature_path),
                    "matrix_bytes": matrix_path.stat().st_size,
                    "matrix_mtime_ns": matrix_path.stat().st_mtime_ns,
                },
            )
        ready = read_json(ready_path)
        if (
            any(ready.get(key) != value for key, value in metadata.items())
            or ready["source_sha256"] != file_sha256(signature_path)
            or ready["matrix_bytes"] != matrix_path.stat().st_size
            or ready["matrix_mtime_ns"] != matrix_path.stat().st_mtime_ns
            or not np.isfinite(ready["target_sum"])
            or ready["target_sum"] <= 0
        ):
            raise ValueError("Completed final test cache metadata changed")
    finally:
        reader.close()
    matrix = np.load(matrix_path, mmap_mode="r")
    if matrix.shape != tuple(ready["shape"]) or matrix.dtype != np.float32:
        close_memmap(matrix)
        raise ValueError("Incomplete final test matrix")
    try:
        if not (folder / "real_means.npy").exists():
            np.save(folder / "real_means.npy", grouped_means(matrix, labels, names))
        if not (folder / "real_de.parquet").exists():
            differential_expression(matrix, labels, genes, folder / "real-de-blocks").write_parquet(
                folder / "real_de.parquet"
            )
        means = np.load(folder / "real_means.npy")
        if means.shape != (len(names), len(genes)) or not np.isfinite(means).all():
            raise ValueError("Invalid cached test pseudobulk means")
        return (
            matrix,
            labels,
            names,
            genes,
            means,
            pl.read_parquet(folder / "real_de.parquet"),
            ready,
        )
    except Exception:
        close_memmap(matrix)
        raise


def control_sampling_plan(labels, names, control_count: int) -> dict:
    """Exactly the existing sorted-name, seed-2025 control sampling protocol."""
    if list(names) != sorted(set(names)) or set(names) != set(np.unique(labels)):
        raise ValueError("Control sampling requires all sorted, distinct test group names")
    rng = np.random.default_rng(2025)
    return {
        name: rng.choice(
            control_count,
            int(np.count_nonzero(labels == name)),
            replace=np.count_nonzero(labels == name) > control_count,
        )
        for name in names
    }


def load_selected_predictor(root, row, genes, device):
    """Load ordinary or distilled MLP checkpoints and verify immutable inputs."""
    if row["kind"] == "no_perturbation":
        return None, None, read_json(root / "data/processed/baselines-v1/manifest.json")
    path = project_path(root, row["checkpoint_path"])
    if file_sha256(path) != row["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint changed before prediction")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model, manifest, model_genes, features, controls = load_inputs(
        root, checkpoint["config"], device
    )
    close_memmap(controls)
    if (
        model_genes != genes
        or checkpoint["preprocessing"] != manifest
        or checkpoint["seed"] != row["seed"]
    ):
        raise ValueError("Selected checkpoint gene/preprocessing/seed provenance differs")
    if {**checkpoint["config"], "seed": row["seed"]} != read_json(path.parent / "config.json"):
        raise ValueError("Selected checkpoint and original run configuration differ")
    summary = read_json(path.parent / "summary.json")
    if (
        summary["checkpoint_sha256"] != row["checkpoint_sha256"]
        or summary["best_epoch"] != checkpoint["epoch"]
    ):
        raise ValueError("Selected checkpoint and original training summary differ")
    if row["model"] == "distillation":
        if (
            checkpoint["config"].get("model") != "mlp"
            or checkpoint["config"].get("protocol") != "distillation-v1"
        ):
            raise ValueError("The distilled student must use the original MLP architecture")
        current_distillation = read_json(root / "data/processed/distillation-v1/manifest.json")
        if not checkpoint.get("distillation_cache") or not (
            checkpoint["distillation_cache"]
            == summary.get("distillation_cache")
            == current_distillation
        ):
            raise ValueError("The student checkpoint and original teacher-label provenance differ")
    elif checkpoint["config"].get("model") != row["model"]:
        raise ValueError("Selected architecture differs from checkpoint configuration")
    if checkpoint["config"].get("variant") != row["variant"]:
        raise ValueError("Selected loss differs from checkpoint configuration")
    if (
        hasattr(model, "feature_manifest")
        and checkpoint.get("frozen_features") != model.feature_manifest
    ):
        raise ValueError("Selected checkpoint frozen feature provenance differs")
    for name, value in model.named_buffers():
        if name not in checkpoint["state_dict"] or not torch.equal(
            value, checkpoint["state_dict"][name]
        ):
            raise ValueError(f"Selected checkpoint buffer differs from fixed inputs: {name}")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    if not all(
        torch.isfinite(v).all() for v in model.state_dict().values() if v.is_floating_point()
    ):
        raise ValueError("Selected checkpoint contains nonfinite tensors")
    return model.eval(), features, manifest


@torch.no_grad()
def predict_final(root, folder, row, labels, names, genes, test_scale, device=None):
    device = torch.device("cuda") if device is None else device
    model, features, fitted = load_selected_predictor(root, row, genes, device)
    controls = np.load(root / "data/processed/baselines-v1/controls.npy", mmap_mode="r")
    plan = control_sampling_plan(labels, names, len(controls))
    prediction_path = folder / "predictions.npy"
    needed = max(
        0,
        len(labels) * len(genes) * 4
        + 1024
        - (prediction_path.stat().st_size if prediction_path.exists() else 0),
    )
    ensure_space(root, needed)
    prediction = np.lib.format.open_memmap(
        prediction_path, mode="w+", dtype=np.float32, shape=(len(labels), len(genes))
    )
    indices = {gene: i for i, gene in enumerate(genes)}
    started = time.perf_counter()
    try:
        for target in names:
            rows = np.flatnonzero(labels == target)
            delta = (
                torch.zeros((1, len(genes)), device=device)
                if model is None or target == CONTROL
                else model(torch.from_numpy(features[indices[target]][None]).to(device))
            )
            for start in range(0, len(rows), 64):
                stop = min(start + 64, len(rows))
                control = torch.from_numpy(np.array(controls[plan[target][start:stop]])).to(device)
                values = apply_residual(control, delta, fitted["target_sum"])[0]
                values = torch.log1p(values.expm1() * (test_scale / fitted["target_sum"]))
                if not torch.isfinite(values).all():
                    raise FloatingPointError("Nonfinite final test prediction")
                prediction[rows[start:stop]] = values.cpu().numpy()
            print(f"FINAL PRED {row['label']} {target}: {len(rows)} cells", flush=True)
        prediction.flush()
        np.savez_compressed(folder / "control_sampling.npz", **plan)
        atomic_json(
            folder / "prediction-info.json",
            {
                "shape": list(prediction.shape),
                "genes": len(genes),
                "target_sum": test_scale,
                "control_sampling_seed": 2025,
                "control_counts_match_official": True,
                "seconds": time.perf_counter() - started,
                "test_used": True,
            },
        )
        return prediction
    except Exception:
        close_memmap(prediction)
        raise
    finally:
        close_memmap(controls)
        del model
        torch.cuda.empty_cache()


def publish_final_run(root: Path, label: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", label):
        raise ValueError("Unsafe final test label")
    folder = root / "outputs/final-test" / label
    small = root / "results/final-test" / label
    sources = [
        ("metrics.json", "metrics.json"),
        ("results.csv", "per_perturbation.csv"),
        ("distribution_per_perturbation.csv", "distribution_per_perturbation.csv"),
        ("prediction-info.json", "prediction-info.json"),
        ("run-source.json", "run-source.json"),
        ("control_sampling.npz", "control_sampling.npz"),
    ]
    for source, _ in sources:
        if not (folder / source).is_file():
            raise FileNotFoundError(f"Incomplete final test publication: {folder / source}")
    small.mkdir(parents=True, exist_ok=True)
    for source, destination in sources:
        shutil.copy2(folder / source, small / destination)
    (folder / "predictions.npy").unlink(missing_ok=True)


def evaluate_final_run(root: Path, selection_path: Path, label: str) -> dict:
    selection, selection_sha = validate_selection(root, selection_path)
    selected = [row for row in selection["runs"] if row["label"] == label]
    if len(selected) != 1:
        raise ValueError("Only a frozen selected label may be evaluated")
    row = selected[0]
    for package, expected in [("cell-eval", "0.6.6"), ("pdex", "0.1.27")]:
        if importlib.metadata.version(package) != expected:
            raise RuntimeError(f"Final test requires pinned {package} {expected}")
    verified_test_download(root, selection_sha)
    seed_all(2025)
    started = time.perf_counter()
    real, labels, names, genes, real_means, real_de, ready = prepare_test_cache(
        root, selection_path
    )
    folder = root / "outputs/final-test" / label
    folder.mkdir(parents=True, exist_ok=True)
    signature = {
        "protocol": PROTOCOL,
        "selection_sha256": selection_sha,
        "run": row,
        "real_cache_manifest_sha256": file_sha256(
            root / "outputs/final-test-cache/real-ready.json"
        ),
        "test_used": True,
    }
    source_path = folder / "run-source.json"
    if source_path.exists() and read_json(source_path) != signature:
        close_memmap(real)
        raise ValueError("Final run sources changed; never reuse old predictions/DE blocks")
    atomic_json(source_path, signature)
    metrics_path = folder / "metrics.json"
    if metrics_path.exists():
        close_memmap(real)
        metrics = read_json(metrics_path)
        if metrics.get("signature") != signature:
            raise ValueError("Completed final run provenance changed")
        publish_final_run(root, label)
        print(f"FINAL already complete: {label}", flush=True)
        return metrics
    if not torch.cuda.is_available():
        close_memmap(real)
        raise RuntimeError("Use the existing experiment GPU for complete final prediction")
    prediction = None
    try:
        prediction = predict_final(root, folder, row, labels, names, genes, ready["target_sum"])
        from .distribution import distribution_scores

        distribution_rows = distribution_scores(prediction, real, labels, names, 64)
        pd.DataFrame(distribution_rows).to_csv(
            folder / "distribution_per_perturbation.csv", index=False
        )
        pred_de = differential_expression(prediction, labels, genes, folder / "de-blocks")
        pred_de.write_parquet(folder / "pred_de.parquet")
        pred_means = grouped_means(prediction, labels, names)
        np.save(folder / "pred_means.npy", pred_means)
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
        if not set(METRICS) <= set(scores.columns) or scores.height != 100:
            raise ValueError("Final official scores must cover all 100 test perturbations")
        scored = scores.to_pandas()
        if (
            set(scored.perturbation) != set(ready["perturbations"])
            or scored.perturbation.duplicated().any()
        ):
            raise ValueError("Final official scores have missing or duplicate test targets")
        values = {metric: float(scored[metric].mean()) for metric in METRICS}
        if not np.isfinite(scored[METRICS].to_numpy()).all():
            raise FloatingPointError("Nonfinite final official metric")
        metrics = {
            "protocol": PROTOCOL,
            "split": "official test",
            "test_used": True,
            "label": label,
            "model": row["model"],
            "role": row["role"],
            "seed": row.get("seed"),
            "variant": row.get("variant"),
            "metrics": values,
            "checkpoint_sha256": row.get("checkpoint_sha256"),
            "gene_count": len(genes),
            "predicted_cell_count": len(labels),
            "perturbation_count": 100,
            "control_cell_count": ready["control_cell_count"],
            "profile": "vcc",
            "cell_eval_version": "0.6.6",
            "pdex_version": "0.1.27",
            "de": "validated asymptotic Mann-Whitney acceleration",
            "global_fdr": "BH over all genes and all test perturbations",
            "evaluation_seconds": time.perf_counter() - started,
            "distribution": {
                "energy_mmd_rms": float(
                    np.mean([item["energy_mmd_rms"] for item in distribution_rows])
                ),
                "cells_per_set": 64,
                "prediction_seed": 2026,
                "observation_seed": 2027,
                "kernel": "negative Euclidean distance divided by sqrt(gene_count)",
                "scope": "supplementary sampled distribution metric; not an official VCC score",
            },
            "signature": signature,
        }
        atomic_json(metrics_path, metrics)
    finally:
        if prediction is not None:
            close_memmap(prediction)
        close_memmap(real)
        gc.collect()
    publish_final_run(root, label)
    print(f"FINAL COMPLETE {label}: {values}", flush=True)
    return metrics
