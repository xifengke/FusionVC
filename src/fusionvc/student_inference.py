"""Small standalone student package: no teacher, raw data or encoder assets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .io import file_sha256, read_json, write_json
from .models import ResidualPredictor, apply_residual


def export_student_bundle(root: Path, run: Path, destination: Path, control_cells=256) -> dict:
    if control_cells < 1:
        raise ValueError("A student demo needs a positive control-bank size")
    checkpoint_path = run / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    if config.get("model") != "mlp" or config.get("protocol") != "distillation-v1":
        raise ValueError("Only a formally distilled MLP may be exported as the student")
    source_hash = file_sha256(checkpoint_path)
    ready = destination / "manifest.json"
    if ready.exists():
        saved = read_json(ready)
        if (
            saved["source_checkpoint_sha256"] != source_hash
            or saved["control_cells"] != control_cells
        ):
            raise ValueError(
                "An existing bundle has different sources; choose another output directory"
            )
        load_student_bundle(destination, "cpu")
        return saved
    base = root / "data/processed/baselines-v1"
    source_hashes = {
        k.replace("\\", "/"): v
        for k, v in checkpoint["distillation_cache"]["signature"]["source_sha256"].items()
    }
    for name in ["genes.json", "gene_descriptors.npy"]:
        relative = f"data/processed/baselines-v1/{name}"
        if file_sha256(base / name) != source_hashes[relative]:
            raise ValueError("Student deployment features differ from the training source")
    genes = read_json(base / "genes.json")
    features = np.load(base / "gene_descriptors.npy")
    selected = np.load(base / "input_gene_indices.npy")
    if not torch.equal(
        torch.from_numpy(features[selected]), checkpoint["state_dict"]["context_features"]
    ):
        raise ValueError("Student input identities differ from the saved model")
    controls = np.load(base / "controls.npy", mmap_mode="r")
    if control_cells > len(controls):
        raise ValueError("Demo control bank cannot exceed available training controls")
    chosen = np.sort(np.random.default_rng(2025).choice(len(controls), control_cells, False))
    bank = np.array(controls[chosen])
    if (
        features.shape[0] != len(genes)
        or bank.shape != (control_cells, len(genes))
        or not np.isfinite(bank).all()
    ):
        raise ValueError("Invalid student deployment gene or control dimensions")
    destination.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": config,
            "state_dict": checkpoint["state_dict"],
            "target_sum": checkpoint["preprocessing"]["target_sum"],
        },
        destination / "student.pt",
    )
    np.save(destination / "target_features.npy", features)
    np.save(destination / "controls.npy", bank)
    np.save(destination / "training_control_indices.npy", chosen)
    write_json(destination / "genes.json", genes)
    files = [
        "student.pt",
        "target_features.npy",
        "controls.npy",
        "training_control_indices.npy",
        "genes.json",
    ]
    manifest = {
        "protocol": "student-inference-v1",
        "source_checkpoint_sha256": source_hash,
        "genes": len(genes),
        "control_cells": control_cells,
        "target_sum": checkpoint["preprocessing"]["target_sum"],
        "files": {name: file_sha256(destination / name) for name in files},
        "total_bytes": sum((destination / name).stat().st_size for name in files),
        "inference_dependencies": "FusionVC code, PyTorch, NumPy and these five local bundle files; no teacher, foundation encoders, original training matrix or teacher delta cache",
        "sampling_scope": "fixed subset of training controls for a standalone demo; not the full control bank used in official evaluation",
    }
    write_json(ready, manifest)
    return manifest


def load_student_bundle(folder: Path, device="cpu"):
    manifest = read_json(folder / "manifest.json")
    if manifest["protocol"] != "student-inference-v1":
        raise ValueError("Unknown student inference package")
    required = {
        "student.pt",
        "target_features.npy",
        "controls.npy",
        "training_control_indices.npy",
        "genes.json",
    }
    if set(manifest["files"]) != required:
        raise ValueError("Student inference package file list is incomplete")
    for name, digest in manifest["files"].items():
        if file_sha256(folder / name) != digest:
            raise ValueError(f"Student inference package changed: {name}")
    checkpoint = torch.load(folder / "student.pt", map_location="cpu", weights_only=True)
    state = checkpoint["state_dict"]
    genes = read_json(folder / "genes.json")
    features = np.load(folder / "target_features.npy")
    controls = np.load(folder / "controls.npy", mmap_mode="r")
    if (
        len(genes) != manifest["genes"]
        or features.shape[0] != len(genes)
        or controls.shape != (manifest["control_cells"], len(genes))
        or checkpoint["target_sum"] != manifest["target_sum"]
        or checkpoint["config"].get("model") != "mlp"
        or checkpoint["config"].get("protocol") != "distillation-v1"
        or not np.isfinite(features).all()
    ):
        raise ValueError("Student bundle dimensions/configuration changed")
    model = ResidualPredictor(
        checkpoint["config"], state["context_features"], state["context_expression"], len(genes)
    )
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False).to(torch.device(device)).eval()
    return model, genes, features, controls, manifest


@torch.no_grad()
def predict_student_bundle(folder: Path, target: str, cells=32, seed=2025, device="cpu"):
    if cells < 1:
        raise ValueError("Requested cell count must be positive")
    model, genes, features, controls, manifest = load_student_bundle(folder, device)
    if target not in genes:
        raise ValueError(f"Unknown official gene symbol: {target}")
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(controls), cells, replace=cells > len(controls))
    selected = torch.from_numpy(np.array(controls[indices])).to(device)
    query = torch.from_numpy(features[genes.index(target)][None]).to(device)
    delta = model(query)
    result = apply_residual(selected, delta, manifest["target_sum"])[0].cpu().numpy()
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite standalone student prediction")
    return (
        result,
        genes,
        {
            "target": target,
            "cells": cells,
            "seed": seed,
            "expression_space": "continuous log1p expression at training library scale",
            "target_sum": manifest["target_sum"],
            "source_checkpoint_sha256": manifest["source_checkpoint_sha256"],
            "scope": manifest["sampling_scope"],
        },
    )
