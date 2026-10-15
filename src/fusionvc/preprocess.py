"""Fit inputs on training controls, collect unpaired labels by perturbation."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.utils.extmath import randomized_svd
from threadpoolctl import threadpool_limits

from .io import CountsReader, normalize_log, read_json, write_json

CONTROL = "non-targeting"


def library_sizes(reader: CountsReader, block_size: int) -> np.ndarray:
    values = np.empty(reader.shape[0], dtype=np.float64)
    for start in range(0, len(values), block_size):
        stop = min(start + block_size, len(values))
        values[start:stop] = np.asarray(reader.block(start, stop).sum(axis=1)).ravel()
    return values


def collect(
    reader: CountsReader, target_sum: float, out: Path, controls_path: Path | None, block_size: int
) -> dict:
    n, g = reader.shape
    sums = np.zeros((len(reader.categories), g), dtype=np.float64)
    squares = np.zeros_like(sums)
    counts = np.bincount(reader.codes, minlength=len(reader.categories))
    ctrl_code = reader.categories.index(CONTROL)
    controls = None
    if controls_path is not None:
        controls = np.lib.format.open_memmap(
            controls_path, mode="w+", dtype="float32", shape=(int(counts[ctrl_code]), g)
        )
    pos = 0
    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        x = normalize_log(reader.block(start, stop), target_sum)
        codes = reader.codes[start:stop]
        grouping = csr_matrix(
            (np.ones(len(codes), dtype=np.float32), (codes, np.arange(len(codes)))),
            shape=(len(counts), len(codes)),
        )
        sums += (grouping @ x).toarray()
        squared = x.copy()
        squared.data **= 2
        squares += (grouping @ squared).toarray()
        if controls is not None:
            selected = x[codes == ctrl_code].toarray()
            controls[pos : pos + len(selected)] = selected
            pos += len(selected)
        if start % (block_size * 16) == 0:
            print(f"PREP {out.stem}: {stop}/{n} cells", flush=True)
    means = (sums / counts[:, None]).astype(np.float32)
    variances = np.maximum(squares / counts[:, None] - means.astype(np.float64) ** 2, 0)
    if controls is not None:
        controls.flush()
        assert pos == len(controls)
    np.savez_compressed(
        out,
        names=np.array(reader.categories),
        counts=counts,
        mean=means,
        variance=variances.astype(np.float32),
    )
    return {
        "cells": n,
        "genes": g,
        "control_cells": int(counts[ctrl_code]),
        "perturbations": len(counts) - 1,
    }


def prepare(
    root: Path,
    block_size: int = 512,
    tokens: int = 256,
    components: int = 32,
    feature_cells: int = 2048,
    seed: int = 2025,
) -> None:
    out = root / "data/processed/baselines-v1"
    manifest = out / "manifest.json"
    if manifest.exists():
        previous = read_json(manifest)
        expected = (tokens, components, feature_cells, seed)
        actual = tuple(previous[k] for k in ["tokens", "components", "feature_cells", "seed"])
        if actual != expected:
            raise ValueError("Existing preprocessing differs. Use a new cache version.")
        print("PREP already complete; using recorded cache", flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    train = CountsReader(root / "data/raw/vcc2025/train/adata_Training.h5ad")
    val = CountsReader(root / "data/raw/vcc2025/validation/adata_Validation.h5ad")
    try:
        assert train.genes == val.genes
        tr = set(train.categories) - {CONTROL}
        va = set(val.categories) - {CONTROL}
        assert not tr & va, "Perturbation-label leakage"
        assert (tr | va) <= set(train.genes), "Missing target descriptor"
        print("PREP library sizes", flush=True)
        train_lib = library_sizes(train, block_size)
        val_lib = library_sizes(val, block_size)
        scale = float(np.median(train_lib[train_lib > 0]))
        val_scale = float(np.median(val_lib[val_lib > 0]))
        train_meta = collect(
            train, scale, out / "train_stats.npz", out / "controls.npy", block_size
        )
        # Validation labels are stored only for loss/metrics, never fitted into inputs.
        val_meta = collect(val, scale, out / "validation_stats.npz", None, block_size)
        stats = np.load(out / "train_stats.npz")
        ctrl_idx = list(stats["names"]).index(CONTROL)
        mean, var = stats["mean"][ctrl_idx], stats["variance"][ctrl_idx]
        hvg = np.argsort(-var, kind="stable")[:tokens]
        controls = np.load(out / "controls.npy", mmap_mode="r")
        rng = np.random.default_rng(seed)
        chosen = np.sort(rng.choice(len(controls), min(feature_cells, len(controls)), False))
        std = np.sqrt(var + 1e-4)
        profile = (np.asarray(controls[chosen]) - mean) / std
        with threadpool_limits(limits=4):
            _, _, vt = randomized_svd(profile, n_components=components, n_iter=4, random_state=seed)
        # Gene loadings are descriptors, not a table of trainable target IDs.
        loadings = vt.T * np.sqrt(len(train.genes))
        descriptors = np.column_stack(
            [loadings, mean / (mean.std() + 1e-6), std / (std.std() + 1e-6)]
        ).astype(np.float32)
        np.save(out / "gene_descriptors.npy", descriptors)
        np.save(out / "input_gene_indices.npy", hvg)
        write_json(out / "genes.json", train.genes)
        shared_controls = set(
            np.array(train.cell_ids)[train.codes == train.categories.index(CONTROL)]
        )
        shared_controls &= set(np.array(val.cell_ids)[val.codes == val.categories.index(CONTROL)])
        split = {
            "train": sorted(tr),
            "validation": sorted(va),
            "test": "not loaded; reserved for final eight-model comparison",
            "control": CONTROL,
        }
        write_json(root / "configs/splits/vcc2025.json", split)
        write_json(
            manifest,
            {
                "version": "baselines-v1",
                "target_sum": scale,
                "validation_eval_target_sum": val_scale,
                "tokens": tokens,
                "components": components,
                "feature_cells": feature_cells,
                "seed": seed,
                "input_fit_source": "training non-targeting cells only",
                "shared_control_cell_ids": len(shared_controls),
                "train": train_meta,
                "validation": val_meta,
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        print(f"PREP complete; normalization scale={scale}, eval scale={val_scale}", flush=True)
    finally:
        train.close()
        val.close()
