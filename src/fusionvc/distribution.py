"""Conditional, unpaired cell-set comparison using the State energy kernel."""

from __future__ import annotations

import math
from pathlib import Path
import shutil

import numpy as np
import torch

from .io import CountsReader, normalize_log, read_json, write_json
from .preprocess import CONTROL


def energy_mmd(prediction: torch.Tensor, observation: torch.Tensor) -> torch.Tensor:
    """Biased empirical MMD^2 with k(x,y)=-||x-y||/sqrt(number of genes).

    Inputs are [conditions, cells, genes], or a single [cells, genes] set.
    Conditions are compared separately and then weighted equally. The RMS
    scaling is a constant kernel rescaling, not a per-cell normalization.
    Diagonals are included in the V-statistic and their distance is exactly zero.
    """
    if prediction.ndim == 2 and observation.ndim == 2:
        prediction, observation = prediction[None], observation[None]
    if prediction.ndim != 3 or observation.ndim != 3:
        raise ValueError("Expected [conditions, cells, genes] cell sets")
    if prediction.shape[0] != observation.shape[0] or prediction.shape[2] != observation.shape[2]:
        raise ValueError("Conditions and output genes must match")
    if min(*prediction.shape, *observation.shape) < 1:
        raise ValueError("Empty cell sets are not supported")
    if prediction.dtype != observation.dtype or prediction.device != observation.device:
        raise ValueError("Cell sets must have the same dtype and device")
    if not prediction.is_floating_point():
        raise ValueError("Cell expressions must be floating point")
    # Matrix-based cdist avoids a [conditions,cells,cells,genes] backward buffer.
    # Mask self diagonals to remove cancellation error in Euclidean distances.
    kwargs = {"p": 2, "compute_mode": "use_mm_for_euclid_dist"}
    pp = torch.cdist(prediction, prediction, **kwargs)
    oo = torch.cdist(observation, observation, **kwargs)
    po = torch.cdist(prediction, observation, **kwargs)
    pred_diagonal = torch.eye(pp.shape[-1], dtype=torch.bool, device=pp.device)
    obs_diagonal = torch.eye(oo.shape[-1], dtype=torch.bool, device=oo.device)
    pp = pp.masked_fill(pred_diagonal, 0)
    oo = oo.masked_fill(obs_diagonal, 0)
    value = 2 * po.mean((-2, -1)) - pp.mean((-2, -1)) - oo.mean((-2, -1))
    # The V-statistic is nonnegative; clamp only possible floating-point negatives.
    return (value / math.sqrt(prediction.shape[-1])).clamp_min(0).mean()


def cache_signature(root: Path, reader: CountsReader, scale: float) -> dict:
    source = root / "data/raw/vcc2025/train/adata_Training.h5ad"
    stat = source.stat()
    return {
        "version": "mmd-v1",
        "shape": list(reader.shape),
        "target_sum": scale,
        "source": "training cells only",
        "source_bytes": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "dtype": "float32",
    }


def prepare_training_cell_cache(root: Path, block_size: int = 512) -> None:
    """Reusable, exact float32 log-expression cache; never read validation cells."""
    cache = root / "data/processed/mmd-v1"
    cache.mkdir(parents=True, exist_ok=True)
    fitted = read_json(root / "data/processed/baselines-v1/manifest.json")
    reader = CountsReader(root / "data/raw/vcc2025/train/adata_Training.h5ad")
    try:
        expected = cache_signature(root, reader, fitted["target_sum"])
        ready = cache / "training-ready.json"
        if ready.exists():
            if read_json(ready) != expected:
                raise ValueError("Training cache differs from source or normalization")
            saved = np.load(cache / "training_cells.npy", mmap_mode="r")
            if saved.shape != reader.shape or saved.dtype != np.float32:
                raise ValueError("Incomplete training cell cache")
            saved._mmap.close()
            print("PREP-MMD reusing exact training cell cache", flush=True)
            return
        required = math.prod(reader.shape) * 4
        if shutil.disk_usage(root).free < required + 18 * 10**9:
            raise RuntimeError("Training cache requires its full size plus 18GB evaluation reserve")
        temporary = cache / "training_cells.partial.npy"
        saved = np.lib.format.open_memmap(
            temporary, mode="w+", dtype=np.float32, shape=reader.shape
        )
        try:
            for start in range(0, reader.shape[0], block_size):
                stop = min(start + block_size, reader.shape[0])
                saved[start:stop] = normalize_log(
                    reader.block(start, stop), fitted["target_sum"]
                ).toarray()
                if start % (block_size * 32) == 0:
                    print(f"PREP-MMD cells {stop}/{reader.shape[0]}", flush=True)
            saved.flush()
        finally:
            saved._mmap.close()
        if cache_signature(root, reader, fitted["target_sum"]) != expected:
            raise ValueError("Training source changed during cache preparation")
        temporary.replace(cache / "training_cells.npy")
        write_json(ready, expected)
        print(
            f"PREP-MMD complete: {required / 10**9:.2f}GB; all training cells retained", flush=True
        )
    finally:
        reader.close()


class TrainingCellSampler:
    """Sample actual cells from training targets; never open held-out labels."""

    def __init__(self, root: Path, genes: list[str], names: list[str], scale: float):
        self.reader = CountsReader(root / "data/raw/vcc2025/train/adata_Training.h5ad")
        try:
            if self.reader.genes != genes:
                raise ValueError("Training cell output genes differ from fitted inputs")
            if CONTROL in names or not set(names) <= set(self.reader.categories):
                raise ValueError("Only observed training perturbations can supply cells")
            self.rows = {
                name: np.flatnonzero(self.reader.codes == self.reader.categories.index(name))
                for name in names
            }
            if any(len(rows) == 0 for rows in self.rows.values()):
                raise ValueError("Training target has no cells")
            self.scale = scale
            self.cache = None
            ready = root / "data/processed/mmd-v1/training-ready.json"
            if ready.exists():
                if read_json(ready) != cache_signature(root, self.reader, scale):
                    raise ValueError("Training cell cache source or scale mismatch")
                self.cache = np.load(ready.parent / "training_cells.npy", mmap_mode="r")
                if self.cache.shape != self.reader.shape or self.cache.dtype != np.float32:
                    raise ValueError("Training cell cache shape or dtype mismatch")
        except Exception:
            self.reader.close()
            raise

    def sample(self, names: list[str], cells: int, rng: np.random.Generator) -> np.ndarray:
        if cells < 1:
            raise ValueError("Cell-set size must be positive")
        chosen = [
            rng.choice(self.rows[name], cells, replace=len(self.rows[name]) < cells)
            for name in names
        ]
        selected = np.concatenate(chosen)
        if self.cache is None:
            raw = self.reader.rows(selected)
            log = normalize_log(raw, self.scale).toarray()
        else:
            log = np.array(self.cache[selected])
        return log.reshape(len(names), cells, len(self.reader.genes))

    def close(self) -> None:
        self.reader.close()
        if self.cache is not None:
            self.cache._mmap.close()


@torch.no_grad()
def distribution_scores(prediction, observation, labels, names, cells: int = 64) -> list[dict]:
    """Fixed, independent cell-set samples on the evaluator's expression scale."""
    rng_pred = np.random.default_rng(2026)
    rng_real = np.random.default_rng(2027)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for name in names:
        if name == CONTROL:
            continue
        indices = np.flatnonzero(labels == name)
        pred_indices = rng_pred.choice(indices, cells, replace=len(indices) < cells)
        real_indices = rng_real.choice(indices, cells, replace=len(indices) < cells)
        p = torch.from_numpy(np.array(prediction[pred_indices], dtype=np.float32)).to(device)
        o = torch.from_numpy(np.array(observation[real_indices], dtype=np.float32)).to(device)
        rows.append({"perturbation": name, "energy_mmd_rms": energy_mmd(p, o).item()})
    return rows
