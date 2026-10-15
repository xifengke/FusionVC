"""Bounded-memory DE with pdex 0.1.27 semantics and validated reference parity.

Reuse sorted controls. Two-sided asymptotic U, tie/continuity correction, pooled BH.
Groups requiring exact tests are rejected. No experimental integer approximation.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from numba import njit, prange, set_num_threads
from scipy.stats import false_discovery_control

from .preprocess import CONTROL


@njit(cache=True, parallel=True)
def reference_ties(sorted_ref):
    result = np.zeros(len(sorted_ref), dtype=np.float64)
    for gene in prange(len(sorted_ref)):
        x = sorted_ref[gene]
        pos, total = 0, 0.0
        while pos < len(x):
            end = pos + 1
            while end < len(x) and x[end] == x[pos]:
                end += 1
            n = float(end - pos)
            total += n * (n * n - 1)
            pos = end
        result[gene] = total
    return result


@njit(cache=True, parallel=True)
def ranksum_reusing_reference(sorted_target, sorted_ref, ref_ties):
    n1, n2 = sorted_target.shape[1], sorted_ref.shape[1]
    p = np.ones(len(sorted_target), dtype=np.float64)
    u = np.zeros(len(sorted_target), dtype=np.float64)
    total_n = n1 + n2
    for gene in prange(len(sorted_target)):
        target, ref = sorted_target[gene], sorted_ref[gene]
        pos, stat, ties = 0, 0.0, ref_ties[gene]
        while pos < n1:
            value = target[pos]
            end = pos + 1
            while end < n1 and target[end] == value:
                end += 1
            t = float(end - pos)
            lo = np.searchsorted(ref, value, side="left")
            hi = np.searchsorted(ref, value, side="right")
            r = float(hi - lo)
            stat += t * (lo + 0.5 * r)
            ties += t * (t * t - 1) + 3 * t * r * (t + r)
            pos = end
        u[gene] = stat
        variance = n1 * n2 / 12.0 * (total_n + 1 - ties / (total_n * (total_n - 1)))
        if variance > 0:
            z = (max(stat, n1 * n2 - stat) - n1 * n2 * 0.5 - 0.5) / math.sqrt(variance)
            p[gene] = min(1.0, math.erfc(z / math.sqrt(2.0)))
    return p, u


def differential_expression(
    matrix: np.ndarray,
    labels: np.ndarray,
    genes: list[str],
    cache_dir: Path | None = None,
    block_size: int = 256,
) -> pl.DataFrame:
    set_num_threads(4)
    targets = [str(p) for p in pd.unique(labels) if p != CONTROL]
    masks = {p: np.flatnonzero(labels == p) for p in targets + [CONTROL]}
    if min(len(m) for m in masks.values()) <= 8:
        raise ValueError("Fast path requires all groups >8 cells (asymptotic pdex test)")
    frames = []
    for start in range(0, len(genes), block_size):
        stop = min(start + block_size, len(genes))
        path = cache_dir / f"genes_{start:05d}_{stop:05d}.parquet" if cache_dir else None
        if path and path.exists():
            frames.append(pl.read_parquet(path))
            continue
        block = np.array(matrix[:, start:stop], dtype=np.float32)
        ref = np.ascontiguousarray(block[masks[CONTROL]].T)
        ref_mean = np.expm1(ref.mean(1))
        sorted_ref = np.sort(ref, axis=1)
        ties = reference_ties(sorted_ref)
        rows = []
        for ordinal, target in enumerate(targets):
            values = np.ascontiguousarray(block[masks[target]].T)
            mean = np.expm1(values.mean(1))
            pvalue, statistic = ranksum_reusing_reference(np.sort(values, axis=1), sorted_ref, ties)
            with np.errstate(divide="ignore", invalid="ignore"):
                fc = mean / ref_mean
                fc = np.where(ref_mean == 0, 20.0, fc)
                fc = np.where(mean == 0, 1 / 20.0, fc)
                fc = np.where((mean == 0) & (ref_mean == 0), 1.0, fc)
                percent = np.where(ref_mean == 0, np.nan, (mean - ref_mean) / ref_mean)
            rows.append(
                pl.DataFrame(
                    {
                        "target": [target] * (stop - start),
                        "reference": [CONTROL] * (stop - start),
                        "feature": genes[start:stop],
                        "target_mean": mean,
                        "reference_mean": ref_mean,
                        "percent_change": percent,
                        "fold_change": fc,
                        "p_value": pvalue,
                        "statistic": statistic,
                        "_target_order": [ordinal] * (stop - start),
                        "_gene_order": np.arange(start, stop),
                    }
                )
            )
        frame = pl.concat(rows)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.write_parquet(path)
        frames.append(frame)
        print(f"DE genes={stop}/{len(genes)} targets={len(targets)}", flush=True)
    combined = pl.concat(frames).sort(["_target_order", "_gene_order"])
    fdr = false_discovery_control(combined["p_value"].to_numpy(), method="bh")
    return combined.drop("_target_order", "_gene_order").with_columns(pl.Series("fdr", fdr))
