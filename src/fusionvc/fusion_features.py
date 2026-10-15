"""Gene-identified control-token prototypes for frozen-model fusion.

Each encoder runs in its own native order. Only its contextualized outputs are
grouped by gene identity; Geneformer's ranked inputs are never reordered.
"""

from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np
import torch

from .io import file_sha256, read_json, seed_all, write_json

CACHE_VERSION = "fusion-tokens-v1"
FUSION_MODELS = {"concat", "gated", "cross_attention"}


def accumulate_tokens(sums, counts, hidden, gene_indices, valid):
    """Aggregate by explicit output-gene ID, ignoring all padded positions."""
    values = np.asarray(hidden)
    ids = np.broadcast_to(gene_indices, values.shape[:2])
    mask = np.broadcast_to(valid, values.shape[:2])
    if np.any(ids[mask] < 0) or np.any(ids[mask] >= len(sums)):
        raise ValueError("Token gene ID outside output gene space")
    np.add.at(sums, ids[mask], values[mask])
    np.add.at(counts, ids[mask], 1)


def source_features(root):
    from .geneformer import load_geneformer_features
    from .scgpt import load_scgpt_features

    sc, sc_context, sc_manifest = load_scgpt_features(root)
    gf, gf_context, gf_manifest = load_geneformer_features(root)
    base = root / "data/processed"
    sc_ids = np.load(base / "scgpt-frozen-v1/control_indices.npy")
    gf_ids = np.load(base / "geneformer-frozen-v1/control_indices.npy")
    if not np.array_equal(sc_ids, gf_ids) or len(sc) != len(gf):
        raise ValueError("Fusion requires the same training controls and output gene order")
    if sc_manifest["signature"]["inputs"] != gf_manifest["signature"]["inputs"]:
        raise ValueError("Frozen feature sources use different baseline inputs")
    return (sc, gf), (sc_context, gf_context), (sc_manifest, gf_manifest)


def token_signature(root, manifests):
    return {
        "version": CACHE_VERSION,
        "adapter_sha256": file_sha256(Path(__file__)),
        "source_manifests": dict(zip(["scgpt", "geneformer"], manifests)),
        "gene_order_sha256": file_sha256(root / "data/processed/baselines-v1/genes.json"),
        "aggregation": "per output-gene ID; float64 sum over 2048 training controls; nonpadding only",
        "feature_seed": 2025,
        "test_used": False,
    }


def prepare_fusion(root: Path):
    started = time.perf_counter()
    _, _, manifests = source_features(root)
    signature = token_signature(root, manifests)
    folder = root / f"data/processed/{CACHE_VERSION}"
    ready = folder / "manifest.json"
    if ready.exists():
        load_token_assets(root, manifests)
        print("FUSION token cache already complete; sources and hashes verified", flush=True)
        return read_json(ready)
    if not torch.cuda.is_available():
        raise RuntimeError("Fusion token extraction requires the experiment GPU")
    folder.mkdir(parents=True, exist_ok=True)
    base = root / "data/processed/baselines-v1"
    genes = read_json(base / "genes.json")
    selected = np.load(base / "input_gene_indices.npy")
    chosen = np.load(root / "data/processed/scgpt-frozen-v1/control_indices.npy")
    seed_all(2025)
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    from .scgpt import bin_expression, load_frozen_encoder as load_sc

    model, args, vocab, _ = load_sc(root / "models/scgpt/whole-human", device)
    known = np.array([int(i) for i in selected if genes[i] in vocab])
    tokens = torch.tensor([vocab[genes[i]] for i in known], device=device)
    controls = np.load(base / "controls.npy", mmap_mode="r")
    sums = np.zeros((len(genes), args["embsize"]), dtype=np.float64)
    counts = np.zeros(len(genes), dtype=np.int64)
    bin_rng = np.random.default_rng(2025)
    pooled = []
    for start in range(0, len(chosen), 8):
        values = bin_expression(np.array(controls[chosen[start:start + 8]][:, known]), 51, bin_rng)
        hidden = model(tokens[None].expand(len(values), -1), torch.from_numpy(values).to(device))
        pooled.append(hidden.mean(1).cpu().numpy())
        hidden = hidden.cpu().numpy()
        accumulate_tokens(sums, counts, hidden, known, np.ones(len(known), dtype=bool))
    reference = np.load(root / "data/processed/scgpt-frozen-v1/control_embeddings.npy")
    sc_error = float(np.max(np.abs(np.concatenate(pooled) - reference)))
    if sc_error > 1e-5:
        raise ValueError(f"scGPT token extraction differs from fixed feature cache: {sc_error}")
    arrays = {
        "scgpt_tokens.npy": (sums[known] / counts[known, None]).astype(np.float32),
        "scgpt_gene_indices.npy": known,
        "scgpt_valid.npy": counts[known] > 0,
        "scgpt_observation_counts.npy": counts[known],
    }
    del model, hidden, sums, pooled
    gc.collect()
    torch.cuda.empty_cache()
    print("FUSION scGPT contextualized gene tokens complete", flush=True)
    from .geneformer import load_frozen_encoder as load_gf, mean_nonpadding

    model, config, _ = load_gf(root / "models/geneformer-v1", device)
    gf_base = root / "data/processed/geneformer-frozen-v1"
    known = np.load(gf_base / "input_gene_indices.npy")
    gene_tokens = np.load(gf_base / "gene_token_ids.npy")
    # Canonical token aliases share one prototype, located at its first candidate gene.
    token_to_gene = {}
    for index in known:
        token_to_gene.setdefault(int(gene_tokens[index]), int(index))
    native_tokens = np.load(gf_base / "control_token_ids.npy")
    lengths = np.load(gf_base / "control_token_lengths.npy")
    table = np.full(config["vocab_size"], -1, dtype=np.int64)
    for token, index in token_to_gene.items():
        table[token] = index
    sums = np.zeros((len(genes), config["hidden_size"]), dtype=np.float64)
    counts = np.zeros(len(genes), dtype=np.int64)
    pooled = []
    for start in range(0, len(chosen), 16):
        lens = lengths[start:start + 16]
        size = int(lens.max())
        batch = native_tokens[start:start + 16, :size]
        mask = np.arange(size)[None] < lens[:, None]
        hidden = model(torch.from_numpy(batch).to(device), torch.from_numpy(mask).to(device))
        pooled.append(mean_nonpadding(hidden, torch.from_numpy(lens).to(device)).cpu().numpy())
        accumulate_tokens(sums, counts, hidden.cpu().numpy(), table[batch], mask)
    reference = np.load(gf_base / "control_embeddings.npy")
    gf_error = float(np.max(np.abs(np.concatenate(pooled) - reference)))
    if gf_error > 1e-5:
        raise ValueError(f"Geneformer token extraction differs from fixed cache: {gf_error}")
    ids = np.array(list(token_to_gene.values()), dtype=np.int64)
    valid = counts[ids] > 0
    prototypes = (sums[ids] / np.maximum(counts[ids, None], 1)).astype(np.float32)
    arrays.update({
        "geneformer_tokens.npy": prototypes,
        "geneformer_gene_indices.npy": ids,
        "geneformer_valid.npy": valid,
        "geneformer_observation_counts.npy": counts[ids],
        "control_indices.npy": chosen,
    })
    for name, array in arrays.items():
        if not np.isfinite(array).all():
            raise ValueError(f"Nonfinite fusion feature: {name}")
        np.save(folder / name, array)
    mapping = {
        name: [{"output_index": int(i), "symbol": genes[i], "valid": bool(v)}
               for i, v in zip(arrays[f"{name}_gene_indices.npy"], arrays[f"{name}_valid.npy"])]
        for name in ["scgpt", "geneformer"]
    }
    write_json(folder / "gene_mapping.json", mapping)
    torch.cuda.synchronize()
    manifest = {
        "signature": signature,
        "feature_hashes": {name: file_sha256(folder / name) for name in [*arrays, "gene_mapping.json"]},
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(),
        "control_cells": len(chosen),
        "scgpt_tokens": len(arrays["scgpt_tokens.npy"]),
        "geneformer_tokens": len(ids),
        "geneformer_valid_tokens": int(valid.sum()),
        "pooled_cache_max_abs_difference": {"scgpt": sc_error, "geneformer": gf_error},
        "test_used": False,
    }
    write_json(ready, manifest)
    write_json(root / "results/fusion/token-features.json", manifest)
    print("FUSION gene-token prototypes complete", flush=True)
    return manifest


def load_token_assets(root, manifests):
    folder = root / f"data/processed/{CACHE_VERSION}"
    manifest = read_json(folder / "manifest.json")
    if manifest["signature"] != token_signature(root, manifests):
        raise ValueError("Fusion token sources changed; use a new cache version")
    for name, digest in manifest["feature_hashes"].items():
        if file_sha256(folder / name) != digest:
            raise ValueError(f"Corrupted fusion token cache: {name}")
    assets = {name[:-4]: np.load(folder / name)
              for name in manifest["feature_hashes"] if name.endswith(".npy")}
    return assets, manifest


def load_fusion_features(root: Path, config: dict):
    if (config.get("protocol") != "fusion-v1" or config.get("encoder_frozen") is not True
            or config["model"] not in FUSION_MODELS):
        raise ValueError("Fusion requires the explicit fusion-v1 protocol and frozen encoders")
    targets, contexts, manifests = source_features(root)
    assets = dict(zip(["scgpt_context", "geneformer_context"], contexts))
    token_manifest = None
    if config["model"] == "cross_attention":
        token_assets, token_manifest = load_token_assets(root, manifests)
        assets.update(token_assets)
    manifest = {
        "protocol": "fusion-v1", "feature_dims": [a.shape[-1] for a in targets],
        "source_manifests": dict(zip(["scgpt", "geneformer"], manifests)),
        "token_manifest": token_manifest,
        "frozen_parameters": sum(m["frozen_parameters"] for m in manifests),
        "elapsed_seconds": sum(m["elapsed_seconds"] for m in manifests)
        + (token_manifest["elapsed_seconds"] if token_manifest else 0),
        "fit_source": "same 2048 training non-targeting controls; no perturbation labels",
        "test_used": False,
    }
    return np.concatenate(targets, axis=-1), assets, manifest
