"""Frozen Geneformer V1-10M, rank tokens, and training-control-only features."""

from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from safetensors.torch import load_file
from torch import nn
from transformers import BertConfig, BertModel

from .io import CountsReader, file_sha256, read_json, seed_all, strings, write_json
from .preprocess import CONTROL

CACHE_VERSION = "geneformer-frozen-v1"
DICTIONARIES = [
    "token_dictionary_gc30M.pkl",
    "gene_median_dictionary_gc30M.pkl",
    "ensembl_mapping_dict_gc30M.pkl",
]


class FrozenGeneformer(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        if config["model_type"] != "bert" or config["position_embedding_type"] != "absolute":
            raise ValueError("Geneformer V1 requires BERT with absolute position embeddings")
        bert_config = BertConfig.from_dict(config)
        bert_config._attn_implementation = "eager"
        self.bert = BertModel(bert_config, add_pooling_layer=False)
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, tokens: torch.Tensor, attention_mask: torch.Tensor):
        if (attention_mask.sum(1) == 0).any():
            raise ValueError("Cannot encode an empty Geneformer sequence")
        return self.bert(input_ids=tokens, attention_mask=attention_mask).last_hidden_state


def load_frozen_encoder(folder: Path, device: torch.device):
    config = read_json(folder / "config.json")
    model = FrozenGeneformer(config)
    source = load_file(folder / "model.safetensors", device="cpu")
    legacy = source.pop("bert.embeddings.position_ids", None)
    if legacy is not None and not torch.equal(
        legacy, torch.arange(config["max_position_embeddings"])[None]
    ):
        raise ValueError("Checkpoint legacy position_ids do not match absolute positions")
    weights = {key[5:]: value for key, value in source.items() if key.startswith("bert.")}
    model.bert.load_state_dict(weights, strict=True)
    ignored = sorted(key for key in source if not key.startswith("bert."))
    if any(not key.startswith("cls.predictions.") for key in ignored):
        raise ValueError(f"Unexpected non-encoder tensors: {ignored}")
    info = {
        "loaded_encoder_tensors": len(weights),
        "ignored_non_encoder_tensors": ignored,
        "legacy_position_ids_validated": legacy is not None,
        "strict_load": True,
        "frozen_parameters": sum(p.numel() for p in model.parameters()),
        "transformers_version": transformers.__version__,
        "attention_backend": "eager",
    }
    return model.to(device).eval(), config, info


def load_dictionaries(folder: Path):
    # Only pinned, checksum-verified local upstream resources are unpickled.
    return [pickle.loads((folder / name).read_bytes()) for name in DICTIONARIES]


def map_gene_ids(gene_ids: list[str], tokens: dict, medians: dict, mapping: dict):
    canonical = [mapping.get(gene, gene) for gene in gene_ids]
    covered = np.array([gene in tokens and gene in medians for gene in canonical])
    token_ids = np.array(
        [int(tokens[gene]) if known else 0 for gene, known in zip(canonical, covered)]
    )
    factors = np.array(
        [float(medians[gene]) if known else 1.0 for gene, known in zip(canonical, covered)]
    )
    if np.any(~np.isfinite(factors)) or np.any(factors <= 0):
        raise ValueError("Geneformer gene medians must be finite and positive")
    return canonical, covered, token_ids, factors


def rank_tokenize(counts: np.ndarray, token_ids: np.ndarray, medians: np.ndarray, max_length=2048):
    """Official nonzero rank encoding, including median scaling and ID collapse.

    Counts are restricted to the fixed baseline input genes. Library scaling is
    positive and constant within each row, so cannot change its ranks.
    """
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim != 2 or counts.shape[1] != len(token_ids) or len(medians) != len(token_ids):
        raise ValueError("Counts, token IDs and medians must have matching gene dimensions")
    if np.any(~np.isfinite(counts)) or np.any(counts < 0) or max_length < 1:
        raise ValueError("Rank encoding requires finite nonnegative counts and positive length")
    if np.any(~np.isfinite(medians)) or np.any(medians <= 0) or np.any(token_ids <= 1):
        raise ValueError("Rank encoding requires valid gene tokens and positive medians")
    # Preserve first occurrence order, summing columns that map to one token.
    unique = list(dict.fromkeys(int(token) for token in token_ids))
    if len(unique) != len(token_ids):
        collapsed, factors = [], []
        for token in unique:
            indices = np.flatnonzero(token_ids == token)
            if not np.all(medians[indices] == medians[indices[0]]):
                raise ValueError("Duplicate canonical IDs have inconsistent medians")
            collapsed.append(counts[:, indices].sum(1))
            factors.append(medians[indices[0]])
        counts, medians = np.column_stack(collapsed), np.array(factors)
        token_ids = np.array(unique)
    sequences = []
    for row in counts:
        nonzero = np.flatnonzero(row)
        if not len(nonzero):
            raise ValueError("No recognized expressed genes in a control cell")
        order = np.argsort(-(row[nonzero] / medians[nonzero]))
        sequences.append(token_ids[nonzero[order]][:max_length])
    lengths = np.array([len(sequence) for sequence in sequences], dtype=np.int64)
    if not len(lengths):
        raise ValueError("Cannot tokenize an empty cell collection")
    padded = np.zeros((len(sequences), int(lengths.max())), dtype=np.int64)
    for index, sequence in enumerate(sequences):
        padded[index, : len(sequence)] = sequence
    return padded, lengths


def mean_nonpadding(hidden: torch.Tensor, lengths: torch.Tensor):
    mask = torch.arange(hidden.shape[1], device=hidden.device)[None] < lengths[:, None]
    return hidden.masked_fill(~mask[:, :, None], 0).sum(1) / lengths[:, None]


def feature_signature(root: Path):
    base = root / "data/processed/baselines-v1"
    assets = root / "models/geneformer-v1"
    recorded = {
        item["path"]: item for item in read_json(root / "docs/download-status.json")["files"]
    }
    names = ["config.json", "model.safetensors"] + [f"dictionaries/{name}" for name in DICTIONARIES]
    hashes = {}
    for name in names:
        key = f"models/geneformer-v1/{name}"
        digest = file_sha256(assets / name)
        if digest != recorded[key]["sha256"]:
            raise ValueError(f"Downloaded Geneformer asset hash mismatch: {key}")
        hashes[name] = digest
    raw = root / "data/raw/vcc2025/train/adata_Training.h5ad"
    return {
        "version": CACHE_VERSION,
        "adapter_sha256": file_sha256(Path(__file__)),
        "upstream_tokenizer_sha256": file_sha256(
            root / "vendor/Geneformer/geneformer/tokenizer.py"
        ),
        "assets": hashes,
        "inputs": {
            name: file_sha256(base / name)
            for name in [
                "manifest.json",
                "genes.json",
                "gene_descriptors.npy",
                "input_gene_indices.npy",
            ]
        },
        "raw_training_source": {"bytes": raw.stat().st_size, "mtime_ns": raw.stat().st_mtime_ns},
        "transformers_version": transformers.__version__,
        "feature_seed": 2025,
        "feature_cells": 2048,
        "input_gene_policy": "fixed baseline 256 candidates; covered, nonzero genes only",
        "gene_mapping": "official training var/gene_id; pinned 30M canonical-ID dictionary",
        "rank_encoding": "raw training counts / Genecorpus-30M gene median; descending np.argsort",
        "max_length": 2048,
        "special_tokens": False,
        "pooling": "last encoder layer; mean nonpadding tokens, then mean training control cells",
        "target_embedding": "frozen pretrained word embedding + train-only descriptor + coverage",
        "fit_source": "training non-targeting cells only; no perturbation labels",
        "precision": "float32 frozen feature extraction; dropout disabled; eager attention",
    }


def prepare_geneformer(root: Path, batch_size=16):
    started = time.perf_counter()
    signature = feature_signature(root)
    folder = root / f"data/processed/{CACHE_VERSION}"
    ready = folder / "manifest.json"
    if ready.exists():
        _, _, manifest = load_geneformer_features(root)
        print("GENEFORMER features already complete; hashes verified", flush=True)
        return manifest
    if not torch.cuda.is_available():
        raise RuntimeError("Geneformer feature extraction requires the experiment GPU")
    folder.mkdir(parents=True, exist_ok=True)
    seed_all(2025)
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    model, config, info = load_frozen_encoder(root / "models/geneformer-v1", device)
    tokens, medians, mapping = load_dictionaries(root / "models/geneformer-v1/dictionaries")
    if len(tokens) != config["vocab_size"] or int(tokens["<pad>"]) != config["pad_token_id"]:
        raise ValueError("Checkpoint and 30M token dictionary do not match")
    base = root / "data/processed/baselines-v1"
    genes = read_json(base / "genes.json")
    selected = np.load(base / "input_gene_indices.npy")
    reader = CountsReader(root / "data/raw/vcc2025/train/adata_Training.h5ad")
    try:
        if genes != reader.genes:
            raise ValueError("Raw training gene order differs from the baseline cache")
        gene_ids = strings(reader.file["var/gene_id"][:])
        canonical, coverage, all_tokens, all_medians = map_gene_ids(
            gene_ids, tokens, medians, mapping
        )
        known = selected[coverage[selected]]
        if not len(known):
            raise ValueError("No baseline candidate input genes covered by Geneformer")
        control_rows = np.flatnonzero(reader.codes == reader.categories.index(CONTROL))
        controls = np.load(base / "controls.npy", mmap_mode="r")
        if len(control_rows) != len(controls):
            raise ValueError("Training-control counts differ from the baseline cache")
        chosen = np.sort(
            np.random.default_rng(2025).choice(len(controls), min(2048, len(controls)), False)
        )
        raw_rows = control_rows[chosen]
        counts = reader.rows(raw_rows).toarray()[:, known]
    finally:
        reader.close()
    padded, lengths = rank_tokenize(
        counts, all_tokens[known], all_medians[known], config["max_position_embeddings"]
    )
    contexts = np.empty((len(chosen), config["hidden_size"]), dtype=np.float32)
    for start in range(0, len(chosen), batch_size):
        stop = min(start + batch_size, len(chosen))
        size = int(lengths[start:stop].max())
        batch = torch.from_numpy(padded[start:stop, :size]).to(device)
        lens = torch.from_numpy(lengths[start:stop]).to(device)
        mask = torch.arange(size, device=device)[None] < lens[:, None]
        contexts[start:stop] = mean_nonpadding(model(batch, mask), lens).cpu().numpy()
        if start % (batch_size * 16) == 0:
            print(f"GENEFORMER extracting training controls {stop}/{len(chosen)}", flush=True)
    with torch.no_grad():
        gene_features = (
            model.bert.embeddings.word_embeddings(torch.from_numpy(all_tokens).to(device))
            .cpu()
            .numpy()
        )
    gene_features[~coverage] = 0
    descriptors = np.load(base / "gene_descriptors.npy")
    target_features = np.column_stack([gene_features, descriptors, coverage]).astype(np.float32)
    if not np.isfinite(contexts).all() or not np.isfinite(target_features).all():
        raise FloatingPointError("Non-finite frozen Geneformer features")
    arrays = {
        "control_indices.npy": chosen,
        "raw_control_rows.npy": raw_rows,
        "control_token_ids.npy": padded,
        "control_token_lengths.npy": lengths,
        "input_gene_indices.npy": known,
        "gene_token_ids.npy": all_tokens,
        "control_embeddings.npy": contexts,
        "context.npy": contexts.mean(0, dtype=np.float64).astype(np.float32),
        "target_features.npy": target_features,
    }
    for name, array in arrays.items():
        np.save(folder / name, array)
    mapping_file = "gene_mapping.json"
    write_json(
        folder / mapping_file,
        [
            {
                "symbol": name,
                "ensembl_id": original,
                "canonical_id": mapped,
                "token_id": int(token),
                "covered": bool(covered),
            }
            for name, original, mapped, token, covered in zip(
                genes, gene_ids, canonical, all_tokens, coverage
            )
        ],
    )
    split = read_json(root / "configs/splits/vcc2025.json")
    by_name = dict(zip(genes, coverage))
    torch.cuda.synchronize()
    manifest = {
        "signature": signature,
        "feature_hashes": {name: file_sha256(folder / name) for name in [*arrays, mapping_file]},
        **info,
        "input_gene_count": len(selected),
        "recognized_input_genes": len(known),
        "missing_input_genes": [genes[i] for i in selected if not coverage[i]],
        "output_genes": len(genes),
        "covered_output_genes": int(coverage.sum()),
        "target_coverage": {
            part: {
                "total": len(split[part]),
                "covered": sum(bool(by_name[g]) for g in split[part]),
                "missing": [g for g in split[part] if not by_name[g]],
            }
            for part in ["train", "validation"]
        },
        "missing_target_policy": "zero pretrained embedding plus train-only descriptor and coverage=0",
        "collapsed_input_columns": len(known) - len(set(all_tokens[known])),
        "control_cells": len(chosen),
        "sequence_length_min": int(lengths.min()),
        "sequence_length_max": int(lengths.max()),
        "sequence_length_mean": float(lengths.mean()),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(),
        "test_used": False,
    }
    write_json(ready, manifest)
    write_json(root / "results/geneformer/features.json", manifest)
    print(
        f"GENEFORMER features complete: {len(known)}/{len(selected)} inputs; {len(genes)} outputs",
        flush=True,
    )
    return manifest


def load_geneformer_features(root: Path):
    folder = root / f"data/processed/{CACHE_VERSION}"
    if not (folder / "manifest.json").exists():
        raise RuntimeError("Run fusionvc prepare-geneformer before Geneformer training")
    manifest = read_json(folder / "manifest.json")
    if manifest["signature"] != feature_signature(root):
        raise ValueError("Frozen Geneformer sources changed; use a new cache version")
    for name, digest in manifest["feature_hashes"].items():
        if file_sha256(folder / name) != digest:
            raise ValueError(f"Corrupted frozen Geneformer features: {name}")
    return np.load(folder / "target_features.npy"), np.load(folder / "context.npy"), manifest
