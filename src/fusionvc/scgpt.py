"""Strict, dependency-light frozen scGPT adapter and train-control-only cache.

Matches the downloaded paper-2024 PyTorch encoder, including post-norm/ReLU
and the official Wqkv -> in_proj conversion. No upstream pretrained prediction
decoder is used. All supervised heads are trained by fusionvc.train.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .io import file_sha256, read_json, seed_all, write_json

CACHE_VERSION = "scgpt-frozen-v1"
ENCODER_PREFIXES = ("encoder.", "value_encoder.", "transformer_encoder.")


class GeneEncoder(nn.Module):
    def __init__(self, tokens: int, width: int, pad: int):
        super().__init__()
        self.embedding = nn.Embedding(tokens, width, padding_idx=pad)
        self.enc_norm = nn.LayerNorm(width)

    def forward(self, tokens):
        return self.enc_norm(self.embedding(tokens))


class ValueEncoder(nn.Module):
    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(1, width)
        self.linear2 = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values):
        values = values.unsqueeze(-1).clamp(max=512)
        return self.dropout(self.norm(self.linear2(torch.relu(self.linear1(values)))))


class FrozenScGPT(nn.Module):
    def __init__(self, args: dict, vocab: dict):
        super().__init__()
        if args["input_emb_style"] != "continuous" or args["input_style"] != "binned":
            raise ValueError("This adapter requires the recorded continuous/binned checkpoint")
        if args.get("pre_norm", False):
            raise ValueError("This adapter supports the recorded post-norm checkpoint only")
        width = args["embsize"]
        self.encoder = GeneEncoder(len(vocab), width, vocab[args["pad_token"]])
        self.value_encoder = ValueEncoder(width, args["dropout"])
        layer = nn.TransformerEncoderLayer(
            width,
            args["nheads"],
            args["d_hid"],
            args["dropout"],
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            layer, args["nlayers"], enable_nested_tensor=False
        )
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        # An enclosing model.train() must never enable pretrained dropout.
        return super().train(False)

    @torch.no_grad()
    def forward(self, tokens, values, padding_mask=None):
        sequence = self.encoder(tokens) + self.value_encoder(values)
        return self.transformer_encoder(sequence, src_key_padding_mask=padding_mask)


def load_frozen_encoder(folder: Path, device: torch.device):
    args = read_json(folder / "args.json")
    vocab = read_json(folder / "vocab.json")
    model = FrozenScGPT(args, vocab)
    source = torch.load(folder / "best_model.pt", map_location="cpu", weights_only=True)
    weights = {
        key.replace("Wqkv.", "in_proj_"): value
        for key, value in source.items()
        if key.startswith(ENCODER_PREFIXES)
    }
    # Strict checking prevents a silently random attention layer on Windows.
    model.load_state_dict(weights, strict=True)
    ignored = sorted(key for key in source if not key.startswith(ENCODER_PREFIXES))
    if any(not key.startswith(("flag_encoder.", "decoder.", "mvc_decoder.")) for key in ignored):
        raise ValueError(f"Unexpected non-encoder checkpoint tensors: {ignored}")
    info = {
        "loaded_encoder_tensors": len(weights),
        "ignored_non_encoder_tensors": ignored,
        "frozen_parameters": sum(p.numel() for p in model.parameters()),
        "strict_load": True,
        "qkv_mapping": "Wqkv. -> in_proj_; official PyTorch backend conversion",
    }
    return model.to(device).eval(), args, vocab, info


def bin_expression(rows: np.ndarray, bins: int, rng: np.random.Generator) -> np.ndarray:
    """Official per-cell nonzero quantiles and random tie spread, using local RNG."""
    if bins < 3 or np.any(~np.isfinite(rows)) or np.any(rows < 0):
        raise ValueError("Binning requires finite nonnegative expression and at least 3 bins")
    result = np.zeros_like(rows, dtype=np.float32)
    for index, row in enumerate(rows):
        selected = np.flatnonzero(row)
        if not len(selected):
            continue
        values = row[selected]
        edges = np.quantile(values, np.linspace(0, 1, bins - 1))
        left = np.digitize(values, edges)
        right = np.digitize(values, edges, right=True)
        result[index, selected] = np.ceil(rng.random(len(values)) * (right - left) + left)
    return result


def feature_signature(root: Path) -> dict:
    base = root / "data/processed/baselines-v1"
    assets = root / "models/scgpt/whole-human"
    recorded = {
        item["path"]: item for item in read_json(root / "docs/download-status.json")["files"]
    }
    asset_hashes = {}
    for name in ["best_model.pt", "args.json", "vocab.json"]:
        path = assets / name
        digest = file_sha256(path)
        key = f"models/scgpt/whole-human/{name}"
        if digest != recorded[key]["sha256"]:
            raise ValueError(f"Downloaded asset hash mismatch: {key}")
        asset_hashes[name] = digest
    controls = base / "controls.npy"
    return {
        "version": CACHE_VERSION,
        "adapter_sha256": file_sha256(Path(__file__)),
        "upstream_model_sha256": file_sha256(root / "vendor/scGPT-paper2024/scgpt/model/model.py"),
        "assets": asset_hashes,
        "inputs": {
            name: file_sha256(base / name)
            for name in [
                "manifest.json",
                "genes.json",
                "gene_descriptors.npy",
                "input_gene_indices.npy",
            ]
        },
        "control_source": {
            "bytes": controls.stat().st_size,
            "mtime_ns": controls.stat().st_mtime_ns,
        },
        "feature_seed": 2025,
        "feature_cells": 2048,
        "bins": 51,
        "include_zero_gene": True,
        "pooling": "mean of all recognized input tokens, then mean of training control cells",
        "target_embedding": "frozen normalized pretrained gene embedding + train-only descriptor + coverage",
        "fit_source": "training non-targeting cells only; no perturbation labels",
        "precision": "float32 frozen feature extraction; dropout disabled",
    }


def prepare_scgpt(root: Path, batch_size: int = 8) -> dict:
    started = time.perf_counter()
    folder = root / f"data/processed/{CACHE_VERSION}"
    signature = feature_signature(root)
    ready = folder / "manifest.json"
    if ready.exists():
        saved = read_json(ready)
        if saved["signature"] != signature:
            raise ValueError("scGPT feature sources changed; use a new cache version")
        for name, digest in saved["feature_hashes"].items():
            if file_sha256(folder / name) != digest:
                raise ValueError(f"Corrupted scGPT feature cache: {name}")
        print("SCGPT features already complete; hashes verified", flush=True)
        return saved
    if not torch.cuda.is_available():
        raise RuntimeError("scGPT feature extraction requires the experiment GPU")
    folder.mkdir(parents=True, exist_ok=True)
    seed_all(2025)
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    model, args, vocab, info = load_frozen_encoder(root / "models/scgpt/whole-human", device)
    base = root / "data/processed/baselines-v1"
    genes = read_json(base / "genes.json")
    selected = np.load(base / "input_gene_indices.npy")
    known = [int(i) for i in selected if genes[i] in vocab]
    missing = [genes[i] for i in selected if genes[i] not in vocab]
    if not known:
        raise ValueError("No selected input gene is covered by scGPT")
    controls = np.load(base / "controls.npy", mmap_mode="r")
    chosen = np.sort(
        np.random.default_rng(2025).choice(len(controls), min(2048, len(controls)), False)
    )
    np.save(folder / "control_indices.npy", chosen)
    input_tokens = torch.tensor([vocab[genes[i]] for i in known], device=device)
    all_tokens = torch.tensor(
        [vocab.get(g, vocab[args["pad_token"]]) for g in genes], device=device
    )
    coverage = np.array([g in vocab for g in genes], dtype=np.float32)
    gene_features = model.encoder(all_tokens).cpu().numpy()
    gene_features[coverage == 0] = 0
    descriptors = np.load(base / "gene_descriptors.npy")
    target_features = np.column_stack([gene_features, descriptors, coverage]).astype(np.float32)
    contexts = np.empty((len(chosen), args["embsize"]), dtype=np.float32)
    bin_rng = np.random.default_rng(2025)
    for start in range(0, len(chosen), batch_size):
        stop = min(start + batch_size, len(chosen))
        values = bin_expression(
            np.array(controls[chosen[start:stop]][:, known]), args["n_bins"], bin_rng
        )
        values = torch.from_numpy(values).to(device)
        tokens = input_tokens[None].expand(len(values), -1)
        with torch.no_grad():
            contexts[start:stop] = model(tokens, values).mean(1).cpu().numpy()
        if start % (batch_size * 8) == 0:
            print(f"SCGPT extracting training controls {stop}/{len(chosen)}", flush=True)
    if not np.isfinite(contexts).all() or not np.isfinite(target_features).all():
        raise FloatingPointError("Non-finite frozen scGPT features")
    np.save(folder / "control_embeddings.npy", contexts)
    np.save(folder / "context.npy", contexts.mean(0, dtype=np.float64).astype(np.float32))
    np.save(folder / "target_features.npy", target_features)
    split = read_json(root / "configs/splits/vcc2025.json")
    target_coverage = {
        part: {
            "total": len(split[part]),
            "covered": sum(g in vocab for g in split[part]),
            "missing": [g for g in split[part] if g not in vocab],
        }
        for part in ["train", "validation"]
    }
    feature_files = [
        "control_indices.npy",
        "control_embeddings.npy",
        "context.npy",
        "target_features.npy",
    ]
    manifest = {
        "signature": signature,
        "feature_hashes": {name: file_sha256(folder / name) for name in feature_files},
        **info,
        "input_gene_count": len(selected),
        "recognized_input_genes": len(known),
        "missing_input_genes": missing,
        "output_genes": len(genes),
        "covered_output_genes": int(coverage.sum()),
        "target_coverage": target_coverage,
        "missing_target_policy": "zero pretrained embedding plus train-only descriptor and coverage=0",
        "control_cells": len(chosen),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(),
        "test_used": False,
    }
    write_json(ready, manifest)
    write_json(root / "results/scgpt/features.json", manifest)
    print(
        f"SCGPT features complete: {len(known)}/{len(selected)} inputs; {len(genes)} outputs",
        flush=True,
    )
    return manifest


def load_scgpt_features(root: Path):
    folder = root / f"data/processed/{CACHE_VERSION}"
    if not (folder / "manifest.json").exists():
        raise RuntimeError("Run fusionvc prepare-scgpt before scGPT training")
    manifest = read_json(folder / "manifest.json")
    if manifest["signature"] != feature_signature(root):
        raise ValueError("Frozen scGPT feature sources changed; use a new cache version")
    # Small cache arrays are cheap to verify on every train/evaluate entry.
    for name, digest in manifest["feature_hashes"].items():
        if file_sha256(folder / name) != digest:
            raise ValueError(f"Corrupted frozen scGPT features: {name}")
    return np.load(folder / "target_features.npy"), np.load(folder / "context.npy"), manifest
