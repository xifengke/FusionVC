"""Compare pinned official rank/pooling functions and real Hugging Face BERT."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import BertForMaskedLM


def functions_from_source(path, names, namespace):
    tree = ast.parse(path.read_text("utf-8"))
    selected = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if {node.name for node in selected} != set(names):
        raise RuntimeError("Pinned upstream functions missing")
    # Execute the exact downloaded functions without importing unrelated plotting,
    # loom, distributed-training and fine-tuning dependencies.
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def main():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    from fusionvc.geneformer import (
        load_frozen_encoder,
        load_geneformer_features,
        load_dictionaries,
        map_gene_ids,
        mean_nonpadding,
        rank_tokenize,
    )
    from fusionvc.io import CountsReader, file_sha256, seed_all, strings, write_json

    seed_all(2025)
    folder = root / "models/geneformer-v1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config, info = load_frozen_encoder(folder, device)
    tokenizer_path = root / "vendor/Geneformer/geneformer/tokenizer.py"
    official = functions_from_source(tokenizer_path, ["rank_genes", "tokenize_cell"], {"np": np})
    counts = np.array([[0, 2, 4, 4, 9], [1, 0, 0, 7, 2]], dtype=np.float32)
    tokens = np.array([2, 3, 4, 5, 6])
    medians = np.array([1.0, 2.0, 4.0, 4.0, 0.5])
    padded, lengths = rank_tokenize(counts, tokens, medians)
    rank_equal = all(
        np.array_equal(padded[i, : lengths[i]], official["tokenize_cell"](row / medians, tokens))
        for i, row in enumerate(counts)
    )
    # from_pretrained is the official Geneformer pretrained-model load route.
    reference, loading = BertForMaskedLM.from_pretrained(
        folder,
        local_files_only=True,
        output_loading_info=True,
        attn_implementation="eager",
        output_hidden_states=True,
    )
    if loading["missing_keys"] or loading["mismatched_keys"] or loading["error_msgs"]:
        raise RuntimeError(f"Official reference did not load completely: {loading}")
    if set(loading["unexpected_keys"]) - {"bert.embeddings.position_ids"}:
        raise RuntimeError(f"Unexpected official checkpoint keys: {loading}")
    reference.to(device).eval().requires_grad_(False)
    generator = torch.Generator().manual_seed(2025)
    batch = torch.randint(2, config["vocab_size"], (3, 37), generator=generator).to(device)
    lens = torch.tensor([37, 23, 11], device=device)
    mask = torch.arange(37, device=device)[None] < lens[:, None]
    batch[~mask] = 0
    with torch.no_grad():
        adapted = model(batch, mask)
        original = reference(batch, attention_mask=mask).hidden_states[-1]
    torch.testing.assert_close(adapted, original, rtol=1e-5, atol=1e-6)
    pooling_path = root / "vendor/Geneformer/geneformer/perturber_utils.py"
    official_pool = functions_from_source(pooling_path, ["mean_nonpadding_embs"], {"torch": torch})
    pooled = mean_nonpadding(adapted, lens)
    expected_pool = official_pool["mean_nonpadding_embs"](original, lens)
    torch.testing.assert_close(pooled, expected_pool, rtol=1e-6, atol=1e-7)
    if not rank_equal:
        raise RuntimeError("Official rank tokenization mismatch")
    cache_checks = {}
    cache = root / "data/processed/geneformer-frozen-v1"
    if (cache / "manifest.json").exists():
        _, _, manifest = load_geneformer_features(root)
        indices = np.load(cache / "control_indices.npy")
        scgpt_indices = np.load(root / "data/processed/scgpt-frozen-v1/control_indices.npy")
        if not np.array_equal(indices, scgpt_indices):
            raise RuntimeError("Control cells differ from frozen scGPT")
        # Replay an original extraction batch, including its cropped pad width;
        # changing GEMM shapes can introduce harmless float32 rounding differences.
        sample = np.arange(min(16, len(indices)))
        raw_rows = np.load(cache / "raw_control_rows.npy")[sample]
        selected = np.load(cache / "input_gene_indices.npy")
        reader = CountsReader(root / "data/raw/vcc2025/train/adata_Training.h5ad")
        try:
            token_dict, median_dict, mapping = load_dictionaries(folder / "dictionaries")
            gene_ids = strings(reader.file["var/gene_id"][:])
            _, _, gene_tokens, factors = map_gene_ids(gene_ids, token_dict, median_dict, mapping)
            raw_counts = reader.rows(raw_rows).toarray()[:, selected]
            if any(reader.categories[reader.codes[row]] != "non-targeting" for row in raw_rows):
                raise RuntimeError("Feature extraction included non-control cells")
        finally:
            reader.close()
        cached_tokens = np.load(cache / "control_token_ids.npy")[sample]
        cached_lengths = np.load(cache / "control_token_lengths.npy")[sample]
        for i, row in enumerate(raw_counts):
            official_tokens = official["tokenize_cell"](
                row / factors[selected], gene_tokens[selected]
            )
            if not np.array_equal(cached_tokens[i, : cached_lengths[i]], official_tokens):
                raise RuntimeError("Cached real-cell tokens differ from official rank encoding")
        batch = torch.from_numpy(cached_tokens[:, : int(cached_lengths.max())]).to(device)
        lens = torch.from_numpy(cached_lengths).to(device)
        mask = torch.arange(batch.shape[1], device=device)[None] < lens[:, None]
        with torch.no_grad():
            reference_hidden = reference(batch, attention_mask=mask).hidden_states[-1]
            reference_context = official_pool["mean_nonpadding_embs"](reference_hidden, lens).cpu()
        saved_context = torch.from_numpy(np.load(cache / "control_embeddings.npy")[sample])
        torch.testing.assert_close(saved_context, reference_context, rtol=1e-5, atol=1e-6)
        cache_checks = {
            "cached_real_control_cells_checked": len(sample),
            "real_count_rank_tokens_equal": True,
            "same_control_cells_as_scgpt": True,
            "only_training_non_targeting": True,
            "cached_embedding_max_absolute_difference": float(
                (saved_context - reference_context).abs().max()
            ),
            "feature_manifest_verified": manifest["test_used"] is False,
        }
    write_json(
        root / "results/geneformer/implementation-parity.json",
        {
            **info,
            "full_encoder_max_absolute_difference": float((adapted - original).abs().max()),
            "pooling_max_absolute_difference": float((pooled - expected_pool).abs().max()),
            "rank_tokens_equal": rank_equal,
            "upstream_tokenizer_sha256": file_sha256(tokenizer_path),
            "upstream_pooling_sha256": file_sha256(pooling_path),
            "reference": "BertForMaskedLM.from_pretrained, pinned local V1-10M; hidden_states[-1]",
            "official_loading_info": loading,
            **cache_checks,
            "all_checks_passed": True,
            "test_used": False,
        },
    )
    print(
        "GENEFORMER parity passed: official rank tokens, BERT encoder and nonpadding pooling",
        flush=True,
    )


if __name__ == "__main__":
    main()
