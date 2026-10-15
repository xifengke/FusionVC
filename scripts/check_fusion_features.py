"""Independently audit prototype IDs/counts against both native input caches."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def main():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    from fusionvc.fusion_features import load_fusion_features
    from fusionvc.io import file_sha256, read_json, write_json

    _, assets, manifest = load_fusion_features(
        root,
        {
            "model": "cross_attention",
            "protocol": "fusion-v1",
            "encoder_frozen": True,
        },
    )
    base = root / "data/processed/baselines-v1"
    genes = read_json(base / "genes.json")
    selected = np.load(base / "input_gene_indices.npy")
    vocab = read_json(root / "models/scgpt/whole-human/vocab.json")
    expected_sc = np.array([i for i in selected if genes[i] in vocab])
    sc_ids_match = np.array_equal(assets["scgpt_gene_indices"], expected_sc)
    sc_counts_match = np.all(assets["scgpt_observation_counts"] == 2048)
    gf_base = root / "data/processed/geneformer-frozen-v1"
    known = np.load(gf_base / "input_gene_indices.npy")
    gene_tokens = np.load(gf_base / "gene_token_ids.npy")
    token_to_gene = {}
    for index in known:
        token_to_gene.setdefault(int(gene_tokens[index]), int(index))
    expected_gf = np.array(list(token_to_gene.values()))
    native = np.load(gf_base / "control_token_ids.npy")
    lengths = np.load(gf_base / "control_token_lengths.npy")
    valid = np.arange(native.shape[1])[None] < lengths[:, None]
    token_counts = dict(zip(*np.unique(native[valid], return_counts=True)))
    expected_counts = np.array([token_counts.get(token, 0) for token in token_to_gene])
    checks = {
        "scgpt_ids_match_fixed_covered_candidates": bool(sc_ids_match),
        "scgpt_counts_equal_2048_per_gene": bool(sc_counts_match),
        "geneformer_ids_match_native_token_identity": bool(
            np.array_equal(assets["geneformer_gene_indices"], expected_gf)
        ),
        "geneformer_counts_match_native_rank_sequences": bool(
            np.array_equal(assets["geneformer_observation_counts"], expected_counts)
        ),
        "geneformer_valid_mask_matches_native_observations": bool(
            np.array_equal(assets["geneformer_valid"], expected_counts > 0)
        ),
        "native_padding_is_zero": bool(np.all(native[~valid] == 0)),
        "native_valid_tokens_are_known_gene_tokens": bool(
            set(native[valid].tolist()) == set(token_to_gene)
        ),
        "source_pooling_within_predeclared_tolerance": all(
            0 <= value <= 1e-5
            for value in manifest["token_manifest"]["pooled_cache_max_abs_difference"].values()
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Fusion identity/count audit failed: {checks}")
    write_json(
        root / "results/fusion/implementation-check.json",
        {
            "checks": checks,
            "all_checks_passed": True,
            "scgpt_genes": len(expected_sc),
            "geneformer_genes": len(expected_gf),
            "native_geneformer_padding_slots_checked": int((~valid).sum()),
            "token_cache_manifest_sha256": file_sha256(
                root / "data/processed/fusion-tokens-v1/manifest.json"
            ),
            "scope": "independent identity/count reconstruction from original native input caches; extraction pooling parity recorded at extraction time",
            "test_used": False,
        },
    )
    print("FUSION native identity/count audit passed", flush=True)


if __name__ == "__main__":
    main()
