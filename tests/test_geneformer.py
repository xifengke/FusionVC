"""Rank encoding, absolute positions, padding, strict load and frozen gradients."""

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from fusionvc.geneformer import (
    FrozenGeneformer,
    load_frozen_encoder,
    map_gene_ids,
    mean_nonpadding,
    rank_tokenize,
)
from fusionvc.io import write_json


def tiny_config():
    return {
        "model_type": "bert",
        "position_embedding_type": "absolute",
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_attention_heads": 2,
        "num_hidden_layers": 2,
        "max_position_embeddings": 12,
        "vocab_size": 9,
        "type_vocab_size": 2,
        "pad_token_id": 0,
        "hidden_act": "relu",
        "layer_norm_eps": 1e-12,
        "hidden_dropout_prob": 0.2,
        "attention_probs_dropout_prob": 0.2,
    }


def test_rank_encoding_medians_nonzero_truncation_and_collapse():
    # The lower count gene moves ahead after corpus-median normalization.
    counts = np.array([[20, 3, 0, 2], [0, 0, 7, 0]])
    tokens = np.array([2, 3, 4, 3])
    medians = np.array([10.0, 1.0, 1.0, 1.0])
    padded, lengths = rank_tokenize(counts, tokens, medians, max_length=2)
    np.testing.assert_array_equal(padded, [[3, 2], [4, 0]])
    np.testing.assert_array_equal(lengths, [2, 1])
    truncated, truncated_lengths = rank_tokenize(counts, tokens, medians, max_length=1)
    np.testing.assert_array_equal(truncated, [[3], [4]])
    np.testing.assert_array_equal(truncated_lengths, [1, 1])
    with pytest.raises(ValueError, match="No recognized"):
        rank_tokenize(np.zeros((1, 4)), tokens, medians)
    with pytest.raises(ValueError, match="finite nonnegative"):
        rank_tokenize(np.array([[np.nan, 1, 1, 1]]), tokens, medians)


def test_explicit_id_mapping_missing_features_and_bad_medians():
    canonical, covered, tokens, medians = map_gene_ids(
        ["old", "unknown", "direct"],
        {"new": 2, "direct": 3},
        {"new": 0.5, "direct": 2.0},
        {"old": "new"},
    )
    assert canonical == ["new", "unknown", "direct"]
    np.testing.assert_array_equal(covered, [True, False, True])
    np.testing.assert_array_equal(tokens, [2, 0, 3])
    np.testing.assert_array_equal(medians, [0.5, 1.0, 2.0])
    with pytest.raises(ValueError, match="finite and positive"):
        map_gene_ids(["A"], {"A": 2}, {"A": 0}, {})


def test_strict_checkpoint_permanent_freeze_and_padding(tmp_path):
    config = tiny_config()
    original = FrozenGeneformer(config)
    source = {f"bert.{k}": v.clone().contiguous() for k, v in original.bert.state_dict().items()}
    source["bert.embeddings.position_ids"] = torch.arange(12)[None]
    source["cls.predictions.bias"] = torch.zeros(9)
    write_json(tmp_path / "config.json", config)
    save_file(source, str(tmp_path / "model.safetensors"))
    model, _, info = load_frozen_encoder(tmp_path, torch.device("cpu"))
    assert info["strict_load"] and info["legacy_position_ids_validated"]
    model.train()
    assert all(not layer.training for layer in model.modules())
    assert all(not p.requires_grad for p in model.parameters())
    tokens = torch.tensor([[2, 3, 4], [3, 2, 0]])
    mask = tokens != 0
    result = model(tokens, mask)
    assert not result.requires_grad
    torch.testing.assert_close(result, original(tokens, mask), rtol=0, atol=0)
    pooled = mean_nonpadding(result, torch.tensor([3, 2]))
    single = model(tokens[1:2, :2], torch.ones(1, 2, dtype=torch.bool)).mean(1)
    torch.testing.assert_close(pooled[1:2], single, rtol=1e-5, atol=1e-6)
    source.pop("bert.encoder.layer.1.attention.self.query.weight")
    save_file(source, str(tmp_path / "model.safetensors"))
    with pytest.raises(RuntimeError, match="Missing key"):
        load_frozen_encoder(tmp_path, torch.device("cpu"))
    source["bert.embeddings.position_ids"] = torch.zeros(1, 12, dtype=torch.long)
    save_file(source, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="position_ids"):
        load_frozen_encoder(tmp_path, torch.device("cpu"))
