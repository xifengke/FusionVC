"""Compare our actual downloaded checkpoint with the pinned official implementation."""

from __future__ import annotations

import ast
import importlib.util
import sys
import types
from pathlib import Path
from typing import Union

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    from fusionvc.io import seed_all, write_json
    from fusionvc.scgpt import bin_expression, load_frozen_encoder

    seed_all(2025)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter, args, vocab, info = load_frozen_encoder(ROOT / "models/scgpt/whole-human", device)
    model_dir = ROOT / "vendor/scGPT-paper2024/scgpt/model"
    package = types.ModuleType("_fusionvc_scgpt_reference")
    package.__path__ = [str(model_dir)]
    sys.modules[package.__name__] = package
    spec = importlib.util.spec_from_file_location(
        package.__name__ + ".model", model_dir / "model.py"
    )
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    reference = (
        official.TransformerModel(
            len(vocab),
            args["embsize"],
            args["nheads"],
            args["d_hid"],
            args["nlayers"],
            vocab=vocab,
            dropout=args["dropout"],
            pad_token=args["pad_token"],
            pad_value=args["pad_value"],
            input_emb_style=args["input_emb_style"],
            cell_emb_style="avg-pool",
            use_fast_transformer=False,
        )
        .to(device)
        .eval()
        .requires_grad_(False)
    )
    reference_state = reference.state_dict()
    reference_state.update(adapter.state_dict())
    reference.load_state_dict(reference_state, strict=True)
    genes = torch.tensor(
        [[vocab["TP53"], vocab["GATA4"], vocab["SOX2"], vocab["NANOG"]]] * 2, device=device
    )
    values = torch.tensor([[0.0, 5.0, 25.0, 50.0], [49.0, 2.0, 3.0, 0.0]], device=device)
    mask = torch.zeros_like(genes, dtype=torch.bool)
    with torch.no_grad():
        expected = reference._encode(genes, values, mask)
        actual = adapter(genes, values, mask)
    difference = (actual - expected).abs().max().item()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    # Independently compare with the unedited vendor's binning functions. Extract
    # only these functions to avoid importing unused scanpy/torchtext packages.
    source = model_dir.parent / "preprocess.py"
    tree = ast.parse(source.read_text("utf-8"))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in {"_digitize", "binning"}
    ]
    namespace = {
        "np": np,
        "torch": torch,
        "Union": Union,
        "logger": types.SimpleNamespace(warning=lambda *a: None),
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
    rows = np.array([[0, 0, 1, 1, 1, 5, 9], [2, 2, 2, 2, 2, 2, 2], [0] * 7], dtype=np.float32)
    np.random.seed(2025)
    expected_bins = np.stack([namespace["binning"](row, args["n_bins"]) for row in rows])
    actual_bins = bin_expression(rows, args["n_bins"], np.random.RandomState(2025))
    np.testing.assert_array_equal(actual_bins, expected_bins)
    adapter.train()
    assert all(not layer.training for layer in adapter.modules())
    assert all(not parameter.requires_grad for parameter in adapter.parameters())
    result = {
        **info,
        "reference_revision": "74603ca5d8a2adf2a7420936c73784a1fef01597",
        "reference": "vendor/scGPT-paper2024/scgpt/model/model.py TransformerModel._encode",
        "full_encoder_max_absolute_difference": difference,
        "forward_rtol": 1e-5,
        "forward_atol": 1e-5,
        "binning_equal_with_identical_random_draws": True,
        "frozen_and_eval_after_train_call": True,
        "device": str(device),
        "test_data_used": False,
    }
    write_json(ROOT / "results/scgpt/implementation-parity.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
