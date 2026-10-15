"""Inspect actual downloaded assets without loading expression matrices in RAM."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import h5py
import numpy as np
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]


def text_values(values):
    return [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in values]


def main() -> None:
    with (ROOT / "data/raw/vcc2025/gene_names.csv").open(encoding="utf-8") as stream:
        genes = [row[0] for row in csv.reader(stream) if row]
    vocab = json.loads((ROOT / "models/scgpt/whole-human/vocab.json").read_text("utf-8"))
    report = {"gene_count": len(genes), "scgpt_vocabulary_size": len(vocab),
              "scgpt_gene_coverage": sum(g in vocab for g in genes),
              "scgpt_missing_genes": [g for g in genes if g not in vocab], "datasets": {}}
    assert len(genes) == len(set(genes)), "Duplicate competition gene names"
    with safe_open(ROOT / "models/geneformer-v1/model.safetensors", framework="np") as weights:
        report["geneformer_tensor_shapes"] = {key: list(weights.get_slice(key).get_shape()) for key in weights.keys()}
    for split, title in [("train", "Training"), ("validation", "Validation")]:
        path = ROOT / f"data/raw/vcc2025/{split}/adata_{title}.h5ad"
        if not path.exists():
            report["datasets"][split] = {"status": "not_downloaded_yet"}
            continue
        with h5py.File(path, "r") as data:
            info = {"status": "readable", "root_keys": list(data), "obs_keys": list(data["obs"]), "var_keys": list(data["var"])}
            var_index = data["var"].attrs.get("_index", "_index")
            stored_genes = text_values(data["var"][var_index][:])
            info["gene_order_matches_official"] = stored_genes == genes
            assert info["gene_order_matches_official"], f"Gene list mismatch in {split}"
            x = data["X"]
            if isinstance(x, h5py.Group):
                info["expression_encoding"] = str(x.attrs.get("encoding-type"))
                info["shape"] = [int(v) for v in x.attrs["shape"]]
                info["nonzero_entries"] = int(x["data"].shape[0])
                sample = x["data"][:4096]
            else:
                info["expression_encoding"] = "dense"
                info["shape"] = list(x.shape)
                sample = x[:8, :512].ravel()
            info["sample_dtype"] = str(sample.dtype)
            info["sample_finite_nonnegative"] = bool(np.all(np.isfinite(sample)) and np.all(sample >= 0))
            info["sample_integer_valued"] = bool(np.all(sample == np.floor(sample)))
            assert info["sample_finite_nonnegative"], f"Invalid sample expression in {split}"
            if "target_gene" in data["obs"]:
                target = data["obs"]["target_gene"]
                if isinstance(target, h5py.Group):
                    categories = text_values(target["categories"][:])
                    codes = target["codes"][:]
                    counts = np.bincount(codes[codes >= 0], minlength=len(categories))
                    info["target_counts"] = {name: int(n) for name, n in zip(categories, counts)}
                else:
                    names, counts = np.unique(text_values(target[:]), return_counts=True)
                    info["target_counts"] = {str(name): int(n) for name, n in zip(names, counts)}
                with (path.parent / f"pert_counts_{title}.csv").open(encoding="utf-8") as stream:
                    metadata = list(csv.DictReader(stream))
                mismatches = [r["target_gene"] for r in metadata if info["target_counts"].get(r["target_gene"]) != int(r["n_cells"])]
                info["metadata_count_mismatches"] = mismatches
                info["metadata_perturbation_count"] = len(metadata)
                assert not mismatches, f"Cell-count metadata mismatch in {split}: {mismatches}"
            report["datasets"][split] = info
            print(split, info["shape"], "gene order OK", flush=True)
    (ROOT / "docs/asset-validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    print("Asset validation report written", flush=True)


if __name__ == "__main__":
    main()
