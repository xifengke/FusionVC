"""Repeat the six added fusion/loss configurations at seed 42 exactly."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import torch
import yaml


def main():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    from fusionvc.io import file_sha256, read_json, write_json

    env = dict(os.environ, PYTHONPATH=str(root / "src"), OMP_NUM_THREADS="4")
    experiments = [
        f"{base}_{variant}"
        for base in ["05_concat", "06_gated", "07_cross_attention"]
        for variant in ["energy", "hybrid"]
    ]
    checks = []
    for experiment in experiments:
        config = yaml.safe_load(
            (root / f"configs/experiments/{experiment}.yaml").read_text("utf-8")
        )
        config["experiment"] = f"repro_check_{experiment}"
        path = root / f"tmp/fusion-mmd-repro/{experiment}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(config, sort_keys=False), "utf-8")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "fusionvc.cli",
                "--root",
                str(root),
                "train",
                "--config",
                str(path),
                "--seed",
                "42",
            ],
            cwd=root,
            env=env,
            check=True,
            timeout=1800,
        )
        original_path = root / f"outputs/{experiment}/seed_42/best.pt"
        repeated_path = root / f"outputs/{config['experiment']}/seed_42/best.pt"
        original = torch.load(original_path, map_location="cpu", weights_only=True)
        repeated = torch.load(repeated_path, map_location="cpu", weights_only=True)
        equal = original["state_dict"].keys() == repeated["state_dict"].keys() and all(
            torch.equal(original["state_dict"][key], repeated["state_dict"][key])
            for key in original["state_dict"]
        )
        epochs_equal = original["epoch"] == repeated["epoch"]
        source_equal = original["frozen_features"] == repeated["frozen_features"]
        checks.append(
            {
                "experiment": experiment,
                "seed": 42,
                "every_tensor_equal": equal,
                "selected_epoch_equal": epochs_equal,
                "frozen_feature_provenance_equal": source_equal,
                "original_checkpoint_sha256": file_sha256(original_path),
                "repeated_checkpoint_sha256": file_sha256(repeated_path),
            }
        )
        if not (equal and epochs_equal and source_equal):
            raise RuntimeError(f"Deterministic rerun mismatch: {experiment}")
        print(f"REPRO {experiment}: exact tensors, epoch and features", flush=True)
    assets_unchanged = True
    for group, asset_path in [
        ("scgpt", "models/scgpt/whole-human"),
        ("geneformer", "models/geneformer-v1"),
    ]:
        manifest = read_json(root / f"results/{group}/features.json")
        assets_unchanged &= all(
            file_sha256(root / asset_path / name) == digest
            for name, digest in manifest["signature"]["assets"].items()
        )
    if not assets_unchanged:
        raise RuntimeError("Pretrained assets changed")
    write_json(
        root / "results/fusion-mmd/reproducibility-check.json",
        {
            "runs": checks,
            "all_tensors_equal": True,
            "pretrained_assets_unchanged": True,
            "scope": "six new architecture/loss groups rerun at seed42; equal tensors so official scoring not repeated; same hardware/software only",
            "test_used": False,
        },
    )


if __name__ == "__main__":
    main()
