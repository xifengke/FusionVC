"""Repeat seed 42 for each loss and verify every learned tensor and selected epoch."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root / "src"))
    from fusionvc.io import file_sha256, read_json, write_json

    env = dict(os.environ)
    env.update(PYTHONPATH=str(root / "src"), OMP_NUM_THREADS="4")
    checks = []
    for experiment in ["03_scgpt", "03_scgpt_energy", "03_scgpt_hybrid"]:
        config = yaml.safe_load(
            (root / f"configs/experiments/{experiment}.yaml").read_text("utf-8")
        )
        config["experiment"] = f"repro_check_{experiment}"
        path = root / f"tmp/scgpt-repro/{experiment}.yaml"
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
        )
        original_path = root / f"outputs/{experiment}/seed_42/best.pt"
        repeated_path = root / f"outputs/{config['experiment']}/seed_42/best.pt"
        original = torch.load(original_path, map_location="cpu", weights_only=True)
        repeated = torch.load(repeated_path, map_location="cpu", weights_only=True)
        equal = original["state_dict"].keys() == repeated["state_dict"].keys()
        equal = equal and all(
            torch.equal(original["state_dict"][key], repeated["state_dict"][key])
            for key in original["state_dict"]
        )
        selected_epoch_equal = original["epoch"] == repeated["epoch"]
        source_equal = original["frozen_features"] == repeated["frozen_features"]
        checks.append(
            {
                "experiment": experiment,
                "seed": 42,
                "every_tensor_equal": equal,
                "selected_epoch_equal": selected_epoch_equal,
                "frozen_feature_provenance_equal": source_equal,
                "original_checkpoint_sha256": file_sha256(original_path),
                "repeated_checkpoint_sha256": file_sha256(repeated_path),
            }
        )
        if not equal or not selected_epoch_equal or not source_equal:
            raise RuntimeError(f"Deterministic rerun mismatch: {experiment}")
    features = read_json(root / "results/scgpt/features.json")
    assets_unchanged = all(
        file_sha256(root / "models/scgpt/whole-human" / name) == digest
        for name, digest in features["signature"]["assets"].items()
    )
    if not assets_unchanged:
        raise RuntimeError("Pretrained assets changed during training")
    write_json(
        root / "results/scgpt/reproducibility-check.json",
        {
            "runs": checks,
            "all_tensors_equal": True,
            "pretrained_assets_unchanged": assets_unchanged,
            "scope": "seed 42 repeated for all three losses; official evaluation not repeated for equal tensors",
            "test_used": False,
        },
    )
    print("SCGPT reproducibility: all three seed-42 models match tensor for tensor", flush=True)


if __name__ == "__main__":
    main()
