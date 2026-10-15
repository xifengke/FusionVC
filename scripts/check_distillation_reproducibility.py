"""Run the zero-KD fairness control and exact seed-42 student rerun."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml


def compare_checkpoints(original_path: Path, repeated_path: Path) -> dict:
    original = torch.load(original_path, map_location="cpu", weights_only=True)
    repeated = torch.load(repeated_path, map_location="cpu", weights_only=True)
    tensors = original["state_dict"].keys() == repeated["state_dict"].keys() and all(
        torch.equal(value, repeated["state_dict"][name])
        for name, value in original["state_dict"].items()
    )
    columns = ["epoch", "train_mse", "val_mae", "val_mse", "lr"]
    old_history = pd.read_csv(original_path.parent / "history.csv")
    new_history = pd.read_csv(repeated_path.parent / "history.csv")
    history_equal = old_history.shape[0] == new_history.shape[0] and np.array_equal(
        old_history[columns].to_numpy(), new_history[columns].to_numpy()
    )
    return {
        "every_tensor_equal": tensors,
        "selected_epoch_equal": original["epoch"] == repeated["epoch"],
        "core_training_history_equal": history_equal,
        "distillation_provenance_equal": original.get("distillation_cache")
        == repeated.get("distillation_cache"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zero-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    from fusionvc.io import file_sha256, write_json

    config = yaml.safe_load((root / "configs/experiments/08_distillation.yaml").read_text("utf-8"))
    env = dict(os.environ, PYTHONPATH=str(root / "src"), OMP_NUM_THREADS="4")
    recipes = [("zero_kd_08_distillation", "01_mlp", 0.0)]
    if not args.zero_only:
        recipes.append(("repro_check_08_distillation", "08_distillation", 1.0))
    checks = []
    for experiment, reference, weight in recipes:
        recipe = {**config, "experiment": experiment, "kd_weight": weight}
        path = root / f"tmp/distillation-repro/{experiment}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(recipe, sort_keys=False), "utf-8")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "fusionvc.cli",
                "--root",
                str(root),
                "train-distillation",
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
        original = root / f"outputs/{reference}/seed_42/best.pt"
        repeated = root / f"outputs/{experiment}/seed_42/best.pt"
        check = {
            "reference_experiment": reference,
            "repeated_experiment": experiment,
            "seed": 42,
            "kd_weight": weight,
            **compare_checkpoints(original, repeated),
            "original_checkpoint_sha256": file_sha256(original),
            "repeated_checkpoint_sha256": file_sha256(repeated),
        }
        required = ["every_tensor_equal", "selected_epoch_equal", "core_training_history_equal"]
        if weight > 0:
            required.append("distillation_provenance_equal")
        if not all(check[name] for name in required):
            raise RuntimeError(f"Distillation reproducibility failed: {check}")
        checks.append(check)
        if weight == 0:
            write_json(
                root / "results/distillation/zero-kd-parity.json",
                {
                    "run": check,
                    "all_tensors_equal": True,
                    "test_used": False,
                    "scope": "independent distillation trainer with zero KD; exact original MLP weights, selected epoch and core training history",
                },
            )
    if not args.zero_only:
        write_json(
            root / "results/distillation/reproducibility-check.json",
            {
                "runs": checks,
                "all_tensors_equal": True,
                "test_used": False,
                "scope": "seed42 zero-KD MLP parity and KD student exact rerun on the same hardware/software; evaluation not repeated for identical tensors",
            },
        )
    print("DISTILL zero-KD parity and requested exact reruns passed", flush=True)


if __name__ == "__main__":
    main()
