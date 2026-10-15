"""Run the complete 2-model x 3-loss x 3-seed MMD comparison."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", choices=["smoke", "train", "evaluate", "report", "all"], default="all"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    for name, value in {
        "PYTHONUTF8": "1",
        "OMP_NUM_THREADS": "4",
        "POLARS_MAX_THREADS": "4",
        "NUMBA_NUM_THREADS": "4",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }.items():
        env[name] = value
    experiments = [
        f"mmd_{prefix}_{model}_{variant}"
        for model, prefix in [("mlp", "01"), ("transformer", "02")]
        for variant in ["mean", "energy", "hybrid"]
    ]
    if args.phase in {"train", "all"}:
        from fusionvc.distribution import prepare_training_cell_cache

        prepare_training_cell_cache(root)
    if args.phase in {"smoke", "train", "all"}:
        for experiment in experiments:
            config_path = root / f"configs/experiments/{experiment}.yaml"
            run_seeds = args.seeds
            if args.phase == "smoke":
                config = yaml.safe_load(config_path.read_text("utf-8"))
                config.update(experiment=f"smoke_{experiment}", epochs=2, patience=2)
                config_path = root / f"tmp/mmd-smoke/{experiment}.yaml"
                config_path.parent.mkdir(parents=True, exist_ok=True)
                config_path.write_text(yaml.safe_dump(config, sort_keys=False), "utf-8")
                run_seeds = [42]
            for seed in run_seeds:
                print(f"RUN train {experiment} seed={seed}", flush=True)
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "fusionvc.cli",
                        "--root",
                        str(root),
                        "train",
                        "--config",
                        str(config_path),
                        "--seed",
                        str(seed),
                    ],
                    cwd=root,
                    env=env,
                    check=True,
                )
        if args.phase != "smoke":
            from fusionvc.mmd_report import verify_mean_parity

            verify_mean_parity(root)
    if args.phase in {"evaluate", "all"}:
        for experiment in experiments:
            for seed in args.seeds:
                print(f"RUN evaluate {experiment} seed={seed}", flush=True)
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "fusionvc.cli",
                        "--root",
                        str(root),
                        "evaluate",
                        "--run",
                        str(root / f"outputs/{experiment}/seed_{seed}"),
                    ],
                    cwd=root,
                    env=env,
                    check=True,
                )
    if args.phase in {"evaluate", "report", "all"}:
        from fusionvc.mmd_report import report_mmd

        report_mmd(root)


if __name__ == "__main__":
    main()
