"""Run the frozen scGPT feature cache, three losses, and three seeds."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", choices=["prepare", "smoke", "train", "evaluate", "report", "all"], default="all"
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["mean", "energy", "hybrid"],
        default=["mean", "energy", "hybrid"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env.update(
        PYTHONPATH=str(root / "src"),
        POLARS_MAX_THREADS="2",
        NUMBA_NUM_THREADS="4",
        OMP_NUM_THREADS="4",
    )
    sys.path.insert(0, str(root / "src"))
    command = [sys.executable, "-m", "fusionvc.cli", "--root", str(root)]
    if args.phase in {"prepare", "smoke", "train", "all"}:
        subprocess.run(command + ["prepare-scgpt"], cwd=root, env=env, check=True)
    if args.phase in {"train", "all"} and any(v != "mean" for v in args.variants):
        subprocess.run(command + ["prepare-mmd"], cwd=root, env=env, check=True)
    experiments = {"mean": "03_scgpt", "energy": "03_scgpt_energy", "hybrid": "03_scgpt_hybrid"}
    if args.phase in {"smoke", "train", "all"}:
        for variant in args.variants:
            experiment = experiments[variant]
            path = root / f"configs/experiments/{experiment}.yaml"
            seeds = args.seeds
            if args.phase == "smoke":
                config = yaml.safe_load(path.read_text("utf-8"))
                config.update(experiment=f"smoke_{experiment}", epochs=2, patience=2)
                path = root / f"tmp/scgpt-smoke/{experiment}.yaml"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(yaml.safe_dump(config, sort_keys=False), "utf-8")
                seeds = [42]
            for seed in seeds:
                subprocess.run(
                    command + ["train", "--config", str(path), "--seed", str(seed)],
                    cwd=root,
                    env=env,
                    check=True,
                )
    if args.phase in {"evaluate", "all"}:
        for variant in args.variants:
            for seed in args.seeds:
                run = root / f"outputs/{experiments[variant]}/seed_{seed}"
                subprocess.run(
                    command + ["evaluate", "--run", str(run)], cwd=root, env=env, check=True
                )
    if args.phase in {"report", "evaluate", "all"}:
        from fusionvc.scgpt_report import report_scgpt

        report_scgpt(root)


if __name__ == "__main__":
    main()
