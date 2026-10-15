"""Sequential frozen fusion training/evaluation, with resumable completed runs."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

EXPERIMENTS = {"concat": "05_concat", "gated": "06_gated", "cross_attention": "07_cross_attention"}
VARIANTS = ["mean", "energy", "hybrid"]


def experiment_name(model, variant):
    base = EXPERIMENTS[model]
    return base if variant == "mean" else f"{base}_{variant}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", choices=["prepare", "smoke", "train", "evaluate", "report", "all"], default="all"
    )
    parser.add_argument("--models", nargs="+", choices=list(EXPERIMENTS), default=list(EXPERIMENTS))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    env = dict(
        os.environ,
        PYTHONPATH=str(root / "src"),
        POLARS_MAX_THREADS="2",
        NUMBA_NUM_THREADS="4",
        OMP_NUM_THREADS="4",
    )
    command = [sys.executable, "-m", "fusionvc.cli", "--root", str(root)]

    def run(arguments):
        subprocess.run(command + arguments, cwd=root, env=env, check=True, timeout=1800)

    if args.phase in {"prepare", "smoke", "train", "all"} and "cross_attention" in args.models:
        run(["prepare-fusion"])
    if args.phase in {"prepare", "smoke", "train", "all"} and any(
        v != "mean" for v in args.variants
    ):
        run(["prepare-mmd"])
    for name in args.models:
        for variant in args.variants:
            experiment = experiment_name(name, variant)
            path = root / f"configs/experiments/{experiment}.yaml"
            if args.phase in {"smoke", "train", "all"}:
                seeds = args.seeds
                if args.phase == "smoke":
                    config = yaml.safe_load(path.read_text("utf-8"))
                    config.update(experiment=f"smoke_{experiment}", epochs=2, patience=2)
                    path = root / f"tmp/fusion-smoke/{experiment}.yaml"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(yaml.safe_dump(config, sort_keys=False), "utf-8")
                    seeds = [42]
                for seed in seeds:
                    run(["train", "--config", str(path), "--seed", str(seed)])
            if args.phase in {"evaluate", "all"}:
                for seed in args.seeds:
                    run(["evaluate", "--run", str(root / f"outputs/{experiment}/seed_{seed}")])
    if args.phase in {"report", "all"}:
        variants = VARIANTS if any(v != "mean" for v in args.variants) else ["mean"]
        complete = all(
            (
                root / f"results/fusion/{experiment_name(model, variant)}/seed_{seed}/metrics.json"
            ).exists()
            for model in EXPERIMENTS
            for variant in variants
            for seed in [42, 43, 44]
        )
        if args.phase == "report" or complete:
            run(["report-fusion-mmd" if len(variants) == 3 else "report-fusion"])
        else:
            print(
                "Selected runs complete; the full report awaits all requested loss groups and seeds 42/43/44.",
                flush=True,
            )


if __name__ == "__main__":
    main()
