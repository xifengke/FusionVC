"""Bounded, sequential distillation recipe; each stage is separately resumable."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=[
            "prepare",
            "smoke",
            "train",
            "evaluate",
            "diagnostics",
            "benchmark",
            "verify",
            "report",
            "all",
        ],
        default="all",
    )
    parser.add_argument("--seeds", nargs="+", type=int, choices=[42, 43, 44], default=[42, 43, 44])
    args = parser.parse_args()
    if args.phase == "all" and sorted(set(args.seeds)) != [42, 43, 44]:
        parser.error("The complete recipe requires all three formal seeds 42/43/44")
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    from fusionvc.io import file_sha256, read_json, write_json

    env = dict(
        os.environ,
        PYTHONPATH=str(root / "src"),
        PYTHONUTF8="1",
        PYTHONIOENCODING="utf-8",
        POLARS_MAX_THREADS="2",
        NUMBA_NUM_THREADS="4",
        OMP_NUM_THREADS="4",
    )
    log_dir = root / "outputs/distillation-recipe/logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    receipts = []

    def run(arguments, label):
        log = log_dir / f"{label}-{time.time_ns()}.log"
        command = [sys.executable, *arguments]
        started = time.perf_counter()
        print(f"DISTILL stage {label}; log={log}", flush=True)
        code, failure = None, None
        with log.open("w", encoding="utf-8") as stream:
            try:
                completed = subprocess.run(
                    command,
                    cwd=root,
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=1800,
                )
                code = completed.returncode
            except subprocess.TimeoutExpired:
                failure = "1800 second stage timeout"
        receipt = {
            "label": label,
            "command": command,
            "log": str(log.relative_to(root)),
            "output_sha256": file_sha256(log),
            "exit_code": code,
            "seconds": time.perf_counter() - started,
            "failure": failure,
        }
        receipts.append(receipt)
        write_json(
            root / "results/distillation/latest-recipe.json",
            {"stages": receipts, "test_used": False},
        )
        if code != 0 or failure:
            print(log.read_text("utf-8")[-12000:], flush=True)
            raise RuntimeError(f"Distillation stage failed: {receipt}")
        print(f"DISTILL stage {label} complete ({receipt['seconds']:.1f}s)", flush=True)
        return receipt, log.read_text("utf-8")

    def cli(arguments, label):
        return run(["-m", "fusionvc.cli", "--root", str(root), *arguments], label)

    path = root / "configs/experiments/08_distillation.yaml"
    if args.phase in {"prepare", "smoke", "all"}:
        cli(["prepare-distillation"], "prepare")
    if args.phase == "smoke":
        config = yaml.safe_load(path.read_text("utf-8"))
        config.update(experiment="smoke_08_distillation", epochs=2, patience=2)
        path = root / "tmp/distillation-smoke/08_distillation.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(config, sort_keys=False), "utf-8")
        cli(["train-distillation", "--config", str(path), "--seed", "42"], "smoke")
    if args.phase == "all":
        run(["scripts/check_distillation_reproducibility.py", "--zero-only"], "zero-kd-parity")
    if args.phase in {"train", "all"}:
        for seed in args.seeds:
            cli(["train-distillation", "--config", str(path), "--seed", str(seed)], f"train-{seed}")
    if args.phase in {"evaluate", "all"}:
        for seed in args.seeds:
            cli(
                ["evaluate", "--run", str(root / f"outputs/08_distillation/seed_{seed}")],
                f"evaluate-{seed}",
            )
    if args.phase in {"diagnostics", "all"}:
        run(["scripts/check_distillation_reproducibility.py"], "reproducibility")
    if args.phase in {"benchmark", "all"}:
        run(["scripts/benchmark_distillation.py"], "benchmark")
    if args.phase in {"verify", "all"}:
        tests, output = run(
            [
                "-m",
                "pytest",
                "-q",
                "tests/test_distillation.py",
                "--basetemp",
                str(root / "tmp/pytest-distillation"),
            ],
            "tests",
        )
        passed = re.search(r"(\d+) passed", output)
        warnings = re.search(r"(\d+) warnings?", output)
        if not passed:
            raise RuntimeError("Successful pytest exit did not include a passed-test count")
        ruff, _ = run(["-m", "ruff", "check", "--no-cache", "src", "scripts", "tests"], "ruff")
        write_json(
            root / "results/distillation/validation-checks.json",
            {
                "tests": {
                    **tests,
                    "passed": int(passed.group(1)),
                    "warnings": int(warnings.group(1)) if warnings else 0,
                },
                "ruff_passed": ruff["exit_code"] == 0,
                "comparison_plot_visually_inspected": False,
                "test_used": False,
            },
        )
        run(["scripts/verify_distillation_artifacts.py"], "artifact-verification")
    if args.phase in {"report", "all"}:
        cli(["report-distillation"], "report")
    if args.phase == "all":
        passport = read_json(root / "results/distillation/report-provenance.json")[
            "material_passport"
        ]
        if passport["verification_status"] != "VERIFIED":
            raise RuntimeError("Full distillation recipe finished without a VERIFIED report")


if __name__ == "__main__":
    main()
