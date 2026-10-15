"""Sequential subprocesses for exactly the validation-frozen test selection."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fusionvc.final_evaluation import evaluate_final_run, prepare_test_cache, validate_selection  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--phase", choices=["prepare", "evaluate", "all"], default="all")
    parser.add_argument(
        "--label", help="Evaluate only this frozen label, without changing selection"
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = args.root.resolve()
    selection_path = (args.selection or root / "docs/final-model-selection.json").resolve()
    selection, _ = validate_selection(root, selection_path)
    if args.label and args.label not in {row["label"] for row in selection["runs"]}:
        raise ValueError("Unknown frozen final label")
    if args.worker:
        if args.phase == "prepare":
            matrix, *_ = prepare_test_cache(root, selection_path)
            matrix._mmap.close()
        else:
            if not args.label:
                raise ValueError("A final evaluation worker requires one frozen label")
            evaluate_final_run(root, selection_path, args.label)
        return
    env = dict(
        os.environ,
        PYTHONPATH=str(root / "src"),
        PYTHONUTF8="1",
        PYTHONIOENCODING="utf-8",
        POLARS_MAX_THREADS="2",
        NUMBA_NUM_THREADS="4",
        OMP_NUM_THREADS="4",
    )
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--root",
        str(root),
        "--selection",
        str(selection_path),
        "--worker",
    ]
    if args.phase in {"prepare", "all"}:
        subprocess.run(
            command + ["--phase", "prepare"], cwd=root, env=env, check=True, timeout=1800
        )
    if args.phase in {"evaluate", "all"}:
        for row in selection["runs"]:
            if not args.label or row["label"] == args.label:
                subprocess.run(
                    command + ["--phase", "evaluate", "--label", row["label"]],
                    cwd=root,
                    env=env,
                    check=True,
                    timeout=1800,
                )


if __name__ == "__main__":
    main()
