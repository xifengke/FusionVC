"""Export the fixed seed42 student or predict from its self-contained demo files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fusionvc.student_inference import export_student_bundle, predict_student_bundle  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=ROOT / "outputs/deployment/student-v1")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--control-cells", type=int, default=256)
    parser.add_argument("--target")
    parser.add_argument("--cells", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/demo/student_prediction.npz")
    args = parser.parse_args()
    if not args.export and not args.target:
        parser.error("Choose --export, --target, or both")
    if args.export:
        manifest = export_student_bundle(
            ROOT, ROOT / "outputs/08_distillation/seed_42", args.bundle, args.control_cells
        )
        print(f"Student package: {args.bundle}; {manifest['total_bytes']} bytes", flush=True)
    if args.target:
        expression, genes, metadata = predict_student_bundle(
            args.bundle, args.target, args.cells, args.seed, args.device
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output,
            expression=expression,
            genes=np.asarray(genes),
            metadata=json.dumps(metadata, ensure_ascii=False),
        )
        print(f"Standalone student prediction {expression.shape}: {args.output}", flush=True)


if __name__ == "__main__":
    main()
