"""Complete the no-perturbation DES and compare all four encoders."""

import os
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    os.environ.setdefault("POLARS_MAX_THREADS", "2")
    os.environ.setdefault("NUMBA_NUM_THREADS", "4")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    from fusionvc.control_evaluation import evaluate_no_perturbation
    from fusionvc.control_report import report_controls

    evaluate_no_perturbation(root)
    report_controls(root)


if __name__ == "__main__":
    main()
