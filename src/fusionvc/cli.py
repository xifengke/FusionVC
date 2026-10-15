"""Command-line stages are explicit: prepare, train, evaluate, report."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(prog="fusionvc")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    sub.add_parser("prepare-mmd")
    sub.add_parser("prepare-scgpt")
    sub.add_parser("prepare-geneformer")
    sub.add_parser("prepare-fusion")
    training = sub.add_parser("train")
    training.add_argument("--config", type=Path, required=True)
    training.add_argument("--seed", type=int, required=True)
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--run", type=Path, required=True)
    sub.add_parser("report")
    sub.add_parser("report-mmd")
    sub.add_parser("report-scgpt")
    sub.add_parser("report-geneformer")
    sub.add_parser("report-fusion")
    sub.add_parser("report-fusion-mmd")
    sub.add_parser("prepare-distillation")
    sub.add_parser("freeze-final-selection")
    sub.add_parser("report-final-test")
    sub.add_parser("report-distillation")
    distillation = sub.add_parser("train-distillation")
    distillation.add_argument("--config", type=Path, required=True)
    distillation.add_argument("--seed", type=int, required=True)
    sub.add_parser("controls")
    sub.add_parser("controls-des")
    sub.add_parser("report-controls")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "prepare":
        from .preprocess import prepare

        prepare(root)
    elif args.command == "prepare-mmd":
        from .distribution import prepare_training_cell_cache

        prepare_training_cell_cache(root)
    elif args.command == "train":
        from .train import train

        train(root, args.config.resolve(), args.seed)
    elif args.command == "prepare-scgpt":
        from .scgpt import prepare_scgpt

        prepare_scgpt(root)
    elif args.command == "evaluate":
        from .evaluate import evaluate

        evaluate(root, args.run.resolve())
    elif args.command == "prepare-geneformer":
        from .geneformer import prepare_geneformer

        prepare_geneformer(root)
    elif args.command == "prepare-fusion":
        from .fusion_features import prepare_fusion

        prepare_fusion(root)
    elif args.command == "report-fusion":
        from .fusion_report import report_fusion

        report_fusion(root)
    elif args.command == "report-fusion-mmd":
        from .fusion_mmd_report import report_fusion_mmd

        report_fusion_mmd(root)
    elif args.command == "prepare-distillation":
        from .distillation import prepare_distillation

        prepare_distillation(root)
    elif args.command == "freeze-final-selection":
        from .final_selection import freeze_final_selection

        freeze_final_selection(root)
    elif args.command == "report-final-test":
        from .final_report import report_final_test

        report_final_test(root)
    elif args.command == "train-distillation":
        from .distillation import train_distillation

        train_distillation(root, args.config.resolve(), args.seed)
    elif args.command == "report-distillation":
        from .distillation_report import report_distillation

        report_distillation(root)
    elif args.command == "controls":
        from .controls import auxiliary_controls

        auxiliary_controls(root)
    elif args.command == "controls-des":
        from .control_evaluation import evaluate_no_perturbation

        evaluate_no_perturbation(root)
    elif args.command == "report-controls":
        from .control_report import report_controls

        report_controls(root)
    elif args.command == "report-mmd":
        from .mmd_report import report_mmd

        report_mmd(root)
    elif args.command == "report-scgpt":
        from .scgpt_report import report_scgpt

        report_scgpt(root)
    elif args.command == "report-geneformer":
        from .geneformer_report import report_geneformer

        report_geneformer(root)
    else:
        from .report import report

        report(root)


if __name__ == "__main__":
    main()
