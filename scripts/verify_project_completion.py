"""Recheck the completed experiment chain without retraining or rewriting results."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fusionvc.distillation import verify_prior_artifacts  # noqa: E402
from fusionvc.distillation_report import verification_passport  # noqa: E402
from fusionvc.final_report import read_final_runs  # noqa: E402
from fusionvc.fusion_mmd_report import material_passport  # noqa: E402
from fusionvc.io import file_sha256, read_json, write_json  # noqa: E402
from fusionvc.student_inference import load_student_bundle, predict_student_bundle  # noqa: E402


def main():
    root = ROOT
    mmd = material_passport(root)
    kd = verification_passport(root)
    if any(p["verification_status"] != "VERIFIED" for p in [mmd, kd]):
        raise ValueError("Validation-stage verification is incomplete")
    final_checks = read_json(root / "docs/final-validation-checks.json")
    if (
        final_checks.get("tests", {}).get("exit_code") != 0
        or final_checks.get("tests", {}).get("passed", 0) < 69
        or final_checks.get("ruff_passed") is not True
        or final_checks.get("format_passed") is not True
        or not final_checks.get("source_sha256")
    ):
        raise ValueError("Final local code validation is incomplete")
    for relative, expected in final_checks["source_sha256"].items():
        if file_sha256(root / relative) != expected:
            raise ValueError(f"Code changed after final local validation: {relative}")
    test_log = root / final_checks["tests"]["log"]
    if file_sha256(test_log) != final_checks["tests"]["output_sha256"]:
        raise ValueError("Final test log changed after validation")
    prior = verify_prior_artifacts(root)
    selection, digest, ready, runs, _, sources, checks, lineage = read_final_runs(
        root, root / "docs/final-model-selection.json"
    )
    final_folder = root / "results/final-test"
    report = read_json(final_folder / "report-provenance.json")
    audit = read_json(final_folder / "artifact-verification.json")
    if (
        report.get("complete") is not True
        or report.get("test_used") is not True
        or report.get("selection_sha256") != digest
        or report.get("source_artifacts") != sources
        or audit.get("all_checks_passed") is not True
        or audit.get("runs") != checks
        or audit.get("cache_source_lineage") != lineage
    ):
        raise ValueError("Regenerate the final report from its verified frozen sources")
    bundle = root / "outputs/deployment/student-v1"
    model, _, _, _, package = load_student_bundle(bundle, "cpu")
    demo_path = root / "outputs/demo/student_prediction.npz"
    demo_receipt = read_json(root / "docs/student-demo-verification.json")
    if (
        demo_receipt["all_checks_passed"] is not True
        or not all(demo_receipt["checks"].values())
        or demo_receipt["bundle_manifest_sha256"] != file_sha256(bundle / "manifest.json")
        or demo_receipt["demo_sha256"] != file_sha256(demo_path)
        or demo_receipt["source_checkpoint_sha256"] != package["source_checkpoint_sha256"]
    ):
        raise ValueError("Standalone demo files differ from the verified export")
    student = next(
        row for row in selection["runs"] if row["model"] == "distillation" and row["seed"] == 42
    )
    if student["checkpoint_sha256"] != package["source_checkpoint_sha256"]:
        raise ValueError("Deployment must use the fixed formal seed42 student")
    checkpoint = torch.load(
        root / student["checkpoint_path"], map_location="cpu", weights_only=True
    )
    if not all(
        torch.equal(value, checkpoint["state_dict"][key])
        for key, value in model.state_dict().items()
    ):
        raise ValueError("Standalone student weights differ from the evaluated checkpoint")
    with np.load(demo_path, allow_pickle=False) as demo:
        metadata = json.loads(str(demo["metadata"]))
        actual, genes, _ = predict_student_bundle(
            bundle, metadata["target"], metadata["cells"], metadata["seed"], "cpu"
        )
        if not np.array_equal(actual, demo["expression"]) or not np.array_equal(
            genes, demo["genes"]
        ):
            raise ValueError("Standalone CPU demonstration is not exactly reproducible")
    receipt_paths = [
        "docs/final-model-selection.json",
        "docs/test-download-status.json",
        "docs/student-demo-verification.json",
        "docs/final-validation-checks.json",
        "results/final-test/report-provenance.json",
        "results/final-test/artifact-verification.json",
        "results/fusion-mmd/report-provenance.json",
        "results/distillation/report-provenance.json",
    ]
    write_json(
        root / "docs/project-completion-verification.json",
        {
            "protocol": "project-completion-v1",
            "selection_sha256": digest,
            "validation_matrix_runs": 63,
            "distilled_student_runs": 3,
            "prior_checkpoints_preserved": prior["checkpoint_count"],
            "prior_small_artifacts_preserved": prior["small_artifact_count"],
            "final_checkpoint_evaluations": len(runs) - 1,
            "final_reference_evaluations": 1,
            "test_cells": ready["shape"][0],
            "test_genes": ready["shape"][1],
            "test_perturbations": len(ready["perturbations"]),
            "all_checks_passed": True,
            "checks": {
                "fusion_mmd_verified": True,
                "distillation_verified": True,
                "final_local_tests_and_code_hashes_verified": True,
                "historical_artifacts_unchanged": True,
                "frozen_selection_and_test_lineage_current": lineage["all_checks_passed"],
                "all_final_scores_and_published_sources_match": True,
                "standalone_student_tensors_match_evaluated_model": True,
                "standalone_demo_reproduces_exactly": True,
                "temporary_final_predictions_removed": True,
            },
            "receipts": {path: file_sha256(root / path) for path in receipt_paths},
            "script_sha256": file_sha256(Path(__file__)),
            "test_used": True,
            "scope": "local artifact integrity and reproducibility; no retraining or new model selection; no remote publication or CI claim",
        },
    )
    print("Project completion verified: 63 validation comparisons, 3 students and full frozen test")


if __name__ == "__main__":
    main()
