"""Freeze validation-only model selection before opening held-out test responses."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path

from .io import file_sha256, read_json, write_json

MODELS = ["mlp", "transformer", "scgpt", "geneformer", "concat", "gated", "cross_attention"]
VARIANTS = ["mean", "energy", "hybrid"]
SEEDS = [42, 43, 44]
PRETEST_RECEIPTS = [
    "results/fusion-mmd/report-provenance.json",
    "results/fusion-mmd/artifact-verification.json",
    "results/fusion-mmd/reproducibility-check.json",
    "results/fusion-mmd/validation-checks.json",
    "results/fusion/implementation-check.json",
    "results/distillation/teacher-selection.json",
    "results/distillation/report-provenance.json",
    "results/distillation/artifact-verification.json",
    "results/distillation/reproducibility-check.json",
    "results/distillation/zero-kd-parity.json",
    "results/distillation/validation-checks.json",
    "results/distillation/cost-benchmark.json",
]


def rank_validation_groups(rows: list[dict]) -> list[dict]:
    expected = {(m, v, s) for m in MODELS for v in VARIANTS for s in SEEDS}
    identities = [(r["model"], r["variant"], r["seed"]) for r in rows]
    if len(rows) != 63 or len(set(identities)) != 63 or set(identities) != expected:
        raise ValueError(
            "Final selection requires exactly seven architectures, three losses and three seeds"
        )
    ranked = []
    for model in MODELS:
        for variant in VARIANTS:
            subset = sorted(
                (r for r in rows if r["model"] == model and r["variant"] == variant),
                key=lambda r: r["seed"],
            )
            experiments = {r["experiment"] for r in subset}
            scores = [float(r["mae"]) for r in subset]
            if len(experiments) != 1 or not all(math.isfinite(x) and x >= 0 for x in scores):
                raise ValueError(
                    "Candidate groups require one experiment and finite validation MAE"
                )
            ranked.append(
                {
                    "model": model,
                    "variant": variant,
                    "experiment": experiments.pop(),
                    "validation_mae_mean": math.fsum(scores) / 3,
                    "validation_mae_by_seed": dict(zip(map(str, SEEDS), scores)),
                }
            )
    return sorted(
        ranked, key=lambda r: (MODELS.index(r["model"]), r["validation_mae_mean"], r["experiment"])
    )


def freeze_final_selection(root: Path) -> dict:
    from .distillation_report import verification_passport
    from .fusion_mmd_report import load_run, material_passport, run_spec

    if material_passport(root)["verification_status"] != "VERIFIED":
        raise RuntimeError("Finish fusion MMD verification before freezing final selection")
    if verification_passport(root)["verification_status"] != "VERIFIED":
        raise RuntimeError(
            "Finish the full distillation audit, fairness, tests and cost checks before test selection"
        )
    for relative in [
        "results/fusion-mmd/report-provenance.json",
        "results/distillation/report-provenance.json",
    ]:
        report = read_json(root / relative)
        if (
            report.get("complete") is not True
            or report.get("material_passport", {}).get("verification_status") != "VERIFIED"
        ):
            raise RuntimeError(
                "Publish both complete VERIFIED validation-stage reports before test selection"
            )
    distillation_folder = root / "results/distillation"
    audit = read_json(distillation_folder / "artifact-verification.json")
    repro = read_json(distillation_folder / "reproducibility-check.json")
    if audit.get("all_checks_passed") is not True or repro.get("all_tensors_equal") is not True:
        raise RuntimeError("Finish distillation artifact and tensor reproducibility checks first")
    if audit.get("test_used") is not False or repro.get("test_used") is not False:
        raise ValueError("Final model selection must precede test use")
    rows, source_rows = [], {}
    for model in MODELS:
        for variant in VARIANTS:
            for seed in SEEDS:
                row, _, _, _ = load_run(root, model, variant, seed)
                rows.append(row)
                source_rows[(model, variant, seed)] = row
    ranked = rank_validation_groups(rows)
    chosen = [next(r for r in ranked if r["model"] == model) for model in MODELS]
    runs = []

    def add_run(model, variant, seed, role):
        group, experiment, _ = run_spec(model, variant)
        row = source_rows[(model, variant, seed)]
        metrics_path = root / f"results/{group}/{experiment}/seed_{seed}/metrics.json"
        checkpoint = root / f"outputs/{experiment}/seed_{seed}/best.pt"
        runs.append(
            {
                "label": f"{model}_{variant}_seed_{seed}",
                "kind": "checkpoint",
                "model": model,
                "variant": variant,
                "seed": seed,
                "experiment": experiment,
                "role": role,
                "checkpoint_path": checkpoint.relative_to(root).as_posix(),
                "checkpoint_sha256": file_sha256(checkpoint),
                "validation_metrics_path": metrics_path.relative_to(root).as_posix(),
                "validation_metrics_sha256": file_sha256(metrics_path),
                "validation_mae": row["mae"],
            }
        )

    for choice in chosen:
        for seed in SEEDS:
            add_run(choice["model"], choice["variant"], seed, "main")
    if chosen[0]["variant"] != "mean":
        for seed in SEEDS:
            add_run("mlp", "mean", seed, "kd_control")
    for seed in SEEDS:
        experiment = "08_distillation"
        checkpoint = root / f"outputs/{experiment}/seed_{seed}/best.pt"
        metrics_path = distillation_folder / f"{experiment}/seed_{seed}/metrics.json"
        metrics = read_json(metrics_path)
        training = read_json(distillation_folder / f"{experiment}/seed_{seed}/training.json")
        digest = file_sha256(checkpoint)
        if (
            metrics["test_used"] is not False
            or metrics["gene_count"] != 18080
            or metrics["predicted_cell_count"] != 98927
            or digest != metrics["checkpoint_sha256"]
            or digest != training["checkpoint_sha256"]
        ):
            raise ValueError("Student checkpoint and complete validation scores differ")
        runs.append(
            {
                "label": f"distillation_mean_kd_seed_{seed}",
                "kind": "checkpoint",
                "model": "distillation",
                "variant": "mean_kd",
                "seed": seed,
                "experiment": experiment,
                "role": "main",
                "checkpoint_path": checkpoint.relative_to(root).as_posix(),
                "checkpoint_sha256": digest,
                "validation_metrics_path": metrics_path.relative_to(root).as_posix(),
                "validation_metrics_sha256": file_sha256(metrics_path),
                "validation_mae": metrics["metrics"]["mae"],
            }
        )
    runs.append(
        {
            "label": "no_perturbation",
            "kind": "no_perturbation",
            "model": "no_perturbation",
            "variant": "reference",
            "role": "reference",
        }
    )
    candidate_runs = []
    for row in rows:
        group, experiment, _ = run_spec(row["model"], row["variant"])
        metrics_path = root / f"results/{group}/{experiment}/seed_{row['seed']}/metrics.json"
        candidate_runs.append(
            {
                "model": row["model"],
                "variant": row["variant"],
                "experiment": experiment,
                "seed": row["seed"],
                "mae": row["mae"],
                "checkpoint_sha256": row["checkpoint_sha256"],
                "validation_metrics_path": metrics_path.relative_to(root).as_posix(),
                "validation_metrics_sha256": file_sha256(metrics_path),
            }
        )
    payload = {
        "protocol": "final-test-v1",
        "producer": "fusionvc.final_selection-v1",
        "frozen": True,
        "test_used": False,
        "selection_basis": "official_validation_mae_mean_over_seeds",
        "selection_rule": "lowest three-seed official validation MAE group per architecture; exact ties by experiment lexicographic order; no seed selection",
        "seeds": SEEDS,
        "candidate_groups": ranked,
        "candidate_runs": candidate_runs,
        "selected_groups": chosen,
        "student": {
            "model": "distillation",
            "variant": "mean_kd",
            "experiment": "08_distillation",
            "kd_weight": 1.0,
        },
        "teacher_selection": read_json(distillation_folder / "teacher-selection.json"),
        "validation_receipts": [
            {"path": p, "sha256": file_sha256(root / p)} for p in PRETEST_RECEIPTS
        ],
        "runs": runs,
    }
    path = root / "docs/final-model-selection.json"
    if path.exists():
        previous = read_json(path)
        if {k: v for k, v in previous.items() if k != "frozen_at_utc"} != payload:
            raise ValueError(
                "Final selection already frozen with different sources or models; do not alter it after test access"
            )
        return previous
    payload["frozen_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(path, payload)
    print(
        f"FROZEN {len(runs) - 1} checkpoint evaluations plus no-perturbation; test expression has not been accessed",
        flush=True,
    )
    return payload
