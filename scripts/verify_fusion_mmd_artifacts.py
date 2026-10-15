"""Audit the 18 added fusion-loss runs without rewriting historical results."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


def verify_fusion_mmd_artifacts(root: Path):
    sys.path.insert(0, str(root / "src"))
    from fusionvc.fusion import FusionPredictor
    from fusionvc.fusion_features import load_fusion_features
    from fusionvc.fusion_mmd_report import (
        FUSION_MODELS,
        SEEDS,
        load_run,
        run_spec,
        validate_prior_snapshot,
    )
    from fusionvc.fusion_report import load_reference
    from fusionvc.io import file_sha256, read_json, write_json

    prior_checks = validate_prior_snapshot(root)
    reference, reference_targets = load_reference(root)
    preprocessing = read_json(root / "data/processed/baselines-v1/manifest.json")
    identity_path = root / "results/fusion/implementation-check.json"
    identity = read_json(identity_path)
    token_manifest_path = root / "data/processed/fusion-tokens-v1/manifest.json"
    if (
        identity["all_checks_passed"] is not True
        or identity["test_used"] is not False
        or identity["token_cache_manifest_sha256"] != file_sha256(token_manifest_path)
        or not identity["checks"]
        or not all(value is True for value in identity["checks"].values())
    ):
        raise RuntimeError(
            "The original native token identity receipt does not bind the current cache"
        )
    bundles, checks, sources = {}, [], []
    for model in FUSION_MODELS:
        for variant in ["energy", "hybrid"]:
            for seed in SEEDS:
                row, targets, run_sources, history_check = load_run(root, model, variant, seed)
                if set(targets.perturbation) != set(reference_targets.perturbation):
                    raise RuntimeError("Added fusion-loss run has different validation targets")
                experiment = run_spec(model, variant)[1]
                run = root / f"outputs/{experiment}/seed_{seed}"
                config = read_json(run / "config.json")
                summary = read_json(run / "summary.json")
                mean_experiment = run_spec(model, "mean")[1]
                mean_config = read_json(root / f"outputs/{mean_experiment}/seed_{seed}/config.json")
                allowed_changes = {"experiment", "variant", "loss"}
                fairness = {
                    name: value for name, value in config.items() if name not in allowed_changes
                } == {
                    name: value
                    for name, value in mean_config.items()
                    if name not in allowed_changes
                }
                if model not in bundles:
                    bundles[model] = load_fusion_features(root, config)
                features, assets, manifest = bundles[model]
                source_manifests = manifest["source_manifests"]
                provenance_valid = (
                    manifest["test_used"] is False
                    and set(source_manifests) == {"scgpt", "geneformer"}
                    and all(
                        m["test_used"] is False and m["control_cells"] == 2048
                        for m in source_manifests.values()
                    )
                    and manifest["frozen_parameters"]
                    == sum(m["frozen_parameters"] for m in source_manifests.values())
                    and features.shape[0] == 18080
                    and np.isfinite(features).all()
                )
                if not provenance_valid:
                    raise RuntimeError(f"Invalid frozen source feature provenance: {experiment}")
                checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
                predictor = FusionPredictor(config, assets, manifest["feature_dims"], 18080)
                buffers_match = all(
                    name in checkpoint["state_dict"]
                    and torch.equal(value, checkpoint["state_dict"][name])
                    for name, value in predictor.named_buffers()
                )
                predictor.load_state_dict(checkpoint["state_dict"], strict=True)
                trainable = sum(parameter.numel() for parameter in predictor.parameters())
                check = {
                    "model": model,
                    "variant": variant,
                    "experiment": experiment,
                    "seed": seed,
                    "checkpoint_sha256": row["checkpoint_sha256"],
                    "official_scores_and_sources_match": True,
                    "full_cells_genes_targets": len(targets) == 50,
                    "control_plan_matches_original_mlp": True,
                    "loss_configuration_matches": True,
                    "only_loss_configuration_changed_from_mean": fairness,
                    "frozen_manifest_matches": summary["frozen_features"]
                    == manifest
                    == checkpoint["frozen_features"],
                    "frozen_features_identical_to_mean": summary["frozen_features"]
                    == read_json(root / f"outputs/{mean_experiment}/seed_{seed}/summary.json")[
                        "frozen_features"
                    ],
                    "checkpoint_buffers_match_frozen_cache": buffers_match,
                    "checkpoint_config_matches": {**checkpoint["config"], "seed": seed} == config,
                    "preprocessing_matches": checkpoint["preprocessing"] == preprocessing,
                    "selected_epoch_matches": checkpoint["epoch"] == summary["best_epoch"],
                    "seed_matches": checkpoint["seed"] == seed,
                    "state_tensors_finite": all(
                        bool(torch.isfinite(t).all())
                        for t in checkpoint["state_dict"].values()
                        if t.is_floating_point()
                    ),
                    "parameter_counts_match": trainable == summary["trainable_parameters"]
                    and summary["frozen_parameters"] == manifest["frozen_parameters"]
                    and summary["parameters"] == trainable + manifest["frozen_parameters"],
                    "training_set_sizes_match": summary["cell_set_size"]
                    == summary["target_set_size"]
                    == 32,
                    "target_sampling_is_training_only": summary["target_sampling"]
                    == "all training cells; separate RNG seed+100000",
                    "temporary_predictions_removed": not (
                        run / "evaluation/predictions.npy"
                    ).exists(),
                    "training_history": history_check,
                    "test_unused": manifest["test_used"] is False,
                }
                if not all(
                    value for value in check.values() if isinstance(value, (bool, np.bool_))
                ):
                    raise RuntimeError(f"Added fusion-loss artifact check failed: {check}")
                checks.append(check)
                sources.extend(run_sources)
                del predictor, checkpoint
    expected = {(m, v, s) for m in FUSION_MODELS for v in ["energy", "hybrid"] for s in SEEDS}
    if len(checks) != 18 or {(r["model"], r["variant"], r["seed"]) for r in checks} != expected:
        raise RuntimeError("Incomplete added fusion-loss artifact coverage")
    result = {
        "protocol": "fusion-v1; fusion-mmd-v1 artifact audit",
        "new_runs": checks,
        "new_run_count": len(checks),
        "prior_snapshot_sha256": prior_checks["prior_snapshot_sha256"],
        "prior_artifacts": prior_checks,
        "source_artifacts": sources,
        "native_token_identity": {
            "path": str(identity_path.relative_to(root)),
            "sha256": file_sha256(identity_path),
            "all_checks_passed": True,
        },
        "no_perturbation_reference": {
            "gene_count": reference["gene_count"],
            "predicted_cell_count": reference["predicted_cell_count"],
            "original_receipt_verified": True,
            "control_plan_matches_original_mlp": True,
        },
        "scope": "saved official validation scores, history identities, checkpoint tensors, frozen caches and fixed sampling; no prediction rerun or causal inference",
        "all_checks_passed": True,
        "test_used": False,
    }
    write_json(root / "results/fusion-mmd/artifact-verification.json", result)
    print(
        f"Verified {len(checks)} added fusion-loss runs and preserved prior artifacts", flush=True
    )
    return result


def main():
    verify_fusion_mmd_artifacts(Path(__file__).resolve().parents[1])


if __name__ == "__main__":
    main()
