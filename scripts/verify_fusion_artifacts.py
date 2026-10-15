"""Audit fusion checkpoints, frozen feature caches and complete official scores."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def verify_fusion_artifacts(root: Path):
    sys.path.insert(0, str(root / "src"))
    from fusionvc.fusion import FusionPredictor
    from fusionvc.fusion_features import load_fusion_features
    from fusionvc.fusion_report import (
        FUSION_MODELS,
        MODELS,
        RUN_SPECS,
        SEEDS,
        load_reference,
        load_run,
        validate_old_checkpoints,
    )
    from fusionvc.io import file_sha256, read_json, write_json

    reference, reference_targets = load_reference(root)
    feature_bundles = {}
    fusion_checks, comparison_checks = [], []
    gate_rows, gate_diagnostics = [], []
    genes = read_json(root / "data/processed/baselines-v1/genes.json")
    gene_indices = {gene: i for i, gene in enumerate(genes)}
    split = read_json(root / "configs/splits/vcc2025.json")
    if (
        len(split["train"]) != 150
        or len(split["validation"]) != 50
        or len(set(split["train"])) != 150
        or len(set(split["validation"])) != 50
        or set(split["train"]) & set(split["validation"])
    ):
        raise RuntimeError(
            "Expected disjoint, fixed train/validation target IDs for gate diagnostics"
        )
    preprocessing = read_json(root / "data/processed/baselines-v1/manifest.json")
    for model in MODELS:
        for seed in SEEDS:
            # This checks official evaluator versions, full genes/cells, source
            # scores, checkpoint hashes, aggregates and actual sampled controls.
            row, targets, sources = load_run(root, model, seed)
            if set(targets.perturbation) != set(reference_targets.perturbation):
                raise RuntimeError(f"Reference target mismatch: {model}, seed {seed}")
            comparison_checks.append(
                {
                    "model": model,
                    "experiment": row["experiment"],
                    "seed": seed,
                    "checkpoint_sha256": row["checkpoint_sha256"],
                    "targets_scored": len(targets),
                    "official_scores_match_full_run": True,
                    "aggregates_match_target_scores": True,
                    "full_genes_and_cells": True,
                    "control_plan_matches_original_mlp": True,
                    "source_artifacts": sources,
                    "test_used": False,
                }
            )
            if model not in FUSION_MODELS:
                continue
            experiment = RUN_SPECS[model][1]
            run = root / f"outputs/{experiment}/seed_{seed}"
            config = read_json(run / "config.json")
            summary = read_json(run / "summary.json")
            if model not in feature_bundles:
                feature_bundles[model] = load_fusion_features(root, config)
            features, assets, feature_manifest = feature_bundles[model]
            source_manifests = feature_manifest["source_manifests"]
            if (
                feature_manifest["test_used"] is not False
                or set(source_manifests) != {"scgpt", "geneformer"}
                or any(m["test_used"] is not False for m in source_manifests.values())
                or any(m["control_cells"] != 2048 for m in source_manifests.values())
                or feature_manifest["frozen_parameters"]
                != sum(m["frozen_parameters"] for m in source_manifests.values())
                or features.shape[0] != 18080
                or not np.isfinite(features).all()
            ):
                raise RuntimeError(f"Invalid frozen feature provenance: {model}")
            source_control_ids = [
                np.load(root / f"data/processed/{name}-frozen-v1/control_indices.npy")
                for name in ["scgpt", "geneformer"]
            ]
            same_controls = len(source_control_ids[0]) == 2048 and np.array_equal(
                *source_control_ids
            )
            if not same_controls:
                raise RuntimeError("The two frozen encoders used different training controls")
            token_checks = None
            if model == "cross_attention":
                token_manifest = feature_manifest["token_manifest"]
                token_checks = {
                    "control_cells": token_manifest["control_cells"],
                    "same_training_controls": np.array_equal(
                        assets["control_indices"], source_control_ids[0]
                    ),
                    "test_unused": token_manifest["test_used"] is False,
                    "sources_match": token_manifest["signature"]["source_manifests"]
                    == source_manifests,
                    "pooled_cache_parity": all(
                        0 <= value <= 1e-5
                        for value in token_manifest["pooled_cache_max_abs_difference"].values()
                    ),
                }
                for name in ["scgpt", "geneformer"]:
                    tokens = assets[f"{name}_tokens"]
                    indices = assets[f"{name}_gene_indices"]
                    valid = assets[f"{name}_valid"]
                    counts = assets[f"{name}_observation_counts"]
                    token_checks[f"{name}_identified_valid_tokens"] = bool(
                        tokens.ndim == 2
                        and len(tokens) == len(indices) == len(valid) == len(counts)
                        and valid.dtype == np.bool_
                        and np.isfinite(tokens).all()
                        and np.array_equal(valid, counts > 0)
                        and np.all((0 <= counts) & (counts <= 2048))
                        and np.all((0 <= indices) & (indices < 18080))
                        and valid.any()
                        and len(np.unique(indices[valid])) == int(valid.sum())
                    )
                if token_checks["control_cells"] != 2048 or not all(
                    value for key, value in token_checks.items() if key != "control_cells"
                ):
                    raise RuntimeError(f"Invalid identified gene-token prototypes: {token_checks}")
            checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
            predictor = FusionPredictor(config, assets, feature_manifest["feature_dims"], 18080)
            expected_buffers = dict(predictor.named_buffers())
            buffer_matches = all(
                name in checkpoint["state_dict"]
                and torch.equal(value, checkpoint["state_dict"][name])
                for name, value in expected_buffers.items()
            )
            predictor.load_state_dict(checkpoint["state_dict"], strict=True)
            trainable = sum(parameter.numel() for parameter in predictor.parameters())
            finite_state = all(
                bool(torch.isfinite(value).all())
                for value in checkpoint["state_dict"].values()
                if value.is_floating_point()
            )
            check = {
                "experiment": experiment,
                "model": model,
                "seed": seed,
                "checkpoint_sha256": file_sha256(run / "best.pt"),
                "frozen_feature_manifest_matches": summary["frozen_features"]
                == feature_manifest
                == checkpoint["frozen_features"],
                "preprocessing_matches": checkpoint["preprocessing"] == preprocessing,
                "checkpoint_config_matches": {**checkpoint["config"], "seed": seed} == config,
                "selected_epoch_matches": checkpoint["epoch"] == summary["best_epoch"],
                "checkpoint_seed_matches": checkpoint["seed"] == seed,
                "checkpoint_buffers_match_frozen_cache": buffer_matches,
                "state_tensors_finite": finite_state,
                "trainable_parameters_match": trainable == summary["trainable_parameters"],
                "frozen_parameters_match": summary["frozen_parameters"]
                == feature_manifest["frozen_parameters"],
                "total_parameters_match": summary["parameters"]
                == trainable + feature_manifest["frozen_parameters"],
                "source_control_cells_match": same_controls,
                "temporary_prediction_removed": not (run / "evaluation/predictions.npy").exists(),
                "token_prototypes": token_checks,
                "test_unused": feature_manifest["test_used"] is False,
            }
            if not all(value for value in check.values() if isinstance(value, (bool, np.bool_))):
                raise RuntimeError(f"Fusion artifact verification failed: {check}")
            if model == "gated":
                predictor.eval()
                for split_name in ["train", "validation"]:
                    target_names = split[split_name]
                    indices = [gene_indices[name] for name in target_names]
                    # Only gene IDs, the supplied target embeddings and frozen
                    # contexts are read here; no observed response labels.
                    with torch.no_grad():
                        alpha = (
                            predictor.gate_weights(torch.from_numpy(features[indices]))
                            .numpy()
                            .ravel()
                        )
                    if not np.isfinite(alpha).all() or not np.all((alpha >= 0) & (alpha <= 1)):
                        raise RuntimeError(f"Invalid gate weights: seed {seed}, {split_name}")
                    for name, value in zip(target_names, alpha):
                        gate_rows.append(
                            {
                                "experiment": experiment,
                                "seed": seed,
                                "split": split_name,
                                "perturbation": name,
                                "alpha_scgpt": float(value),
                                "alpha_geneformer": 1.0 - float(value),
                                "checkpoint_sha256": check["checkpoint_sha256"],
                            }
                        )
                    gate_diagnostics.append(
                        {
                            "seed": seed,
                            "split": split_name,
                            "targets": len(alpha),
                            "alpha_scgpt_min": float(alpha.min()),
                            "alpha_scgpt_max": float(alpha.max()),
                            "alpha_scgpt_mean": float(alpha.mean()),
                            "alpha_scgpt_std_population": float(alpha.std()),
                            "fraction_alpha_below_0_1": float(np.mean(alpha < 0.1)),
                            "fraction_alpha_above_0_9": float(np.mean(alpha > 0.9)),
                        }
                    )
            fusion_checks.append(check)
            del predictor, checkpoint
    original_checks = validate_old_checkpoints(root)
    gate_frame = pd.DataFrame(gate_rows)
    if len(gate_frame) != 600 or gate_frame.duplicated(["seed", "split", "perturbation"]).any():
        raise RuntimeError(
            "Expected 600 distinct gate weights across three seeds and both target splits"
        )
    gate_path = root / "results/fusion/gate_weights.csv"
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    gate_frame.to_csv(gate_path, index=False)
    result = {
        "protocol": "fusion-v1",
        "fusion_runs": fusion_checks,
        "comparison_runs": comparison_checks,
        "original_runs": original_checks,
        "gate_weights": {
            "path": str(gate_path.relative_to(root)),
            "sha256": file_sha256(gate_path),
            "rows": len(gate_frame),
            "diagnostics": gate_diagnostics,
            "scope": "descriptive per-target scGPT branch alpha and complementary Geneformer weight from each fitted gate; target IDs and frozen features only; no response labels read; no causal or mechanistic interpretation",
        },
        "no_perturbation_reference": {
            "metrics_sha256": file_sha256(root / "results/controls/no_perturbation/metrics.json"),
            "targets_sha256": file_sha256(
                root / "results/controls/no_perturbation/per_perturbation.csv"
            ),
            "original_receipt_verified": True,
            "original_reference_data_sources_verified": True,
            "control_plan_matches_original_mlp": True,
            "gene_count": reference["gene_count"],
            "predicted_cell_count": reference["predicted_cell_count"],
            "test_used": False,
        },
        "verification_scope": "saved official scores, checkpoint tensors, immutable feature caches and sampling; does not re-run predictions or establish biological independence",
        "all_checks_passed": True,
        "test_used": False,
    }
    write_json(root / "results/fusion/artifact-verification.json", result)
    print(
        f"Verified {len(fusion_checks)} fusion runs, {len(comparison_checks)} full mean-loss comparisons "
        f"and {len(original_checks)} unchanged original checkpoints",
        flush=True,
    )
    return result


def main():
    verify_fusion_artifacts(Path(__file__).resolve().parents[1])


if __name__ == "__main__":
    main()
