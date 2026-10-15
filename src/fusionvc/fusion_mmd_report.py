"""Immutable-run comparison of seven architectures crossed with three losses."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .fusion_report import (
    BASE_MODELS,
    FUSION_MODELS,
    METRICS as OFFICIAL_METRICS,
    MODEL_LABELS,
    MODELS,
    RUN_SPECS,
    SEEDS,
    control_plan_matches,
    load_reference,
    validate_evaluation,
    validated_target_scores,
)
from .io import file_sha256, read_json, write_json

METRICS = [*OFFICIAL_METRICS, "energy_mmd_rms"]
VARIANTS = ["mean", "energy", "hybrid"]
LOSS_LABELS = {"mean": "Mean MSE", "energy": "MMD", "hybrid": "Mean MSE + 0.1 MMD"}
LOSSES = {
    "mean": "unpaired_pseudobulk_mse",
    "energy": "unpaired_energy_mmd",
    "hybrid": "unpaired_mean_energy_mmd",
}
LOG_RTOL, LOG_ATOL = 1e-6, 1e-9


def run_spec(model: str, variant: str):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown loss variant: {variant}")
    group, base, protocol = RUN_SPECS[model]
    experiment = (
        f"{base.removesuffix('_mean')}_{variant}"
        if model in {"mlp", "transformer"}
        else base + ("" if variant == "mean" else f"_{variant}")
    )
    return group, experiment, protocol


def validate_training_history(path: Path, config: dict, summary: dict) -> dict:
    frame = pd.read_csv(path)
    variant = config["variant"]
    columns = ["epoch", "train_mse", "val_mae", "val_mse", "train_loss"]
    if variant != "mean":
        columns += ["train_energy_mmd_rms"]
    if (
        len(frame) != summary["epochs_run"]
        or not np.array_equal(frame["epoch"].to_numpy(), np.arange(1, len(frame) + 1))
        or not np.isfinite(frame[columns].to_numpy()).all()
        or not 1 <= summary["best_epoch"] <= len(frame)
        or not np.isclose(
            frame.loc[frame.epoch == summary["best_epoch"], "val_mae"].iloc[0],
            summary["selection_validation_mae"],
            rtol=1e-12,
            atol=1e-12,
        )
    ):
        raise ValueError(f"Incomplete or inconsistent training history: {path}")
    expected = frame["train_mse"]
    if variant == "energy":
        expected = frame["train_energy_mmd_rms"]
    elif variant == "hybrid":
        expected = frame["train_mse"] + config["energy_weight"] * frame["train_energy_mmd_rms"]
    error = np.abs(frame["train_loss"] - expected)
    if not np.allclose(frame["train_loss"], expected, rtol=LOG_RTOL, atol=LOG_ATOL):
        raise ValueError(f"Logged objective does not match configured loss: {path}")
    return {
        "epochs": len(frame),
        "logged_loss_matches_objective": True,
        "max_absolute_objective_difference": float(error.max()),
        "tolerance": {"rtol": LOG_RTOL, "atol": LOG_ATOL},
    }


def load_run(root: Path, model: str, variant: str, seed: int):
    group, experiment, protocol = run_spec(model, variant)
    small = root / f"results/{group}/{experiment}/seed_{seed}"
    run = root / f"outputs/{experiment}/seed_{seed}"
    config = read_json(small / "config.json")
    training = read_json(small / "training.json")
    evaluation = read_json(small / "metrics.json")
    validate_evaluation(evaluation)
    expected = {
        "model": model,
        "variant": variant,
        "experiment": experiment,
        "seed": seed,
        "protocol": protocol,
        "results_group": group,
        "loss": LOSSES[variant],
        "selection": "validation_pseudobulk_mae",
        "width": 64,
        "control_bank_size": 32,
        "target_bank_size": 32,
        "distribution_eval_cells": 64,
        "energy_weight": 0.1,
    }
    if any(config.get(name) != value for name, value in expected.items()):
        raise ValueError(f"Unexpected loss comparison configuration: {small}")
    if model not in {"mlp", "transformer"} and config.get("encoder_frozen") is not True:
        raise ValueError(f"Expected frozen source encoders: {small}")
    if any(
        training.get(name) != expected[name]
        for name in ["model", "variant", "experiment", "seed", "protocol", "loss"]
    ):
        raise ValueError(f"Published training summary has different identity/loss: {small}")
    for path, value in [
        (run / "config.json", config),
        (run / "summary.json", training),
        (run / "evaluation/metrics.json", evaluation),
    ]:
        if read_json(path) != value:
            raise ValueError(f"Published artifact differs from original run: {path}")
    digest = file_sha256(run / "best.pt")
    if digest != training["checkpoint_sha256"] or digest != evaluation["checkpoint_sha256"]:
        raise ValueError(f"Checkpoint provenance mismatch: {run}")
    prediction = read_json(run / "evaluation/prediction-info.json")
    if (
        prediction["shape"] != [98927, 18080]
        or prediction["genes"] != 18080
        or prediction["control_sampling_seed"] != 2025
        or prediction["control_counts_match_official"] is not True
        or not control_plan_matches(
            run / "evaluation/control_sampling.npz",
            root / "outputs/01_mlp/seed_42/evaluation/control_sampling.npz",
        )
    ):
        raise ValueError(f"Full-cell prediction/control sampling mismatch: {run}")
    targets = validated_target_scores(small / "per_perturbation.csv", evaluation)
    for published, original in [
        (small / "per_perturbation.csv", run / "evaluation/results.csv"),
        (
            small / "distribution_per_perturbation.csv",
            run / "evaluation/distribution_per_perturbation.csv",
        ),
        (small / "history.csv", run / "history.csv"),
    ]:
        if file_sha256(published) != file_sha256(original):
            raise ValueError(f"Published scores/history differ from full run: {published}")
    distribution = pd.read_csv(small / "distribution_per_perturbation.csv")
    if (
        len(distribution) != 50
        or distribution.perturbation.duplicated().any()
        or set(distribution.perturbation) != set(targets.perturbation)
        or not np.isfinite(distribution.energy_mmd_rms).all()
        or not np.isclose(
            distribution.energy_mmd_rms.mean(),
            evaluation["distribution"]["energy_mmd_rms"],
            rtol=1e-12,
            atol=1e-12,
        )
        or evaluation["distribution"]["cells_per_set"] != 64
        or evaluation["distribution"]["prediction_seed"] != 2026
        or evaluation["distribution"]["observation_seed"] != 2027
    ):
        raise ValueError(f"Supplementary distribution evaluation mismatch: {small}")
    history = validate_training_history(small / "history.csv", config, training)
    targets = targets.merge(distribution, on="perturbation", validate="one_to_one")
    targets["model"], targets["variant"], targets["seed"] = model, variant, seed
    row = {
        "model": model,
        "variant": variant,
        "experiment": experiment,
        "seed": seed,
        **{name: evaluation["metrics"][name] for name in OFFICIAL_METRICS},
        "energy_mmd_rms": evaluation["distribution"]["energy_mmd_rms"],
        **{
            name: training[name]
            for name in [
                "parameters",
                "trainable_parameters",
                "best_epoch",
                "epochs_run",
                "training_seconds",
                "checkpoint_bytes",
            ]
        },
        "frozen_parameters": training.get("frozen_parameters", 0),
        "evaluation_seconds": evaluation["evaluation_seconds"],
        "peak_gpu_mib": training["peak_allocated_gpu_bytes"] / 1024**2,
        "feature_extraction_seconds_shared": training.get("feature_extraction_seconds", 0),
        "checkpoint_sha256": digest,
    }
    if row["parameters"] != row["trainable_parameters"] + row["frozen_parameters"]:
        raise ValueError(f"Parameter accounting mismatch: {small}")
    sources = [
        {"path": str((small / name).relative_to(root)), "sha256": file_sha256(small / name)}
        for name in [
            "metrics.json",
            "config.json",
            "training.json",
            "per_perturbation.csv",
            "distribution_per_perturbation.csv",
            "history.csv",
        ]
    ]
    return row, targets, sources, history


def paired_comparisons(targets: pd.DataFrame, reference_targets: pd.DataFrame) -> dict:
    averaged = targets.groupby(["model", "variant", "perturbation"])[METRICS].mean()
    rng = np.random.default_rng(2025)
    loss_contrasts, reference_contrasts = [], []

    def compare(model, variant, baseline, metrics, reference):
        candidate = averaged.loc[(model, variant)].sort_index()
        if len(candidate) != 50 or set(candidate.index) != set(baseline.index):
            raise ValueError("Paired intervals require the same 50 validation targets")
        baseline = baseline.reindex(candidate.index)
        draws = rng.integers(0, 50, (5000, 50))
        rows = []
        for metric in metrics:
            delta = (candidate[metric] - baseline[metric]).to_numpy()
            if not np.isfinite(delta).all():
                raise ValueError("Non-finite paired difference")
            low, high = np.quantile(delta[draws].mean(1), [0.025, 0.975])
            rows.append(
                {
                    "candidate": f"{model}/{variant}",
                    "reference": reference,
                    "metric": metric,
                    "difference": float(delta.mean()),
                    "target_bootstrap_95": [float(low), float(high)],
                    "candidate_better_targets": int(
                        (delta < 0).sum()
                        if metric in {"mae", "energy_mmd_rms"}
                        else (delta > 0).sum()
                    ),
                    "targets": 50,
                }
            )
        return rows

    for model in MODELS:
        for variant in ["energy", "hybrid"]:
            loss_contrasts += compare(
                model, variant, averaged.loc[(model, "mean")], METRICS, f"{model}/mean"
            )
    reference = reference_targets.set_index("perturbation")
    for model in FUSION_MODELS:
        for variant in VARIANTS:
            reference_contrasts += compare(
                model, variant, reference, OFFICIAL_METRICS, "no_perturbation"
            )
    return {
        "loss_contrasts": loss_contrasts,
        "no_perturbation_contrasts": reference_contrasts,
        "scope": "seed means within the same 50 validation targets; 5000 paired target bootstraps; conditional on fitted models, checkpoint selection and fixed controls; no multiplicity adjustment; seeds are not biological replicates",
        "bootstrap_seed": 2025,
        "test_used": False,
    }


def validate_prior_snapshot(root: Path) -> dict:
    """Check the pre-run snapshot without editing any historical artifact."""
    path = root / "results/fusion-mmd/prior-artifact-snapshot.json"
    snapshot = read_json(path)
    expected_runs = {
        (run_spec(model, variant)[1], seed)
        for model in BASE_MODELS
        for variant in VARIANTS
        for seed in SEEDS
    }
    expected_runs |= {
        (run_spec(model, "mean")[1], seed) for model in FUSION_MODELS for seed in SEEDS
    }
    expected_runs |= {(name, seed) for name in ["01_mlp", "02_transformer"] for seed in SEEDS}
    if (
        snapshot["test_used"] is not False
        or snapshot["checkpoint_count"] != 51
        or len(snapshot["checkpoints"]) != 51
        or len(snapshot["small_artifacts"]) != 351
        or {(r["experiment"], r["seed"]) for r in snapshot["checkpoints"]} != expected_runs
    ):
        raise ValueError(
            "Prior snapshot must cover exactly 51 old checkpoints and 351 old small files"
        )
    expected_small = set()
    for experiment, seed in expected_runs:
        group = (
            "baselines"
            if experiment in {"01_mlp", "02_transformer"}
            else "mmd"
            if experiment.startswith("mmd_")
            else "scgpt"
            if experiment.startswith("03_")
            else "geneformer"
            if experiment.startswith("04_")
            else "fusion"
        )
        names = [
            "config.json",
            "environment.json",
            "history.csv",
            "metrics.json",
            "per_perturbation.csv",
            "training.json",
        ]
        if group != "baselines":
            names.append("distribution_per_perturbation.csv")
        expected_small |= {f"results/{group}/{experiment}/seed_{seed}/{name}" for name in names}
    if {r["path"].replace("\\", "/") for r in snapshot["small_artifacts"]} != expected_small:
        raise ValueError("Prior small-artifact snapshot has missing or unexpected paths")
    checkpoint_checks, small_checks = [], []
    for key, output in [("checkpoints", checkpoint_checks), ("small_artifacts", small_checks)]:
        for row in snapshot[key]:
            relative = row["path"].replace("\\", "/")
            source = (root / relative).resolve()
            if not source.is_relative_to(root.resolve()):
                raise ValueError("Snapshot path is outside the project")
            if (
                key == "checkpoints"
                and relative != f"outputs/{row['experiment']}/seed_{row['seed']}/best.pt"
            ):
                raise ValueError("Snapshot checkpoint path disagrees with its run identity")
            actual = file_sha256(source)
            if actual != row["sha256"]:
                raise ValueError(f"Previously published artifact changed: {relative}")
            output.append({"path": relative, "sha256": actual, "hash_matches_snapshot": True})
    return {
        "prior_snapshot_sha256": file_sha256(path),
        "checkpoint_count": len(checkpoint_checks),
        "small_artifact_count": len(small_checks),
        "checkpoints": checkpoint_checks,
        "small_artifacts": small_checks,
        "all_hashes_match": True,
        "test_used": False,
    }


def material_passport(root: Path) -> dict:
    """Bind verification labels to current artifacts and exact rerun coverage."""
    prior = validate_prior_snapshot(root)
    folder = root / "results/fusion-mmd"
    paths = {
        "artifacts": folder / "artifact-verification.json",
        "reproducibility": folder / "reproducibility-check.json",
        "validation": folder / "validation-checks.json",
        "native_token_identity": root / "results/fusion/implementation-check.json",
    }
    receipts = {name: read_json(path) for name, path in paths.items() if path.exists()}
    reasons = [f"missing {name} receipt" for name in paths if name not in receipts]
    expected = {
        (run_spec(m, v)[1], s) for m in FUSION_MODELS for v in ["energy", "hybrid"] for s in SEEDS
    }
    artifacts = receipts.get("artifacts", {})
    new_runs = artifacts.get("new_runs", [])
    covered = {(r.get("experiment"), r.get("seed")) for r in new_runs}
    artifact_valid = (
        artifacts.get("all_checks_passed") is True
        and artifacts.get("test_used") is False
        and len(new_runs) == 18
        and covered == expected
        and artifacts.get("prior_artifacts", {}).get("prior_snapshot_sha256")
        == prior["prior_snapshot_sha256"]
        and all(
            r.get("checkpoint_sha256")
            == file_sha256(root / f"outputs/{r['experiment']}/seed_{r['seed']}/best.pt")
            for r in new_runs
        )
        and all(
            file_sha256(root / r["path"]) == r["sha256"]
            for r in artifacts.get("source_artifacts", [])
        )
        and len(artifacts.get("source_artifacts", [])) == 108
    )
    if not artifact_valid:
        reasons.append("18-run artifact coverage, sources or current hashes not verified")
    repro = receipts.get("reproducibility", {})
    repeated = repro.get("runs", [])
    expected_repro = {(experiment, 42) for experiment, _ in expected}
    repro_valid = (
        repro.get("all_tensors_equal") is True
        and repro.get("pretrained_assets_unchanged") is True
        and repro.get("test_used") is False
        and len(repeated) == 6
        and {(r.get("experiment"), r.get("seed")) for r in repeated} == expected_repro
    )
    if repro_valid:
        for row in repeated:
            original = root / f"outputs/{row['experiment']}/seed_42/best.pt"
            repeat = root / f"outputs/repro_check_{row['experiment']}/seed_42/best.pt"
            repro_valid = repro_valid and (
                all(
                    row.get(key) is True
                    for key in [
                        "every_tensor_equal",
                        "selected_epoch_equal",
                        "frozen_feature_provenance_equal",
                    ]
                )
                and original.exists()
                and repeat.exists()
                and row.get("original_checkpoint_sha256") == file_sha256(original)
                and row.get("repeated_checkpoint_sha256") == file_sha256(repeat)
            )
    if not repro_valid:
        reasons.append("six seed-42 tensor reruns or their current checkpoint hashes not verified")
    validation = receipts.get("validation", {})
    tests_valid = (
        validation.get("test_used") is False
        and validation.get("ruff_passed") is True
        and validation.get("tests", {}).get("exit_code") == 0
        and validation.get("tests", {}).get("passed", 0) > 0
    )
    if not tests_valid:
        reasons.append("test exit status and Ruff success not verified")
    identity = receipts.get("native_token_identity", {})
    identity_valid = (
        identity.get("test_used") is False
        and identity.get("all_checks_passed") is True
        and bool(identity.get("checks"))
        and all(v is True for v in identity.get("checks", {}).values())
        and identity.get("token_cache_manifest_sha256")
        == file_sha256(root / "data/processed/fusion-tokens-v1/manifest.json")
    )
    if not identity_valid:
        reasons.append("native token identity receipt does not bind current cache")
    return {
        "origin_skill": "academic-research-suite/experiment-agent",
        "stage": "run",
        "date": "2026-10-15",
        "version": "fusion-v1 / fusion-mmd-v1",
        "verification_status": "VERIFIED"
        if artifact_valid and repro_valid and tests_valid and identity_valid
        else "UNVERIFIED",
        "unverified_reasons": reasons,
        "prior_snapshot_sha256": prior["prior_snapshot_sha256"],
        "plot_visually_inspected": validation.get("comparison_plot_visually_inspected") is True,
        "receipts": {
            name: {"path": str(path.relative_to(root)), "sha256": file_sha256(path)}
            for name, path in paths.items()
            if path.exists()
        },
    }


def report_fusion_mmd(root: Path):
    reference, reference_targets = load_reference(root)
    rows, frames, sources = [], [], []
    for model in MODELS:
        for variant in VARIANTS:
            for seed in SEEDS:
                row, frame, source, _ = load_run(root, model, variant, seed)
                if set(frame.perturbation) != set(reference_targets.perturbation):
                    raise ValueError("Official comparison target sets differ")
                rows.append(row)
                frames.append(frame)
                sources.extend(source)
    runs, targets = pd.DataFrame(rows), pd.concat(frames, ignore_index=True)
    expected = {(m, v, s) for m in MODELS for v in VARIANTS for s in SEEDS}
    if (
        len(runs) != 63
        or set(runs[["model", "variant", "seed"]].itertuples(index=False, name=None)) != expected
    ):
        raise ValueError("Expected all 63 official model/loss/seed runs")
    for model in MODELS:
        for name in ["parameters", "trainable_parameters", "frozen_parameters"]:
            if runs.loc[runs.model == model, name].nunique() != 1:
                raise ValueError(f"Architecture parameter counts differ across loss/seed: {model}")
    grouped = runs.groupby(["model", "variant"])[METRICS].agg(["mean", "std", "count"])
    grouped.columns = ["_".join(column) for column in grouped.columns]
    contrasts = paired_comparisons(targets, reference_targets)
    folder = root / "results/fusion-mmd"
    folder.mkdir(parents=True, exist_ok=True)
    passport = material_passport(root)
    runs.to_csv(folder / "comparison_runs.csv", index=False)
    fusion = runs[runs.model.isin(FUSION_MODELS)]
    fusion.to_csv(folder / "runs.csv", index=False)
    fusion[fusion.variant != "mean"].to_csv(folder / "new_runs.csv", index=False)
    grouped.to_csv(folder / "summary.csv")
    reference_frame = reference_targets.copy()
    reference_frame["model"], reference_frame["variant"], reference_frame["seed"] = (
        "no_perturbation",
        "reference",
        np.nan,
    )
    pd.concat([targets, reference_frame], ignore_index=True).to_csv(
        folder / "per_perturbation.csv", index=False
    )
    write_json(folder / "paired-comparisons.json", contrasts)
    comparison = [
        {"model": "no_perturbation", "variant": "reference", "seeds": 0, **reference["metrics"]}
    ]
    for model in MODELS:
        for variant in VARIANTS:
            row = {"model": model, "variant": variant, "seeds": 3}
            for metric in METRICS:
                row[metric] = grouped.loc[(model, variant), f"{metric}_mean"]
                row[f"{metric}_std"] = grouped.loc[(model, variant), f"{metric}_std"]
            comparison.append(row)
    pd.DataFrame(comparison).to_csv(folder / "comparison.csv", index=False)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)
    for ax, metric, title in zip(
        axes,
        METRICS,
        ["MAE (lower)", "PDS (higher)", "DES (higher)", "Energy MMD (lower; sampled)"],
    ):
        for offset, variant in enumerate(VARIANTS):
            ax.bar(
                np.arange(7) + (offset - 1) * 0.25,
                [grouped.loc[(m, variant), f"{metric}_mean"] for m in MODELS],
                yerr=[grouped.loc[(m, variant), f"{metric}_std"] for m in MODELS],
                width=0.23,
                capsize=2,
                label=LOSS_LABELS[variant],
            )
        if metric in OFFICIAL_METRICS:
            ax.axhline(
                reference["metrics"][metric],
                color="black",
                linestyle="--",
                linewidth=1,
                label="No perturbation",
            )
        ax.set_xticks(np.arange(7), [MODEL_LABELS[m] for m in MODELS], rotation=40, ha="right")
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=7)
    fig.savefig(folder / "comparison.png", dpi=180)
    plt.close(fig)

    def metric_values(model, variant):
        row = grouped.loc[(model, variant)]
        return " | ".join(f"{row[f'{m}_mean']:.6f} ± {row[f'{m}_std']:.6f}" for m in METRICS)

    lines = [
        "# 三种融合的 MMD 与混合损失完整验证比较",
        "",
        "## Material Passport",
        "",
        "| Field | Value |",
        "| --- | --- |",
        "| Origin Skill | academic-research-suite/experiment-agent |",
        "| Stage | run |",
        "| Date | 2026-10-15 |",
        "| Version | fusion-v1 / fusion-mmd-v1 |",
        f"| Verification Status | {passport['verification_status']} |",
        "",
        "本轮新增三种融合 × 纯 MMD/混合损失 × 三个种子，共 18/18 次完整官方验证；已有 9 次融合均值组保持原样。",
        "三融合共有 27 次运行，与原四架构的 36 次三损失运行组成 63 次完整比较。测试集未使用。",
        "Verification Status 只表示新增产物、六组 seed-42 重训、身份检查和测试记录通过；不表示生物机制或正式测试集泛化已验证。",
        "",
        "## 三融合的三种损失",
        "",
        "| 方法 | 损失 | MAE ↓ | PDS ↑ | DES ↑ | Energy MMD ↓ |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
        "| 无扰动对照 | 固定一次完整评价 | "
        + " | ".join(f"{reference['metrics'][m]:.6f}" for m in OFFICIAL_METRICS)
        + " | — |",
    ]
    for model in FUSION_MODELS:
        for variant in VARIANTS:
            lines.append(
                f"| {MODEL_LABELS[model]} | {LOSS_LABELS[variant]} | {metric_values(model, variant)} |"
            )
    lines += [
        "",
        "神经模型为三个种子 42/43/44 的均值 ± 样本标准差，无扰动对照不复制为多个种子。",
        "Mean MSE 使用全训练细胞 pseudobulk 均值；纯 MMD 使用无配对预测/真实细胞集合，混合组为 Mean MSE + 0.1 MMD。",
        "训练 MMD 集合大小为 32+32，附加评价 Energy MMD 固定抽取 64+64 个细胞，不是官方 VCC 指标或综合得分。",
        "",
        "## 新损失相对自身均值组的配对差值",
        "",
        "以下是新损失减自身 mean；MAE/附加 MMD 负数更好，PDS/DES 正数更好。方括号为条件性 95% 靶点 bootstrap 区间。",
        "",
        "| 方法 | 损失 | MAE 差值 [95%] | PDS 差值 [95%] | DES 差值 [95%] | Energy MMD 差值 [95%] |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for model in FUSION_MODELS:
        for variant in ["energy", "hybrid"]:
            subset = [
                c for c in contrasts["loss_contrasts"] if c["candidate"] == f"{model}/{variant}"
            ]
            values = []
            for metric in METRICS:
                item = next(c for c in subset if c["metric"] == metric)
                low, high = item["target_bootstrap_95"]
                values.append(f"{item['difference']:+.6f} [{low:+.6f}, {high:+.6f}]")
            lines.append(
                f"| {MODEL_LABELS[model]} | {LOSS_LABELS[variant]} | " + " | ".join(values) + " |"
            )
    for model in FUSION_MODELS:
        for variant in ["energy", "hybrid"]:
            change = (
                grouped.loc[(model, variant), "mae_mean"] / grouped.loc[(model, "mean"), "mae_mean"]
                - 1
            ) * 100
            relation = "降低" if change < 0 else "升高" if change > 0 else "不变"
            item = next(
                c
                for c in contrasts["loss_contrasts"]
                if c["candidate"] == f"{model}/{variant}" and c["metric"] == "mae"
            )
            low, high = item["target_bootstrap_95"]
            interval = "跨零" if low <= 0 <= high else "未跨零"
            ref_mae = grouped.loc[(model, variant), "mae_mean"] - reference["metrics"]["mae"]
            lines += [
                "",
                f"{MODEL_LABELS[model]} 的 {LOSS_LABELS[variant]}：MAE 相对自身 mean {relation} {abs(change):.2f}%，靶点差值区间{interval}；与无扰动对照的 MAE 差值为 {ref_mae:+.6f}。",
            ]
    lines += [
        "",
        "区间先平均三个种子，再对 50 个相同靶点做 5,000 次配对重采样；仅对当前模型、checkpoint 选择、固定控制抽样和验证靶点成立。",
        "三个种子不是生物重复；同一验证集同时用于早停与架构/损失比较，存在选择乐观偏差。区间未纳入重新训练/选择不确定性，也未作多重比较校正。",
        "应逐项解释 MAE、PDS、DES 与附加分布指标，单项均值或区间变化不能当作整体能力显著提高。七架构损失配对和各融合损失对无扰动的完整区间见 JSON。",
        "",
        "![Seven architectures and three losses](comparison.png)",
        "",
        "## 七架构全部成绩",
        "",
        "| 方法 | 损失 | MAE ↓ | PDS ↑ | DES ↑ | Energy MMD ↓ |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for model in MODELS:
        for variant in VARIANTS:
            lines.append(
                f"| {MODEL_LABELS[model]} | {LOSS_LABELS[variant]} | {metric_values(model, variant)} |"
            )
    lines += [
        "",
        "## 训练选择、公平性与成本",
        "",
        "同一架构的三个损失组保持模型、冻结特征、解码宽度 64、优化器和预算上限相同；混合权重固定为 0.1。所有组仍按验证 pseudobulk MAE 选择 checkpoint，可能限制分布损失的收益。",
        "新增损失不会改变当前模型对同一扰动输出共同表达位移的限制；细胞异质性来自输入对照，没有新增细胞状态对应的独立扰动响应。",
        "跨架构的容量与输入信息粒度不同。Cross-Attention 使用同 2,048 个训练对照、按真实基因 ID 聚合的上下文 token 原型；不能将差异单独归因于注意力或预训练互补。",
        "所有输出均覆盖 18,080 基因，完整评价使用 98,927 验证细胞、50 个 held-out 扰动及相同控制抽样；输入仍受固定 256 候选基因和预训练词表覆盖限制。",
        "",
        "| 融合方法 | 损失 | 可训练参数 | 冻结参数 | 训练秒均值 ± SD | 评价秒均值 ± SD | 训练GPU峰值 MiB |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in FUSION_MODELS:
        for variant in VARIANTS:
            subset = fusion[(fusion.model == model) & (fusion.variant == variant)]
            first = subset.iloc[0]
            lines.append(
                f"| {MODEL_LABELS[model]} | {LOSS_LABELS[variant]} | {int(first.trainable_parameters):,} | {int(first.frozen_parameters):,} | {subset.training_seconds.mean():.2f} ± {subset.training_seconds.std():.2f} | {subset.evaluation_seconds.mean():.2f} ± {subset.evaluation_seconds.std():.2f} | {subset.peak_gpu_mib.max():.1f} |"
            )
    lines += [
        "",
        "训练 GPU 峰值不含离线编码器提取；共享特征提取不按每个种子重复计费，详见逐次成本 CSV。墙钟时间受缓存和硬件负载影响。",
        "旧 mean 报告保持原样；旧 51 个 checkpoint 与 351 个历史小产物逐一匹配训练前快照。新增运行及完整性检查保存在新目录。训练日志核验 train_loss 与相应 mean/energy 目标关系，允许 float32 舍入误差。",
        "",
        "[原 mean 主比较](../fusion/README.md)、[融合27次](runs.csv)、[新增18次](new_runs.csv)、[全部63次与成本](comparison_runs.csv)、[21组均值标准差](summary.csv)、",
        "[逐靶点结果](per_perturbation.csv)、[条件性配对区间](paired-comparisons.json)、[产物核验](artifact-verification.json)、[逐tensor复现](reproducibility-check.json)、",
        "[测试记录](validation-checks.json)、[旧产物快照](prior-artifact-snapshot.json)、[报告来源](report-provenance.json)。",
    ]
    if passport["unverified_reasons"]:
        lines += ["", "尚未满足的验证条件：" + "；".join(passport["unverified_reasons"]) + "。"]
    (folder / "README.md").write_text("\n".join(lines) + "\n", "utf-8")
    write_json(
        folder / "report-provenance.json",
        {
            "protocol": "fusion-v1; supplemental fusion-mmd-v1 report; existing architecture/loss results retained",
            "model_runs": 63,
            "fusion_runs": 27,
            "new_runs": 18,
            "complete": True,
            "prior_snapshot_sha256": passport["prior_snapshot_sha256"],
            "source_artifacts": sources,
            "material_passport": passport,
            "reference_metrics_sha256": file_sha256(
                root / "results/controls/no_perturbation/metrics.json"
            ),
            "reference_targets_sha256": file_sha256(
                root / "results/controls/no_perturbation/per_perturbation.csv"
            ),
            "test_used": False,
        },
    )
    print(pd.DataFrame(comparison).to_string(index=False), flush=True)
