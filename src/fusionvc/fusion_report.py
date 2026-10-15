"""Validation-only comparison of three frozen-feature fusion heads."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .io import file_sha256, read_json, write_json

METRICS = ["mae", "discrimination_score_l1", "overlap_at_N"]
SEEDS = [42, 43, 44]
BASE_MODELS = ["mlp", "transformer", "scgpt", "geneformer"]
FUSION_MODELS = ["concat", "gated", "cross_attention"]
MODELS = BASE_MODELS + FUSION_MODELS
MODEL_LABELS = {
    "mlp": "MLP",
    "transformer": "Transformer",
    "scgpt": "Frozen scGPT",
    "geneformer": "Frozen Geneformer",
    "concat": "Concat",
    "gated": "Gated Fusion",
    "cross_attention": "Cross-Attention",
    "no_perturbation": "无扰动对照",
}
RUN_SPECS = {
    "mlp": ("mmd", "mmd_01_mlp_mean", "mmd-v1"),
    "transformer": ("mmd", "mmd_02_transformer_mean", "mmd-v1"),
    "scgpt": ("scgpt", "03_scgpt", "scgpt-v1"),
    "geneformer": ("geneformer", "04_geneformer", "geneformer-v1"),
    "concat": ("fusion", "05_concat", "fusion-v1"),
    "gated": ("fusion", "06_gated", "fusion-v1"),
    "cross_attention": ("fusion", "07_cross_attention", "fusion-v1"),
}


def control_plan_matches(path: Path, reference: Path) -> bool:
    """Compare actual sampled row IDs rather than compressed-file byte hashes."""
    with np.load(path, allow_pickle=False) as plan, np.load(reference, allow_pickle=False) as base:
        return set(plan.files) == set(base.files) and all(
            np.array_equal(plan[name], base[name]) for name in plan.files
        )


def validated_target_scores(path: Path, evaluation: dict) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if (
        len(frame) != 50
        or frame["perturbation"].isna().any()
        or frame["perturbation"].duplicated().any()
        or not np.isfinite(frame[METRICS].to_numpy()).all()
    ):
        raise ValueError(f"Expected 50 distinct, finite validation target scores: {path}")
    for metric in METRICS:
        if not np.isclose(
            frame[metric].mean(), evaluation["metrics"][metric], atol=1e-12, rtol=1e-12
        ):
            raise ValueError(f"Official aggregate differs from target scores: {path}: {metric}")
    return frame


def validate_evaluation(evaluation: dict) -> None:
    if (
        evaluation["test_used"] is not False
        or evaluation["gene_count"] != 18080
        or evaluation["predicted_cell_count"] != 98927
        or evaluation["profile"] != "vcc"
        or evaluation["cell_eval_version"] != "0.6.6"
        or evaluation["pdex_version"] != "0.1.27"
        or not np.isfinite([evaluation["metrics"][metric] for metric in METRICS]).all()
    ):
        raise ValueError("Expected complete official VCC validation with the pinned evaluator")


def load_reference(root: Path):
    """Validate the immutable reference artifacts against their original receipt.

    The receipt records the source used when the reference was evaluated. Adding
    fusion branches changes current source hashes without changing these saved
    reference predictions/scores, so this reader does not rebind old provenance.
    """
    folder = root / "results/controls/no_perturbation"
    evaluation = read_json(folder / "metrics.json")
    receipt = read_json(folder / "verification.json")
    validate_evaluation(evaluation)
    if (
        receipt["all_checks_passed"] is not True
        or receipt["test_used"] is not False
        or receipt["full_genes_and_cells"] is not True
        or receipt["targets_scored"] != 50
        or receipt["control_plan_matches_original_mlp"] is not True
        or evaluation["signature"]["test_used"] is not False
    ):
        raise ValueError("The saved no-perturbation reference failed its provenance checks")
    for name, digest in receipt["small_artifact_hashes"].items():
        if file_sha256(folder / name) != digest:
            raise ValueError(f"Reference artifact changed: {name}")
    required = {"metrics.json", "per_perturbation.csv", "de_gene_counts.csv"}
    if not required <= set(receipt["small_artifact_hashes"]):
        raise ValueError("Incomplete original reference hash receipt")
    for name, digest in evaluation["signature"]["source_sha256"].items():
        if not name.startswith("src/") and file_sha256(root / name) != digest:
            raise ValueError(f"Reference data source changed: {name}")
    for key, path in [
        ("training_controls", root / "data/processed/baselines-v1/controls.npy"),
        ("validation_matrix", root / "outputs/evaluation-cache/validation_log.npy"),
    ]:
        current = {"bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
        if current != evaluation["signature"][key]:
            raise ValueError(f"Reference matrix signature changed: {path}")
    if not control_plan_matches(
        root / "outputs/auxiliary/no_perturbation_full/control_sampling.npz",
        root / "outputs/01_mlp/seed_42/evaluation/control_sampling.npz",
    ):
        raise ValueError("Reference sampling differs from the original MLP")
    return evaluation, validated_target_scores(folder / "per_perturbation.csv", evaluation)


def validate_old_checkpoints(root: Path) -> list[dict]:
    checks = []
    for group, pattern in [
        ("baselines", "0*/seed_*/training.json"),
        ("mmd", "mmd_*/seed_*/training.json"),
        ("scgpt", "03_scgpt*/seed_*/training.json"),
        ("geneformer", "04_geneformer*/seed_*/training.json"),
    ]:
        for path in sorted((root / "results" / group).glob(pattern)):
            training = read_json(path)
            checkpoint = root / f"outputs/{training['experiment']}/seed_{training['seed']}/best.pt"
            digest = file_sha256(checkpoint)
            if digest != training["checkpoint_sha256"]:
                raise ValueError(f"Existing checkpoint changed: {checkpoint}")
            checks.append(
                {
                    "experiment": training["experiment"],
                    "seed": training["seed"],
                    "checkpoint_sha256": digest,
                    "checkpoint_unchanged": True,
                }
            )
    identities = {(row["experiment"], row["seed"]) for row in checks}
    if len(checks) != 42 or len(identities) != 42:
        raise ValueError(f"Expected exactly 42 distinct original checkpoints; found {len(checks)}")
    return checks


def load_run(root: Path, model: str, seed: int):
    group, experiment, protocol = RUN_SPECS[model]
    small = root / f"results/{group}/{experiment}/seed_{seed}"
    run = root / f"outputs/{experiment}/seed_{seed}"
    evaluation = read_json(small / "metrics.json")
    training = read_json(small / "training.json")
    config = read_json(small / "config.json")
    validate_evaluation(evaluation)
    if (
        config["experiment"] != experiment
        or config["model"] != model
        or config["seed"] != seed
        or config["protocol"] != protocol
        or config["results_group"] != group
        or config["variant"] != "mean"
        or config["loss"] != "unpaired_pseudobulk_mse"
        or config["selection"] != "validation_pseudobulk_mae"
        or config["width"] != 64
        or training["experiment"] != experiment
        or training["seed"] != seed
    ):
        raise ValueError(f"Unexpected mean-loss comparison protocol: {small}")
    if model not in {"mlp", "transformer"} and config["encoder_frozen"] is not True:
        raise ValueError(f"Expected permanently frozen encoders: {small}")
    for name, expected in [
        ("config.json", config),
        ("summary.json", training),
        ("evaluation/metrics.json", evaluation),
    ]:
        if read_json(run / name) != expected:
            raise ValueError(f"Published artifact differs from full run: {run / name}")
    digest = file_sha256(run / "best.pt")
    if digest != training["checkpoint_sha256"] or digest != evaluation["checkpoint_sha256"]:
        raise ValueError(f"Checkpoint provenance mismatch: {run}")
    info = read_json(run / "evaluation/prediction-info.json")
    if (
        info["shape"] != [98927, 18080]
        or info["genes"] != 18080
        or info["control_counts_match_official"] is not True
        or info["control_sampling_seed"] != 2025
    ):
        raise ValueError(f"Incomplete validation prediction: {run}")
    if not control_plan_matches(
        run / "evaluation/control_sampling.npz",
        root / "outputs/01_mlp/seed_42/evaluation/control_sampling.npz",
    ):
        raise ValueError(f"Control sampling differs from the original MLP: {run}")
    targets = validated_target_scores(small / "per_perturbation.csv", evaluation)
    distribution_path = small / "distribution_per_perturbation.csv"
    for published, original in [
        (small / "per_perturbation.csv", run / "evaluation/results.csv"),
        (distribution_path, run / "evaluation/distribution_per_perturbation.csv"),
    ]:
        if file_sha256(published) != file_sha256(original):
            raise ValueError(f"Published target scores differ from full run: {published}")
    distribution = pd.read_csv(distribution_path)
    if (
        len(distribution) != 50
        or distribution["perturbation"].duplicated().any()
        or set(distribution["perturbation"]) != set(targets["perturbation"])
        or not np.isfinite(distribution["energy_mmd_rms"]).all()
        or not np.isclose(
            distribution["energy_mmd_rms"].mean(),
            evaluation["distribution"]["energy_mmd_rms"],
            atol=1e-12,
            rtol=1e-12,
        )
    ):
        raise ValueError(f"Incomplete supplementary distribution scores: {small}")
    targets = targets.merge(distribution, on="perturbation", validate="one_to_one")
    targets["model"], targets["variant"], targets["seed"] = model, "mean", seed
    row = {
        "model": model,
        "experiment": experiment,
        "variant": "mean",
        "seed": seed,
        **{metric: evaluation["metrics"][metric] for metric in METRICS},
        "energy_mmd_rms": evaluation["distribution"]["energy_mmd_rms"],
        "parameters": training["parameters"],
        "trainable_parameters": training["trainable_parameters"],
        "frozen_parameters": training.get("frozen_parameters", 0),
        "best_epoch": training["best_epoch"],
        "epochs_run": training["epochs_run"],
        "training_seconds": training["training_seconds"],
        "evaluation_seconds": evaluation["evaluation_seconds"],
        "peak_gpu_mib": training["peak_allocated_gpu_bytes"] / 1024**2,
        "feature_extraction_seconds_shared": training.get("feature_extraction_seconds", 0),
        "checkpoint_bytes": training["checkpoint_bytes"],
        "checkpoint_sha256": digest,
    }
    if row["parameters"] != row["trainable_parameters"] + row["frozen_parameters"]:
        raise ValueError(f"Parameter accounting mismatch: {small}")
    sources = [
        {"path": str(path.relative_to(root)), "sha256": file_sha256(path)}
        for path in [
            small / "metrics.json",
            small / "training.json",
            small / "config.json",
            small / "per_perturbation.csv",
            distribution_path,
        ]
    ]
    return row, targets, sources


def paired_comparisons(targets: pd.DataFrame, reference_targets: pd.DataFrame) -> list[dict]:
    averaged = targets.groupby(["model", "perturbation"])[METRICS].mean()
    reference = reference_targets.set_index("perturbation")[METRICS]
    pairs = [(model, ref) for model in FUSION_MODELS for ref in [*BASE_MODELS, "no_perturbation"]]
    pairs += [(model, ref) for i, model in enumerate(FUSION_MODELS) for ref in FUSION_MODELS[:i]]
    rng = np.random.default_rng(2025)
    contrasts = []
    for model, ref in pairs:
        candidate = averaged.loc[model].sort_index()
        baseline = reference if ref == "no_perturbation" else averaged.loc[ref]
        if len(candidate) != 50 or set(candidate.index) != set(baseline.index):
            raise ValueError("Paired comparisons require identical validation targets")
        baseline = baseline.reindex(candidate.index)
        resamples = rng.integers(0, len(candidate), (5000, len(candidate)))
        for metric in METRICS:
            delta = (candidate[metric] - baseline[metric]).to_numpy()
            low, high = np.quantile(delta[resamples].mean(1), [0.025, 0.975])
            contrasts.append(
                {
                    "candidate": f"{model}/mean",
                    "reference": "no_perturbation" if ref == "no_perturbation" else f"{ref}/mean",
                    "metric": metric,
                    "difference": float(delta.mean()),
                    "target_bootstrap_95": [float(low), float(high)],
                    "candidate_better_targets": int(
                        (delta < 0).sum() if metric == "mae" else (delta > 0).sum()
                    ),
                    "targets": 50,
                }
            )
    return contrasts


def report_fusion(root: Path):
    reference, reference_targets = load_reference(root)
    rows, target_frames, sources = [], [], []
    for model in MODELS:
        for seed in SEEDS:
            row, frame, run_sources = load_run(root, model, seed)
            if set(frame.perturbation) != set(reference_targets.perturbation):
                raise ValueError("Reference and fitted models have different target sets")
            rows.append(row)
            target_frames.append(frame)
            sources.extend(run_sources)
    old_checkpoints = validate_old_checkpoints(root)
    runs = pd.DataFrame(rows)
    targets = pd.concat(target_frames, ignore_index=True)
    grouped = runs.groupby("model")[METRICS + ["energy_mmd_rms"]].agg(["mean", "std", "count"])
    grouped.columns = ["_".join(column) for column in grouped.columns]
    for model in MODELS:
        if len(runs[runs.model == model]) != 3:
            raise ValueError(f"Incomplete seeds for {model}")
        for name in ["parameters", "trainable_parameters", "frozen_parameters"]:
            if runs.loc[runs.model == model, name].nunique() != 1:
                raise ValueError(f"Parameter count differs across {model} seeds")
    contrasts = paired_comparisons(targets, reference_targets)
    folder = root / "results/fusion"
    folder.mkdir(parents=True, exist_ok=True)
    verification_paths = {
        "artifacts": folder / "artifact-verification.json",
        "reproducibility": folder / "reproducibility-check.json",
        "native_token_identity": folder / "implementation-check.json",
    }
    receipts = {name: read_json(path) for name, path in verification_paths.items() if path.exists()}
    verified = (
        receipts.get("artifacts", {}).get("all_checks_passed") is True
        and receipts.get("reproducibility", {}).get("all_tensors_equal") is True
        and receipts.get("native_token_identity", {}).get("all_checks_passed") is True
        and all(receipt.get("test_used") is False for receipt in receipts.values())
    )
    verification_status = "VERIFIED" if verified else "UNVERIFIED"
    runs.to_csv(folder / "comparison_runs.csv", index=False)
    runs[runs.model.isin(FUSION_MODELS)].to_csv(folder / "runs.csv", index=False)
    grouped.reindex(MODELS).to_csv(folder / "summary.csv")
    reference_frame = reference_targets.copy()
    reference_frame["model"], reference_frame["variant"], reference_frame["seed"] = (
        "no_perturbation",
        "reference",
        np.nan,
    )
    pd.concat([targets, reference_frame], ignore_index=True).to_csv(
        folder / "per_perturbation.csv", index=False
    )
    comparison_rows = [
        {"model": "no_perturbation", "variant": "reference", "seeds": 0, **reference["metrics"]}
    ]
    for model in MODELS:
        row = {"model": model, "variant": "mean", "seeds": 3}
        for metric in METRICS:
            row[metric] = grouped.loc[model, f"{metric}_mean"]
            row[f"{metric}_std"] = grouped.loc[model, f"{metric}_std"]
        comparison_rows.append(row)
    pd.DataFrame(comparison_rows).to_csv(folder / "main_comparison.csv", index=False)
    write_json(
        folder / "paired-comparisons.json",
        {
            "contrasts": contrasts,
            "scope": "three seeds averaged within each of 50 fixed validation targets; 5000 paired target bootstraps; conditional on fitted models, checkpoint selection and fixed control sample; no multiplicity adjustment; not biological replicates or a held-out test",
            "bootstrap_seed": 2025,
            "test_used": False,
        },
    )
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    colors = [
        "#777777",
        "#3973ac",
        "#d68937",
        "#479858",
        "#9467bd",
        "#278c9e",
        "#d46c91",
        "#81684b",
    ]
    for ax, metric, title in zip(axes, METRICS, ["MAE (lower)", "PDS (higher)", "DES (higher)"]):
        means = [reference["metrics"][metric]] + [grouped.loc[m, f"{metric}_mean"] for m in MODELS]
        stds = [0.0] + [grouped.loc[m, f"{metric}_std"] for m in MODELS]
        ax.bar(np.arange(8), means, yerr=stds, capsize=3, color=colors)
        ax.set_xticks(
            np.arange(8),
            ["No perturbation"] + [MODEL_LABELS[m] for m in MODELS],
            rotation=40,
            ha="right",
        )
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(folder / "comparison.png", dpi=180)
    plt.close(fig)
    lines = [
        "# 三种冻结表示融合的验证比较",
        "",
        "## Material Passport",
        "",
        "| Field | Value |",
        "| --- | --- |",
        "| Origin Skill | academic-research-suite/experiment-agent |",
        "| Stage | run |",
        "| Date | 2026-10-15 |",
        "| Version | fusion-v1 |",
        f"| Verification Status | {verification_status} |",
        "",
        "Verification Status 仅指本轮产物与逐 tensor 复现记录，不表示生物学结论或测试集泛化已获验证。",
        "",
        "已完成 Concat、Gated Fusion、Cross-Attention 各三个种子，共 9/9 次正式训练和完整官方验证。",
        "主比较纳入原四种模型的均值损失组，共 21 次运行；无扰动对照为固定抽样的一次完整评价。测试集未使用。",
        "",
        "## 相同均值损失下的官方成绩",
        "",
        "| 方法 | MAE ↓ | PDS ↑ | DES ↑ |",
        "| --- | ---: | ---: | ---: |",
        "| 无扰动对照 | "
        + " | ".join(f"{reference['metrics'][metric]:.6f}" for metric in METRICS)
        + " |",
    ]
    for model in MODELS:
        row = grouped.loc[model]
        lines.append(
            f"| {MODEL_LABELS[model]} | "
            + " | ".join(
                f"{row[f'{metric}_mean']:.6f} ± {row[f'{metric}_std']:.6f}" for metric in METRICS
            )
            + " |"
        )
    lines += [
        "",
        "神经模型为种子 42/43/44 的均值 ± 样本标准差。无扰动对照不复制为三个训练种子。",
        "所有模型按相同验证 pseudobulk MAE 选择 checkpoint，采用均值 MSE 损失。本轮未训练融合 MMD/混合损失组。",
        "",
        "## 配对差值与条件性结论",
        "",
        "下表差值为融合组减参考组；先平均三个种子，再对 50 个相同验证靶点做 5,000 次配对 bootstrap。",
        "MAE 负数更好，PDS/DES 正数更好；括号内是逐靶点重采样的 95% 区间。",
        "",
        "| 融合方法 | 参考 | MAE 差值 [95%] | PDS 差值 [95%] | DES 差值 [95%] |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for model in FUSION_MODELS:
        for ref in ["no_perturbation", *BASE_MODELS]:
            key = ref if ref == "no_perturbation" else f"{ref}/mean"
            matched = [
                c for c in contrasts if c["candidate"] == f"{model}/mean" and c["reference"] == key
            ]
            values = []
            for metric in METRICS:
                contrast = next(c for c in matched if c["metric"] == metric)
                low, high = contrast["target_bootstrap_95"]
                values.append(f"{contrast['difference']:+.6f} [{low:+.6f}, {high:+.6f}]")
            lines.append(
                f"| {MODEL_LABELS[model]} | {MODEL_LABELS[ref]} | " + " | ".join(values) + " |"
            )
    for model in FUSION_MODELS:
        statements = []
        for metric, label in zip(METRICS, ["MAE", "PDS", "DES"]):
            contrast = next(
                c
                for c in contrasts
                if c["candidate"] == f"{model}/mean"
                and c["reference"] == "no_perturbation"
                and c["metric"] == metric
            )
            low, high = contrast["target_bootstrap_95"]
            relation = (
                "低于"
                if contrast["difference"] < 0
                else "高于"
                if contrast["difference"] > 0
                else "等于"
            )
            interval = "区间跨零" if low <= 0 <= high else "区间未跨零"
            statements.append(f"{label} 均值{relation}无扰动对照，{interval}")
        lines += ["", f"{MODEL_LABELS[model]}：{'；'.join(statements)}。"]
    lines += [
        "",
        "这些区间仅对当前拟合模型、checkpoint 选择、固定对照抽样及验证靶点成立；三个随机种子不是生物重复。",
        "同一验证集用于早停和本轮架构比较，存在选择乐观偏差；区间不含模型选择及重训不确定性，且未做多重比较校正。",
        "三种融合间的全部配对差值同样保存在 paired-comparisons.json。单项改善不能扩大为整体能力提升，也不能据此宣称正式测试集泛化优势。",
        "",
        "![Fusion validation comparison](comparison.png)",
        "",
        "## 架构、特征与公平性范围",
        "",
        "三组共同冻结 scGPT whole-human 与 Geneformer V1-10M，只训练目标/上下文投影、融合头与 64 → 18,080 解码器。",
        "Concat 拼接两个分支；Gated Fusion 为每个扰动学习一个标量门控并加权两个分支。门控权重本身不能证明生物机制。",
        "Cross-Attention 在相同 2,048 个训练 non-targeting 对照细胞上分别运行两个编码器，之后按真实基因 ID 聚合 contextualized gene-token prototypes，再做双向注意力。",
        "Geneformer 保留自身表达秩顺序编码，仅对表达过且非 padding 的 token 按基因 ID 聚合；未把两条池化向量当作 token 序列。",
        "两编码器输入范围仍受固定 256 个训练候选基因及各自词表覆盖限制。跨注意力使用固定训练对照的基因原型，不表示细胞特异或扰动后 token 响应。",
        "所有模型仍输出每个扰动的共同表达位移，细胞异质性来自输入对照细胞，没有额外的细胞状态条件响应。",
        "三种融合头的参数量不同；Cross-Attention 还使用更细的固定 token 信息。共同解码宽度和训练协议不能隔离头容量、信息粒度与融合机制的影响。",
        "因此这轮结果检验当前实现的预测表现；尚不能单独归因于预训练互补或注意力机制，需容量/信息匹配消融再检验。",
    ]
    gate_diagnostics = receipts.get("artifacts", {}).get("gate_weights", {}).get("diagnostics", [])
    if gate_diagnostics:
        lines += [
            "",
            "## 门控权重诊断",
            "",
            "| Seed | 验证靶点 scGPT 权重均值 | 范围 | 权重大于0.9的比例 |",
            "| --- | ---: | ---: | ---: |",
        ]
        for gate in gate_diagnostics:
            if gate["split"] == "validation":
                lines.append(
                    f"| {gate['seed']} | {gate['alpha_scgpt_mean']:.4f} | "
                    f"[{gate['alpha_scgpt_min']:.4f}, {gate['alpha_scgpt_max']:.4f}] | "
                    f"{gate['fraction_alpha_above_0_9']:.1%} |"
                )
        lines += [
            "",
            "门控在不同种子间的拟合偏好不同。权重还受投影后分支尺度影响，不能直接当作信息贡献比例，也不能据此认定某一路无用。",
        ]
    lines += [
        "",
        "## 参数与运行成本",
        "",
        "| 方法 | 可训练参数 | 冻结参数 | 总参数 | 训练秒均值 ± SD | 评价秒均值 ± SD | 训练 GPU 峰值 MiB | 共享特征提取秒 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in MODELS:
        subset = runs[runs.model == model]
        first = subset.iloc[0]
        lines.append(
            f"| {MODEL_LABELS[model]} | {int(first.trainable_parameters):,} | {int(first.frozen_parameters):,} | {int(first.parameters):,} | "
            f"{subset.training_seconds.mean():.2f} ± {subset.training_seconds.std():.2f} | "
            f"{subset.evaluation_seconds.mean():.2f} ± {subset.evaluation_seconds.std():.2f} | "
            f"{subset.peak_gpu_mib.max():.1f} | {first.feature_extraction_seconds_shared:.2f} |"
        )
    lines += [
        "",
        "融合组冻结参数指两个来源编码器的参数总和；它们用于离线特征提取，不会同时加载到训练 GPU。共享特征提取耗时是原缓存记录，不按种子重复计费。",
        "Cross-Attention 的共享提取成本包含新增基因 token 原型。GPU 峰值仅为监督训练阶段，不能当作编码器总推理显存；墙钟差异还受缓存及硬件负载影响。",
        "Energy MMD 仅为固定 64+64 细胞抽样的附加指标，逐次值与均值标准差保存在 CSV，不是官方 VCC 主成绩。",
        "",
        "## 产物核验与复现",
        "",
        "21 次比较均核验发布成绩与原运行一致、50 个逐靶点聚合、18,080 个基因与 98,927 个验证细胞、官方 evaluator 版本、checkpoint SHA256 和原 MLP 控制抽样。",
        f"{len(old_checkpoints)} 个旧 checkpoint 均与各自已发布 SHA256 一致。旧无扰动对照按原核验记录的产物哈希校验，保留其评价时的源码签名。",
        "融合缓存及其源编码器特征的核验见 artifact-verification.json；逐 tensor 复现见 reproducibility-check.json；原始 native token 的身份/观察次数独立重建见 implementation-check.json。Material Passport 仅在三份记录均通过时标记 VERIFIED。",
        "Gated Fusion 的 train/validation 逐靶点权重见 gate_weights.csv，alpha_scgpt 表示 scGPT 分支权重，另一分支权重为 1-alpha。范围与均值诊断只描述当前拟合头，不具有因果或生物机制解释。",
        "",
        "```powershell",
        "./.venv/Scripts/python.exe scripts/verify_fusion_artifacts.py",
        "./.venv/Scripts/python.exe -m fusionvc.cli report-fusion",
        "```",
        "",
        "[主比较](main_comparison.csv)、[全部逐次成绩与成本](comparison_runs.csv)、[融合运行](runs.csv)、[均值标准差](summary.csv)、",
        "[逐靶点成绩](per_perturbation.csv)、[配对差值区间](paired-comparisons.json)、[来源哈希](report-provenance.json)、",
        "[独立产物核验](artifact-verification.json)、[逐 tensor 复现](reproducibility-check.json)、[门控权重](gate_weights.csv)、[融合协议](../../docs/fusion-protocol.md)。",
    ]
    (folder / "README.md").write_text("\n".join(lines) + "\n", "utf-8")
    write_json(
        folder / "report-provenance.json",
        {
            "protocol": "fusion-v1; original mean-loss comparison groups; immutable no-perturbation-full-v1 reference",
            "models": 7,
            "fusion_runs": 9,
            "comparison_runs": 21,
            "seeds": SEEDS,
            "variant": "mean",
            "source_artifacts": sources,
            "reference_metrics_sha256": file_sha256(
                root / "results/controls/no_perturbation/metrics.json"
            ),
            "reference_targets_sha256": file_sha256(
                root / "results/controls/no_perturbation/per_perturbation.csv"
            ),
            "reference_verification_sha256": file_sha256(
                root / "results/controls/no_perturbation/verification.json"
            ),
            "original_checkpoints": old_checkpoints,
            "material_passport": {
                "origin_skill": "academic-research-suite/experiment-agent",
                "stage": "run",
                "date": "2026-10-15",
                "version": "fusion-v1",
                "verification_status": verification_status,
                "receipts": {
                    name: {"path": str(path.relative_to(root)), "sha256": file_sha256(path)}
                    for name, path in verification_paths.items()
                    if path.exists()
                },
            },
            "complete": True,
            "test_used": False,
        },
    )
    print(pd.DataFrame(comparison_rows).to_string(index=False), flush=True)
