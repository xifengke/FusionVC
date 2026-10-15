"""Four-encoder validation comparison; existing reports remain independent."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .io import file_sha256, read_json, write_json

METRICS = ["mae", "discrimination_score_l1", "overlap_at_N", "energy_mmd_rms"]
MODELS = ["mlp", "transformer", "scgpt", "geneformer"]
VARIANTS = ["mean", "energy", "hybrid"]
LABELS = {"mean": "Mean", "energy": "MMD", "hybrid": "Mean + MMD"}
MODEL_LABELS = {
    "mlp": "MLP",
    "transformer": "Transformer",
    "scgpt": "Frozen scGPT",
    "geneformer": "Frozen Geneformer",
}


def report_geneformer(root: Path):
    folder = root / "results/geneformer"
    rows, target_frames, sources = [], [], []
    for group, pattern in [
        ("mmd", "mmd_*/seed_*/metrics.json"),
        ("scgpt", "03_scgpt*/seed_*/metrics.json"),
        ("geneformer", "04_geneformer*/seed_*/metrics.json"),
    ]:
        for path in sorted((root / "results" / group).glob(pattern)):
            evaluation = read_json(path)
            training = read_json(path.parent / "training.json")
            config = read_json(path.parent / "config.json")
            if evaluation["test_used"] or evaluation["gene_count"] != 18080:
                raise ValueError("Comparisons require full-gene validation only")
            if group in {"scgpt", "geneformer"} and (
                config["protocol"] != f"{group}-v1" or not config["encoder_frozen"]
            ):
                raise ValueError("Unexpected frozen-model protocol")
            run = root / f"outputs/{training['experiment']}/seed_{training['seed']}"
            digest = file_sha256(run / "best.pt")
            if digest != training["checkpoint_sha256"] or digest != evaluation["checkpoint_sha256"]:
                raise ValueError(f"Checkpoint provenance mismatch: {run}")
            rows.append(
                {
                    "experiment": training["experiment"],
                    "model": config["model"],
                    "variant": config["variant"],
                    "seed": training["seed"],
                    **evaluation["metrics"],
                    "energy_mmd_rms": evaluation["distribution"]["energy_mmd_rms"],
                    "parameters": training["parameters"],
                    "trainable_parameters": training["trainable_parameters"],
                    "frozen_parameters": training.get("frozen_parameters", 0),
                    "best_epoch": training["best_epoch"],
                    "epochs_run": training["epochs_run"],
                    "training_seconds": training["training_seconds"],
                    "evaluation_seconds": evaluation["evaluation_seconds"],
                    "peak_gpu_mib": training["peak_allocated_gpu_bytes"] / 1024**2,
                    "feature_extraction_seconds_shared": training.get(
                        "feature_extraction_seconds", 0
                    ),
                    "checkpoint_sha256": digest,
                }
            )
            frame = pd.read_csv(path.parent / "per_perturbation.csv").merge(
                pd.read_csv(path.parent / "distribution_per_perturbation.csv"),
                on="perturbation",
                validate="one_to_one",
            )
            if len(frame) != 50 or not np.isfinite(frame[METRICS]).all().all():
                raise ValueError(f"Incomplete target metrics: {path}")
            frame["model"], frame["variant"], frame["seed"] = (
                config["model"],
                config["variant"],
                training["seed"],
            )
            target_frames.append(frame)
            sources.append({"path": str(path.relative_to(root)), "sha256": file_sha256(path)})
    expected = {
        (model, variant, seed) for model in MODELS for variant in VARIANTS for seed in [42, 43, 44]
    }
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No completed comparisons")
    actual = set(df[["model", "variant", "seed"]].itertuples(index=False, name=None))
    if len(df) != 36 or actual != expected:
        raise RuntimeError(
            f"Comparison incomplete: {len(df)}/36; missing {sorted(expected - actual)}"
        )
    df = df.sort_values(["model", "variant", "seed"])
    df.to_csv(folder / "comparison_runs.csv", index=False)
    df[df.model == "geneformer"].to_csv(folder / "runs.csv", index=False)
    grouped = df.groupby(["model", "variant"])[METRICS].agg(["mean", "std", "count"])
    grouped.columns = ["_".join(column) for column in grouped.columns]
    grouped.to_csv(folder / "summary.csv")
    targets = pd.concat(target_frames, ignore_index=True)
    targets.to_csv(folder / "per_perturbation.csv", index=False)
    averaged = targets.groupby(["model", "variant", "perturbation"])[METRICS].mean()
    contrasts, rng = [], np.random.default_rng(2025)
    pairs = [("mean", model, "mean") for model in MODELS if model != "geneformer"]
    pairs += [(variant, "geneformer", "mean") for variant in ["energy", "hybrid"]]
    for variant, reference_model, reference_variant in pairs:
        candidate = averaged.loc[("geneformer", variant)]
        reference = averaged.loc[(reference_model, reference_variant)]
        if set(candidate.index) != set(reference.index):
            raise ValueError("Target sets differ across comparisons")
        reference = reference.reindex(candidate.index)
        for metric in METRICS:
            delta = candidate[metric] - reference[metric]
            bootstrap = delta.to_numpy()[rng.integers(0, len(delta), (5000, len(delta)))].mean(1)
            contrasts.append(
                {
                    "candidate": f"geneformer/{variant}",
                    "reference": f"{reference_model}/{reference_variant}",
                    "metric": metric,
                    "difference": float(delta.mean()),
                    "target_bootstrap_95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
                    "targets": len(delta),
                    "candidate_better_targets": int(
                        (delta < 0).sum()
                        if metric in {"mae", "energy_mmd_rms"}
                        else (delta > 0).sum()
                    ),
                }
            )
    write_json(
        folder / "paired-comparisons.json",
        {
            "contrasts": contrasts,
            "scope": "seed-averaged paired validation targets; conditional on fitted models; 5000 target bootstraps",
            "test_used": False,
        },
    )
    aux = pd.read_csv(root / "results/baselines/auxiliary.csv").set_index("reference")
    no_perturbation = float(aux.loc["no_perturbation", "mae"])
    reference = aux.loc["no_perturbation"]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.7), constrained_layout=True)
    for ax, metric, title in zip(
        axes, METRICS, ["MAE (lower)", "PDS (higher)", "DES (higher)", "Energy MMD (lower)"]
    ):
        for offset, variant in enumerate(VARIANTS):
            ax.bar(
                np.arange(len(MODELS)) + (offset - 1) * 0.25,
                [grouped.loc[(model, variant), f"{metric}_mean"] for model in MODELS],
                yerr=[grouped.loc[(model, variant), f"{metric}_std"] for model in MODELS],
                width=0.23,
                capsize=3,
                label=LABELS[variant],
            )
        if metric in reference.index and pd.notna(reference[metric]):
            ax.axhline(
                float(reference[metric]),
                color="black",
                linestyle="--",
                linewidth=1,
                label="No perturbation",
            )
        ax.set_xticks(
            np.arange(len(MODELS)),
            [MODEL_LABELS[model] for model in MODELS],
            rotation=25,
            ha="right",
        )
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=7)
    fig.savefig(folder / "comparison.png", dpi=180)
    plt.close(fig)
    features = read_json(folder / "features.json")
    parity = read_json(folder / "implementation-parity.json")

    def values(row, metrics):
        return " | ".join(
            f"{row[f'{metric}_mean']:.6f} ± {row[f'{metric}_std']:.6f}" for metric in metrics
        )

    lines = [
        "# Frozen Geneformer 验证比较",
        "",
        "已完成 Geneformer 三种损失 × 三个种子，共 9/9 次正式训练和完整官方验证。",
        "原 MLP/Transformer/scGPT 的 27 次损失比较保留，总计 36 次。测试集未使用。",
        "",
        "## 主比较：相同均值损失",
        "",
        "| Model | Seeds | MAE ↓ | PDS ↑ | DES ↑ |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    if pd.notna(reference.get("overlap_at_N")):
        lines.append(
            "| No perturbation | fixed sample | "
            + " | ".join(f"{float(reference[metric]):.6f}" for metric in METRICS[:3])
            + " |"
        )
    for model in MODELS:
        lines.append(
            f"| {MODEL_LABELS[model]} | 3 | {values(grouped.loc[(model, 'mean')], METRICS[:3])} |"
        )
    mean_mae = grouped.loc[("geneformer", "mean"), "mae_mean"]
    reference_des = reference.get("overlap_at_N")
    reference_note = (
        f"无扰动对照 DES = **{reference_des:.6f}**，已使用相同完整细胞、DE 和 FDR 流程补算。"
        if pd.notna(reference_des)
        else "无扰动对照的 DES 尚未评价。"
    )
    lines += ["", f"无扰动辅助对照 MAE = **{no_perturbation:.6f}**。{reference_note}"]
    for model in MODELS[:-1]:
        reference_mae = grouped.loc[(model, "mean"), "mae_mean"]
        percent = (mean_mae - reference_mae) / reference_mae * 100
        direction = "升高" if percent >= 0 else "降低"
        contrast = next(
            item
            for item in contrasts
            if item["reference"] == f"{model}/mean" and item["metric"] == "mae"
        )
        low, high = contrast["target_bootstrap_95"]
        lines.append(
            f"Geneformer 主组 MAE 相比 {MODEL_LABELS[model]} {direction} {abs(percent):.2f}%；"
            f"50 靶点配对差值 95% 区间 [{low:.6f}, {high:.6f}]。"
        )
    relation = "高于" if mean_mae > no_perturbation else "低于"
    lines += [
        f"主组 MAE {relation}无扰动对照。区间先平均种子，再重采样验证靶点，仅对当前拟合模型成立。",
        "PDS/DES 需同时比较，单项改善不代表整体能力提升。",
        "",
        "## 损失对照",
        "",
        "| Model | Loss | Seeds | MAE ↓ | PDS ↑ | DES ↑ | Energy MMD ↓ |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in MODELS:
        for variant in VARIANTS:
            lines.append(
                f"| {MODEL_LABELS[model]} | {LABELS[variant]} | 3 | {values(grouped.loc[(model, variant)], METRICS)} |"
            )
    lines += [
        "",
        "表中为三个种子的均值 ± 标准差；控制抽样固定。Energy MMD 是固定 64+64 细胞的附加抽样指标。",
        "所有组按验证 pseudobulk MAE 选择 checkpoint，混合系数 0.1 预先固定。",
    ]
    for variant in ["energy", "hybrid"]:
        row = grouped.loc[("geneformer", variant)]
        percent = (row["mae_mean"] - mean_mae) / mean_mae * 100
        direction = "升高" if percent >= 0 else "降低"
        changes = ", ".join(
            f"{label} {'升高' if row[f'{metric}_mean'] > grouped.loc[('geneformer', 'mean'), f'{metric}_mean'] else '降低'}"
            for metric, label in zip(METRICS[1:], ["PDS", "DES", "Energy MMD"])
        )
        lines.append(
            f"Geneformer {LABELS[variant]} 相比其均值组 MAE {direction} {abs(percent):.2f}%；{changes}。"
        )
    if all(
        grouped.loc[("geneformer", variant), "mae_mean"] > no_perturbation for variant in VARIANTS
    ):
        lines.append(
            "Geneformer 三组 MAE 均未超过无扰动对照，本轮没有显示对平均表达误差的稳定优势。"
        )
    lines += [
        "",
        "![Four-encoder comparison](comparison.png)",
        "",
        "## 实现、来源与成本",
        "",
        f"使用固定 V1-10M，冻结 {features['frozen_parameters']:,} 个参数，{features['loaded_encoder_tensors']} 个编码器张量严格加载。",
        f"真实 BERT 编码器对照最大绝对差异 {parity['full_encoder_max_absolute_difference']:.8g}；官方排序和非 pad pooling 检查通过。",
        f"提取 {features['control_cells']:,} 个训练对照细胞耗时 {features['elapsed_seconds']:.2f} 秒，GPU 峰值 {features['peak_allocated_gpu_bytes'] / 1024**2:.1f} MiB，九次训练共享。",
        f"候选输入覆盖 {features['recognized_input_genes']}/{features['input_gene_count']}；缺失输入 {', '.join(features['missing_input_genes'])}。",
        f"输出保留 {features['output_genes']:,} 个基因，预训练词表覆盖 {features['covered_output_genes']:,} 个；",
        f"训练靶点覆盖 {features['target_coverage']['train']['covered']}/{features['target_coverage']['train']['total']}，验证靶点覆盖 {features['target_coverage']['validation']['covered']}/{features['target_coverage']['validation']['total']}。",
        f"按原始 var/gene_id 映射，非零 token 长度 {features['sequence_length_min']}–{features['sequence_length_max']}，均值 {features['sequence_length_mean']:.2f}。",
        "输入只由训练对照和预训练资源构建。最后一层非 pad 均值作为 context，靶点使用冻结 word embedding、训练描述与覆盖标记。",
        "投影及统一 64 → 18,080 解码器可训练；编码器完全冻结。输出保持共同条件位移。",
        "输入覆盖、秩编码、预训练语料和预测头维度存在差异，不能完全隔离纯预训练效果。受限输入不能等同于完整转录组标准零样本 Geneformer。",
        "",
        "[逐次成本](runs.csv)、[全部比较](comparison_runs.csv)、[逐靶点结果](per_perturbation.csv)、",
        "[配对差值区间](paired-comparisons.json)、[实现检查](implementation-parity.json)、",
        "[逐 tensor 复现](reproducibility-check.json)、[产物核验](artifact-verification.json)、",
        "[特征来源](features.json)、[实验协议](../../docs/geneformer-protocol.md)。",
        "[无扰动与四模型的完整三指标比较](../controls/README.md)。",
    ]
    (folder / "README.md").write_text("\n".join(lines) + "\n", "utf-8")
    write_json(
        folder / "report-provenance.json",
        {
            "protocol": "geneformer-v1; baseline-v1 outputs/selection; mmd-v1 loss definitions",
            "geneformer_runs": 9,
            "comparison_runs": 36,
            "complete": True,
            "test_used": False,
            "source_metrics": sources,
            "main_comparison": "same mean loss; MLP/Transformer/scGPT/Geneformer",
        },
    )
    print(grouped.to_string(), flush=True)
