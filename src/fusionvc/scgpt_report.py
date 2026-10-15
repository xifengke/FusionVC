"""Compare frozen scGPT with existing encoders while keeping old results intact."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .io import file_sha256, read_json, write_json

METRICS = ["mae", "discrimination_score_l1", "overlap_at_N", "energy_mmd_rms"]
MODELS = ["mlp", "transformer", "scgpt"]
VARIANTS = ["mean", "energy", "hybrid"]
LABELS = {"mean": "Mean", "energy": "MMD", "hybrid": "Mean + MMD"}
MODEL_LABELS = {"mlp": "MLP", "transformer": "Transformer", "scgpt": "Frozen scGPT"}


def report_scgpt(root: Path) -> None:
    folder = root / "results/scgpt"
    rows, target_frames, sources = [], [], []
    for group, pattern in [
        ("mmd", "mmd_*/seed_*/metrics.json"),
        ("scgpt", "03_scgpt*/seed_*/metrics.json"),
    ]:
        for path in sorted((root / "results" / group).glob(pattern)):
            evaluation = read_json(path)
            training = read_json(path.parent / "training.json")
            config = read_json(path.parent / "config.json")
            if evaluation["test_used"] or evaluation["gene_count"] != 18080:
                raise ValueError("All compared runs must use full-gene validation only")
            if group == "scgpt" and (
                config["protocol"] != "scgpt-v1" or not config["encoder_frozen"]
            ):
                raise ValueError("Unexpected scGPT protocol or unfrozen encoder")
            run = root / "outputs" / training["experiment"] / f"seed_{training['seed']}"
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
            frame = pd.read_csv(path.parent / "per_perturbation.csv")
            distribution = pd.read_csv(path.parent / "distribution_per_perturbation.csv")
            frame = frame.merge(distribution, on="perturbation", validate="one_to_one")
            if len(frame) != 50 or frame[METRICS].isna().any().any():
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
    if len(df) != 27 or actual != expected:
        raise RuntimeError(
            f"Comparison incomplete: {len(df)}/27 runs; missing {sorted(expected - actual)}"
        )
    df = df.sort_values(["model", "variant", "seed"])
    df.to_csv(folder / "comparison_runs.csv", index=False)
    df[df.model == "scgpt"].to_csv(folder / "runs.csv", index=False)
    grouped = df.groupby(["model", "variant"])[METRICS].agg(["mean", "std", "count"])
    grouped.columns = ["_".join(column) for column in grouped.columns]
    grouped.to_csv(folder / "summary.csv")
    targets = pd.concat(target_frames, ignore_index=True)
    targets.to_csv(folder / "per_perturbation.csv", index=False)
    averaged = targets.groupby(["model", "variant", "perturbation"])[METRICS].mean()
    contrasts = []
    pairs = [("scgpt", "mean", model, "mean") for model in ["mlp", "transformer"]]
    pairs += [("scgpt", variant, "scgpt", "mean") for variant in ["energy", "hybrid"]]
    rng = np.random.default_rng(2025)
    for model, variant, reference_model, reference_variant in pairs:
        candidate = averaged.loc[(model, variant)]
        reference = averaged.loc[(reference_model, reference_variant)]
        if set(candidate.index) != set(reference.index):
            raise ValueError("Target sets differ across comparisons")
        reference = reference.reindex(candidate.index)
        for metric in METRICS:
            delta = candidate[metric] - reference[metric]
            bootstrap = delta.to_numpy()[rng.integers(0, len(delta), (5000, len(delta)))].mean(1)
            contrasts.append(
                {
                    "candidate": f"{model}/{variant}",
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
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.5), constrained_layout=True)
    for ax, metric, title in zip(
        axes, METRICS, ["MAE (lower)", "PDS (higher)", "DES (higher)", "Energy MMD (lower)"]
    ):
        for offset, variant in enumerate(VARIANTS):
            means = [grouped.loc[(model, variant), f"{metric}_mean"] for model in MODELS]
            errors = [grouped.loc[(model, variant), f"{metric}_std"] for model in MODELS]
            ax.bar(
                np.arange(3) + (offset - 1) * 0.25,
                means,
                yerr=errors,
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
        ax.set_xticks(np.arange(3), ["MLP", "Transformer", "Frozen scGPT"], rotation=15)
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=7)
    fig.savefig(folder / "comparison.png", dpi=180)
    plt.close(fig)
    features = read_json(folder / "features.json")
    parity = read_json(folder / "implementation-parity.json")
    lines = [
        "# Frozen scGPT 验证比较",
        "",
        "已完成 scGPT 三种损失 × 三个种子，共 9/9 次正式训练和完整官方验证。",
        "原 MLP/Transformer 共 18 次损失对照作为固定参考，结果保留。测试集未使用。",
        "",
        "## 主比较：相同均值损失",
        "",
        "| Model | Seeds | MAE ↓ | PDS ↑ | DES ↑ |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    def values(row, metrics):
        return " | ".join(
            f"{row[f'{metric}_mean']:.6f} ± {row[f'{metric}_std']:.6f}" for metric in metrics
        )

    if pd.notna(reference.get("overlap_at_N")):
        lines.append(
            "| No perturbation | fixed sample | "
            + " | ".join(f"{float(reference[metric]):.6f}" for metric in METRICS[:3])
            + " |"
        )
    for model in MODELS:
        row = grouped.loc[(model, "mean")]
        lines.append(f"| {MODEL_LABELS[model]} | 3 | {values(row, METRICS[:3])} |")
    scgpt_mean = grouped.loc[("scgpt", "mean"), "mae_mean"]
    reference_des = reference.get("overlap_at_N")
    reference_note = (
        f"无扰动对照 DES = **{reference_des:.6f}**，已使用相同完整细胞、DE 和 FDR 流程补算。"
        if pd.notna(reference_des)
        else "无扰动对照的 DES 尚未评价。"
    )
    lines += ["", f"无扰动对照 MAE = **{no_perturbation:.6f}**。{reference_note}"]
    for model in ["mlp", "transformer"]:
        reference = grouped.loc[(model, "mean"), "mae_mean"]
        percent = (scgpt_mean - reference) / reference * 100
        direction = "升高" if percent >= 0 else "降低"
        lines.append(
            f"Frozen scGPT 主组 MAE 相比 {MODEL_LABELS[model]} {direction} {abs(percent):.2f}%。"
        )
    relation = "高于" if scgpt_mean > no_perturbation else "低于"
    lines += [
        f"其 MAE {relation}无扰动对照；PDS/DES 请同时查看，单项改善不代表整体能力提升。",
        "相对 MLP，主组 MAE 的 50 靶点配对 bootstrap 95% 差值区间包含 0，改善未显示稳定优势；",
        "DES 的配对差值区间高于 0，但其 DES 仍低于随机 Transformer。区间仅对本轮拟合模型与验证靶点成立。",
        "",
        "## 损失对照",
        "",
        "| Model | Loss | Seeds | MAE ↓ | PDS ↑ | DES ↑ | Energy MMD ↓ |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in MODELS:
        for variant in VARIANTS:
            row = grouped.loc[(model, variant)]
            lines.append(
                f"| {MODEL_LABELS[model]} | {LABELS[variant]} | 3 | {values(row, METRICS)} |"
            )
    lines += [
        "",
        "表中为三个种子的均值 ± 标准差；控制抽样固定。Energy MMD 是固定 64+64 细胞的附加抽样指标。",
        "所有组继续按验证 pseudobulk MAE 挑选 checkpoint，混合系数 0.1 预先固定。",
        f"scGPT 纯 MMD 相对其均值组的 MAE 降低 "
        f"{(scgpt_mean - grouped.loc[('scgpt', 'energy'), 'mae_mean']) / scgpt_mean:.2%}，"
        "DES 和 PDS 均下降，附加 Energy MMD 也没有同步改善。",
        "混合组 MAE 同样小幅下降、DES 略降；三组仍未超过无扰动 MAE 对照，本轮没有观察到整体能力的明显提升。",
        "",
        "![Three-encoder comparison](comparison.png)",
        "",
        "## 实现与成本",
        "",
        f"使用已下载 whole-human 权重，冻结参数 {features['frozen_parameters']:,}；"
        f"{features['loaded_encoder_tensors']} 个编码器张量严格加载。",
        f"完整编码器相对官方固定版本的最大绝对差异 {parity['full_encoder_max_absolute_difference']:.8g}，"
        "表达分箱在相同随机数下完全一致；未静默随机初始化缺失层。",
        f"一次提取 {features['control_cells']:,} 个训练对照细胞，耗时 {features['elapsed_seconds']:.2f} 秒，"
        f"GPU 峰值 {features['peak_allocated_gpu_bytes'] / 1024**2:.1f} MiB；该缓存被九次训练共享。",
        f"固定 256 输入基因中 {features['recognized_input_genes']} 个被词表识别；"
        f"全部 {features['output_genes']:,} 个输出保留，词表覆盖 {features['covered_output_genes']:,} 个。",
        "全部 150 个训练靶点和 50 个验证靶点均被词表覆盖；缺失靶点回退策略已实现，本轮没有触发。",
        "对照特征只来自训练 non-targeting 细胞；靶点使用冻结基因 embedding、训练对照描述与覆盖标记。",
        "监督阶段训练投影和统一 64 → 18,080 解码器；平均对照表示保持共同条件位移模型，没有新增状态依赖响应。",
        "这是冻结 scGPT 特征迁移比较；输入可识别基因数和预测头略有差异，不能完全隔离纯预训练效应。",
        "",
        "[每次训练与评价成本](runs.csv)、[全部比较运行](comparison_runs.csv)、[按靶点结果](per_perturbation.csv)、",
        "[配对差值与区间](paired-comparisons.json)、[实现一致性](implementation-parity.json)、",
        "[逐 tensor 复现](reproducibility-check.json)、[完整产物校验](artifact-verification.json)、",
        "[特征来源及覆盖率](features.json)、[实验协议](../../docs/scgpt-protocol.md)。",
        "[无扰动与四模型的完整三指标比较](../controls/README.md)。",
    ]
    (folder / "README.md").write_text("\n".join(lines) + "\n", "utf-8")
    write_json(
        folder / "report-provenance.json",
        {
            "protocol": "scgpt-v1; same outputs/selection/loss definitions as baseline-v1 and mmd-v1",
            "scgpt_runs": 9,
            "comparison_runs": 27,
            "complete": True,
            "test_used": False,
            "source_metrics": sources,
            "main_comparison": "mean loss; scGPT/MLP/Transformer",
        },
    )
    print(grouped.to_string(), flush=True)
