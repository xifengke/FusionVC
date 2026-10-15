"""Compare four models and three losses against one fixed zero-shift reference."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .control_evaluation import METRICS, load_full_reference
from .io import file_sha256, read_json, write_json

MODELS = ["mlp", "transformer", "scgpt", "geneformer"]
VARIANTS = ["mean", "energy", "hybrid"]
MODEL_LABELS = {
    "mlp": "MLP",
    "transformer": "Transformer",
    "scgpt": "Frozen scGPT",
    "geneformer": "Frozen Geneformer",
}
LOSS_LABELS = {"mean": "Mean", "energy": "MMD", "hybrid": "Mean + MMD"}


def report_controls(root: Path):
    folder = root / "results/controls"
    reference, reference_targets = load_full_reference(root)
    ref = reference["metrics"]
    runs_path = root / "results/geneformer/comparison_runs.csv"
    targets_path = root / "results/geneformer/per_perturbation.csv"
    runs, targets = pd.read_csv(runs_path), pd.read_csv(targets_path)
    expected = {(m, v, s) for m in MODELS for v in VARIANTS for s in [42, 43, 44]}
    actual = set(runs[["model", "variant", "seed"]].itertuples(index=False, name=None))
    if len(runs) != 36 or actual != expected:
        raise ValueError("Expected all 36 completed model/loss/seed comparisons")
    sources = []
    for row in runs.itertuples():
        group = "mmd" if row.model in {"mlp", "transformer"} else row.model
        source = root / f"results/{group}/{row.experiment}/seed_{row.seed}/metrics.json"
        evaluation = read_json(source)
        if (
            evaluation["test_used"]
            or evaluation["gene_count"] != 18080
            or evaluation["predicted_cell_count"] != 98927
        ):
            raise ValueError("Compared evaluation is not full-gene/full-cell validation")
        for metric in METRICS:
            if not np.isclose(
                getattr(row, metric), evaluation["metrics"][metric], atol=1e-12, rtol=1e-12
            ):
                raise ValueError("Comparison CSV differs from official scores")
        sources.append({"path": str(source.relative_to(root)), "sha256": file_sha256(source)})
    grouped = runs.groupby(["model", "variant"])[METRICS].agg(["mean", "std", "count"])
    grouped.columns = ["_".join(column) for column in grouped.columns]
    averaged = targets.groupby(["model", "variant", "perturbation"])[METRICS].mean()
    reference_targets = reference_targets.set_index("perturbation")
    contrasts, rows = [], []
    rng = np.random.default_rng(2025)
    for model in MODELS:
        for variant in VARIANTS:
            target = averaged.loc[(model, variant)]
            if set(target.index) != set(reference_targets.index) or len(target) != 50:
                raise ValueError("Reference and model validation targets differ")
            ref_target = reference_targets.reindex(target.index)
            row = {"model": model, "variant": variant, "seeds": 3}
            for metric in METRICS:
                values = target[metric]
                if not np.isfinite(values).all():
                    raise ValueError("Non-finite per-target comparison")
                mean = float(grouped.loc[(model, variant), f"{metric}_mean"])
                if not np.isclose(values.mean(), mean, atol=1e-12, rtol=1e-12):
                    raise ValueError("Target and run aggregates disagree")
                delta = values - ref_target[metric]
                bootstrap = delta.to_numpy()[rng.integers(0, 50, (5000, 50))].mean(1)
                contrasts.append(
                    {
                        "candidate": f"{model}/{variant}",
                        "reference": "no_perturbation",
                        "metric": metric,
                        "difference": float(delta.mean()),
                        "target_bootstrap_95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
                        "candidate_better_targets": int(
                            (delta < 0).sum() if metric == "mae" else (delta > 0).sum()
                        ),
                        "targets": 50,
                    }
                )
                row[metric] = mean
                row[f"{metric}_std"] = float(grouped.loc[(model, variant), f"{metric}_std"])
                row[f"{metric}_minus_reference"] = float(delta.mean())
            rows.append(row)
    rows.insert(0, {"model": "no_perturbation", "variant": "reference", "seeds": 0, **ref})
    comparison = pd.DataFrame(rows)
    comparison.to_csv(folder / "comparison.csv", index=False)
    main = comparison[(comparison.variant == "mean") | (comparison.model == "no_perturbation")]
    main.to_csv(folder / "main_comparison.csv", index=False)
    write_json(
        folder / "paired-comparisons.json",
        {
            "contrasts": contrasts,
            "scope": "three seeds averaged; 5000 paired bootstraps over 50 fixed validation targets; conditional on fitted models and fixed control sample; not biological replicates",
            "test_used": False,
        },
    )
    reference_frame = reference_targets.reset_index()[["perturbation", *METRICS]].copy()
    reference_frame["model"], reference_frame["variant"], reference_frame["seed"] = (
        "no_perturbation",
        "reference",
        np.nan,
    )
    pd.concat(
        [targets[["perturbation", "model", "variant", "seed", *METRICS]], reference_frame]
    ).to_csv(folder / "per_perturbation.csv", index=False)
    # Verify every previously published checkpoint, including duplicated mean controls.
    checkpoint_checks = []
    for group, pattern in [
        ("baselines", "0*/seed_*/training.json"),
        ("mmd", "mmd_*/seed_*/training.json"),
        ("scgpt", "03_scgpt*/seed_*/training.json"),
        ("geneformer", "04_geneformer*/seed_*/training.json"),
    ]:
        for path in sorted((root / "results" / group).glob(pattern)):
            training = read_json(path)
            run = root / f"outputs/{training['experiment']}/seed_{training['seed']}"
            if file_sha256(run / "best.pt") != training["checkpoint_sha256"]:
                raise ValueError(f"Existing checkpoint changed: {run}")
            checkpoint_checks.append(
                {
                    "experiment": training["experiment"],
                    "seed": training["seed"],
                    "checkpoint_unchanged": True,
                }
            )
    write_json(
        folder / "artifact-verification.json",
        {
            "reference": read_json(folder / "no_perturbation/verification.json"),
            "model_runs": 36,
            "existing_checkpoints": checkpoint_checks,
            "all_checks_passed": True,
            "test_used": False,
        },
    )
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), constrained_layout=True)
    labels = ["No perturbation"] + [MODEL_LABELS[model] for model in MODELS]
    colors = ["#777777", "#3973ac", "#d68937", "#479858", "#9467bd"]
    for ax, metric, title in zip(axes, METRICS, ["MAE (lower)", "PDS (higher)", "DES (higher)"]):
        means = [ref[metric]] + [grouped.loc[(model, "mean"), f"{metric}_mean"] for model in MODELS]
        stds = [0] + [grouped.loc[(model, "mean"), f"{metric}_std"] for model in MODELS]
        ax.bar(np.arange(5), means, yerr=stds, capsize=3, color=colors)
        ax.set_xticks(np.arange(5), labels, rotation=25, ha="right")
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(folder / "comparison.png", dpi=180)
    plt.close(fig)

    def formatted(model, variant):
        row = grouped.loc[(model, variant)]
        return " | ".join(
            f"{row[f'{metric}_mean']:.6f} ± {row[f'{metric}_std']:.6f}" for metric in METRICS
        )

    lines = [
        "# 无扰动对照与四种模型的完整验证比较",
        "",
        "已补算无扰动对照 DES：与四种模型使用相同的 98,927 个完整验证细胞、18,080 个基因、50 个扰动、控制抽样和官方 VCC profile。测试集未使用。",
        "无扰动预测对每个条件都使用训练对照细胞、delta=0，保留细胞异质性及原归一化规则；不人为指定 DES=0。",
        "",
        "## 主比较：四种模型均使用均值损失",
        "",
        "| 方法 | MAE ↓ | PDS ↑ | DES ↑ |",
        "| --- | ---: | ---: | ---: |",
        "| 无扰动对照 | " + " | ".join(f"{ref[metric]:.6f}" for metric in METRICS) + " |",
    ]
    for model in MODELS:
        lines.append(f"| {MODEL_LABELS[model]} | {formatted(model, 'mean')} |")
    lines += [
        "",
        "神经模型为三个训练种子的均值 ± 标准差；无扰动对照为同一固定控制抽样的一次完整评价，不人为复制成三个种子。",
        "所有模型仍按原验证 MAE 选择 checkpoint，本次补算未重新训练或按 DES 挑选模型。",
        "",
        "## 与无扰动对照的差值",
        "",
        "| 方法 | MAE 差值（负数更好） | PDS 差值（正数更好） | DES 差值（正数更好） |",
        "| --- | ---: | ---: | ---: |",
    ]
    for model in MODELS:
        row = comparison[(comparison.model == model) & (comparison.variant == "mean")].iloc[0]
        lines.append(
            f"| {MODEL_LABELS[model]} | "
            + " | ".join(f"{row[f'{metric}_minus_reference']:+.6f}" for metric in METRICS)
            + " |"
        )
    lines += ["", "MAE：四个主组均高于无扰动对照；全部三种损失的组均值也高于该对照。"]
    for model in MODELS:
        contrast = next(
            item
            for item in contrasts
            if item["candidate"] == f"{model}/mean" and item["metric"] == "discrimination_score_l1"
        )
        low, high = contrast["target_bootstrap_95"]
        lines.append(
            f"{MODEL_LABELS[model]} 主组 PDS 差值的 50 靶点配对 95% 区间 [{low:.6f}, {high:.6f}]。"
        )
    better_des = [
        MODEL_LABELS[model]
        for model in MODELS
        if grouped.loc[(model, "mean"), "overlap_at_N_mean"] > ref["overlap_at_N"]
    ]
    lines.append(f"DES：{'、'.join(better_des)} 主组高于无扰动对照。")
    lines += [
        "这表示模型找回了一部分真实显著变化基因，不能据此认定整体预测能力已经优于简单对照。",
        "三项指标应分别报告，不能把 MAE 不如对照扩大成三项都不如对照。配对区间仅对当前拟合模型、固定控制抽样和验证靶点成立。",
        "",
        "![Comparison with no perturbation](comparison.png)",
        "",
        "## 三种训练损失的全部比较",
        "",
        "| 方法 | 损失 | MAE ↓ | PDS ↑ | DES ↑ |",
        "| --- | --- | ---: | ---: | ---: |",
        "| 无扰动对照 | — | " + " | ".join(f"{ref[metric]:.6f}" for metric in METRICS) + " |",
    ]
    for model in MODELS:
        for variant in VARIANTS:
            lines.append(
                f"| {MODEL_LABELS[model]} | {LOSS_LABELS[variant]} | {formatted(model, variant)} |"
            )
    verification = read_json(folder / "no_perturbation/verification.json")
    lines += [
        "",
        "## 一致性与复现",
        "",
        f"无扰动预测中被 FDR < 0.05 检出的显著基因总数（基因×靶点）为 {verification['predicted_significant_genes_total']}；具体情况见逐靶点诊断。",
        f"旧 MAE/PDS 的逐靶点最大绝对差异分别为 {verification['legacy_per_target_max_absolute_difference']['mae']:.3g} / {verification['legacy_per_target_max_absolute_difference']['discrimination_score_l1']:.3g}。控制抽样与原 MLP 完全一致。",
        f"{len(checkpoint_checks)} 个旧 checkpoint 经核验未变；正式模型成绩未改动，测试集未使用。",
        "DES 使用同一全细胞 Mann–Whitney/Wilcoxon 实现、tie/continuity correction、全基因×全靶点 BH 校正，显著性阈值 0.05，按绝对 log2 fold change 排序。",
        "",
        "```powershell",
        "./.venv/Scripts/python.exe scripts/evaluate_no_perturbation.py",
        "```",
        "",
        "[主比较数据](main_comparison.csv)、[全部损失比较](comparison.csv)、[逐靶点结果](per_perturbation.csv)、",
        "[配对差值与区间](paired-comparisons.json)、[完整产物核验](artifact-verification.json)、",
        "[无扰动官方成绩](no_perturbation/metrics.json)、[逐靶点 DE 数量](no_perturbation/de_gene_counts.csv)、",
        "[无扰动一致性检查](no_perturbation/verification.json)。",
    ]
    (folder / "README.md").write_text("\n".join(lines) + "\n", "utf-8")
    write_json(
        folder / "report-provenance.json",
        {
            "reference_metrics_sha256": file_sha256(folder / "no_perturbation/metrics.json"),
            "comparison_runs_source_sha256": file_sha256(runs_path),
            "comparison_targets_source_sha256": file_sha256(targets_path),
            "model_metric_sources": sources,
            "models": 4,
            "loss_variants": 3,
            "model_runs": 36,
            "complete": True,
            "test_used": False,
        },
    )
    print(main.to_string(index=False), flush=True)
