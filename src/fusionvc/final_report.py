"""Audit and report the frozen, complete held-out test comparison."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .final_evaluation import validate_selection
from .fusion_report import control_plan_matches
from .io import file_sha256, read_json, write_json

METRICS = ["mae", "discrimination_score_l1", "overlap_at_N", "energy_mmd_rms"]
MODEL_ORDER = [
    "mlp",
    "transformer",
    "scgpt",
    "geneformer",
    "concat",
    "gated",
    "cross_attention",
    "distillation",
]
LABELS = {
    "mlp": "MLP",
    "transformer": "Transformer",
    "scgpt": "Frozen scGPT",
    "geneformer": "Frozen Geneformer",
    "concat": "Concat",
    "gated": "Gated Fusion",
    "cross_attention": "Cross-Attention",
    "distillation": "Distilled MLP",
    "no_perturbation": "No perturbation",
}


def read_final_runs(root: Path, selection_path: Path):
    from .final_evaluation import validate_test_cache_sources

    selection, digest = validate_selection(root, selection_path)
    cache_lineage = validate_test_cache_sources(root, digest)
    ready_path = root / "outputs/final-test-cache/real-ready.json"
    ready = read_json(ready_path)
    reference_label = next(r["label"] for r in selection["runs"] if r["kind"] == "no_perturbation")
    reference_plan = root / f"outputs/final-test/{reference_label}/control_sampling.npz"
    rows, target_frames, sources, checks = [], [], [], []
    expected_targets = set(ready["perturbations"])
    for selected in selection["runs"]:
        label = selected["label"]
        small = root / f"results/final-test/{label}"
        output = root / f"outputs/final-test/{label}"
        metrics = read_json(small / "metrics.json")
        info = read_json(small / "prediction-info.json")
        signature = metrics["signature"]
        if (
            metrics["test_used"] is not True
            or metrics["split"] != "official test"
            or metrics["gene_count"] != 18080
            or metrics["perturbation_count"] != 100
            or metrics["predicted_cell_count"] != ready["shape"][0]
            or metrics["control_cell_count"] != ready["control_cell_count"]
            or metrics["profile"] != "vcc"
            or metrics["cell_eval_version"] != "0.6.6"
            or metrics["pdex_version"] != "0.1.27"
            or signature["selection_sha256"] != digest
            or signature["run"] != selected
            or signature["real_cache_manifest_sha256"] != file_sha256(ready_path)
            or info["shape"] != ready["shape"]
        ):
            raise ValueError(f"Incomplete or rebound final-test provenance: {label}")
        if (
            selected["kind"] == "checkpoint"
            and metrics["checkpoint_sha256"] != selected["checkpoint_sha256"]
        ):
            raise ValueError(f"Final-test checkpoint differs from pre-test selection: {label}")
        for name in [
            "metrics.json",
            "per_perturbation.csv",
            "distribution_per_perturbation.csv",
            "prediction-info.json",
            "run-source.json",
            "control_sampling.npz",
        ]:
            original = output / ("results.csv" if name == "per_perturbation.csv" else name)
            if file_sha256(small / name) != file_sha256(original):
                raise ValueError(f"Published final artifact differs from full run: {label}/{name}")
            sources.append(
                {
                    "path": (small / name).relative_to(root).as_posix(),
                    "sha256": file_sha256(small / name),
                }
            )
        frame = pd.read_csv(small / "per_perturbation.csv")
        distribution = pd.read_csv(small / "distribution_per_perturbation.csv")
        for table in [frame, distribution]:
            if (
                len(table) != 100
                or table.perturbation.duplicated().any()
                or set(table.perturbation) != expected_targets
            ):
                raise ValueError(
                    f"Final target set must exactly match the 100 official targets: {label}"
                )
        frame = frame.merge(distribution, on="perturbation", validate="one_to_one")
        values = {**metrics["metrics"], "energy_mmd_rms": metrics["distribution"]["energy_mmd_rms"]}
        if not np.isfinite(frame[METRICS].to_numpy()).all() or not all(
            np.isclose(frame[m].mean(), values[m], rtol=1e-12, atol=1e-12) for m in METRICS
        ):
            raise ValueError(f"Final aggregate and target scores differ: {label}")
        if not control_plan_matches(output / "control_sampling.npz", reference_plan):
            raise ValueError(f"Final models must use the same sampled training controls: {label}")
        if (output / "predictions.npy").exists():
            raise ValueError(f"Temporary final prediction was not cleaned: {label}")
        row = {
            "label": label,
            "model": selected["model"],
            "variant": selected.get("variant"),
            "seed": selected.get("seed"),
            "role": selected["role"],
            **values,
            "evaluation_seconds": metrics["evaluation_seconds"],
            "checkpoint_sha256": selected.get("checkpoint_sha256"),
        }
        if selected["kind"] == "checkpoint":
            summary = read_json(
                root / f"outputs/{selected['experiment']}/seed_{selected['seed']}/summary.json"
            )
            row.update(
                trainable_parameters=summary["trainable_parameters"],
                parameters=summary["parameters"],
                frozen_parameters=summary.get("frozen_parameters", 0),
                checkpoint_bytes=summary["checkpoint_bytes"],
                validation_mae=selected["validation_mae"],
            )
        rows.append(row)
        frame["label"], frame["model"], frame["role"] = label, selected["model"], selected["role"]
        frame["seed"], frame["variant"] = selected.get("seed"), selected.get("variant")
        target_frames.append(frame)
        checks.append(
            {
                "label": label,
                "complete_cells_genes_targets": True,
                "official_scores_and_aggregates_match": True,
                "frozen_checkpoint_and_selection_match": True,
                "shared_control_sampling": True,
                "published_sources_match": True,
                "temporary_prediction_removed": True,
            }
        )
    return (
        selection,
        digest,
        ready,
        pd.DataFrame(rows),
        pd.concat(target_frames, ignore_index=True),
        sources,
        checks,
        cache_lineage,
    )


def paired_intervals(candidate, reference, candidate_name, reference_name, rng):
    if set(candidate.index) != set(reference.index) or len(candidate) != 100:
        raise ValueError("Final paired comparisons require the same 100 test targets")
    reference = reference.reindex(candidate.index)
    draws = rng.integers(0, 100, size=(5000, 100))
    rows = []
    for metric in METRICS:
        delta = (candidate[metric] - reference[metric]).to_numpy()
        low, high = np.quantile(delta[draws].mean(1), [0.025, 0.975])
        lower = metric in {"mae", "energy_mmd_rms"}
        rows.append(
            {
                "candidate": candidate_name,
                "reference": reference_name,
                "metric": metric,
                "difference": float(delta.mean()),
                "target_bootstrap_95": [float(low), float(high)],
                "candidate_better_targets": int((delta < 0).sum() if lower else (delta > 0).sum()),
                "targets": 100,
            }
        )
    return rows


def report_final_test(root: Path, selection_path: Path | None = None):
    selection_path = selection_path or root / "docs/final-model-selection.json"
    selection, digest, ready, runs, targets, sources, checks, cache_lineage = read_final_runs(
        root, selection_path
    )
    folder = root / "results/final-test"
    main = runs[runs.role == "main"]
    summary = main.groupby("model")[METRICS].agg(["mean", "std", "count"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary = summary.reindex(MODEL_ORDER)
    reference = runs[runs.role == "reference"].iloc[0]
    reference_targets = targets[targets.role == "reference"].set_index("perturbation")
    averaged = targets[targets.role == "main"].groupby(["model", "perturbation"])[METRICS].mean()
    contrasts, rng = [], np.random.default_rng(2025)
    for model in MODEL_ORDER:
        contrasts.extend(
            paired_intervals(averaged.loc[model], reference_targets, model, "no_perturbation", rng)
        )
    mlp_control = targets[(targets.model == "mlp") & (targets.variant == "mean")]
    if len(mlp_control) != 300:
        raise ValueError("The undistilled MLP mean control must contain all three seeds")
    control_means = mlp_control.groupby("perturbation")[METRICS].mean()
    contrasts.extend(
        paired_intervals(
            averaged.loc["distillation"], control_means, "distillation", "mlp/mean_kd_control", rng
        )
    )
    teacher_selection = selection["teacher_selection"]
    teacher_record = root / "data/processed/distillation-v1/manifest.json"
    teacher_cache = read_json(teacher_record)
    teacher_digest = teacher_cache["teacher_checkpoint_sha256"]
    if teacher_selection["teacher_checkpoint_sha256"] != teacher_digest:
        raise ValueError("Actual teacher differs from the frozen validation-only selection")
    teacher_row = next(
        (r for r in selection["runs"] if r.get("checkpoint_sha256") == teacher_digest), None
    )
    if teacher_row is None:
        raise ValueError("The actual teacher checkpoint must be included in final test evaluation")
    teacher_targets = targets[targets.label == teacher_row["label"]].set_index("perturbation")
    contrasts.extend(
        paired_intervals(
            averaged.loc["distillation"],
            teacher_targets,
            "distillation",
            "actual_teacher_seed_42",
            rng,
        )
    )
    runs.to_csv(folder / "runs.csv", index=False)
    summary.to_csv(folder / "summary.csv")
    targets.to_csv(folder / "per_perturbation.csv", index=False)
    write_json(
        folder / "paired-comparisons.json",
        {
            "contrasts": contrasts,
            "scope": "three seeds averaged within each of 100 test targets, except the fixed seed42 teacher and single reference; 5000 paired target bootstraps conditional on fitted models and fixed controls; no multiplicity adjustment; seeds are not biological replicates",
            "test_used": True,
        },
    )
    write_json(
        folder / "artifact-verification.json",
        {
            "all_checks_passed": True,
            "selection_sha256": digest,
            "test_used": True,
            "runs": checks,
            "checkpoint_evaluations": len(runs) - 1,
            "reference_evaluations": 1,
            "source_artifacts": sources,
            "actual_shape": ready["shape"],
            "perturbation_count": 100,
            "scope": "complete saved official scores, frozen sources, target aggregates, shared sampling and prediction cleanup; not a second independent biological validation",
            "cache_source_lineage": cache_lineage,
        },
    )
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    chart_models = ["no_perturbation", *MODEL_ORDER]
    for ax, metric, title in zip(
        axes.flat,
        METRICS,
        ["MAE (lower)", "PDS (higher)", "DES (higher)", "Energy MMD (lower; supplementary)"],
    ):
        means = [reference[metric]] + [summary.loc[m, f"{metric}_mean"] for m in MODEL_ORDER]
        errors = [0.0] + [summary.loc[m, f"{metric}_std"] for m in MODEL_ORDER]
        ax.bar(
            np.arange(9),
            means,
            yerr=errors,
            capsize=3,
            color=[
                "#777777",
                "#3973ac",
                "#d68937",
                "#479858",
                "#9467bd",
                "#278c9e",
                "#d46c91",
                "#81684b",
                "#9cba48",
            ],
        )
        ax.set_xticks(np.arange(9), [LABELS[m] for m in chart_models], rotation=38, ha="right")
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(folder / "comparison.png", dpi=180)
    plt.close(fig)
    lines = [
        "# FusionVC 最终保留测试集结果",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: academic-research-suite / experiment-agent",
        "- Origin Mode: run",
        "- Version Label: final-test-v1",
        "- Verification Status: VERIFIED（仅指冻结来源与完整评价产物核验）",
        "",
        f"模型与损失依据验证集冻结后，完成 {len(runs) - 1} 个checkpoint及一次无扰动对照的完整test评价。",
        f"实际test共 {ready['shape'][0]:,} 个细胞、18,080个基因、100个扰动，其中对照 {ready['control_cell_count']:,} 个。",
        "测试数据仅用于本次最终评价，没有用于训练、Teacher选择或超参数调整。",
        "",
        "## 八组主比较",
        "",
        "| 模型 | 验证选择的训练损失 | MAE ↓ | PDS ↑ | DES ↑ |",
        "| --- | --- | ---: | ---: | ---: |",
        "| 无扰动对照 | 固定一次抽样 | "
        + " | ".join(f"{reference[m]:.6f}" for m in METRICS[:3])
        + " |",
    ]
    for model in MODEL_ORDER:
        variant = main.loc[main.model == model, "variant"].iloc[0]
        values = " | ".join(
            f"{summary.loc[model, f'{m}_mean']:.6f} ± {summary.loc[model, f'{m}_std']:.6f}"
            for m in METRICS[:3]
        )
        lines.append(f"| {LABELS[model]} | {variant} | {values} |")
    lines += [
        "",
        "模型为42/43/44三种子的均值±样本标准差；无扰动是固定抽样的一次完整评价。",
        "每架构在相同的三个损失候选预算内按验证MAE选择，因此本表是验证选择后的模型方案比较，并非隔离架构因素的消融。",
        "MAE、PDS、DES分别解读；附加Energy MMD在CSV另列，不构造历史排行榜总分。",
        "",
        "## 蒸馏的同架构对照",
        "",
        "| 方法 | MAE ↓ | PDS ↑ | DES ↑ |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, frame in [
        ("未蒸馏MLP（mean）", mlp_control),
        ("蒸馏MLP", targets[targets.model == "distillation"]),
    ]:
        seed_metrics = frame.groupby("seed")[METRICS].mean()
        lines.append(
            f"| {name} | "
            + " | ".join(
                f"{seed_metrics[m].mean():.6f} ± {seed_metrics[m].std():.6f}" for m in METRICS[:3]
            )
            + " |"
        )
    actual_teacher = runs[runs.label == teacher_row["label"]].iloc[0]
    lines.append(
        "| 实际Teacher（固定seed42，非三种子均值） | "
        + " | ".join(f"{actual_teacher[m]:.6f}" for m in METRICS[:3])
        + " |"
    )
    student_mae = summary.loc["distillation", "mae_mean"]
    control_mae = mlp_control.groupby("seed").mae.mean().mean()
    kd_improvement = 100 * (control_mae - student_mae) / control_mae
    lines += [
        "",
        f"蒸馏相对同架构未蒸馏MLP的测试MAE改善约{kd_improvement:.2f}%。",
        "本次结果中PDS基本持平、DES下降；均值误差改善没有带来全面指标提升。",
        "Student的MAE仍高于实际单Teacher和无扰动对照，不能表述为无损迁移或全面超过Teacher。",
        "",
        "| Student−未蒸馏MLP | 差值 | 95% 靶点区间 |",
        "| --- | ---: | --- |",
    ]
    for item in contrasts:
        if item["candidate"] == "distillation" and item["reference"] == "mlp/mean_kd_control":
            low, high = item["target_bootstrap_95"]
            lines.append(
                f"| {item['metric']} | {item['difference']:+.6f} | [{low:+.6f}, {high:+.6f}] |"
            )
    lines += [
        "",
        "Teacher只由融合验证成绩选择，Student只接受150个训练靶点的软标签。Student推理不调用Teacher或两个基础编码器。",
        "推理成本、部署依赖、零KD公平性复现及训练来源详见[蒸馏报告](../distillation/README.md)。",
        "",
        "## 逐项差值与解释边界",
        "",
    ]
    for model in MODEL_ORDER:
        notes = []
        for metric, display in zip(METRICS[:3], ["MAE", "PDS", "DES"]):
            item = next(
                c
                for c in contrasts
                if c["candidate"] == model
                and c["reference"] == "no_perturbation"
                and c["metric"] == metric
            )
            low, high = item["target_bootstrap_95"]
            notes.append(f"{display}差值{item['difference']:+.6f} [{low:+.6f}, {high:+.6f}]")
        lines.append(f"- {LABELS[model]}：" + "；".join(notes) + "。")
    lines += [
        "",
        "差值为模型减无扰动；MAE负数更好，PDS/DES正数更好。区间先平均种子再对100个靶点配对重采样。",
        "区间仅对当前拟合模型和固定控制抽样成立，未作多重比较校正，训练种子不是生物重复。单项改善不能扩大为整体能力提升。",
        "本项目限定于H1 hESC单基因CRISPRi，仍是共同条件位移结构，细胞异质性主要来自输入对照；并不证明通用虚拟细胞能力。",
        "",
        "![Final test comparison](comparison.png)",
        "",
        "## 复现与完整性",
        "",
        "所有test运行绑定同一份冻结选择SHA256、原checkpoint和真实test缓存签名；使用完全相同的控制抽样。",
        "完整18,080基因及100靶点官方评分、逐靶点聚合和小产物核验均通过，每次临时预测矩阵已清理。",
        "训练/验证结果保留原路径，test产物明确标记test_used=true，独立保存在本目录。",
        "",
        "[冻结选择](../../docs/final-model-selection.json)、[最终评价协议](../../docs/final-evaluation-protocol.md)、[测试下载记录](../../docs/test-download-status.json)、",
        "[逐次成绩](runs.csv)、[八组汇总](summary.csv)、[逐靶点结果](per_perturbation.csv)、[条件性差值区间](paired-comparisons.json)、[产物核验](artifact-verification.json)。",
    ]
    (folder / "README.md").write_text("\n".join(lines) + "\n", "utf-8")
    write_json(
        folder / "report-provenance.json",
        {
            "selection_path": selection_path.relative_to(root).as_posix(),
            "selection_sha256": digest,
            "source_artifacts": sources,
            "real_cache_manifest_sha256": file_sha256(
                root / "outputs/final-test-cache/real-ready.json"
            ),
            "actual_teacher_label": teacher_row["label"],
            "main_models": 8,
            "checkpoint_evaluations": len(runs) - 1,
            "complete": True,
            "test_used": True,
        },
    )
    print(summary.to_string(), flush=True)
