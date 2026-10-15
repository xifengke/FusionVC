"""Compare the fixed teacher, unchanged supervised MLP and distilled students."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .distillation import EXPERIMENT, SEEDS, load_distillation_cache, verify_prior_artifacts
from .fusion_mmd_report import METRICS, load_run
from .fusion_report import (
    control_plan_matches,
    load_reference,
    validate_evaluation,
    validated_target_scores,
)
from .io import file_sha256, read_json, write_json


def load_student_result(root: Path, seed: int):
    run = root / f"outputs/{EXPERIMENT}/seed_{seed}"
    folder = root / f"results/distillation/{EXPERIMENT}/seed_{seed}"
    config, summary, evaluation = [
        read_json(folder / name) for name in ["config.json", "training.json", "metrics.json"]
    ]
    validate_evaluation(evaluation)
    if (
        config["experiment"] != EXPERIMENT
        or config["model"] != "mlp"
        or config["seed"] != seed
        or config["protocol"] != "distillation-v1"
        or config["results_group"] != "distillation"
        or config["variant"] != "mean_kd"
        or config["kd_weight"] != 1.0
        or config["loss"] != "unpaired_pseudobulk_mse"
        or config["selection"] != "validation_pseudobulk_mae"
        or summary["test_used"] is not False
    ):
        raise ValueError("Unexpected formal distilled-student configuration")
    for path, value in [
        (run / "config.json", config),
        (run / "summary.json", summary),
        (run / "evaluation/metrics.json", evaluation),
    ]:
        if read_json(path) != value:
            raise ValueError(f"Published distilled student differs from source: {path}")
    digest = file_sha256(run / "best.pt")
    if digest != summary["checkpoint_sha256"] or digest != evaluation["checkpoint_sha256"]:
        raise ValueError("Distilled-student checkpoint hash differs")
    info = read_json(run / "evaluation/prediction-info.json")
    if (
        info["shape"] != [98927, 18080]
        or info["genes"] != 18080
        or info["control_counts_match_official"] is not True
        or info["control_sampling_seed"] != 2025
        or not control_plan_matches(
            run / "evaluation/control_sampling.npz",
            root / "outputs/01_mlp/seed_42/evaluation/control_sampling.npz",
        )
    ):
        raise ValueError(
            "Student did not use complete validation cells/genes and the common sampling plan"
        )
    targets = validated_target_scores(folder / "per_perturbation.csv", evaluation)
    distribution = pd.read_csv(folder / "distribution_per_perturbation.csv")
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
    ):
        raise ValueError("Student sampled distribution scores are incomplete")
    for published, source in [
        ("history.csv", "history.csv"),
        ("per_perturbation.csv", "evaluation/results.csv"),
        ("distribution_per_perturbation.csv", "evaluation/distribution_per_perturbation.csv"),
    ]:
        if file_sha256(folder / published) != file_sha256(run / source):
            raise ValueError("Published student history/target scores differ from source")
    history = pd.read_csv(folder / "history.csv")
    if (
        len(history) != summary["epochs_run"]
        or not np.array_equal(history.epoch, np.arange(1, len(history) + 1))
        or not np.isfinite(
            history[["train_mse", "train_kd_mse", "train_loss", "val_mae", "val_mse"]]
        )
        .all()
        .all()
        or not np.allclose(
            history.train_loss, history.train_mse + history.train_kd_mse, rtol=1e-6, atol=1e-9
        )
        or not np.isclose(
            history.loc[history.epoch == summary["best_epoch"], "val_mae"].iloc[0],
            summary["selection_validation_mae"],
            rtol=1e-12,
            atol=1e-12,
        )
    ):
        raise ValueError("Student history does not implement mean MSE + 1.0 expression-mean KD")
    targets = targets.merge(distribution, on="perturbation", validate="one_to_one")
    targets["role"], targets["seed"] = "distilled_mlp", seed
    row = {
        "role": "distilled_mlp",
        "experiment": EXPERIMENT,
        "seed": seed,
        **evaluation["metrics"],
        "energy_mmd_rms": evaluation["distribution"]["energy_mmd_rms"],
        **{
            name: summary[name]
            for name in [
                "parameters",
                "trainable_parameters",
                "frozen_parameters",
                "training_seconds",
                "checkpoint_bytes",
                "best_epoch",
            ]
        },
        "peak_gpu_mib": summary["peak_allocated_gpu_bytes"] / 1024**2,
        "checkpoint_sha256": digest,
        "evaluation_seconds": evaluation["evaluation_seconds"],
    }
    sources = [
        {"path": str((folder / name).relative_to(root)), "sha256": file_sha256(folder / name)}
        for name in [
            "config.json",
            "training.json",
            "metrics.json",
            "history.csv",
            "per_perturbation.csv",
            "distribution_per_perturbation.csv",
        ]
    ]
    return row, targets, sources


def collect_distillation(root: Path):
    _, _, manifest = load_distillation_cache(root)
    teacher = manifest["signature"]["teacher"]
    rows, frames, sources = [], [], []
    for seed in SEEDS:
        row, frame, source, _ = load_run(root, "mlp", "mean", seed)
        row["role"], frame["role"] = "supervised_mlp", "supervised_mlp"
        rows.append(row)
        frames.append(frame)
        sources.extend(source)
        row, frame, source = load_student_result(root, seed)
        rows.append(row)
        frames.append(frame)
        sources.extend(source)
    row, frame, source, _ = load_run(root, teacher["model"], teacher["variant"], 42)
    row["role"], frame["role"] = "teacher", "teacher"
    rows.append(row)
    frames.append(frame)
    sources.extend(source)
    return pd.DataFrame(rows), pd.concat(frames, ignore_index=True), sources


def distillation_intervals(targets, reference_targets):
    averaged = targets.groupby(["role", "perturbation"])[METRICS].mean()
    candidate = averaged.loc["distilled_mlp"].sort_index()
    rng, rows = np.random.default_rng(2025), []
    for role in ["supervised_mlp", "teacher", "no_perturbation"]:
        baseline = (
            reference_targets.set_index("perturbation")
            if role == "no_perturbation"
            else averaged.loc[role]
        )
        if len(candidate) != 50 or set(candidate.index) != set(baseline.index):
            raise ValueError("Student contrasts require the same 50 validation targets")
        baseline = baseline.reindex(candidate.index)
        metrics = METRICS[:3] if role == "no_perturbation" else METRICS
        samples = rng.integers(0, 50, (5000, 50))
        for metric in metrics:
            delta = (candidate[metric] - baseline[metric]).to_numpy()
            interval = np.quantile(delta[samples].mean(1), [0.025, 0.975])
            rows.append(
                {
                    "candidate": "distilled_mlp",
                    "reference": role,
                    "metric": metric,
                    "difference": float(delta.mean()),
                    "target_bootstrap_95": interval.tolist(),
                    "targets": 50,
                    "candidate_better_targets": int(
                        (delta < 0).sum()
                        if metric in {"mae", "energy_mmd_rms"}
                        else (delta > 0).sum()
                    ),
                }
            )
    return {
        "contrasts": rows,
        "bootstrap_seed": 2025,
        "scope": "student and supervised MLP three-seed means; one fixed seed42 teacher; paired 5000 bootstraps of 50 validation targets; conditional on models, teacher/checkpoint selection and fixed controls; no multiplicity correction; not biological replicates",
        "test_used": False,
    }


def verification_passport(root: Path):
    folder = root / "results/distillation"
    paths = {
        name: folder / filename
        for name, filename in [
            ("artifacts", "artifact-verification.json"),
            ("reproducibility", "reproducibility-check.json"),
            ("zero_kd", "zero-kd-parity.json"),
            ("validation", "validation-checks.json"),
            ("cost", "cost-benchmark.json"),
        ]
    }
    receipts = {name: read_json(path) for name, path in paths.items() if path.exists()}
    tests = receipts.get("validation", {})
    verified = (
        len(receipts) == len(paths)
        and receipts["artifacts"].get("all_checks_passed") is True
        and receipts["reproducibility"].get("all_tensors_equal") is True
        and receipts["zero_kd"].get("all_tensors_equal") is True
        and tests.get("tests", {}).get("exit_code") == 0
        and tests.get("tests", {}).get("passed", 0) > 0
        and tests.get("ruff_passed") is True
        and all(r.get("test_used") is False for r in receipts.values())
    )
    if verified:
        student_runs = receipts["artifacts"].get("student_runs", [])
        reruns = receipts["reproducibility"].get("runs", [])
        verified &= len(student_runs) == 3 and {r.get("seed") for r in student_runs} == set(SEEDS)
        verified &= len(reruns) == 2 and {
            (r.get("reference_experiment"), r.get("repeated_experiment")) for r in reruns
        } == {("01_mlp", "zero_kd_08_distillation"), (EXPERIMENT, "repro_check_08_distillation")}
        verified &= receipts["artifacts"].get("teacher_cache_manifest_sha256") == file_sha256(
            root / "data/processed/distillation-v1/manifest.json"
        )
        verified &= receipts["artifacts"].get("teacher_selection_sha256") == file_sha256(
            folder / "teacher-selection.json"
        )
        for row in student_runs:
            verified &= row["checkpoint_sha256"] == file_sha256(
                root / f"outputs/{EXPERIMENT}/seed_{row['seed']}/best.pt"
            )
        for row in reruns:
            verified &= row["original_checkpoint_sha256"] == file_sha256(
                root / f"outputs/{row['reference_experiment']}/seed_42/best.pt"
            )
            verified &= row["repeated_checkpoint_sha256"] == file_sha256(
                root / f"outputs/{row['repeated_experiment']}/seed_42/best.pt"
            )
            verified &= all(
                row.get(key) is True
                for key in [
                    "every_tensor_equal",
                    "selected_epoch_equal",
                    "core_training_history_equal",
                ]
            )
        zero = receipts["zero_kd"].get("run", {})
        verified &= bool(zero) and zero.get("original_checkpoint_sha256") == file_sha256(
            root / "outputs/01_mlp/seed_42/best.pt"
        )
        verified &= bool(zero) and zero.get("repeated_checkpoint_sha256") == file_sha256(
            root / "outputs/zero_kd_08_distillation/seed_42/best.pt"
        )
        for row in receipts["cost"].get("rows", []):
            verified &= row["checkpoint_sha256"] == file_sha256(
                root / f"outputs/{row['experiment']}/seed_42/best.pt"
            )
        verified &= len(receipts["cost"].get("rows", [])) == 3
    return {
        "origin_skill": "academic-research-suite/experiment-agent",
        "stage": "run",
        "date": "2026-10-15",
        "version": "distillation-v1",
        "verification_status": "VERIFIED" if verified else "UNVERIFIED",
        "receipts": {
            name: {"path": str(path.relative_to(root)), "sha256": file_sha256(path)}
            for name, path in paths.items()
            if path.exists()
        },
    }


def report_distillation(root: Path):
    prior = verify_prior_artifacts(root)
    reference, reference_targets = load_reference(root)
    runs, targets, sources = collect_distillation(root)
    passport = verification_passport(root)
    grouped = runs.groupby("role")[METRICS].agg(["mean", "std", "count"])
    grouped.columns = ["_".join(column) for column in grouped.columns]
    intervals = distillation_intervals(targets, reference_targets)
    folder = root / "results/distillation"
    runs.to_csv(folder / "runs.csv", index=False)
    grouped.to_csv(folder / "summary.csv")
    targets.to_csv(folder / "per_perturbation.csv", index=False)
    write_json(folder / "paired-comparisons.json", intervals)
    selection = read_json(folder / "teacher-selection.json")
    labels = {
        "supervised_mlp": "Supervised MLP",
        "distilled_mlp": "Distilled MLP",
        "teacher": "Fixed teacher",
    }
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.8), constrained_layout=True)
    roles = ["supervised_mlp", "distilled_mlp", "teacher"]
    for ax, metric, title in zip(
        axes, METRICS, ["MAE (lower)", "PDS (higher)", "DES (higher)", "Sampled Energy MMD (lower)"]
    ):
        means = [grouped.loc[r, f"{metric}_mean"] for r in roles]
        stds = [0.0 if r == "teacher" else grouped.loc[r, f"{metric}_std"] for r in roles]
        ax.bar(np.arange(3), means, yerr=stds, capsize=3, color=["#3973ac", "#278c9e", "#d68937"])
        if metric in reference["metrics"]:
            ax.axhline(
                reference["metrics"][metric],
                linestyle="--",
                color="black",
                linewidth=1,
                label="No perturbation",
            )
        ax.set_xticks(np.arange(3), [labels[r] for r in roles], rotation=25, ha="right")
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=7)
    fig.savefig(folder / "comparison.png", dpi=180)
    plt.close(fig)
    lines = [
        "# 固定融合 Teacher → 原架构 MLP Student 的均值响应蒸馏",
        "",
        "## Material Passport",
        "",
        "| Field | Value |",
        "| --- | --- |",
        "| Origin Skill | academic-research-suite/experiment-agent |",
        "| Stage | run |",
        "| Date | 2026-10-15 |",
        "| Version | distillation-v1 |",
        f"| Verification Status | {passport['verification_status']} |",
        "",
        f"Teacher 为 **{selection['experiment']}/seed_42**：在九个融合架构×损失组中，按三种子官方验证 MAE 均值最小选择，精确并列按 experiment 字典序。",
        f"胜出组三种子 MAE 均值为 {selection['selected_group_validation_mae_mean']:.6f}；实际单 Teacher MAE 为 {selection['selected_checkpoint_validation_mae']:.6f}。所有 Student 共用该固定 checkpoint，不筛选最佳种子或使用 ensemble。",
        "只对 150 个训练靶点离线生成 Teacher delta；训练时在相同 32 个控制细胞上经 apply_residual 转为表达均值，使用 mean MSE + 1.0 × KD MSE。未扩展 Student 的 MMD 任务损失。",
        "Student 与原 MLP 架构、输入、初始化种子、预算及验证 MAE 选择规则一致；推理只用训练对照拟合的基因描述与 MLP，不调用 Teacher 或基础编码器。",
        "",
        "| 方法 | 运行数 | MAE ↓ | PDS ↑ | DES ↑ | 附加 Energy MMD ↓ |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        "| 无扰动对照 | 固定一次 | "
        + " | ".join(f"{reference['metrics'][m]:.6f}" for m in METRICS[:3])
        + " | — |",
    ]
    for role in roles:
        row = grouped.loc[role]
        values = [
            f"{row[f'{m}_mean']:.6f}" + ("" if role == "teacher" else f" ± {row[f'{m}_std']:.6f}")
            for m in METRICS
        ]
        lines.append(
            f"| {labels[role]} | {1 if role == 'teacher' else 3} | " + " | ".join(values) + " |"
        )
    lines += [
        "",
        "MLP 两组为种子 42/43/44 的均值 ± 样本标准差；Teacher 是实际使用的固定单模型，不能把胜出组均值当作它的成绩。",
        "",
        "## 蒸馏学生的配对差值",
        "",
        "| 参考 | 指标 | Student−参考 | 95% 靶点区间 |",
        "| --- | --- | ---: | ---: |",
    ]
    for item in intervals["contrasts"]:
        low, high = item["target_bootstrap_95"]
        lines.append(
            f"| {item['reference']} | {item['metric']} | {item['difference']:+.6f} | [{low:+.6f}, {high:+.6f}] |"
        )
    change = (
        grouped.loc["distilled_mlp", "mae_mean"] / grouped.loc["supervised_mlp", "mae_mean"] - 1
    ) * 100
    lines += [
        "",
        f"蒸馏学生相对同架构纯监督 MLP 的 MAE 均值变化为 {change:+.2f}%。应同时检查 PDS/DES，单项变化不能扩大为整体能力提升。",
        "区间先平均 Student/MLP 的三个种子，再对相同 50 个验证靶点做 5,000 次配对重采样；Teacher 保持固定。区间只对当前模型、Teacher/checkpoint 选择和控制抽样成立，未做多重比较校正。",
        "同一验证集承担融合筛选、早停与本轮比较，存在选择乐观偏差；种子不是生物重复，均值蒸馏不证明分布或细胞状态机制迁移。测试集在本阶段未使用。",
        "",
        "![Distillation validation comparison](comparison.png)",
        "",
        "## 参数与实际成本",
        "",
    ]
    cost_path = folder / "cost-benchmark.json"
    if cost_path.exists():
        cost = read_json(cost_path)
        lines += [
            "基准使用相同 150 个训练靶点、32 个训练对照、18,080 输出基因，float32，目标 batch=4；一轮预热后五轮测量。",
            "",
            "| 方法 | 可训练参数 | 源冻结参数 | checkpoint MiB | 当前缓存加载秒 | 暖预测头毫秒/靶点 | 预测头+完整表达毫秒/靶点 | GPU 峰值 MiB |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in cost["rows"]:
            head = row["measurements"]["warm_cached_head"]["median_seconds_per_perturbation"] * 1000
            expression = (
                row["measurements"]["warm_cached_head_and_expression"][
                    "median_seconds_per_perturbation"
                ]
                * 1000
            )
            lines.append(
                f"| {row['label']} | {row['trainable_parameters']:,} | {row['frozen_parameters']:,} | {row['checkpoint_bytes'] / 1024**2:.2f} | {row['cached_model_and_inputs_load_seconds']:.3f} | {head:.3f} | {expression:.3f} | {row['measured_peak_allocated_gpu_bytes'] / 1024**2:.1f} |"
            )
        teacher_cost = next(r for r in cost["rows"] if r["label"] == "teacher")
        lines += [
            "",
            f"Teacher 原缓存记录的编码器特征提取为 {teacher_cost['existing_recorded_encoder_feature_extraction_seconds']:.2f} 秒；共同预处理原记录为 {teacher_cost['existing_recorded_shared_preprocessing_seconds']:.2f} 秒，均为历史实测的一次性共享成本，并非本次重新测量的冷启动。",
            f"两组依赖共同训练控制矩阵 {teacher_cost['shared_training_control_file_bytes'] / 1024**3:.3f} GiB；静态目标表、缓存来源资产另列于成本 JSON。checkpoint 大小不是完整部署大小。",
            "成本 JSON 分开列出数值预测 payload、研究来源校验实际读取的文件，以及仅 stat 的训练原始矩阵；原始矩阵的字节数不计为计算模型大小。当前研究 loader 的来源核验依赖与最小推理部署包是两种口径，后者另行导出和验证。",
        ]
    else:
        lines.append("成本基准尚未执行，不能报告加速或显存收益。")
    lines += [
        "",
        "Teacher 与 Student 当前暖推理都使用缓存；编码器参数比不等于端到端加速。加载测量包含现有 loader 的文件/来源校验与 GPU 转移，未清除操作系统文件缓存；表达基准不含文件输出或官方 DE 评分。完整包含内容及五次原始测量见成本 JSON。",
        "",
        "## 完整性与复现",
        "",
        f"训练前的 {prior['checkpoint_count']} 个 checkpoint 和 {prior['small_artifact_count']} 个历史结果文件均保持原 SHA256。",
        "独立 trainer 的 KD=0 seed42 须逐 tensor、最佳轮次及核心训练日志复现原 MLP；正式 KD seed42 在独立目录重训。三次正式 Student 均作完整官方验证、相同控制抽样与 checkpoint 哈希核验。",
        "Material Passport 仅在产物核验、KD0公平性、KD复现、测试与成本记录齐全且通过时为 VERIFIED，含义限于本轮可复现产物。",
        "",
        "[Teacher选择](teacher-selection.json)、[训练软标签来源](teacher-cache.json)、[逐次成绩](runs.csv)、[均值标准差](summary.csv)、[逐靶点](per_perturbation.csv)、",
        "[配对区间](paired-comparisons.json)、[KD0公平性](zero-kd-parity.json)、[精确复现](reproducibility-check.json)、[产物核验](artifact-verification.json)、",
        "[成本基准](cost-benchmark.json)、[测试](validation-checks.json)、[协议](../../docs/distillation-protocol.md)。",
    ]
    (folder / "README.md").write_text("\n".join(lines) + "\n", "utf-8")
    write_json(
        folder / "report-provenance.json",
        {
            "material_passport": passport,
            "teacher": selection,
            "sources": sources,
            "prior_snapshot_sha256": prior["snapshot_sha256"],
            "student_runs": 3,
            "supervised_runs": 3,
            "teacher_runs": 1,
            "test_used": False,
            "complete": True,
        },
    )
    print(grouped.to_string(), flush=True)
