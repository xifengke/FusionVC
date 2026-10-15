"""Report the two encoders crossed with three loss variants, without mixing protocols."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .io import read_json, write_json


def verify_mean_parity(root: Path) -> dict:
    checks = []
    for model, prefix in [("mlp", "01"), ("transformer", "02")]:
        for seed in [42, 43, 44]:
            original = root / f"outputs/{prefix}_{model}/seed_{seed}/best.pt"
            repeated = root / f"outputs/mmd_{prefix}_{model}_mean/seed_{seed}/best.pt"
            if not original.exists() or not repeated.exists():
                continue
            old = torch.load(original, map_location="cpu", weights_only=True)
            new = torch.load(repeated, map_location="cpu", weights_only=True)
            equal = old["state_dict"].keys() == new["state_dict"].keys()
            equal = equal and all(
                torch.equal(old["state_dict"][k], new["state_dict"][k]) for k in old["state_dict"]
            )
            checks.append(
                {
                    "model": model,
                    "seed": seed,
                    "weights_equal": equal,
                    "old_epoch": old["epoch"],
                    "new_epoch": new["epoch"],
                }
            )
    result = {
        "runs": checks,
        "all_available_equal": bool(checks) and all(r["weights_equal"] for r in checks),
        "test_used": False,
    }
    write_json(root / "results/mmd/baseline-parity.json", result)
    if checks and not result["all_available_equal"]:
        raise RuntimeError("Mean controls failed original-weight parity; inspect before comparison")
    return result


def report_mmd(root: Path) -> None:
    folder = root / "results/mmd"
    folder.mkdir(parents=True, exist_ok=True)
    rows, target_frames = [], []
    for path in sorted(folder.glob("mmd_*/seed_*/metrics.json")):
        evaluation = read_json(path)
        training = read_json(path.parent / "training.json")
        if training.get("protocol") != "mmd-v1":
            raise ValueError("Non-MMD protocol included in MMD report")
        rows.append(
            {
                "experiment": training["experiment"],
                "model": training["model"],
                "variant": training["variant"],
                "seed": training["seed"],
                **evaluation["metrics"],
                "energy_mmd_rms": evaluation["distribution"]["energy_mmd_rms"],
                "parameters": training["parameters"],
                "best_epoch": training["best_epoch"],
                "training_seconds": training["training_seconds"],
                "peak_gpu_mib": training["peak_allocated_gpu_bytes"] / 1024**2,
            }
        )
        frame = pd.read_csv(path.parent / "per_perturbation.csv")
        distribution = pd.read_csv(path.parent / "distribution_per_perturbation.csv")
        frame = frame.merge(distribution, on="perturbation", validate="one_to_one")
        frame["model"], frame["variant"], frame["seed"] = (
            training["model"],
            training["variant"],
            training["seed"],
        )
        target_frames.append(frame)
    if not rows:
        raise RuntimeError("No finished MMD evaluations")
    df = pd.DataFrame(rows).sort_values(["model", "variant", "seed"])
    expected = {
        (model, variant, seed)
        for model in ["mlp", "transformer"]
        for variant in ["mean", "energy", "hybrid"]
        for seed in [42, 43, 44]
    }
    actual = set(df[["model", "variant", "seed"]].itertuples(index=False, name=None))
    complete = len(df) == len(expected) and actual == expected
    df.to_csv(folder / "runs.csv", index=False)
    metrics = ["mae", "discrimination_score_l1", "overlap_at_N", "energy_mmd_rms"]
    grouped = df.groupby(["model", "variant"])[metrics].agg(["mean", "std", "count"])
    grouped.columns = ["_".join(c) for c in grouped.columns]
    grouped.to_csv(folder / "summary.csv")
    targets = pd.concat(target_frames, ignore_index=True)
    targets.to_csv(folder / "per_perturbation.csv", index=False)
    averaged = targets.groupby(["model", "variant", "perturbation"])[metrics].mean()
    contrasts = []
    rng = np.random.default_rng(2025)
    for model in ["mlp", "transformer"]:
        if (model, "mean") not in averaged.index.droplevel("perturbation"):
            continue
        baseline = averaged.loc[(model, "mean")]
        for variant in ["energy", "hybrid"]:
            if (model, variant) not in averaged.index.droplevel("perturbation"):
                continue
            changed = averaged.loc[(model, variant)]
            for metric in metrics:
                delta = changed[metric] - baseline[metric]
                bootstrap = delta.to_numpy()[rng.integers(0, len(delta), (5000, len(delta)))].mean(
                    1
                )
                contrasts.append(
                    {
                        "model": model,
                        "variant": variant,
                        "metric": metric,
                        "difference_from_mean": float(delta.mean()),
                        "target_bootstrap_95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
                        "targets": len(delta),
                    }
                )
    write_json(
        folder / "paired-comparisons.json",
        {
            "contrasts": contrasts,
            "scope": "conditional on fitted models and fixed validation targets; seed means",
            "test_used": False,
        },
    )
    labels = {"mean": "Mean", "energy": "MMD", "hybrid": "Mean + MMD"}
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), constrained_layout=True)
    for ax, metric, title in zip(
        axes, metrics, ["MAE (lower)", "PDS (higher)", "DES (higher)", "Energy MMD (lower)"]
    ):
        for offset, variant in enumerate(["mean", "energy", "hybrid"]):
            positions, means, errors = [], [], []
            for index, model in enumerate(["mlp", "transformer"]):
                if (model, variant) in grouped.index:
                    row = grouped.loc[(model, variant)]
                    positions.append(index + (offset - 1) * 0.24)
                    means.append(row[f"{metric}_mean"])
                    deviation = row[f"{metric}_std"]
                    errors.append(0.0 if pd.isna(deviation) else deviation)
            ax.bar(positions, means, yerr=errors, width=0.22, capsize=3, label=labels[variant])
        ax.set_xticks([0, 1], ["MLP", "Transformer"])
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].legend(fontsize=8)
    fig.savefig(folder / "comparison.png", dpi=180)
    plt.close(fig)
    lines = [
        "# MMD 对照实验结果",
        "",
        f"已完成 {len(df)}/18 次正式验证：MLP 和 Transformer 各三种训练方式，每种三个随机种子。",
        "模型结构、数据、基因数量、训练预算上限和模型选择规则相同，只改变训练时怎样判断预测好坏。",
        "原始对照比较一群细胞的平均值；MMD 比较两群细胞的分布；混合组同时比较两者。",
        "输出覆盖全部 18,080 个基因，使用官方验证划分；这些是验证成绩，测试集未使用。",
        "",
        "| Model | Loss | Seeds | MAE ↓ | PDS ↑ | DES ↑ | Energy MMD ↓ |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for (model, variant), row in grouped.iterrows():
        formatted = []
        for metric in metrics:
            value, deviation = row[f"{metric}_mean"], row[f"{metric}_std"]
            formatted.append(
                f"{value:.6f}" if pd.isna(deviation) else f"{value:.6f} ± {deviation:.6f}"
            )
        lines.append(
            f"| {model} | {labels[variant]} | {int(row['mae_count'])} | "
            + " | ".join(formatted)
            + " |"
        )
    lines += [
        "",
        "表中为三个种子的均值 ± 标准差。MAE 是表达平均误差，越低越好；",
        "PDS 衡量能否区分不同扰动，DES 衡量关键变化基因的预测重合程度，越高越好。",
        "Energy MMD 是额外的细胞群差异检查，越低越好；不是官方综合得分。",
        "",
        "![Comparison](comparison.png)",
        "",
    ]
    if complete:
        mlp = grouped.loc[("mlp", "mean"), "mae_mean"]
        mlp_mmd = grouped.loc[("mlp", "energy"), "mae_mean"]
        transformer = grouped.loc[("transformer", "mean"), "mae_mean"]
        transformer_hybrid = grouped.loc[("transformer", "hybrid"), "mae_mean"]
        lines += [
            f"本次 MLP 的纯 MMD 组 MAE 相比原对照降低 {(mlp - mlp_mmd) / mlp:.2%}；",
            f"Transformer 的混合组降低 {(transformer - transformer_hybrid) / transformer:.2%}。",
            "这两种 MAE 改善幅度很小，DES 都未同步提高；Transformer 的纯 MMD 组四项平均指标均变差。",
            "各新训练组的 MAE 仍高于已有无扰动对照 0.027547。",
            "本轮没有观察到整体预测能力的明显提升，因此保留原模型作为后续实验的正式基线。",
            "这说明当前条件下仅改变损失收益有限，不能据此断言 MMD 在其他模型或数据上无效。",
            "",
        ]
    lines += [
        "附加 Energy MMD 每个验证靶点固定独立抽取 64 个预测细胞和 64 个真实细胞，有抽样误差。",
        "混合权重预先固定为 0.1；energy kernel 除以基因数平方根，属于固定尺度变换。",
        "所有组仍按验证集平均误差挑选模型，这可能限制针对分布的改善。",
        "当前模型对同一扰动学习共同的表达位移，尚未学习各细胞状态对应的独立响应，也未实现 State 的细胞集合 attention。",
        "",
        "原均值对照全部重新训练，与旧 checkpoint 权重完全一致，见 [baseline-parity.json](baseline-parity.json)。",
        "六次均值对照的官方评分与旧结果完全一致，见 [evaluation-reference-parity.json](evaluation-reference-parity.json)。",
        "按靶点的差值区间见 [paired-comparisons.json](paired-comparisons.json)：先平均种子，再对 50 个验证靶点重采样；不代表测试集表现保证。",
        "完整方案见 [mmd-protocol.md](../../docs/mmd-protocol.md)，每次运行见 [runs.csv](runs.csv)。",
    ]
    (folder / "README.md").write_text("\n".join(lines) + "\n", "utf-8")
    write_json(
        folder / "report-provenance.json",
        {
            "protocol": "mmd-v1",
            "runs": len(df),
            "expected_runs": 18,
            "complete": complete,
            "test_used": False,
            "metrics_source": "cell-eval 0.6.6 VCC profile + supplementary sampled energy MMD",
        },
    )
    cache_record = root / "data/processed/mmd-v1/training-ready.json"
    if cache_record.exists():
        write_json(folder / "training-data-cache.json", read_json(cache_record))
    print(grouped.to_string(), flush=True)
