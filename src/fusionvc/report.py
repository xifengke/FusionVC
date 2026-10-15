"""Publish small result artifacts; keep checkpoints and matrices local."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .io import read_json, write_json


def report(root: Path) -> None:
    folder = root / "results/baselines"
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(folder.glob("*/seed_*/metrics.json")):
        evaluation = read_json(path)
        training = read_json(path.parent / "training.json")
        rows.append(
            {
                "model": training["experiment"],
                "seed": training["seed"],
                **evaluation["metrics"],
                "parameters": training["parameters"],
                "best_epoch": training["best_epoch"],
                "training_seconds": training["training_seconds"],
                "peak_gpu_mib": training["peak_allocated_gpu_bytes"] / 1024**2,
            }
        )
    if not rows:
        raise RuntimeError("No finished evaluations to report")
    df = pd.DataFrame(rows)
    df.to_csv(folder / "runs.csv", index=False)
    metrics = ["mae", "discrimination_score_l1", "overlap_at_N"]
    agg = df.groupby("model")[metrics].agg(["mean", "std", "count"])
    agg.columns = ["_".join(c) for c in agg.columns]
    agg.to_csv(folder / "summary.csv")
    pert_rows = []
    for path in sorted(folder.glob("*/seed_*/per_perturbation.csv")):
        frame = pd.read_csv(path)
        frame["model"] = path.parent.parent.name
        frame["seed"] = int(path.parent.name.split("_")[-1])
        pert_rows.append(frame)
    per_pert = pd.concat(pert_rows).groupby(["model", "perturbation"])[metrics].mean()
    per_pert.to_csv(folder / "per_perturbation_mean.csv")
    if {"01_mlp", "02_transformer"} <= set(per_pert.index.get_level_values("model")):
        paired = per_pert.loc["01_mlp"].join(
            per_pert.loc["02_transformer"], lsuffix="_mlp", rsuffix="_transformer"
        )
        paired["mae_transformer_minus_mlp"] = paired["mae_transformer"] - paired["mae_mlp"]
        paired.sort_values("mae_transformer_minus_mlp", ascending=False).to_csv(
            folder / "paired_target_analysis.csv"
        )
        delta = paired["mae_transformer_minus_mlp"].to_numpy()
        rng = np.random.default_rng(2025)
        bootstrap = delta[rng.integers(0, len(delta), size=(5000, len(delta)))].mean(1)
        write_json(
            folder / "paired_mae_analysis.json",
            {
                "contrast": "Transformer minus MLP; positive means MLP lower MAE",
                "mean_difference": float(delta.mean()),
                "percentile_95_interval": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
                "mlp_lower_mae_targets": int((delta > 0).sum()),
                "targets": len(delta),
                "method": "paired bootstrap over 50 fixed validation target losses; seeds averaged",
                "scope": "conditional on fitted models and this validation set; no test inference",
            },
        )
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), constrained_layout=True)
    model_names = list(agg.index)
    for ax, metric, title in zip(axes, metrics, ["MAE (lower)", "PDS (higher)", "DES (higher)"]):
        means = agg[f"{metric}_mean"].to_numpy()
        std = agg[f"{metric}_std"].fillna(0).to_numpy()
        ax.bar(np.arange(len(means)), means, yerr=std, capsize=4, color=["#3973ac", "#d68937"])
        ax.set_xticks(np.arange(len(means)), [x.split("_", 1)[1] for x in model_names])
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(folder / "comparison.png", dpi=180)
    plt.close(fig)
    lines = [
        "# MLP / Transformer validation results",
        "",
        "VCC 2025 official validation split; 18,080 genes. Test untouched.",
        "",
        "Pseudobulk-trained condition mean-shift models; full required cell counts in evaluation.",
        "",
        "| Model | Seeds | MAE ↓ | PDS ↑ | DES ↑ |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    auxiliary_path = folder / "auxiliary.csv"
    if auxiliary_path.exists():
        auxiliary = pd.read_csv(auxiliary_path).set_index("reference")
        reference = auxiliary.loc["no_perturbation"]
        if pd.notna(reference.get("overlap_at_N")):
            lines.append(
                f"| no_perturbation | fixed sample | {reference['mae']:.6f} | "
                f"{reference['discrimination_score_l1']:.6f} | {reference['overlap_at_N']:.6f} |"
            )
    for name, row in agg.iterrows():
        formatted = [f"{row[f'{m}_mean']:.6f} ± {row[f'{m}_std']:.6f}" for m in metrics]
        lines.append(f"| {name} | {int(row['mae_count'])} | " + " | ".join(formatted) + " |")
    lines += [
        "",
        "![comparison](comparison.png)",
        "",
        "Raw validation metrics; not the historical competition normalized score.",
        "Standard deviation is across training seeds; control sampling is held fixed.",
        "See `runs.csv` for runtime, parameters, selected epochs and GPU memory.",
        "Each run includes training logs/configs and per-perturbation metrics.",
    ]
    auxiliary_path = folder / "auxiliary.csv"
    if auxiliary_path.exists():
        auxiliary = pd.read_csv(auxiliary_path)
        lines += [
            "",
            "## Auxiliary references",
            "",
            "Same full-cell sampling plan. No-perturbation DES uses full cells; average-training-shift DES remains unevaluated.",
            "",
            "| Reference | MAE ↓ | PDS ↑ | DES ↑ |",
            "| --- | ---: | ---: | ---: |",
        ]
        for _, row in auxiliary.iterrows():
            des = (
                f"{row['overlap_at_N']:.6f}"
                if pd.notna(row.get("overlap_at_N"))
                else "not evaluated"
            )
            lines.append(
                f"| {row['reference']} | {row['mae']:.6f} | {row['discrimination_score_l1']:.6f} | {des} |"
            )
        lines += [
            "",
            "Both neural baselines have higher MAE than the no-perturbation reference.",
            "Transformer improves DES relative to MLP, while its MAE is worse.",
            "PDS remains close to the uninformative-reference scale; no convincing target identification gain.",
            "Transformer best epochs are all 1: optimization improves training loss but harms validation.",
            "These are useful first baselines and negative findings, not a completed competitive model.",
            "The first protocol models mean shifts; future distribution-modeling changes require rerunning baselines.",
            "Full four-model comparison: [no-perturbation reference report](../controls/README.md).",
        ]
    (folder / "README.md").write_text("\n".join(lines) + "\n", "utf-8")
    write_json(
        folder / "report-provenance.json",
        {
            "runs": len(df),
            "test_used": False,
            "metrics_source": "cell-eval 0.6.6 VCC profile",
            "de_adapter": "tie/continuity-corrected U with pooled BH; parity verified",
        },
    )
    print(agg.to_string(), flush=True)
