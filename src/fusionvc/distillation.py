"""Train-only response distillation into the unchanged supervised MLP."""

from __future__ import annotations

import csv
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from .io import file_sha256, read_json, seed_all, write_json
from .models import apply_residual

PROTOCOL = "distillation-v1"
EXPERIMENT = "08_distillation"
SEEDS = [42, 43, 44]
TEACHER_RULE = "minimum three-seed official validation MAE mean over nine fusion model/loss groups; exact ties by experiment; fixed seed42 checkpoint; no ensemble"


def choose_teacher(rows: list[dict]) -> dict:
    """Pure selection rule; never select a seed from its observed score."""
    from .fusion_mmd_report import FUSION_MODELS, VARIANTS, run_spec

    frame = pd.DataFrame(rows)
    expected = {(m, v, s) for m in FUSION_MODELS for v in VARIANTS for s in SEEDS}
    if (
        len(frame) != 27
        or set(frame[["model", "variant", "seed"]].itertuples(index=False, name=None)) != expected
    ):
        raise ValueError("Teacher selection requires all 27 distinct fusion runs")
    if not np.isfinite(frame.mae).all():
        raise ValueError("Teacher scores must be finite")
    for row in frame.itertuples():
        if row.experiment != run_spec(row.model, row.variant)[1]:
            raise ValueError("Teacher run identity differs from its declared model/loss")
    grouped = (
        frame.groupby(["experiment", "model", "variant"], as_index=False)
        .agg(
            validation_mae_mean=("mae", "mean"),
            validation_mae_std=("mae", "std"),
            seeds=("seed", "count"),
        )
        .sort_values(["validation_mae_mean", "experiment"], kind="stable")
    )
    winner = grouped.iloc[0]
    selected = frame[(frame.experiment == winner.experiment) & (frame.seed == 42)].iloc[0]
    return {
        "rule": TEACHER_RULE,
        "candidates": grouped.to_dict(orient="records"),
        "experiment": str(winner.experiment),
        "model": str(winner.model),
        "variant": str(winner.variant),
        "seed": 42,
        "checkpoint_sha256": selected["checkpoint_sha256"],
        "teacher_checkpoint_sha256": selected["checkpoint_sha256"],
        "teacher_experiment": str(winner.experiment),
        "teacher_seed": 42,
        "selected_checkpoint_validation_mae": float(selected.mae),
        "selected_group_validation_mae_mean": float(winner.validation_mae_mean),
        "test_used": False,
    }


def snapshot_prior_artifacts(root: Path) -> dict:
    """Freeze all existing result files and the 69 previously published weights."""
    from .fusion_mmd_report import MODELS, VARIANTS, run_spec

    path = root / "results/distillation/prior-artifact-snapshot.json"
    if path.exists():
        return verify_prior_artifacts(root)
    checkpoints = []
    identities = {(run_spec(m, v)[1], s) for m in MODELS for v in VARIANTS for s in SEEDS}
    identities |= {(name, s) for name in ["01_mlp", "02_transformer"] for s in SEEDS}
    for experiment, seed in sorted(identities):
        checkpoint = root / f"outputs/{experiment}/seed_{seed}/best.pt"
        checkpoints.append(
            {
                "path": str(checkpoint.relative_to(root)),
                "sha256": file_sha256(checkpoint),
                "experiment": experiment,
                "seed": seed,
            }
        )
    files = []
    for group in ["baselines", "mmd", "scgpt", "geneformer", "fusion", "controls", "fusion-mmd"]:
        for source in sorted((root / "results" / group).rglob("*")):
            if source.is_file():
                files.append({"path": str(source.relative_to(root)), "sha256": file_sha256(source)})
    if len(checkpoints) != 69 or not files:
        raise ValueError("Expected all 69 old checkpoints before distillation")
    write_json(path, {"checkpoints": checkpoints, "small_artifacts": files, "test_used": False})
    return verify_prior_artifacts(root)


def verify_prior_artifacts(root: Path) -> dict:
    from .fusion_mmd_report import MODELS, VARIANTS, run_spec

    path = root / "results/distillation/prior-artifact-snapshot.json"
    snapshot = read_json(path)
    expected = {(run_spec(m, v)[1], s) for m in MODELS for v in VARIANTS for s in SEEDS}
    expected |= {(name, s) for name in ["01_mlp", "02_transformer"] for s in SEEDS}
    if (
        snapshot["test_used"] is not False
        or len(snapshot["checkpoints"]) != 69
        or {(r["experiment"], r["seed"]) for r in snapshot["checkpoints"]} != expected
        or len(snapshot["small_artifacts"]) < 477
    ):
        raise ValueError("Incomplete pre-distillation checkpoint snapshot")
    checks = []
    for kind in ["checkpoints", "small_artifacts"]:
        paths = [r["path"] for r in snapshot[kind]]
        if not paths or len(set(paths)) != len(paths):
            raise ValueError("Snapshot contains duplicate or missing paths")
        for row in snapshot[kind]:
            source = (root / row["path"]).resolve()
            if not source.is_relative_to(root.resolve()) or file_sha256(source) != row["sha256"]:
                raise ValueError(f"Pre-distillation artifact changed: {row['path']}")
            checks.append({**row, "hash_matches_snapshot": True})
    return {
        "snapshot_sha256": file_sha256(path),
        "checkpoint_count": 69,
        "small_artifact_count": len(snapshot["small_artifacts"]),
        "checks": checks,
        "all_hashes_match": True,
        "test_used": False,
    }


def _cache_signature(root: Path, selection: dict) -> dict:
    base = root / "data/processed/baselines-v1"
    return {
        "protocol": PROTOCOL,
        "teacher": selection,
        "source_sha256": {
            str(path.relative_to(root)): file_sha256(path)
            for path in [
                Path(__file__),
                base / "manifest.json",
                base / "genes.json",
                base / "gene_descriptors.npy",
                base / "input_gene_indices.npy",
                base / "train_stats.npz",
                base / "validation_stats.npz",
                root / "src/fusionvc/train.py",
                root / "src/fusionvc/models.py",
                root / "configs/experiments/01_mlp.yaml",
                root / "configs/splits/vcc2025.json",
                root / f"outputs/{selection['experiment']}/seed_42/best.pt",
            ]
        },
        "target_policy": "exactly the 150 fixed training perturbation gene IDs; no validation or test soft labels",
        "kd_space": "mean apply_residual expression on the same sampled training control bank",
        "teacher_precision": "float32 eval/no_grad; raw deltas cached only to evaluate expression means on varying banks",
        "test_used": False,
    }


def load_distillation_cache(root: Path):
    folder = root / f"data/processed/{PROTOCOL}"
    manifest = read_json(folder / "manifest.json")
    selection = read_json(root / "results/distillation/teacher-selection.json")
    if manifest["signature"] != _cache_signature(root, selection):
        raise ValueError("Distillation sources changed; use a new protocol/cache version")
    for name, digest in manifest["cache_sha256"].items():
        if file_sha256(folder / name) != digest:
            raise ValueError(f"Distillation cache changed: {name}")
    targets = read_json(folder / "training_targets.json")["targets"]
    split = read_json(root / "configs/splits/vcc2025.json")
    delta = np.load(folder / "teacher_delta.npy", allow_pickle=False)
    if (
        len(targets) != 150
        or len(set(targets)) != 150
        or targets != split["train"]
        or set(targets) & set(split["validation"])
        or delta.shape != (150, 18080)
        or delta.dtype != np.float32
        or not np.isfinite(delta).all()
        or manifest["test_used"] is not False
        or manifest["teacher_checkpoint_sha256"] != selection["checkpoint_sha256"]
        or manifest["teacher_experiment"] != selection["experiment"]
        or manifest["teacher_seed"] != 42
    ):
        raise ValueError("Invalid training-only teacher soft labels")
    return targets, delta, manifest


def prepare_distillation(root: Path) -> dict:
    """Select only after all 63 completed comparisons and their audits pass."""
    from .fusion_mmd_report import MODELS, VARIANTS, FUSION_MODELS, load_run, material_passport
    from .train import load_inputs

    report_path = root / "results/fusion-mmd/report-provenance.json"
    if (
        material_passport(root)["verification_status"] != "VERIFIED"
        or not report_path.exists()
        or read_json(report_path).get("material_passport", {}).get("verification_status")
        != "VERIFIED"
        or read_json(report_path).get("model_runs") != 63
    ):
        raise RuntimeError(
            "Finish and verify the full fusion MMD comparison before selecting a teacher"
        )
    rows = [load_run(root, m, v, s)[0] for m in MODELS for v in VARIANTS for s in SEEDS]
    selection = choose_teacher([r for r in rows if r["model"] in FUSION_MODELS])
    folder = root / f"data/processed/{PROTOCOL}"
    if (folder / "manifest.json").exists():
        _, _, manifest = load_distillation_cache(root)
        if manifest["signature"]["teacher"] != selection:
            raise ValueError("Recorded teacher differs from the immutable selection rule")
        verify_prior_artifacts(root)
        print("DISTILL cache already complete; teacher and sources verified", flush=True)
        return manifest
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Prepare official teacher labels in the existing GPU experiment environment"
        )
    snapshot_prior_artifacts(root)
    teacher_run = root / f"outputs/{selection['experiment']}/seed_42"
    teacher_config = read_json(teacher_run / "config.json")
    checkpoint = torch.load(teacher_run / "best.pt", map_location="cpu", weights_only=True)
    started = time.perf_counter()
    teacher, _, genes, features, _ = load_inputs(root, teacher_config, torch.device("cuda"))
    if checkpoint.get("frozen_features") != teacher.feature_manifest:
        raise ValueError("Teacher checkpoint and frozen features differ")
    teacher.load_state_dict(checkpoint["state_dict"], strict=True)
    teacher.eval().requires_grad_(False)
    names = read_json(root / "configs/splits/vcc2025.json")["train"]
    index = {gene: i for i, gene in enumerate(genes)}
    values = []
    with torch.no_grad():
        for start in range(0, len(names), 4):
            q = torch.from_numpy(
                features[[index[name] for name in names[start : start + 4]]]
            ).cuda()
            values.append(teacher(q).float().cpu().numpy())
    delta = np.concatenate(values)
    if delta.shape != (150, 18080) or not np.isfinite(delta).all():
        raise ValueError("Teacher must produce finite all-gene deltas for 150 training targets")
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / "teacher_delta.npy", delta)
    write_json(folder / "training_targets.json", {"targets": names, "test_used": False})
    write_json(root / "results/distillation/teacher-selection.json", selection)
    manifest = {
        "signature": _cache_signature(root, selection),
        "shape": list(delta.shape),
        "teacher_checkpoint_sha256": selection["checkpoint_sha256"],
        "teacher_experiment": selection["experiment"],
        "teacher_seed": 42,
        "cache_sha256": {
            name: file_sha256(folder / name)
            for name in ["training_targets.json", "teacher_delta.npy"]
        },
        "generation_seconds": time.perf_counter() - started,
        "teacher_frozen_features": teacher.feature_manifest,
        "test_used": False,
    }
    write_json(folder / "manifest.json", manifest)
    write_json(root / "results/distillation/teacher-cache.json", manifest)
    del teacher, checkpoint
    torch.cuda.empty_cache()
    load_distillation_cache(root)
    print(
        f"DISTILL teacher={selection['experiment']}/seed42; train-only cache complete", flush=True
    )
    return manifest


def student_objective(prediction, truth, controls, teacher_delta, target_sum, kd_weight):
    """No arbitrary cell pairing: both terms compare conditional expression means."""
    if not np.isfinite(kd_weight) or kd_weight < 0:
        raise ValueError("KD weight must be finite and nonnegative")
    student_mean = prediction.mean(1)
    mean_loss = (student_mean - truth).square().mean()
    # The exact original mean-loss branch also preserves its autograd graph.
    if kd_weight == 0:
        return mean_loss, mean_loss, mean_loss.new_zeros(())
    with torch.no_grad():
        teacher_mean = apply_residual(controls, teacher_delta.detach(), target_sum).mean(1)
    kd_loss = (student_mean - teacher_mean).square().mean()
    return mean_loss + kd_weight * kd_loss, mean_loss, kd_loss


def train_distillation(root: Path, config_path: Path, seed: int) -> Path:
    """Independent loop with the original MLP initialization and RNG draw order."""
    from .train import label_tensors, load_inputs, validate

    config = yaml.safe_load(config_path.read_text("utf-8"))
    baseline = yaml.safe_load((root / "configs/experiments/01_mlp.yaml").read_text("utf-8"))
    if (
        config.get("protocol") != PROTOCOL
        or config.get("model") != "mlp"
        or config.get("results_group") != "distillation"
        or config.get("variant") != "mean_kd"
        or config.get("loss") != "unpaired_pseudobulk_mse"
        or config.get("kd_weight") not in {0.0, 1.0}
        or seed not in SEEDS
    ):
        raise ValueError("Unexpected distillation protocol")
    smoke = config["experiment"] == "smoke_08_distillation"
    allowed = {"experiment"} | ({"epochs", "patience"} if smoke else set())
    if any(config.get(key) != value for key, value in baseline.items() if key not in allowed):
        raise ValueError("The student must preserve the original MLP training configuration")
    target_names, teacher_delta, distillation_manifest = load_distillation_cache(root)
    out = root / f"outputs/{config['experiment']}/seed_{seed}"
    if (out / "summary.json").exists():
        previous = read_json(out / "summary.json")
        if (
            read_json(out / "config.json") != {**config, "seed": seed}
            or previous["distillation_cache"] != distillation_manifest
            or file_sha256(out / "best.pt") != previous["checkpoint_sha256"]
        ):
            raise ValueError("Completed distillation run has different configuration or sources")
        print(f"DISTILL train already complete: {out}", flush=True)
        return out
    if not torch.cuda.is_available():
        raise RuntimeError("Official distillation training requires the experiment GPU")
    seed_all(seed)
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    model, manifest, genes, features, ctrl_np = load_inputs(root, config, device)
    train_names, train_q, train_y = label_tensors(root, "train", genes, features, device)
    _, val_q, val_y = label_tensors(root, "validation", genes, features, device)
    if set(train_names) != set(target_names):
        raise ValueError("Teacher cache and supervised training targets differ")
    position = {name: i for i, name in enumerate(target_names)}
    teacher_tensor = torch.from_numpy(teacher_delta[[position[name] for name in train_names]]).to(
        device
    )
    rng, fixed_rng = np.random.default_rng(seed), np.random.default_rng(2025)
    val_idx = fixed_rng.choice(len(ctrl_np), config["validation_control_cells"], False)
    val_ctrl = torch.from_numpy(np.array(ctrl_np[val_idx])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config["epochs"])
    use_amp = config["mixed_precision"] == "bf16"
    if use_amp and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The unchanged MLP recipe requires bfloat16 support")
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "config.json", {**config, "seed": seed})
    write_json(
        out / "environment.json",
        {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(),
            "platform": platform.platform(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "attention_backend": "math",
        },
    )
    parameters = sum(p.numel() for p in model.parameters())
    started = time.perf_counter()
    best, best_epoch, stale = float("inf"), 0, 0
    fields = [
        "epoch",
        "train_mse",
        "train_kd_mse",
        "train_loss",
        "val_mae",
        "val_mse",
        "lr",
        "elapsed_seconds",
    ]
    with (out / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, config["epochs"] + 1):
            model.train()
            total, total_kd, total_loss = 0.0, 0.0, 0.0
            order = rng.permutation(len(train_q))
            for start in range(0, len(order), config["batch_size"]):
                selected = torch.tensor(order[start : start + config["batch_size"]], device=device)
                ctrl_idx = rng.choice(len(ctrl_np), config["control_bank_size"], False)
                controls = torch.from_numpy(np.array(ctrl_np[ctrl_idx])).to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    delta = model(train_q[selected])
                prediction = apply_residual(controls, delta.float(), manifest["target_sum"])
                loss, mean_loss, kd_loss = student_objective(
                    prediction,
                    train_y[selected],
                    controls,
                    teacher_tensor[selected],
                    manifest["target_sum"],
                    config["kd_weight"],
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite distillation objective")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
                optimizer.step()
                total += mean_loss.item() * len(selected)
                total_kd += kd_loss.item() * len(selected)
                total_loss += loss.item() * len(selected)
            val_mae, val_mse = validate(
                model, val_q, val_y, val_ctrl, manifest["target_sum"], config["batch_size"]
            )
            if not np.isfinite(val_mae):
                raise FloatingPointError("Non-finite validation loss")
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_mse": total / len(order),
                    "train_kd_mse": total_kd / len(order),
                    "train_loss": total_loss / len(order),
                    "val_mae": val_mae,
                    "val_mse": val_mse,
                    "lr": optimizer.param_groups[0]["lr"],
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            stream.flush()
            if val_mae < best - 1e-6:
                best, best_epoch, stale = val_mae, epoch, 0
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "config": config,
                        "seed": seed,
                        "epoch": epoch,
                        "preprocessing": manifest,
                        "distillation_cache": distillation_manifest,
                    },
                    out / "best.pt",
                )
            else:
                stale += 1
            scheduler.step()
            print(
                f"DISTILL seed={seed} epoch={epoch} mse={total / len(order):.6f} kd={total_kd / len(order):.6f} val_mae={val_mae:.6f} best={best_epoch}",
                flush=True,
            )
            if stale >= config["patience"]:
                break
    torch.cuda.synchronize()
    summary = {
        "experiment": config["experiment"],
        "seed": seed,
        "protocol": PROTOCOL,
        "model": "mlp",
        "variant": "mean_kd",
        "loss": config["loss"],
        "kd_weight": config["kd_weight"],
        "epochs_run": epoch,
        "best_epoch": best_epoch,
        "selection_validation_mae": best,
        "parameters": parameters,
        "trainable_parameters": parameters,
        "frozen_parameters": 0,
        "training_seconds": time.perf_counter() - started,
        "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_gpu_bytes": torch.cuda.max_memory_reserved(),
        "checkpoint_bytes": (out / "best.pt").stat().st_size,
        "checkpoint_sha256": file_sha256(out / "best.pt"),
        "distillation_cache": distillation_manifest,
        "cell_set_size": config["control_bank_size"],
        "scope": "train-only teacher labels; same condition mean-shift MLP; validation MAE selection; no online teacher or foundation encoder at student inference",
        "test_used": False,
    }
    write_json(out / "summary.json", summary)
    print(f"DISTILL complete: {out}", flush=True)
    return out
