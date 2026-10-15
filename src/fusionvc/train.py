"""Train perturbation-level baselines; validation is used only for selection."""

from __future__ import annotations

import csv
import platform
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from .io import file_sha256, read_json, seed_all, write_json
from .models import FrozenFeaturePredictor, ResidualPredictor, apply_residual
from .preprocess import CONTROL
from .fusion_features import FUSION_MODELS

FROZEN_MODELS = {"scgpt", "geneformer"} | FUSION_MODELS


def load_frozen_features(root: Path, config: dict):
    name = config["model"]
    if name in FUSION_MODELS:
        from .fusion_features import load_fusion_features

        return load_fusion_features(root, config)
    if config.get("protocol") != f"{name}-v1" or config.get("encoder_frozen") is not True:
        raise ValueError(f"{name} requires its explicit v1 protocol and a frozen encoder")
    if name == "scgpt":
        from .scgpt import load_scgpt_features

        return load_scgpt_features(root)
    if name == "geneformer":
        from .geneformer import load_geneformer_features

        return load_geneformer_features(root)
    raise ValueError(f"Unknown frozen encoder: {name}")


def load_inputs(root: Path, config: dict, device: torch.device):
    cache = root / "data/processed/baselines-v1"
    manifest = read_json(cache / "manifest.json")
    genes = read_json(cache / "genes.json")
    if config["model"] in FUSION_MODELS:
        from .fusion import FusionPredictor

        features, assets, feature_manifest = load_frozen_features(root, config)
        model = FusionPredictor(config, assets, feature_manifest["feature_dims"], len(genes)).to(
            device
        )
        model.feature_manifest = feature_manifest
    elif config["model"] in {"scgpt", "geneformer"}:
        features, context, feature_manifest = load_frozen_features(root, config)
        model = FrozenFeaturePredictor(
            config, torch.from_numpy(context), features.shape[-1], len(genes)
        ).to(device)
        model.feature_manifest = feature_manifest
    else:
        features = np.load(cache / "gene_descriptors.npy")
        indices = np.load(cache / "input_gene_indices.npy")
        stats = np.load(cache / "train_stats.npz")
        ctrl = list(stats["names"]).index(CONTROL)
        context = stats["mean"][ctrl, indices]
        model = ResidualPredictor(
            config, torch.from_numpy(features[indices]), torch.from_numpy(context), len(genes)
        ).to(device)
    controls = np.load(cache / "controls.npy", mmap_mode="r")
    return model, manifest, genes, features, controls


def label_tensors(root: Path, split: str, genes: list, features: np.ndarray, device: torch.device):
    stats = np.load(root / f"data/processed/baselines-v1/{split}_stats.npz")
    idx = [i for i, p in enumerate(stats["names"]) if p != CONTROL]
    names = [str(stats["names"][i]) for i in idx]
    gene_index = {g: i for i, g in enumerate(genes)}
    q = torch.from_numpy(features[[gene_index[p] for p in names]]).to(device)
    y = torch.from_numpy(stats["mean"][idx]).to(device)
    return names, q, y


@torch.no_grad()
def validate(model, q, truth, controls, scale: float, batch_size: int) -> tuple[float, float]:
    model.eval()
    errors = []
    squares = []
    # Small target batches bound the full-gene control-bank intermediate.
    for start in range(0, len(q), min(batch_size, 4)):
        delta = model(q[start : start + min(batch_size, 4)])
        pred = apply_residual(controls, delta, scale).mean(1)
        diff = pred - truth[start : start + len(pred)]
        errors.append(diff.abs().mean().item() * len(pred))
        squares.append(diff.square().mean().item() * len(pred))
    return sum(errors) / len(q), sum(squares) / len(q)


def train(root: Path, config_path: Path, seed: int) -> Path:
    config = yaml.safe_load(config_path.read_text("utf-8"))
    valid_losses = {"unpaired_pseudobulk_mse", "unpaired_energy_mmd", "unpaired_mean_energy_mmd"}
    if config["loss"] not in valid_losses:
        raise ValueError(f"Unknown training loss: {config['loss']}")
    use_mmd = config["loss"] != "unpaired_pseudobulk_mse"
    if use_mmd and config.get("protocol") not in {
        "mmd-v1",
        "scgpt-v1",
        "geneformer-v1",
        "fusion-v1",
    }:
        raise ValueError("Distribution losses require an explicit supported protocol")
    energy_weight = float(config.get("energy_weight", 0.1))
    if use_mmd and (not np.isfinite(energy_weight) or energy_weight <= 0):
        raise ValueError("Energy weight must be finite and positive")
    out = root / f"outputs/{config['experiment']}/seed_{seed}"
    if (out / "summary.json").exists():
        existing = read_json(out / "config.json")
        if existing != {**config, "seed": seed}:
            raise ValueError("Completed run has a different config; use a new experiment name")
        if config["model"] in FROZEN_MODELS:
            _, _, features_manifest = load_frozen_features(root, config)
            summary = read_json(out / "summary.json")
            if summary["frozen_features"] != features_manifest:
                raise ValueError("Completed run used different frozen features")
        if file_sha256(out / "best.pt") != read_json(out / "summary.json")["checkpoint_sha256"]:
            raise ValueError("Completed checkpoint hash mismatch")
        print(f"TRAIN already complete: {out}", flush=True)
        return out
    out.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU PyTorch is required for this experiment configuration")
    seed_all(seed)
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    model, manifest, genes, features, ctrl_np = load_inputs(root, config, device)
    train_names, train_q, train_y = label_tensors(root, "train", genes, features, device)
    _, val_q, val_y = label_tensors(root, "validation", genes, features, device)
    rng = np.random.default_rng(seed)
    fixed_rng = np.random.default_rng(2025)
    val_idx = fixed_rng.choice(len(ctrl_np), config["validation_control_cells"], False)
    val_ctrl = torch.from_numpy(np.array(ctrl_np[val_idx])).to(device)
    sampler = None
    target_rng = np.random.default_rng(seed + 100000)
    if use_mmd:
        from .distribution import TrainingCellSampler

        sampler = TrainingCellSampler(root, genes, train_names, manifest["target_sum"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config["epochs"])
    use_amp = config["mixed_precision"] == "bf16"
    if use_amp and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This configuration requires bfloat16-capable GPU")
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
    parameter_count = sum(p.numel() for p in model.parameters())
    frozen_features = getattr(model, "feature_manifest", None)
    record_distribution = config.get("protocol") in {
        "mmd-v1",
        "scgpt-v1",
        "geneformer-v1",
        "fusion-v1",
    }
    started = time.perf_counter()
    best, best_epoch, stale = float("inf"), 0, 0
    path = out / "history.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        fields = ["epoch", "train_mse", "val_mae", "val_mse", "lr", "elapsed_seconds"]
        if record_distribution:
            fields += ["train_loss", "train_energy_mmd_rms"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, config["epochs"] + 1):
            model.train()
            total = 0.0
            total_loss, total_energy = 0.0, 0.0
            order = rng.permutation(len(train_q))
            for start in range(0, len(order), config["batch_size"]):
                selected = torch.tensor(order[start : start + config["batch_size"]], device=device)
                ctrl_idx = rng.choice(len(ctrl_np), config["control_bank_size"], False)
                controls = torch.from_numpy(np.array(ctrl_np[ctrl_idx])).to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    delta = model(train_q[selected])
                # Full-gene nonlinearity and set mean are evaluated in float32.
                prediction = apply_residual(controls, delta.float(), manifest["target_sum"])
                mean_loss = (prediction.mean(1) - train_y[selected]).square().mean()
                energy_loss = None
                if use_mmd:
                    from .distribution import energy_mmd

                    target_names = [train_names[i] for i in order[start : start + len(selected)]]
                    observed = sampler.sample(target_names, config["target_bank_size"], target_rng)
                    truth = torch.from_numpy(observed).to(device)
                    energy_loss = energy_mmd(prediction, truth)
                    if config["loss"] == "unpaired_energy_mmd":
                        loss = energy_loss
                    else:
                        loss = mean_loss + energy_weight * energy_loss
                else:
                    loss = mean_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
                optimizer.step()
                total += mean_loss.item() * len(selected)
                total_loss += loss.item() * len(selected)
                if energy_loss is not None:
                    total_energy += energy_loss.item() * len(selected)
            val_mae, val_mse = validate(
                model, val_q, val_y, val_ctrl, manifest["target_sum"], config["batch_size"]
            )
            if not np.isfinite(val_mae):
                raise FloatingPointError("Non-finite validation loss")
            row = {
                "epoch": epoch,
                "train_mse": total / len(order),
                "val_mae": val_mae,
                "val_mse": val_mse,
                "lr": optimizer.param_groups[0]["lr"],
                "elapsed_seconds": time.perf_counter() - started,
            }
            if record_distribution:
                row.update(
                    train_loss=total_loss / len(order),
                    train_energy_mmd_rms=total_energy / len(order) if use_mmd else None,
                )
            writer.writerow(row)
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
                        **({"frozen_features": frozen_features} if frozen_features else {}),
                    },
                    out / "best.pt",
                )
            else:
                stale += 1
            scheduler.step()
            print(
                f"TRAIN {config['model']} seed={seed} epoch={epoch} "
                f"train_mse={total / len(order):.6f} val_mae={val_mae:.6f} "
                f"best={best_epoch} time={time.perf_counter() - started:.0f}s",
                flush=True,
            )
            if stale >= config["patience"]:
                break
    if sampler is not None:
        sampler.close()
    torch.cuda.synchronize()
    summary = {
        "experiment": config["experiment"],
        "seed": seed,
        "epochs_run": epoch,
        "best_epoch": best_epoch,
        "selection_validation_mae": best,
        "parameters": parameter_count,
        "trainable_parameters": parameter_count,
        "training_seconds": time.perf_counter() - started,
        "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_gpu_bytes": torch.cuda.max_memory_reserved(),
        "checkpoint_bytes": (out / "best.pt").stat().st_size,
        "checkpoint_sha256": file_sha256(out / "best.pt"),
        "scope": "validation only; test untouched; pseudobulk-trained mean-shift baseline",
    }
    if record_distribution:
        summary.update(
            protocol=config["protocol"],
            loss=config["loss"],
            model=config["model"],
            variant=config["variant"],
            scope="validation only; test untouched; condition mean-shift model",
            cell_set_size=config["control_bank_size"],
            target_set_size=config.get("target_bank_size") if use_mmd else None,
            energy_kernel="negative Euclidean distance divided by sqrt(18080)",
            target_sampling=(
                "all training cells; separate RNG seed+100000"
                if use_mmd
                else "not sampled; all-cell pseudobulk means"
            ),
        )
    if frozen_features:
        summary.update(
            frozen_features=frozen_features,
            frozen_parameters=frozen_features["frozen_parameters"],
            parameters=parameter_count + frozen_features["frozen_parameters"],
            feature_extraction_seconds=frozen_features["elapsed_seconds"],
            feature_extraction_shared_across_runs=True,
            architecture=f"frozen {config['model']} train-control context + gene embedding; shared condition mean shift",
        )
    write_json(out / "summary.json", summary)
    print(f"TRAIN COMPLETE {out}", flush=True)
    return out
