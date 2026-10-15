"""Measure cached loading, warm heads and the same all-gene expression workload."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch


def loader_dependency_inventory(root: Path, run: Path, feature_manifest: dict | None):
    """Separate numerical payload from research-source integrity requirements.

    These are project data/signature paths used by the present cached loader.
    Python/package/runtime dependencies are described by the environment instead
    of counting the installed environment as a model artifact.
    """
    inventory = {}

    def add(path, access, purpose):
        relative = str(path.relative_to(root))
        record = inventory.setdefault(
            relative,
            {"path": relative, "bytes": path.stat().st_size, "access": set(), "purpose": set()},
        )
        record["access"].add(access)
        record["purpose"].add(purpose)

    add(run / "best.pt", "payload_read", "selected model weights and fixed buffers")
    add(run / "config.json", "payload_read", "selected architecture configuration")
    base = root / "data/processed/baselines-v1"
    for name in ["manifest.json", "genes.json", "controls.npy"]:
        add(
            base / name, "payload_read", "common normalization/gene IDs and prediction control bank"
        )
    if feature_manifest is None:
        for name in ["gene_descriptors.npy", "input_gene_indices.npy", "train_stats.npz"]:
            add(
                base / name,
                "payload_read",
                "current MLP loader builds target descriptors and fixed control context",
            )
    else:
        add(
            root / "docs/download-status.json",
            "provenance_read",
            "recorded original asset checksums",
        )
        for name, manifest in feature_manifest["source_manifests"].items():
            folder = root / f"data/processed/{name}-frozen-v1"
            asset_folder = root / (
                "models/scgpt/whole-human" if name == "scgpt" else "models/geneformer-v1"
            )
            add(
                folder / "manifest.json",
                "provenance_read",
                "immutable frozen feature/source receipt",
            )
            for filename in manifest["feature_hashes"]:
                add(folder / filename, "sha256_read", "cached-feature integrity validation")
            for filename in ["target_features.npy", "context.npy", "control_indices.npy"]:
                add(
                    folder / filename,
                    "payload_read",
                    "current fusion loader target/context/control identity arrays",
                )
            for filename in manifest["signature"]["assets"]:
                add(
                    asset_folder / filename,
                    "sha256_read",
                    "research source validation; no foundation-model forward pass",
                )
            add(root / f"src/fusionvc/{name}.py", "sha256_read", "adapter source signature")
        for filename in [
            "manifest.json",
            "genes.json",
            "gene_descriptors.npy",
            "input_gene_indices.npy",
        ]:
            add(base / filename, "sha256_read", "frozen feature baseline-input signature")
        add(
            root / "vendor/scGPT-paper2024/scgpt/model/model.py",
            "sha256_read",
            "pinned upstream source signature",
        )
        add(
            root / "vendor/Geneformer/geneformer/tokenizer.py",
            "sha256_read",
            "pinned upstream tokenizer signature",
        )
        add(
            root / "data/raw/vcc2025/train/adata_Training.h5ad",
            "stat_only",
            "Geneformer source size/mtime identity only; counts are not opened",
        )
        if feature_manifest.get("token_manifest"):
            add(
                root / "src/fusionvc/fusion_features.py",
                "sha256_read",
                "token extraction adapter signature",
            )
            folder = root / "data/processed/fusion-tokens-v1"
            add(folder / "manifest.json", "provenance_read", "token feature source receipt")
            for filename in feature_manifest["token_manifest"]["feature_hashes"]:
                add(folder / filename, "sha256_read", "gene-token cache integrity validation")
                if filename.endswith(".npy"):
                    add(
                        folder / filename,
                        "payload_read",
                        "current token loader reads all numerical token/cache arrays",
                    )
    rows = []
    for path in sorted(inventory):
        row = inventory[path]
        rows.append({**row, "access": sorted(row["access"]), "purpose": sorted(row["purpose"])})
    return {
        "files": rows,
        "numerical_payload_and_configuration_bytes": sum(
            r["bytes"] for r in rows if "payload_read" in r["access"]
        ),
        "provenance_only_read_bytes": sum(
            r["bytes"]
            for r in rows
            if "payload_read" not in r["access"] and r["access"] != ["stat_only"]
        ),
        "stat_only_source_bytes": sum(r["bytes"] for r in rows if r["access"] == ["stat_only"]),
        "scope": "current cached research-loader file inventory, not a minimal inference bundle; stat-only source size is never counted as numerical model payload; installed Python/packages are excluded",
    }


def benchmark_distillation(root: Path):
    sys.path.insert(0, str(root / "src"))
    from fusionvc.distillation import load_distillation_cache
    from fusionvc.io import file_sha256, read_json, write_json
    from fusionvc.models import apply_residual
    from fusionvc.train import load_inputs

    if not torch.cuda.is_available():
        raise RuntimeError("Benchmark with the same GPU as the formal runs")
    names, _, manifest = load_distillation_cache(root)
    selection = manifest["signature"]["teacher"]
    rows = []
    for label, experiment in [
        ("supervised_mlp", "01_mlp"),
        ("distilled_mlp", "08_distillation"),
        ("teacher", selection["experiment"]),
    ]:
        run = root / f"outputs/{experiment}/seed_42"
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        config = read_json(run / "config.json")
        checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
        model, preprocessing, genes, features, controls = load_inputs(
            root, config, torch.device("cuda")
        )
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval()
        index = {gene: i for i, gene in enumerate(genes)}
        q = torch.from_numpy(features[[index[name] for name in names]]).cuda()
        chosen = np.random.default_rng(2025).choice(len(controls), 32, replace=False)
        bank = torch.from_numpy(np.array(controls[chosen])).cuda()
        torch.cuda.synchronize()
        loading_seconds = time.perf_counter() - started

        @torch.inference_mode()
        def workload(expression: bool, predictor, queries, control_bank, target_sum):
            checksum = torch.zeros((), device="cuda")
            for start in range(0, 150, 4):
                delta = predictor(queries[start : start + 4])
                output = apply_residual(control_bank, delta, target_sum) if expression else delta
                checksum += output.sum()
            return checksum

        measurements = {}
        for name, expression in [
            ("warm_cached_head", False),
            ("warm_cached_head_and_expression", True),
        ]:
            workload(expression, model, q, bank, preprocessing["target_sum"])
            torch.cuda.synchronize()
            samples = []
            for _ in range(5):
                torch.cuda.synchronize()
                started = time.perf_counter()
                checksum = workload(expression, model, q, bank, preprocessing["target_sum"])
                torch.cuda.synchronize()
                samples.append(time.perf_counter() - started)
            measurements[name] = {
                "seconds": samples,
                "median_seconds": float(np.median(samples)),
                "p95_seconds": float(np.quantile(samples, 0.95)),
                "median_seconds_per_perturbation": float(np.median(samples) / 150),
                "checksum": float(checksum.item()),
            }
        training = read_json(run / "summary.json")
        if label == "teacher":
            source = manifest["teacher_frozen_features"]
            source_feature_bytes = sum(
                (root / f"data/processed/{name}-frozen-v1/target_features.npy").stat().st_size
                for name in ["scgpt", "geneformer"]
            )
            source_weights_bytes = sum(
                (root / folder / name).stat().st_size
                for folder, name in [
                    ("models/scgpt/whole-human", "best_model.pt"),
                    ("models/geneformer-v1", "model.safetensors"),
                ]
            )
            existing_extraction_seconds = source["elapsed_seconds"]
        else:
            source_feature_bytes = (
                (root / "data/processed/baselines-v1/gene_descriptors.npy").stat().st_size
            )
            source_weights_bytes, existing_extraction_seconds = 0, 0.0
        dependency_inventory = loader_dependency_inventory(
            root, run, source if label == "teacher" else None
        )
        rows.append(
            {
                "label": label,
                "experiment": experiment,
                "seed": 42,
                "checkpoint_sha256": file_sha256(run / "best.pt"),
                "parameters": training["parameters"],
                "trainable_parameters": training["trainable_parameters"],
                "frozen_parameters": training.get("frozen_parameters", 0),
                "checkpoint_bytes": (run / "best.pt").stat().st_size,
                "target_feature_file_bytes": source_feature_bytes,
                "offline_source_encoder_weight_bytes": source_weights_bytes,
                "shared_training_control_file_bytes": (
                    root / "data/processed/baselines-v1/controls.npy"
                )
                .stat()
                .st_size,
                "current_research_loader_dependencies": dependency_inventory,
                "cached_model_and_inputs_load_seconds": loading_seconds,
                "measured_peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(),
                "existing_recorded_shared_preprocessing_seconds": preprocessing["elapsed_seconds"],
                "existing_recorded_encoder_feature_extraction_seconds": existing_extraction_seconds,
                "measurements": measurements,
            }
        )
        del model, checkpoint, q, bank, checksum, controls, features
        torch.cuda.empty_cache()
    result = {
        "rows": rows,
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "workload": {
            "perturbations": 150,
            "targets_source": "fixed training split IDs; no response expression labels",
            "controls": 32,
            "control_sampling_seed": 2025,
            "output_genes": 18080,
            "target_batch_size": 4,
            "warmup_rounds": 1,
            "measured_rounds": 5,
            "precision": "float32 eval",
        },
        "scope": {
            "loading": "wall time for checkpoint/config, current cached-input loader including source hash checks, target lookup and transfer, controls transfer and CUDA synchronization; initialized process; OS page caches not cleared; not a fresh machine cold start",
            "head": "150 cached-feature condition deltas; no encoding, apply_residual, disk output or official scoring",
            "expression": "same 150 targets by 32 training controls by 18080 genes; head and shared apply_residual; no output-file I/O, validation/test responses or DE scoring",
            "peak_memory": "includes model/input loading and both inference workloads; excludes historical encoder extraction",
            "historical_costs": "existing measured source-manifest preprocessing/encoder extraction, shared across seeds; not newly measured cold-start timings",
            "deployment": "both teacher and student use cached inputs in this protocol; frozen source encoder counts/weights do not measure warm speedups; current research-loader numerical payload, provenance checksum reads and stat-only raw training source are inventoried separately; none is a measured minimal inference bundle",
            "harness_setup": "the benchmark validates the distillation cache and obtains its 150 training IDs before timers; that experimental setup is separate from Student inference dependencies; no response labels are used in measured workloads",
        },
        "test_used": False,
    }
    write_json(root / "results/distillation/cost-benchmark.json", result)
    print(
        "DISTILL measured loading and equal 150-target/32-control cached inference workloads",
        flush=True,
    )
    return result


if __name__ == "__main__":
    benchmark_distillation(Path(__file__).resolve().parents[1])
