"""Synthetic CPU checks; these tests never access actual held-out test data."""

import base64
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fusionvc import final_evaluation as final
from fusionvc.io import file_sha256, read_json, write_json
from fusionvc.preprocess import CONTROL
from fusionvc.train import load_inputs


def frozen_selection(root, mlp_variant="mean"):
    runs = []
    for model in sorted(final.MAIN_MODELS):
        variant = (
            "mean_kd" if model == "distillation" else mlp_variant if model == "mlp" else "mean"
        )
        for seed in [42, 43, 44]:
            path = root / f"outputs/{model}_{variant}/seed_{seed}/best.pt"
            path.parent.mkdir(parents=True)
            path.write_bytes(f"synthetic {model} {variant} {seed}".encode())
            runs.append(
                {
                    "label": f"{model}_{variant}_{seed}",
                    "model": model,
                    "variant": variant,
                    "seed": seed,
                    "kind": "checkpoint",
                    "role": "main",
                    "checkpoint_path": path.relative_to(root).as_posix(),
                    "checkpoint_sha256": file_sha256(path),
                }
            )
    runs.append(
        {
            "label": "no_perturbation",
            "model": "no_perturbation",
            "role": "reference",
            "kind": "no_perturbation",
        }
    )
    selection = {
        "protocol": final.PROTOCOL,
        "producer": "fusionvc.final_selection-v1",
        "selection_basis": final.SELECTION_BASIS,
        "frozen": True,
        "test_used": False,
        "runs": runs,
    }
    path = root / "docs/final-model-selection.json"
    write_json(path, selection)
    return path, selection


def test_selection_gate_precedes_any_test_reader_and_binds_checkpoint(tmp_path, monkeypatch):
    path, selection = frozen_selection(tmp_path)
    # Isolate structural checks; the complete evidence gate is tested below.
    monkeypatch.setattr(final, "_validate_pretest_evidence", lambda *args: None)
    final.validate_selection(tmp_path, path)
    selection["frozen"] = False
    write_json(path, selection)
    monkeypatch.setattr(
        final, "CountsReader", lambda _: pytest.fail("Actual test data must remain unopened")
    )
    with pytest.raises(ValueError, match="frozen"):
        final.prepare_test_cache(tmp_path, path)
    selection["frozen"] = True
    write_json(path, selection)
    (tmp_path / selection["runs"][0]["checkpoint_path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint changed"):
        final.validate_selection(tmp_path, path)


def test_selection_requires_kd_mean_controls_and_safe_labels(tmp_path, monkeypatch):
    path, selection = frozen_selection(tmp_path, mlp_variant="energy")
    monkeypatch.setattr(final, "_validate_pretest_evidence", lambda *args: None)
    with pytest.raises(ValueError, match="MLP mean KD controls"):
        final.validate_selection(tmp_path, path)
    for seed in [42, 43, 44]:
        checkpoint = tmp_path / f"outputs/mlp_mean/seed_{seed}/best.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"mean-control")
        selection["runs"].append(
            {
                "label": f"mlp_mean_{seed}",
                "kind": "checkpoint",
                "model": "mlp",
                "variant": "mean",
                "seed": seed,
                "role": "kd_control",
                "checkpoint_path": checkpoint.relative_to(tmp_path).as_posix(),
                "checkpoint_sha256": file_sha256(checkpoint),
            }
        )
    write_json(path, selection)
    final.validate_selection(tmp_path, path)
    selection["runs"][0]["label"] = "../../validation"
    write_json(path, selection)
    with pytest.raises(ValueError, match="safe ASCII"):
        final.validate_selection(tmp_path, path)


def test_flags_and_arbitrary_checkpoints_do_not_unlock_actual_test_reader(tmp_path, monkeypatch):
    path, selection = frozen_selection(tmp_path)
    monkeypatch.setattr(
        final, "CountsReader", lambda *_: pytest.fail("No test reader before full proof")
    )
    with pytest.raises(ValueError, match="all 63"):
        final.prepare_test_cache(tmp_path, path)
    selection.pop("producer")
    write_json(path, selection)
    with pytest.raises(ValueError, match="frozen"):
        final.validate_selection(tmp_path, path)


def test_validation_record_requires_canonical_official_validation_source(tmp_path):
    row = {"model": "mlp", "variant": "mean", "seed": 42, "experiment": "mmd_01_mlp_mean"}
    with pytest.raises(ValueError, match="canonical official"):
        final._validate_validation_record(tmp_path, row, candidate=True)


def test_final_cache_lineage_checks_current_sources_without_reading_expression(
    tmp_path, monkeypatch
):
    folder = tmp_path / "outputs/final-test-cache"
    folder.mkdir(parents=True)
    download = {"files": [{"synthetic": "pinned raw identity"}]}
    write_json(tmp_path / final.DOWNLOAD_RECEIPT, download)
    monkeypatch.setattr(final, "verified_test_download", lambda *_: download)
    monkeypatch.setattr(
        final, "CountsReader", lambda *_: pytest.fail("Lineage audit must not open raw expression")
    )
    baseline = {}
    for name in ["manifest.json", "genes.json", "gene_descriptors.npy", "input_gene_indices.npy"]:
        path = tmp_path / "data/processed/baselines-v1" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic-small-source")
        baseline[name] = file_sha256(path)
    scoring = {}
    for name in [
        "final_evaluation.py",
        "evaluate.py",
        "fast_de.py",
        "models.py",
        "train.py",
        "io.py",
        "distribution.py",
    ]:
        path = tmp_path / "src/fusionvc" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"# synthetic scoring source\n")
        scoring[name] = file_sha256(path)
    controls = tmp_path / "data/processed/baselines-v1/controls.npy"
    controls.write_bytes(b"synthetic-control-matrix-provenance")
    matrix = folder / "test_log.npy"
    matrix.write_bytes(b"synthetic-test-matrix-provenance")
    source = {
        "protocol": final.PROTOCOL,
        "test_used": True,
        "selection_sha256": "selection",
        "download_receipt_sha256": file_sha256(tmp_path / final.DOWNLOAD_RECEIPT),
        "raw_files": download["files"],
        "baseline_sources": baseline,
        "scoring_sources": scoring,
        "training_controls": {
            "bytes": controls.stat().st_size,
            "mtime_ns": controls.stat().st_mtime_ns,
        },
    }
    write_json(folder / "source.json", source)
    write_json(
        folder / "real-ready.json",
        {
            "test_used": True,
            "source_sha256": file_sha256(folder / "source.json"),
            "matrix_bytes": matrix.stat().st_size,
            "matrix_mtime_ns": matrix.stat().st_mtime_ns,
            "shape": [1100, 18080],
            "perturbations": [f"g{i}" for i in range(100)],
            "control_cell_count": 100,
            "perturbed_cell_count": 1000,
        },
    )
    assert final.validate_test_cache_sources(tmp_path, "selection")["all_checks_passed"] is True
    (tmp_path / "src/fusionvc/models.py").write_bytes(b"# changed source\n")
    with pytest.raises(ValueError, match="source changed"):
        final.validate_test_cache_sources(tmp_path, "selection")


def test_metadata_counts_actual_controls_in_addition_to_csv_perturbations(tmp_path):
    genes = [f"G{i:05d}" for i in range(18080)]
    write_json(tmp_path / "data/processed/baselines-v1/genes.json", genes)
    write_json(
        tmp_path / "configs/splits/vcc2025.json", {"train": ["TRAIN"], "validation": ["VALIDATION"]}
    )
    counts = tmp_path / "data/raw/vcc2025/test/pert_counts_Test.csv"
    counts.parent.mkdir(parents=True)
    counts.write_text(
        "target_gene,n_cells,median_umi_per_cell\n"
        + "".join(f"{gene},9,30\n" for gene in genes[:100]),
        "utf-8",
    )
    codes = np.repeat(np.arange(101), [*([9] * 100), 17])
    reader = SimpleNamespace(
        categories=[*genes[:100], CONTROL], codes=codes, genes=genes, shape=(917, 18080)
    )
    info = final.inspect_test_metadata(tmp_path, reader)
    assert info["shape"] == [917, 18080]
    assert info["perturbed_cell_count"] == 900 and info["control_cell_count"] == 17
    assert len(info["perturbations"]) == 100 and info["test_used"] is True
    reader.codes[0] = 100
    with pytest.raises(ValueError, match="labels/counts"):
        final.inspect_test_metadata(tmp_path, reader)


def test_sampling_matches_existing_sorted_seed_protocol_and_all_groups():
    labels = np.array(["B"] * 7 + [CONTROL] * 5 + ["A"] * 3)
    names = sorted(set(labels))
    plan = final.control_sampling_plan(labels, names, 6)
    rng = np.random.default_rng(2025)
    for name in names:
        count = int((labels == name).sum())
        np.testing.assert_array_equal(plan[name], rng.choice(6, count, replace=count > 6))
    with pytest.raises(ValueError, match="sorted"):
        final.control_sampling_plan(labels, list(reversed(names)), 6)


def test_publication_resumes_and_deletes_only_new_run_prediction(tmp_path):
    label = "synthetic_selected_run"
    out = tmp_path / "outputs/final-test" / label
    out.mkdir(parents=True)
    for name in ["results.csv", "distribution_per_perturbation.csv", "control_sampling.npz"]:
        (out / name).write_bytes(b"saved-score")
    for name in ["metrics.json", "prediction-info.json", "run-source.json"]:
        write_json(out / name, {"split": "official test", "test_used": True})
    prediction = out / "predictions.npy"
    prediction.write_bytes(b"temporary")
    preserved = out / "pred_de.parquet"
    preserved.write_bytes(b"keep")
    validation = tmp_path / "outputs/evaluation-cache/validation_log.npy"
    validation.parent.mkdir(parents=True)
    validation.write_bytes(b"keep-validation")
    (out / "run-source.json").unlink()
    with pytest.raises(FileNotFoundError):
        final.publish_final_run(tmp_path, label)
    assert prediction.exists()
    write_json(out / "run-source.json", {"split": "official test", "test_used": True})
    final.publish_final_run(tmp_path, label)
    final.publish_final_run(tmp_path, label)
    assert not prediction.exists() and preserved.read_bytes() == b"keep"
    assert validation.read_bytes() == b"keep-validation"
    assert read_json(tmp_path / "results/final-test" / label / "metrics.json")["test_used"] is True


def test_disk_preflight_retains_eighteen_gb_reserve(tmp_path, monkeypatch):
    monkeypatch.setattr(final.shutil, "disk_usage", lambda _: SimpleNamespace(free=30 * 10**9))
    final.ensure_space(tmp_path, 12 * 10**9)
    with pytest.raises(RuntimeError, match="18.00 GB reserve"):
        final.ensure_space(tmp_path, 12 * 10**9 + 1)


def test_distilled_student_loads_as_original_mlp_and_validates_frozen_inputs(tmp_path):
    base = tmp_path / "data/processed/baselines-v1"
    base.mkdir(parents=True)
    genes = [f"G{i}" for i in range(11)]
    write_json(base / "genes.json", genes)
    write_json(base / "manifest.json", {"target_sum": 30.0})
    np.save(base / "gene_descriptors.npy", np.ones((11, 3), np.float32))
    np.save(base / "input_gene_indices.npy", np.arange(4))
    np.save(base / "controls.npy", np.ones((3, 11), np.float32))
    np.savez(base / "train_stats.npz", names=np.array([CONTROL]), mean=np.ones((1, 11), np.float32))
    config = {
        "model": "mlp",
        "protocol": "distillation-v1",
        "variant": "mean_kd",
        "width": 4,
        "dropout": 0.0,
    }
    model, fitted, _, _, controls = load_inputs(tmp_path, config, torch.device("cpu"))
    final.close_memmap(controls)
    path = tmp_path / "outputs/08_distillation/seed_42/best.pt"
    path.parent.mkdir(parents=True)
    teacher_cache = {"signature": {"teacher": "synthetic training-only provenance"}}
    write_json(tmp_path / "data/processed/distillation-v1/manifest.json", teacher_cache)
    checkpoint = {
        "config": config,
        "seed": 42,
        "epoch": 1,
        "preprocessing": fitted,
        "state_dict": model.state_dict(),
        "distillation_cache": teacher_cache,
    }
    torch.save(checkpoint, path)
    row = {
        "kind": "checkpoint",
        "model": "distillation",
        "variant": "mean_kd",
        "seed": 42,
        "checkpoint_path": path.relative_to(tmp_path).as_posix(),
        "checkpoint_sha256": file_sha256(path),
    }
    write_json(path.parent / "config.json", {**config, "seed": 42})
    write_json(
        path.parent / "summary.json",
        {
            "checkpoint_sha256": row["checkpoint_sha256"],
            "best_epoch": 1,
            "distillation_cache": teacher_cache,
        },
    )
    loaded, features, _ = final.load_selected_predictor(tmp_path, row, genes, torch.device("cpu"))
    torch.testing.assert_close(
        loaded(torch.from_numpy(features[:2])), model(torch.from_numpy(features[:2]))
    )
    np.save(base / "gene_descriptors.npy", np.zeros((11, 3), np.float32))
    with pytest.raises(ValueError, match="buffer differs"):
        final.load_selected_predictor(tmp_path, row, genes, torch.device("cpu"))


def test_download_specs_are_pinned_public_anonymous_endpoints():
    specs = final.test_asset_specs()
    assert [s["size_bytes"] for s in specs] == [11950739168, 1894]
    assert [s["crc32c"] for s in specs] == ["JMP5RQ==", "dCaUQw=="]
    assert all(
        s["url"].startswith("https://storage.googleapis.com/") and "?generation=" in s["url"]
        for s in specs
    )
    assert all("userProject" not in s["url"] for s in specs)


def test_zero_shift_prediction_uses_all_cells_and_maps_train_to_test_scale(tmp_path):
    base = tmp_path / "data/processed/baselines-v1"
    base.mkdir(parents=True)
    write_json(base / "manifest.json", {"target_sum": 30.0})
    controls = np.log1p(np.array([[2, 3, 5], [1, 2, 7], [4, 3, 3]], np.float32))
    np.save(base / "controls.npy", controls)
    labels = np.array(["B"] * 4 + [CONTROL] * 2 + ["A"] * 3)
    names = sorted(set(labels))
    folder = tmp_path / "outputs/final-test/no_perturbation"
    folder.mkdir(parents=True)
    prediction = final.predict_final(
        tmp_path,
        folder,
        {"kind": "no_perturbation", "label": "no_perturbation"},
        labels,
        names,
        ["A", "B", "G"],
        90.0,
        device=torch.device("cpu"),
    )
    try:
        plan = final.control_sampling_plan(labels, names, len(controls))
        for name in names:
            rows = np.flatnonzero(labels == name)
            expected = final.apply_residual(
                torch.from_numpy(controls[plan[name]]), torch.zeros(1, 3), 30.0
            )[0]
            expected = torch.log1p(expected.expm1() * 3)
            np.testing.assert_array_equal(prediction[rows], expected.numpy())
        np.testing.assert_allclose(np.expm1(prediction).sum(1), 90.0, rtol=1e-6)
        assert read_json(folder / "prediction-info.json")["shape"] == [9, 3]
    finally:
        final.close_memmap(prediction)


def downloader_module():
    path = Path(__file__).resolve().parents[1] / "scripts/download_test_assets.py"
    spec = importlib.util.spec_from_file_location("synthetic_test_downloader", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_anonymous_range_download_checks_complete_crc_and_atomic_rename(tmp_path, monkeypatch):
    downloader = downloader_module()
    content = b"synthetic-range-bytes"
    crc = base64.b64encode(downloader.google_crc32c.Checksum(content).digest()).decode("ascii")
    spec = {
        "path": "data/raw/vcc2025/test/synthetic.bin",
        "url": "https://example.invalid/pinned?generation=42",
        "generation": "42",
        "size_bytes": len(content),
        "crc32c": crc,
    }
    monkeypatch.setattr(downloader, "CHUNK_SIZE", 3)
    monkeypatch.setattr(downloader, "validate_remote", lambda _: {"anonymous_access": True})
    monkeypatch.setattr(downloader, "ensure_space", lambda *args: None)

    class RangeResponse:
        status_code = 206

        def __init__(self, lo, hi):
            self.lo, self.hi = lo, hi
            self.headers = {
                "Content-Range": f"bytes {lo}-{hi}/{len(content)}",
                "x-goog-generation": "42",
            }

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def raise_for_status(self):
            return None

        def iter_content(self, _):
            yield content[self.lo : self.hi + 1]

    def get(url, headers, **kwargs):
        assert (
            url == spec["url"]
            and "Authorization" not in headers
            and "x-goog-user-project" not in headers
        )
        lo, hi = map(int, headers["Range"].removeprefix("bytes=").split("-"))
        return RangeResponse(lo, hi)

    monkeypatch.setattr(downloader.requests, "get", get)
    receipt = downloader.download_object(tmp_path, spec, "selection-sha")
    destination = tmp_path / spec["path"]
    assert destination.read_bytes() == content
    assert receipt["crc32c"] == crc and receipt["sha256"] == file_sha256(destination)
    assert not destination.with_name(destination.name + ".part").exists()
    assert not destination.with_name(destination.name + ".resume.json").exists()


def test_complete_download_receipt_is_reused_without_network_or_rewrite(tmp_path, monkeypatch):
    downloader = downloader_module()
    receipt_path = tmp_path / final.DOWNLOAD_RECEIPT
    final.atomic_json(
        receipt_path, {"status": "complete", "selection_sha256": "frozen-sha", "files": []}
    )
    before = file_sha256(receipt_path), receipt_path.stat().st_mtime_ns
    monkeypatch.setattr(downloader, "validate_selection", lambda *args: ({}, "frozen-sha"))
    checked = []
    monkeypatch.setattr(downloader, "verified_test_download", lambda *args: checked.append(args))
    monkeypatch.setattr(
        downloader.requests,
        "head",
        lambda *args, **kwargs: pytest.fail("No repeat network request"),
    )
    monkeypatch.setattr(sys, "argv", ["download_test_assets.py", "--root", str(tmp_path)])
    downloader.main()
    assert checked and (file_sha256(receipt_path), receipt_path.stat().st_mtime_ns) == before
