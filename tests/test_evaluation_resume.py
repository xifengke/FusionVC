"""A scored evaluation must finish publishing when resumed after interruption."""

import pytest

from fusionvc.evaluate import evaluate
from fusionvc.io import file_sha256, read_json, write_json


def scored_run(root):
    run = root / "outputs/05_concat/seed_42"
    out = run / "evaluation"
    out.mkdir(parents=True)
    (run / "best.pt").write_bytes(b"already-scored-checkpoint")
    write_json(run / "config.json", {"results_group": "fusion"})
    for name in ["summary.json", "environment.json"]:
        write_json(run / name, {"saved": True})
    (run / "history.csv").write_text("epoch,val_mae\n1,0.1\n", "utf-8")
    (out / "results.csv").write_text("perturbation,mae\nA,0.1\n", "utf-8")
    (out / "distribution_per_perturbation.csv").write_text(
        "perturbation,energy_mmd_rms\nA,0.1\n", "utf-8"
    )
    write_json(
        out / "metrics.json",
        {
            "checkpoint_sha256": file_sha256(run / "best.pt"),
            "distribution": {"energy_mmd_rms": 0.1},
        },
    )
    (out / "predictions.npy").write_bytes(b"temporary-prediction")
    return run


def test_scored_resume_repairs_publication_without_recomputing(tmp_path):
    run = scored_run(tmp_path)
    evaluate(tmp_path, run)
    small = tmp_path / "results/fusion/05_concat/seed_42"
    assert read_json(small / "metrics.json")["distribution"]["energy_mmd_rms"] == 0.1
    assert (small / "training.json").is_file()
    assert (small / "per_perturbation.csv").is_file()
    assert (small / "distribution_per_perturbation.csv").is_file()
    assert not (run / "evaluation/predictions.npy").exists()
    evaluate(tmp_path, run)


def test_failed_publication_keeps_prediction_and_hash_mismatch_rejected(tmp_path):
    run = scored_run(tmp_path)
    (run / "environment.json").unlink()
    with pytest.raises(FileNotFoundError, match="Incomplete scored evaluation"):
        evaluate(tmp_path, run)
    assert (run / "evaluation/predictions.npy").is_file()
    (run / "best.pt").write_bytes(b"modified")
    with pytest.raises(ValueError, match="Checkpoint changed"):
        evaluate(tmp_path, run)
