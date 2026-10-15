"""CPU checks for selection, response-space KD and unchanged MLP training."""

import copy
from unittest.mock import patch

import numpy as np
import pytest
import torch

from fusionvc.distillation import choose_teacher, prepare_distillation, student_objective
from fusionvc.fusion_mmd_report import FUSION_MODELS, SEEDS, VARIANTS, run_spec
from fusionvc.io import write_json
from fusionvc.models import ResidualPredictor, apply_residual


def candidates():
    rows = []
    for model in FUSION_MODELS:
        for variant in VARIANTS:
            for seed in SEEDS:
                rows.append(
                    {
                        "model": model,
                        "variant": variant,
                        "seed": seed,
                        "experiment": run_spec(model, variant)[1],
                        "mae": 0.3,
                        "checkpoint_sha256": f"{model}-{variant}-{seed}",
                    }
                )
    return rows


def test_teacher_selects_group_mean_with_exact_tie_and_fixed_seed():
    rows = candidates()
    for row in rows:
        if row["variant"] == "mean" and row["model"] in {"concat", "gated"}:
            row["mae"] = 0.1
        if row["model"] == "cross_attention" and row["seed"] == 43:
            row["mae"] = 0.0  # Best single seed must not determine the group.
    selected = choose_teacher(rows)
    assert selected["experiment"] == "05_concat" and selected["seed"] == 42
    assert selected["checkpoint_sha256"] == "concat-mean-42"
    assert selected == choose_teacher(list(reversed(rows)))


@pytest.mark.parametrize("bad", ["missing", "duplicate", "nonfinite", "wrong_identity"])
def test_incomplete_or_invalid_teacher_candidates_are_rejected(bad):
    rows = candidates()
    if bad == "missing":
        rows.pop()
    elif bad == "duplicate":
        rows[-1] = rows[0].copy()
    elif bad == "nonfinite":
        rows[0]["mae"] = float("nan")
    else:
        rows[0]["experiment"] = "unrelated"
    with pytest.raises(ValueError):
        choose_teacher(rows)


def test_prepare_refuses_before_verified_full_mmd_report_without_selecting_teacher(tmp_path):
    with (
        patch(
            "fusionvc.fusion_mmd_report.material_passport",
            return_value={"verification_status": "UNVERIFIED"},
        ),
        patch(
            "fusionvc.fusion_mmd_report.load_run",
            side_effect=AssertionError("selected before comparison finished"),
        ),
        patch(
            "torch.cuda.is_available",
            side_effect=AssertionError("touched GPU before protocol gate"),
        ),
    ):
        with pytest.raises(RuntimeError, match="Finish and verify"):
            prepare_distillation(tmp_path)


def test_kd_uses_transformed_same_bank_mean_and_freezes_teacher():
    controls = torch.tensor([[0.1, 1.0, 2.0], [0.4, 1.2, 0.5]], dtype=torch.float64)
    student_delta = torch.tensor([[0.2, -0.3, 0.1]], dtype=torch.float64, requires_grad=True)
    teacher_delta = torch.tensor([[0.5, -0.1, -0.2]], dtype=torch.float64, requires_grad=True)
    prediction = apply_residual(controls, student_delta, 10.0)
    truth = torch.tensor([[0.7, 1.1, 1.5]], dtype=torch.float64)
    loss, task, kd = student_objective(prediction, truth, controls, teacher_delta, 10.0, 1.0)
    raw = np.expm1(np.clip(controls.numpy() + teacher_delta.detach().numpy(), 0, 14))
    teacher_mean = np.log1p(raw * (10 / raw.sum(axis=1, keepdims=True))).mean(axis=0)
    expected_kd = np.mean((prediction.detach().numpy().mean(axis=1) - teacher_mean) ** 2)
    assert kd.item() == pytest.approx(expected_kd, rel=1e-12, abs=1e-12)
    assert loss.item() == pytest.approx(task.item() + kd.item())
    assert kd.item() != pytest.approx((student_delta - teacher_delta).square().mean().item())
    loss.backward()
    assert student_delta.grad is not None and student_delta.grad.abs().sum() > 0
    assert teacher_delta.grad is None


def test_equal_expression_after_clipping_has_zero_kd_despite_different_raw_delta():
    controls = torch.tensor([[0.2, 0.8, 1.2], [0.5, 0.7, 1.0]])
    student = torch.full((1, 3), 20.0, requires_grad=True)
    teacher = torch.full((1, 3), 16.0)
    prediction = apply_residual(controls, student, 10.0)
    loss, task, kd = student_objective(
        prediction, prediction.detach().mean(1), controls, teacher, 10.0, 1.0
    )
    assert (student - teacher).square().mean() > 0
    assert loss.item() == task.item() == kd.item() == 0


@pytest.mark.parametrize("weight", [-1.0, float("nan"), float("inf")])
def test_invalid_kd_weights_are_rejected(weight):
    with pytest.raises(ValueError):
        student_objective(
            torch.ones(1, 2, 3), torch.ones(1, 3), torch.ones(2, 3), torch.ones(1, 3), 10.0, weight
        )


def test_zero_kd_matches_original_loss_gradients_and_optimizer_states_exactly():
    torch.manual_seed(1729)
    config = {"model": "mlp", "width": 4, "dropout": 0.1}
    original = ResidualPredictor(config, torch.randn(2, 2), torch.rand(2), 5)
    student = copy.deepcopy(original)
    left = torch.optim.AdamW(original.parameters(), lr=0.001, weight_decay=0.0001)
    right = torch.optim.AdamW(student.parameters(), lr=0.001, weight_decay=0.0001)
    q, controls, truth = torch.randn(3, 2), torch.rand(2, 5), torch.rand(3, 5)
    for step in range(3):
        torch.manual_seed(step)
        left.zero_grad(set_to_none=True)
        prediction = apply_residual(controls, original(q), 10.0)
        old_loss = (prediction.mean(1) - truth).square().mean()
        old_loss.backward()
        torch.nn.utils.clip_grad_norm_(original.parameters(), 1.0)
        left.step()
        torch.manual_seed(step)
        right.zero_grad(set_to_none=True)
        prediction = apply_residual(controls, student(q), 10.0)
        # NaN teacher data is intentionally unused by the exact zero branch.
        new_loss, _, kd = student_objective(
            prediction, truth, controls, torch.full((3, 5), float("nan")), 10.0, 0.0
        )
        assert torch.equal(old_loss, new_loss) and kd.item() == 0
        new_loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        right.step()
        for name, value in original.state_dict().items():
            assert torch.equal(value, student.state_dict()[name])
        for a, b in zip(left.state.values(), right.state.values()):
            for key in a:
                assert torch.equal(a[key], b[key])


def test_student_input_loader_works_without_teacher_or_foundation_assets(tmp_path):
    from fusionvc.train import load_inputs

    base = tmp_path / "data/processed/baselines-v1"
    base.mkdir(parents=True)
    write_json(base / "manifest.json", {"target_sum": 10.0})
    write_json(base / "genes.json", ["A", "B", "C"])
    np.save(base / "gene_descriptors.npy", np.ones((3, 2), dtype=np.float32))
    np.save(base / "input_gene_indices.npy", np.array([0, 2]))
    np.savez(
        base / "train_stats.npz",
        names=np.array(["non-targeting"]),
        mean=np.ones((1, 3), dtype=np.float32),
    )
    np.save(base / "controls.npy", np.ones((2, 3), dtype=np.float32))
    config = {"model": "mlp", "width": 4, "dropout": 0.0, "protocol": "distillation-v1"}
    with (
        patch(
            "fusionvc.train.load_frozen_features", side_effect=AssertionError("unexpected encoder")
        ),
        patch(
            "fusionvc.distillation.load_distillation_cache",
            side_effect=AssertionError("unexpected teacher"),
        ),
    ):
        model, _, _, features, _ = load_inputs(tmp_path, config, torch.device("cpu"))
        assert model(torch.from_numpy(features[:1])).shape == (1, 3)
