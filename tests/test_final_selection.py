"""Selection uses group validation means, never an individually lucky seed."""

import pytest

from fusionvc.final_selection import MODELS, SEEDS, VARIANTS, rank_validation_groups


def candidate_rows():
    return [
        {
            "model": m,
            "variant": v,
            "seed": s,
            "experiment": f"{m}_{v}",
            "mae": {"mean": 0.03, "energy": 0.02, "hybrid": 0.04}[v],
        }
        for m in MODELS
        for v in VARIANTS
        for s in SEEDS
    ]


def test_group_selection_retains_all_seeds_and_uses_exact_tie_rule():
    rows = candidate_rows()
    # A lucky single seed cannot outweigh two poor seeds.
    for row in rows:
        if row["model"] == "mlp" and row["variant"] == "mean":
            row["mae"] = 0.001 if row["seed"] == 42 else 0.04
    ranked = rank_validation_groups(rows)
    assert next(r for r in ranked if r["model"] == "mlp")["variant"] == "energy"
    for row in rows:
        if row["variant"] == "hybrid":
            row["mae"] = 0.02
    ranked = rank_validation_groups(rows)
    for model in MODELS:
        assert next(r for r in ranked if r["model"] == model)["variant"] == "energy"


def test_incomplete_duplicate_or_nonfinite_validation_candidates_rejected():
    rows = candidate_rows()
    with pytest.raises(ValueError, match="exactly"):
        rank_validation_groups(rows[:-1])
    with pytest.raises(ValueError, match="exactly"):
        rank_validation_groups(rows[:-1] + [rows[0]])
    rows[0]["mae"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        rank_validation_groups(rows)
