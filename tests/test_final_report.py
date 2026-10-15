"""Independent target alignment and direction checks for final paired intervals."""

import numpy as np
import pandas as pd
import pytest

from fusionvc.final_report import METRICS, paired_intervals


def test_final_intervals_align_target_ids_and_preserve_metric_directions():
    names = [f"target_{i}" for i in range(100)]
    baseline = pd.DataFrame(
        np.arange(100)[:, None] + np.ones((100, 4)), index=names, columns=METRICS
    )
    change = np.array([-0.1, 0.2, 0.3, -0.4])
    candidate = baseline + change
    reference = baseline.iloc[::-1]
    result = paired_intervals(candidate, reference, "student", "control", np.random.default_rng(42))
    for row, expected in zip(result, change):
        assert row["difference"] == pytest.approx(expected)
        assert row["candidate_better_targets"] == 100
        assert row["target_bootstrap_95"] == pytest.approx([expected, expected])


def test_final_intervals_reject_mismatched_target_sets():
    data = pd.DataFrame(np.ones((100, 4)), columns=METRICS)
    with pytest.raises(ValueError, match="same 100"):
        paired_intervals(data, data.iloc[:-1], "a", "b", np.random.default_rng(42))
