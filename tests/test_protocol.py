"""Guard scientific invariants rather than snapshot implementation details."""

import numpy as np
import torch
from scipy.sparse import csr_matrix

from fusionvc.io import normalize_log
from fusionvc.models import apply_residual


def test_normalization_preserves_zeros_and_library_scale():
    raw = csr_matrix(np.array([[0, 3, 7], [4, 0, 1]], dtype=np.float32))
    log = normalize_log(raw, 10000).toarray()
    assert np.all(log[raw.toarray() == 0] == 0)
    np.testing.assert_allclose(np.expm1(log).sum(1), 10000, rtol=1e-5)


def test_residual_is_unpaired_and_permutation_invariant():
    counts = torch.tensor([[2000.0, 8000.0, 0.0], [0.0, 4000.0, 6000.0]])
    controls = counts.log1p()
    delta = torch.tensor([[0.1, -0.2, 0.05], [-0.2, 0.3, 0.4]])
    pred = apply_residual(controls, delta, 10000)
    permuted = apply_residual(controls.flip(0), delta, 10000)
    torch.testing.assert_close(pred.mean(1), permuted.mean(1))
    torch.testing.assert_close(
        pred.expm1().sum(-1), torch.full((2, 2), 10000.0), rtol=1e-5, atol=0.1
    )
    assert torch.all(pred >= 0)


def test_identity_shift_retains_cell_variability():
    ctrl = torch.tensor([[100.0, 900.0], [700.0, 300.0]]).log1p()
    pred = apply_residual(ctrl, torch.zeros((1, 2)), 1000)[0]
    torch.testing.assert_close(pred, ctrl)
    assert not torch.equal(pred[0], pred[1])
