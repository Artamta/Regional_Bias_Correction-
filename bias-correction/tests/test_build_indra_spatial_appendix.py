"""Synthetic contract tests for the INDRA spatial appendix builder."""

from __future__ import annotations

import numpy as np

from build_indra_spatial_appendix import (
    empirical_crps,
    jjas_midpoint_mask,
    select_nearest_quantile,
)


def test_empirical_crps_for_two_member_ensemble() -> None:
    members = np.asarray([[[[[0.0]]], [[[2.0]]]]], dtype=np.float32)
    truth = np.asarray([[[[1.0]]]], dtype=np.float32)
    observed = empirical_crps(members, truth)
    assert observed.shape == (1, 1, 1, 1)
    assert np.allclose(observed, 0.5)


def test_quantile_selection_is_nearest_and_deterministic() -> None:
    values = np.asarray([-2.0, -1.0, 1.0, 4.0])
    assert select_nearest_quantile(values, 0.5) == 1
    assert select_nearest_quantile(values, 0.1) == 0


def test_jjas_assignment_uses_valid_period_midpoint() -> None:
    targets = np.empty((2, 6, 7), dtype="datetime64[D]")
    starts = np.asarray(["2024-05-29", "2024-09-28"], dtype="datetime64[D]")
    offsets = np.arange(1, 43).reshape(1, 6, 7).astype("timedelta64[D]")
    targets[:] = starts[:, None, None] + offsets
    mask = jjas_midpoint_mask(targets)
    assert mask[0, 0]
    assert not mask[1, 0]
