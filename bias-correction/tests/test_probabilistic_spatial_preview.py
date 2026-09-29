"""Focused numerical checks for the probabilistic spatial preview."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


EVALUATE = Path(__file__).resolve().parents[1] / "evaluate"
if str(EVALUATE) not in sys.path:
    sys.path.insert(0, str(EVALUATE))

from generate_probabilistic_spatial_preview import (  # noqa: E402
    apply_affine_log_calibration,
    ensemble_crps,
    select_median_skill_case,
)


def test_affine_identity_returns_raw_members() -> None:
    rng = np.random.default_rng(7)
    raw = rng.gamma(2.0, 2.0, size=(5, 6, 27, 27)).astype(np.float32)
    corrected = apply_affine_log_calibration(
        raw,
        np.zeros((6, 27, 27), dtype=np.float32),
        np.ones((6, 27, 27), dtype=np.float32),
    )
    np.testing.assert_allclose(corrected, raw, rtol=2.0e-6, atol=2.0e-6)


def test_deterministic_crps_equals_absolute_error() -> None:
    forecast = np.asarray([[[1.0, 4.0], [8.0, 3.0]]])
    truth = np.asarray([[2.0, 1.0], [8.0, 5.0]])
    np.testing.assert_allclose(ensemble_crps(forecast, truth), np.abs(forecast[0] - truth))


def test_median_skill_selection_is_mechanical() -> None:
    skills = np.asarray([-0.2, 0.4, 0.1, 0.8, 0.2])
    candidates = np.asarray([11, 15, 19, 23, 27])
    assert select_median_skill_case(skills, candidates) == 27
