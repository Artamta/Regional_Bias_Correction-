"""Leakage and uncertainty tests for blocked OOF calibration."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parents[1]
NEURAL_SRC = HERE.parent / "neural_adapter" / "src"
for path in (HERE, NEURAL_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import fuxi_imd_blocked_oof_affine as oof  # noqa: E402


def test_four_folds_partition_calibration_years_without_overlap() -> None:
    oof.validate_fold_design()
    outer = []
    for fold in oof.FOLDS:
        assert len(fold.train_years) == 10
        assert len(fold.inner_validation_years) == 2
        assert len(fold.outer_years) == 4
        assert not set(fold.train_years) & set(fold.inner_validation_years)
        assert not set(fold.train_years) & set(fold.outer_years)
        assert not set(fold.inner_validation_years) & set(fold.outer_years)
        outer.extend(fold.outer_years)
    assert sorted(outer) == list(range(2002, 2018))


def test_sealed_archive_configuration_excludes_all_post_2017_years() -> None:
    oof.configure_calibration_archive(oof.FOLDS[2].train_years)
    assert oof.base.TRAIN_YEARS == oof.FOLDS[2].train_years
    assert oof.base.ALL_YEARS == tuple(range(2002, 2018))
    assert not set(oof.base.ALL_YEARS) & set(range(2018, 2026))


def test_fold_truth_identity_treats_shared_missing_cells_as_equal() -> None:
    first = np.asarray([1.0, np.nan, 2.0], dtype=np.float32)
    second = np.asarray([1.0, np.nan, 2.0], dtype=np.float32)
    assert np.array_equal(first, second, equal_nan=True)
    second[-1] = 3.0
    assert not np.array_equal(first, second, equal_nan=True)


def _case_metrics(candidate_delta: float) -> pd.DataFrame:
    rows = []
    for configuration, shift in (("candidate", candidate_delta), ("reference", 0.0)):
        for day in range(8):
            for lead in range(1, 7):
                rows.append(
                    {
                        "configuration": configuration,
                        "member": "ensemble",
                        "case_id": pd.Timestamp("2018-06-01") + pd.Timedelta(days=day),
                        "lead": lead,
                        "rmse": 5.0 + shift,
                        "mae": 3.0 + shift,
                        "bias": -1.0 + shift,
                        "acc": 0.3 - shift,
                    }
                )
    return pd.DataFrame(rows)


def test_bootstrap_resamples_initializations_with_all_leads_and_is_deterministic() -> None:
    metrics = _case_metrics(-0.1)
    first = oof.paired_case_bootstrap(
        metrics, "candidate", "reference", repetitions=200, seed=13
    )
    second = oof.paired_case_bootstrap(
        metrics, "candidate", "reference", repetitions=200, seed=13
    )
    pd.testing.assert_frame_equal(first, second)
    assert set(first.metric) == {"rmse", "mae", "bias_abs", "acc"}
    assert first.bootstrap_unit.str.contains("all six leads").all()
    assert np.isfinite(first[["ci_lower_2p5", "ci_upper_97p5"]]).all().all()
