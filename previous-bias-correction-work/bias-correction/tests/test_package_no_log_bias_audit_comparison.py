from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


EVALUATE = Path(__file__).resolve().parents[1] / "evaluate"
if str(EVALUATE) not in sys.path:
    sys.path.insert(0, str(EVALUATE))

import package_no_log_bias_audit_comparison as audit  # noqa: E402


def test_audit_effect_uses_positive_raw_identity_skill() -> None:
    reference = np.full((4, 2), 10.0)
    candidate = np.full((4, 2), 9.0)
    sampled = np.tile(np.arange(4), (20, 1))

    effect, lower, upper, units = audit.summarize_effect(
        candidate,
        reference,
        sampled,
        metric="rmse_mm_day",
    )
    assert effect == pytest.approx(10.0)
    assert lower == pytest.approx(10.0)
    assert upper == pytest.approx(10.0)
    assert "percent reduction" in units

    acc, lower, upper, _ = audit.summarize_effect(
        np.full((4, 2), 0.25),
        np.full((4, 2), 0.20),
        sampled,
        metric="acc",
    )
    assert acc == pytest.approx(0.05)
    assert lower == pytest.approx(0.05)
    assert upper == pytest.approx(0.05)


def test_audit_comparison_rejects_baseline_drift() -> None:
    rows = []
    for method in (audit.RAW, audit.LOG_BIAS, audit.SOURCE_MODEL):
        rows.append(
            {
                "method": method,
                "method_label": method,
                "init": "2022-06-01",
                "year": 2022,
                "lead_week": 1,
                "region": "all_india",
                "region_label": "All India",
                "rmse_mm_day": 5.0,
                "mae_mm_day": 3.0,
                "bias_mm_day": -0.5,
                "acc": 0.3,
                "valid_cell_count": 171,
                "effective_area_km2": 1.0,
            }
        )
    anchored = pd.DataFrame(rows)
    raw_identity = anchored.copy()
    raw_identity.loc[raw_identity.method.eq(audit.RAW), "rmse_mm_day"] += 0.01

    with pytest.raises(AssertionError):
        audit.combine_cases(anchored, raw_identity)
