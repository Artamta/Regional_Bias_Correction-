"""Contracts for the one-shot deployable CDF hybrid validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import fuxi_lead_gated_cdf_hybrid_validation as hybrid


def _validation_dates() -> np.ndarray:
    return np.concatenate(
        (
            np.datetime64("2018-01-01") + np.arange(104),
            np.datetime64("2019-01-01") + np.arange(92),
        )
    ).astype("datetime64[D]")


def test_frozen_plan_hash_and_parent_inputs_are_exact() -> None:
    assert hybrid.sha256_file(hybrid.PLAN_PATH) == hybrid.PLAN_SHA256
    inputs = hybrid.validate_frozen_inputs()
    assert inputs["cache_declared_sha256"] == hybrid.CACHE_DATA_SHA256
    assert inputs["pbc_manifest"]["contract"]["sealed_2025_target_opened"] is False


def test_base_adjustment_checkpoints_are_tensor_identical_to_accepted_neural() -> None:
    receipts = hybrid.verify_checkpoint_identity()
    assert set(receipts) == {"42", "43", "44"}
    assert all(receipt["tensor_identity"] for receipt in receipts.values())
    assert all(receipt["tensor_count"] == 16 for receipt in receipts.values())


def test_unequal_year_bootstrap_is_reproducible_and_retains_native_sizes() -> None:
    dates = _validation_dates()
    first = hybrid._bootstrap_draws(dates)
    second = hybrid._bootstrap_draws(dates)
    assert len(first) == len(second) == hybrid.BOOTSTRAP_DRAWS
    assert {len(draw) for draw in first} == {184, 196, 208}
    assert all(np.array_equal(left, right) for left, right in zip(first, second))


def test_hybrid_uses_only_neural_w1_and_persistence_w2_w6() -> None:
    neural = np.arange(2 * 6 * 3, dtype=np.float32).reshape(2, 6, 3)
    persistence = -np.ones_like(neural)
    result = hybrid.assemble_hybrid(neural, persistence)
    assert np.array_equal(result[:, 0], neural[:, 0])
    assert np.array_equal(result[:, 1:], persistence[:, 1:])


def _decision_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    bootstrap = pd.DataFrame(
        {
            "method": [hybrid.HYBRID_METHOD, hybrid.HYBRID_METHOD],
            "baseline": [hybrid.PERSISTENCE_METHOD, hybrid.COMBINED_METHOD],
            "rps_reduction_fraction": [0.006, 0.007],
            "ci_lower_95": [0.001, 0.002],
            "ci_upper_95": [0.011, 0.012],
        }
    )
    rows = []
    for year in hybrid.EXPECTED_YEAR_COUNTS:
        rows.extend(
            [
                {"method": hybrid.HYBRID_METHOD, "year": year, "rps": 0.19},
                {"method": hybrid.PERSISTENCE_METHOD, "year": year, "rps": 0.20},
                {"method": hybrid.COMBINED_METHOD, "year": year, "rps": 0.21},
            ]
        )
    return bootstrap, pd.DataFrame(rows)


def test_selection_requires_every_frozen_gate_and_exact_contrasts() -> None:
    bootstrap, yearwise = _decision_tables()
    decision = hybrid.selection_decision(bootstrap, yearwise, cdf_checks_pass=True)
    assert decision["candidate_promoted"] is True

    below_margin = bootstrap.copy()
    below_margin.loc[0, "rps_reduction_fraction"] = 0.004999
    assert (
        hybrid.selection_decision(below_margin, yearwise, cdf_checks_pass=True)[
            "candidate_promoted"
        ]
        is False
    )

    duplicates = pd.concat([bootstrap.iloc[[0]], bootstrap.iloc[[0]]], ignore_index=True)
    with pytest.raises(hybrid.HybridValidationError, match="contrast set moved"):
        hybrid.selection_decision(duplicates, yearwise, cdf_checks_pass=True)

