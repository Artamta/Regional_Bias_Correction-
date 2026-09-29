"""Contracts for the post-hoc score-only neural/PBC lead router."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import fuxi_lead_gated_neural_pbc as router


def test_immutable_sources_build_the_exact_fixed_policy() -> None:
    for spec in router.COHORTS:
        frame = router.build_policy_case_scores(spec)
        hybrid = frame.loc[frame.method.eq(router.HYBRID_METHOD)]
        assert len(hybrid) == spec.expected_cases * len(router.LEADS)
        assert set(hybrid.loc[hybrid.lead_week.eq(1), "component_method"]) == {
            router.NEURAL_METHOD
        }
        assert set(hybrid.loc[hybrid.lead_week.gt(1), "component_method"]) == {
            router.PERSISTENCE_METHOD
        }
        sources = frame.loc[frame.method.ne(router.HYBRID_METHOD)].set_index(
            ["method", "initialization", "lead_week"]
        )
        for row in hybrid.itertuples(index=False):
            expected_method = (
                router.NEURAL_METHOD
                if row.lead_week == 1
                else router.PERSISTENCE_METHOD
            )
            expected_rps = sources.loc[
                (expected_method, row.initialization, row.lead_week), "rps"
            ]
            assert row.rps == expected_rps
        assert hybrid.groupby("lead_week").initialization.nunique().to_dict() == {
            lead: spec.expected_cases for lead in router.LEADS
        }
        assert hybrid.policy_post_hoc.all()


def test_changed_manifest_hash_is_rejected_before_scores_are_read() -> None:
    moved = replace(router.COHORTS[0], manifest_sha256="0" * 64)
    with pytest.raises(router.LeadGateError, match="manifest SHA-256 changed"):
        router.build_policy_case_scores(moved)


def test_equal_year_block_draws_are_reproducible_and_keep_case_clusters() -> None:
    dates = np.concatenate(
        [
            np.datetime64("2018-01-04") + 7 * np.arange(8),
            np.datetime64("2019-01-03") + 7 * np.arange(8),
        ]
    ).astype("datetime64[D]")
    first, diagnostics = router._equal_year_draws(
        dates, draws=20, block_length=3, seed=17
    )
    second, _ = router._equal_year_draws(
        dates, draws=20, block_length=3, seed=17
    )
    assert all(np.array_equal(left, right) for left, right in zip(first, second))
    assert {len(draw) for draw in first} == {len(dates)}
    assert diagnostics["source_years"] == [2018, 2019]
    assert diagnostics["block_length_initializations"] == 3


def test_acceptance_requires_every_pbc_interval_to_clear_zero() -> None:
    rows = []
    for spec in router.COHORTS:
        for baseline in (router.PERSISTENCE_METHOD, router.COMBINED_METHOD):
            rows.append(
                {
                    "cohort": spec.name,
                    "baseline": baseline,
                    "rps_reduction_fraction": 0.01,
                    "ci_lower_95": 0.001,
                    "ci_upper_95": 0.02,
                }
            )
    passing = pd.DataFrame(rows)
    assert router.acceptance_decision(passing)["passes"] is True
    failing = passing.copy()
    failing.loc[0, "ci_lower_95"] = 0.0
    assert router.acceptance_decision(failing)["passes"] is False

    duplicated = pd.concat([passing.iloc[[0]]] * len(passing), ignore_index=True)
    with pytest.raises(router.LeadGateError, match="incomplete or duplicated"):
        router.acceptance_decision(duplicated)


def test_full_score_only_bundle_is_atomic_and_non_overwriting(tmp_path) -> None:
    output = tmp_path / "diagnostic"
    manifest = router.run(output)
    assert manifest["status"] == "complete"
    assert manifest["contract"]["sealed_2025_target_opened"] is False
    assert manifest["contract"]["forecast_probability_arrays_opened"] is False
    assert (output / "metrics/case_scores.csv").is_file()
    persisted = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert persisted["acceptance"] == manifest["acceptance"]
    with pytest.raises(router.LeadGateError, match="fresh output path required"):
        router.run(output)
