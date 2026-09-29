"""Focused contracts for the receipt-gated persistence/PBC evaluator."""

from __future__ import annotations

import copy
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import fuxi_allseason_persistence_augmented as v1
import fuxi_allseason_persistence_mask_control as mask_driver
import fuxi_allseason_persistence_pbc_comparison as comparison
from fuxi_persistence_context import (
    ARMS,
    BASE_ARM,
    PERSISTENCE_LAG_ARM,
    ZERO_LAG_ARM,
)
from fuxi_persistence_mask_control import MASK_ONLY_ARM


REAL_V1_MANIFEST = Path(
    "resultsv2/fuxi_allseason_persistence_augmented/"
    "full_20260822T233542Z/manifest.json"
)
VALIDATION_DATES = np.asarray(["2018-01-04", "2019-01-03"], dtype="datetime64[D]")
CDF_COLUMNS = (
    "cdf_shape_matches_thresholds",
    "forecast_cdf_finite_where_threshold_defined",
    "observed_cdf_finite_where_threshold_defined",
    "forecast_cdf_valid_for_thresholds",
    "observed_cdf_valid_for_thresholds",
)


def _seed_score_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = ("2020-01-02", "2021-01-07")
    for method_index, method in enumerate(ARMS):
        for seed in comparison.SEEDS:
            for initialization in dates:
                for lead_week in range(1, 7):
                    rows.append(
                        {
                            "method": method,
                            "seed": seed,
                            "initialization": initialization,
                            "lead_week": lead_week,
                            "rps": (
                                method_index + lead_week / 100.0 + (seed - 42) / 10.0
                            ),
                        }
                    )
    return pd.DataFrame.from_records(rows)


def _joint_metric_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    parent_rows: list[dict[str, object]] = []
    mask_rows: list[dict[str, object]] = []
    scores = {
        BASE_ARM: (1.0, 1.0),
        ZERO_LAG_ARM: (1.0, 1.0),
        PERSISTENCE_LAG_ARM: (0.99, 0.98),
    }
    for seed in comparison.SEEDS:
        for initialization in VALIDATION_DATES:
            date = np.datetime_as_string(initialization, unit="D")
            for lead_week in range(1, 7):
                common: dict[str, object] = {
                    "split": "validation",
                    "seed": seed,
                    "init": date,
                    "year": int(date[:4]),
                    "lead_week": lead_week,
                    "score_contract": v1.SCORING_CONTRACT_VERSION,
                    **{name: True for name in CDF_COLUMNS},
                }
                for arm in ARMS:
                    crps, rps = scores[arm]
                    parent_rows.append(
                        {
                            **common,
                            "arm": arm,
                            "crps": crps,
                            "quintile_rps": rps,
                        }
                    )
                mask_rows.append(
                    {
                        **common,
                        "arm": MASK_ONLY_ARM,
                        "crps": 1.0,
                        "quintile_rps": 1.0,
                    }
                )
    return pd.DataFrame(parent_rows), pd.DataFrame(mask_rows)


def _bootstrap_case_scores() -> pd.DataFrame:
    dates_by_year = []
    for start in ("2020-01-02", "2021-01-04"):
        weekly = pd.date_range(start, periods=52, freq="7D").to_numpy(
            dtype="datetime64[D]"
        )
        dates_by_year.append(np.sort(np.concatenate((weekly, weekly + 3))))
    dates = np.concatenate(dates_by_year)
    method_scale = {
        BASE_ARM: 0.88,
        ZERO_LAG_ARM: 1.02,
        PERSISTENCE_LAG_ARM: 0.84,
        comparison.COMBINED_PBC: 1.00,
        comparison.PERSISTENCE_PLUS_PLUS: 0.96,
    }
    rows: list[dict[str, object]] = []
    for case_index, initialization in enumerate(dates):
        date = np.datetime_as_string(initialization, unit="D")
        year = int(date[:4])
        case_factor = 1.0 + 0.04 * np.sin(case_index / 9.0)
        for lead_week in range(1, 7):
            lead_factor = 1.0 + lead_week / 100.0
            for method, scale in method_scale.items():
                rows.append(
                    {
                        "method": method,
                        "initialization": date,
                        "year": year,
                        "lead_week": lead_week,
                        "rps": float(scale * case_factor * lead_factor),
                    }
                )
    return pd.DataFrame.from_records(rows)


def test_average_seed_scores_requires_one_complete_paired_grid() -> None:
    frame = _seed_score_frame()
    result = comparison.average_seed_scores(frame)
    assert len(result) == len(ARMS) * 2 * 6
    assert set(result.seed) == {"mean_of_seed_scores_42_43_44"}
    selected = result.loc[
        result.method.eq(BASE_ARM)
        & result.initialization.eq("2020-01-02")
        & result.lead_week.eq(1)
    ]
    assert selected.rps.item() == pytest.approx(0.11)

    with pytest.raises(ValueError, match="paired|lacks every"):
        comparison.average_seed_scores(frame.drop(index=frame.index[0]))
    nonfinite = frame.copy()
    nonfinite.loc[0, "rps"] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        comparison.average_seed_scores(nonfinite)


def test_frozen_bootstrap_is_deterministic_42_rows_and_claims_use_full_ci() -> None:
    scores = _bootstrap_case_scores()
    first = comparison.paired_pbc_bootstrap(
        scores,
        ARMS,
        samples=comparison.BOOTSTRAP_SAMPLES,
        block_length=comparison.BOOTSTRAP_BLOCK_LENGTH,
        seed=comparison.BOOTSTRAP_SEED,
    )
    second = comparison.paired_pbc_bootstrap(
        scores,
        ARMS,
        samples=comparison.BOOTSTRAP_SAMPLES,
        block_length=comparison.BOOTSTRAP_BLOCK_LENGTH,
        seed=comparison.BOOTSTRAP_SEED,
    )
    pd.testing.assert_frame_equal(first, second, check_exact=True)
    assert len(first) == 3 * 2 * 7 == 42
    assert set(first.lead_scope) == {
        "W1-W6",
        "W1",
        "W2",
        "W3",
        "W4",
        "W5",
        "W6",
    }
    assert set(first.bootstrap_samples) == {2000}
    assert set(first.block_length_initializations) == {13}
    assert set(first.bootstrap_seed) == {20260823}
    assert set(first.paired_initializations) == {208}

    claims = comparison.classify_pbc_claims(
        first, {"joint_validation_selected_arm": BASE_ARM}
    )
    selected = claims["joint_validation_selected_arm"]
    assert selected["method"] == BASE_ARM
    assert selected["better_than_strongest_classical_baseline_pooled"] is True
    assert {row["claim_label"] for row in selected["comparisons"]} == {"better"}

    crossed = first.loc[first.method.eq(BASE_ARM)].copy()
    pooled_combined = crossed.lead_scope.eq("W1-W6") & crossed.baseline.eq(
        comparison.COMBINED_PBC
    )
    crossed.loc[pooled_combined, "ci_lower_95"] = -1.0e-6
    crossed_claim = comparison.classify_pbc_claims(
        crossed, {"joint_validation_selected_arm": BASE_ARM}
    )["joint_validation_selected_arm"]
    combined = next(
        row
        for row in crossed_claim["comparisons"]
        if row["lead_scope"] == "W1-W6" and row["baseline"] == comparison.COMBINED_PBC
    )
    assert combined["claim_label"] == "unresolved"
    assert crossed_claim["better_than_combined_pbc_pooled"] is False


def test_parser_makes_joint_selection_manifest_mandatory() -> None:
    parser = comparison.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--persistence-manifest", "parent.json"])
    parsed = parser.parse_args(
        [
            "--persistence-manifest",
            "parent.json",
            "--joint-selection-manifest",
            "joint.json",
        ]
    )
    assert parsed.joint_selection_manifest == Path("joint.json")


def test_real_immutable_v1_receipt_is_rejected_after_alignment_withdrawal() -> None:
    if not REAL_V1_MANIFEST.is_file() or not comparison.DEFAULT_PBC_MANIFEST.is_file():
        pytest.skip("immutable local experiment receipts are unavailable")
    with pytest.raises(
        comparison.PersistencePBCComparisonError,
        match="live source differs from persistence receipt",
    ):
        comparison.validate_persistence_receipt(REAL_V1_MANIFEST)


def test_joint_selector_recomputation_detects_tampering() -> None:
    parent, mask = _joint_metric_frames()
    locked = mask_driver.select_joint_arm(
        parent,
        mask,
        expected_seeds=comparison.SEEDS,
        expected_initializations=VALIDATION_DATES.tolist(),
    )
    locked["scientific_selection"] = True
    evidence = comparison.validate_joint_selection_reproduction(
        locked,
        parent,
        mask,
        VALIDATION_DATES.tolist(),
    )
    assert evidence["all_recomputed_gate_fields_exact"] is True
    assert evidence["selected_arm_exact"] is True
    assert evidence["development_indices_accessed"] is False

    tampered = copy.deepcopy(locked)
    tampered["selected_arm"] = BASE_ARM
    with pytest.raises(
        comparison.PersistencePBCComparisonError,
        match="selected_arm",
    ):
        comparison.validate_joint_selection_reproduction(
            tampered,
            parent,
            mask,
            VALIDATION_DATES.tolist(),
        )


def test_all_receipt_and_replay_gates_precede_the_sole_development_unlock(
    tmp_path: Path,
) -> None:
    public_source = inspect.getsource(comparison.run_evaluation)
    assert list(inspect.signature(comparison.run_evaluation).parameters) == [
        "args",
        "output",
    ]
    persistence_gate = public_source.index("validate_persistence_receipt")
    pbc_gate = public_source.index("validate_accepted_pbc_receipt")
    joint_gate = public_source.index("validate_joint_selection_receipt")
    private_call = public_source.index("_evaluate_validated")
    assert persistence_gate < pbc_gate < joint_gate < private_call

    fresh_source = inspect.getsource(comparison._fresh_identity_gates)
    assert "validate_validation_adjustment_replay" in fresh_source
    assert "splits.test" not in fresh_source
    evaluator_source = inspect.getsource(comparison._evaluate_validated)
    assert evaluator_source.count("splits.test") == 1
    assert evaluator_source.index("_fresh_identity_gates") < evaluator_source.index(
        "splits.test"
    )
    assert evaluator_source.index("splits.test") < evaluator_source.index(
        "load_accepted_pbc_inputs"
    )
    assert "joint.selected_arm" in evaluator_source
    assert '"claim_eligible_methods": [joint.selected_arm]' in evaluator_source

    joint_source = inspect.getsource(comparison.validate_joint_selection_receipt)
    for receipt_field in (
        "current_gpu_receipt",
        "parent_gpu_receipt",
        "gpu_model_memory_driver_exact",
        "mask_only_context_provenance_sha256",
        "mask_only_context_contract",
    ):
        assert receipt_field in joint_source

    snapshot = comparison._source_snapshot(tmp_path)
    assert len(snapshot) == 16
    assert "code/tests/test_allseason_persistence_pbc_comparison.py" in snapshot
    assert "code/slurm/evaluate_allseason_persistence_pbc_comparison.sbatch" in snapshot
