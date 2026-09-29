"""Focused contracts for the one-shot all-season physical-context screen."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import fuxi_allseason_ensemble_calibration as base
import fuxi_allseason_physical_context as physical


def _base_context(case_count: int) -> base.ContextBundle:
    support = np.ones((27, 27), dtype=bool)
    return base.ContextBundle(
        normalized_climatology=np.zeros(
            (case_count, 6, 27, 27), dtype=np.float32
        ),
        climatology_mean_by_lead=np.zeros(6, dtype=np.float32),
        climatology_std_by_lead=np.ones(6, dtype=np.float32),
        latitude_scaled=np.linspace(1.0, -1.0, 27, dtype=np.float32),
        longitude_scaled=np.linspace(-1.0, 1.0, 27, dtype=np.float32),
        season_sin=np.zeros((case_count, 6), dtype=np.float32),
        season_cos=np.ones((case_count, 6), dtype=np.float32),
        lead_scaled=np.linspace(0.0, 1.0, 6, dtype=np.float32),
        support=support,
    )


def _raw_fields(case_count: int = 4) -> np.ndarray:
    case = np.arange(case_count, dtype=np.float32)[:, None, None, None, None]
    lead = np.arange(6, dtype=np.float32)[None, :, None, None, None]
    feature = np.arange(10, dtype=np.float32)[None, None, :, None, None]
    row = np.arange(27, dtype=np.float32)[None, None, None, :, None]
    column = np.arange(27, dtype=np.float32)[None, None, None, None, :]
    return np.asarray(
        10.0 + case + 0.2 * lead + 2.0 * feature + 0.01 * row + 0.001 * column,
        dtype=np.float32,
    )


def test_exact_feature_order_widths_and_parameter_counts() -> None:
    assert physical.PHYSICAL_CONTEXT_FEATURE_NAMES == (
        "t2m_mean",
        "tcwv_mean",
        "q850_mean",
        "u850_mean",
        "v850_mean",
        "q850_u850_flux_mean",
        "q850_v850_flux_mean",
        "z500_mean",
        "msl_mean",
        "olr_mean",
    )
    assert physical.PHYSICAL_CONTEXT_CHANNEL_NAMES == (
        *physical.BASE_CONTEXT_CHANNEL_NAMES,
        *physical.PHYSICAL_CONTEXT_FEATURE_NAMES,
    )
    assert [physical.context_channels_for_arm(arm) for arm in physical.ARMS] == [
        7,
        17,
        17,
    ]
    for arm in physical.ARMS:
        model = physical.build_seeded_model(arm, 42)
        assert sum(parameter.numel() for parameter in model.parameters()) == (
            physical.EXPECTED_PARAMETER_COUNTS[arm]
        )
    assert physical.EXPECTED_PARAMETER_COUNTS == {
        physical.BASE_ARM: 42_434,
        physical.ZERO_PHYSICAL_ARM: 48_914,
        physical.PHYSICAL_ARM: 48_914,
    }


def test_wide_arms_share_exact_zero_transplanted_initial_state() -> None:
    zero = physical.build_seeded_model(physical.ZERO_PHYSICAL_ARM, 43)
    candidate = physical.build_seeded_model(physical.PHYSICAL_ARM, 43)
    for name, values in zero.state_dict().items():
        assert torch.equal(values, candidate.state_dict()[name])
    base_model = physical.build_seeded_model(physical.BASE_ARM, 43)
    wide_weight = zero.state_dict()["backbone.input.weight"]
    base_weight = base_model.state_dict()["backbone.input.weight"]
    assert torch.equal(wide_weight[:, : base_weight.shape[1]], base_weight)
    assert torch.count_nonzero(wide_weight[:, base_weight.shape[1] :]) == 0


def test_train_only_feature_lead_normalization_ignores_validation_mutation() -> None:
    raw = _raw_fields()
    train = np.asarray([0, 1], dtype=np.int64)
    weights = np.ones((27, 27), dtype=np.float64)
    original = physical.build_physical_context_bundle(
        _base_context(4), raw, train, weights
    )
    changed_raw = raw.copy()
    changed_raw[2:] += np.float32(1_000_000.0)
    changed = physical.build_physical_context_bundle(
        _base_context(4), changed_raw, train, weights
    )

    np.testing.assert_array_equal(
        original.mean_by_lead_feature, changed.mean_by_lead_feature
    )
    np.testing.assert_array_equal(
        original.std_by_lead_feature, changed.std_by_lead_feature
    )
    for case in train:
        np.testing.assert_array_equal(
            physical.context_for_case(original, int(case), physical.PHYSICAL_ARM),
            physical.context_for_case(changed, int(case), physical.PHYSICAL_ARM),
        )
    assert not np.array_equal(
        physical.context_for_case(original, 2, physical.PHYSICAL_ARM),
        physical.context_for_case(changed, 2, physical.PHYSICAL_ARM),
    )


def test_context_materialization_has_exact_base_zero_and_physical_schemas() -> None:
    raw = _raw_fields()
    bundle = physical.build_physical_context_bundle(
        _base_context(4),
        raw,
        np.asarray([0, 1], dtype=np.int64),
        np.ones((27, 27), dtype=np.float64),
    )
    base_context = physical.context_for_case(bundle, 2, physical.BASE_ARM)
    zero_context = physical.context_for_case(bundle, 2, physical.ZERO_PHYSICAL_ARM)
    candidate = physical.context_for_case(bundle, 2, physical.PHYSICAL_ARM)
    assert base_context.shape == (6, 7, 27, 27)
    assert zero_context.shape == candidate.shape == (6, 17, 27, 27)
    np.testing.assert_array_equal(zero_context[:, :7], base_context)
    np.testing.assert_array_equal(candidate[:, :7], base_context)
    assert np.count_nonzero(zero_context[:, 7:]) == 0
    expected = (
        raw[2] - bundle.mean_by_lead_feature[:, :, None, None]
    ) / bundle.std_by_lead_feature[:, :, None, None]
    np.testing.assert_allclose(candidate[:, 7:], expected, rtol=1.0e-6, atol=1.0e-6)


def test_physical_cache_alignment_binds_order_dimensions_dates_and_grid() -> None:
    starts = np.asarray(["2002-01-01", "2002-01-04"], dtype="datetime64[D]")
    latitude = np.linspace(39.0, 0.0, 27)
    longitude = np.linspace(60.0, 99.0, 27)
    member_cache = base.MemberCache(
        members=np.zeros((2, 1, 6, 27, 27), dtype=np.float32),
        initializations=starts,
        latitude=latitude,
        longitude=longitude,
        member_labels=np.asarray([0]),
        cache_root=Path("/tmp/member-cache"),
        members_path=Path("/tmp/members.npy"),
        metadata_path=Path("/tmp/metadata.npz"),
        manifest_path=None,
    )
    fields = np.zeros((2, 6, 10, 27, 27), dtype=np.float32)
    metadata = {
        "feature_names": list(physical.PHYSICAL_CONTEXT_FEATURE_NAMES),
        "dims": ["init", "lead_week", "feature", "lat", "lon"],
        "normalization": "none",
        "initializations": ["2002-01-01", "2002-01-04"],
        "latitude": latitude.tolist(),
        "longitude": longitude.tolist(),
        "source_fingerprint": "synthetic",
        "data_file": "physical.npy",
        "data_sha256": "a" * 64,
        "feature_contract_sha256": "b" * 64,
    }
    receipt = physical.validate_physical_cache_alignment(
        fields, metadata, member_cache
    )
    assert receipt["alignment_exact"] is True
    moved = dict(metadata)
    moved["feature_names"] = list(reversed(metadata["feature_names"]))
    with pytest.raises(physical.PhysicalContextExperimentError, match="order"):
        physical.validate_physical_cache_alignment(fields, moved, member_cache)


def _metrics(
    *,
    rps: dict[str, float] | None = None,
    crps: dict[str, float] | None = None,
    overrides: dict[tuple[str, int, int, int], float] | None = None,
) -> pd.DataFrame:
    rps = rps or {
        physical.BASE_ARM: 1.000,
        physical.ZERO_PHYSICAL_ARM: 1.001,
        physical.PHYSICAL_ARM: 0.980,
    }
    crps = crps or {
        physical.BASE_ARM: 1.000,
        physical.ZERO_PHYSICAL_ARM: 1.001,
        physical.PHYSICAL_ARM: 1.003,
    }
    overrides = overrides or {}
    rows: list[dict[str, object]] = []
    for arm in physical.ARMS:
        for seed in base.SEEDS:
            for year in base.VALIDATION_YEARS:
                initialization = f"{year}-01-01"
                for lead in range(1, 7):
                    rows.append(
                        {
                            "split": "validation",
                            "arm": arm,
                            "seed": seed,
                            "init": initialization,
                            "year": year,
                            "lead_week": lead,
                            "crps": crps[arm],
                            "quintile_rps": overrides.get(
                                (arm, seed, year, lead), rps[arm]
                            ),
                            "score_contract": physical.SCORING_CONTRACT_VERSION,
                        }
                    )
    return pd.DataFrame(rows)


def test_selector_promotes_only_after_every_frozen_guard() -> None:
    selection = physical.select_physical_arm(_metrics())
    assert selection["selected_arm"] == physical.PHYSICAL_ARM
    assert selection["candidate_promoted"] is True
    assert selection["zero_physical_selectable"] is False
    assert selection["post_lock_pbc_benchmark_used_for_selection"] is False
    assert selection["zero_physical_control_reproduction"]["passes"] is True
    for comparison in selection["candidate_comparisons"].values():
        assert comparison["pooled_rps_improvement_guard"] is True
        assert comparison["w2_w6_rps_improvement_guard"] is True
        assert comparison["pooled_crps_max_0p5pct_worsening_guard"] is True
        assert comparison["both_years_rps_improvement_guard"] is True
        assert comparison["matched_seed_rps_guard"] is True


@pytest.mark.parametrize(
    ("scenario", "guard"),
    [
        ("late", "w2_w6_rps_improvement_guard"),
        ("crps", "pooled_crps_max_0p5pct_worsening_guard"),
        ("year", "both_years_rps_improvement_guard"),
        ("seed", "matched_seed_rps_guard"),
    ],
)
def test_each_candidate_guard_fails_closed(scenario: str, guard: str) -> None:
    metrics = _metrics()
    if scenario == "late":
        mask = (metrics.arm == physical.PHYSICAL_ARM) & (metrics.lead_week == 1)
        metrics.loc[metrics.arm == physical.PHYSICAL_ARM, "quintile_rps"] = 1.01
        metrics.loc[mask, "quintile_rps"] = 0.80
    elif scenario == "crps":
        metrics.loc[metrics.arm == physical.PHYSICAL_ARM, "crps"] = 1.006
    elif scenario == "year":
        metrics.loc[
            (metrics.arm == physical.PHYSICAL_ARM) & (metrics.year == 2019),
            "quintile_rps",
        ] = 1.01
    elif scenario == "seed":
        metrics.loc[metrics.arm == physical.PHYSICAL_ARM, "quintile_rps"] = 1.01
        metrics.loc[
            (metrics.arm == physical.PHYSICAL_ARM) & (metrics.seed == 42),
            "quintile_rps",
        ] = 0.94
    selection = physical.select_physical_arm(metrics)
    assert selection["selected_arm"] == physical.BASE_ARM
    assert selection["candidate_promoted"] is False
    assert any(
        comparison[guard] is False
        for comparison in selection["candidate_comparisons"].values()
    )


def test_zero_control_must_reproduce_base_before_promotion() -> None:
    metrics = _metrics()
    metrics.loc[metrics.arm == physical.ZERO_PHYSICAL_ARM, "quintile_rps"] = 1.01
    selection = physical.select_physical_arm(metrics)
    assert selection["zero_physical_control_reproduction"]["passes"] is False
    assert selection["candidate_promoted"] is False
    assert selection["selected_arm"] == physical.BASE_ARM


def test_selector_rejects_nonvalidation_or_misaligned_rows() -> None:
    contaminated = _metrics()
    contaminated.loc[0, "split"] = "test"
    with pytest.raises(ValueError, match="validation rows only"):
        physical.select_physical_arm(contaminated)
    incomplete = _metrics().iloc[:-1]
    with pytest.raises(ValueError, match="identical validation cases"):
        physical.select_physical_arm(incomplete)


def _benchmark_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    bootstrap = pd.DataFrame(
        {
            "method": [physical.SELECTED_POOL_METHOD] * 3,
            "baseline": list(physical.BENCHMARK_BASELINES),
            "rps_reduction_fraction": [0.01, 0.02, 0.03],
            "ci_lower_95": [0.001, 0.002, 0.003],
            "ci_upper_95": [0.02, 0.03, 0.04],
        }
    )
    rows = []
    for year in base.VALIDATION_YEARS:
        rows.append(
            {"method": physical.SELECTED_POOL_METHOD, "year": year, "rps": 0.18}
        )
        for index, baseline_name in enumerate(physical.BENCHMARK_BASELINES):
            rows.append(
                {"method": baseline_name, "year": year, "rps": 0.19 + index * 0.01}
            )
    return bootstrap, pd.DataFrame(rows)


def test_post_lock_superiority_requires_promotion_pooled_year_and_interval_gates() -> None:
    bootstrap, yearwise = _benchmark_tables()
    passed = physical.post_lock_superiority_decision(
        bootstrap,
        yearwise,
        candidate_promoted=True,
        selected_arm=physical.PHYSICAL_ARM,
        cdf_checks_pass=True,
    )
    assert passed["physical_candidate_superiority_supported"] is True
    assert passed["used_for_arm_selection"] is False
    not_promoted = physical.post_lock_superiority_decision(
        bootstrap,
        yearwise,
        candidate_promoted=False,
        selected_arm=physical.BASE_ARM,
        cdf_checks_pass=True,
    )
    assert not_promoted["physical_candidate_superiority_supported"] is False
    moved = bootstrap.copy()
    moved.loc[moved.baseline == physical.COMBINED_PBC_METHOD, "ci_lower_95"] = -0.001
    failed_interval = physical.post_lock_superiority_decision(
        moved,
        yearwise,
        candidate_promoted=True,
        selected_arm=physical.PHYSICAL_ARM,
        cdf_checks_pass=True,
    )
    assert failed_interval["physical_candidate_superiority_supported"] is False


def test_validate_args_freezes_full_and_smoke_recipes() -> None:
    full = physical.build_parser().parse_args([])
    physical.validate_args(full)
    assert base._parse_seeds(full.seeds) == (42, 43, 44)
    assert full.max_epochs == 100 and full.patience == 15
    smoke = physical.build_parser().parse_args(["--smoke"])
    physical.validate_args(smoke)
    assert base._parse_seeds(smoke.seeds) == (42,)
    assert smoke.max_epochs == 2 and smoke.patience == 1
    bad = physical.build_parser().parse_args(["--seeds", "42,43"])
    with pytest.raises(ValueError, match="full.*seeds"):
        physical.validate_args(bad)


def test_source_snapshot_binds_plan_tests_cache_dependencies_and_launchers(
    tmp_path: Path,
) -> None:
    receipts = physical.source_snapshot(tmp_path)
    required = {
        "code/src/fuxi_allseason_physical_context.py",
        "code/src/fuxi_allseason_physical_context_cache.py",
        "code/src/fuxi_allseason_member_cache.py",
        "code/src/fuxi_physical_feature_cache.py",
        "code/src/fuxi_persistence_context.py",
        "code/tests/test_allseason_physical_context.py",
        "code/tests/test_allseason_physical_context_cache.py",
        "code/plan/ALLSEASON_PHYSICAL_CONTEXT_ONE_SHOT_20260824.md",
        "code/slurm/build_fuxi_allseason_physical_context_cache.sbatch",
        "code/slurm/run_allseason_physical_context.sbatch",
    }
    assert required <= set(receipts)
    assert all(len(receipts[name]) == 64 for name in required)
    assert all((tmp_path / name).is_file() for name in required)


def test_slurm_launcher_binds_distinct_member_and_physical_sources() -> None:
    launcher = physical.PROJECT_ROOT / "slurm/run_allseason_physical_context.sbatch"
    source = launcher.read_text(encoding="utf-8")
    member_fingerprint = (
        "655ee4b82597daf150a8c28b2ed7b474ba6ce878d00836a6db8c3e75cb7a9dae"
    )
    physical_fingerprint = (
        "7e3bdc0ab304fafb54556558b8fa2872c6d2add71361ee8dbba328c6d01757bb"
    )
    assert f"EXPECTED_MEMBER_SOURCE_FINGERPRINT={member_fingerprint}" in source
    assert f"EXPECTED_PHYSICAL_SOURCE_FINGERPRINT={physical_fingerprint}" in source
    assert source.index("WORK_ROOT=") < source.index("SMOKE_MANIFEST_SHA256=")
    assert "slurm_gate_receipt.json" in source
    subprocess.run(["bash", "-n", str(launcher)], check=True)
