from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


CLEAN_ROOT = Path(__file__).resolve().parents[2]
STUDY = CLEAN_ROOT / "studies/fuxi_imd_adapter_benchmark_v1"
if str(STUDY) not in sys.path:
    sys.path.insert(0, str(STUDY))

import make_ccai_neural_adapter_figures as figures  # noqa: E402


AUDIT = STUDY / "results/full_context_jjas_2022_2024_job91439"
OUTPUT = STUDY / "results/full_context_jjas_2022_2024_ccai_figures_v1"
HISTORICAL_BOUNDARY_LOADER = (
    Path(__file__).resolve().parents[1]
    / "archive"
    / "source_snapshots"
    / "ccai_2022_2024"
    / "plot_physical_validation_results.py"
)


def canonical_cases() -> pd.DataFrame:
    return figures.validate_case_table(pd.read_csv(AUDIT / "case_metrics.csv"))


def test_frozen_case_contract_and_saved_summary_reconcile() -> None:
    cases = canonical_cases()
    assert len(cases) == 9_000
    assert cases["init"].nunique() == 100
    assert not cases.duplicated(["method", "init", "lead_week", "region"]).any()
    summary = pd.read_csv(AUDIT / "summary_by_lead_region.csv")
    figures.reconcile_saved_summary(cases, summary)


def test_duplicate_case_key_is_rejected() -> None:
    cases = pd.read_csv(AUDIT / "case_metrics.csv")
    duplicate = pd.concat([cases, cases.iloc[[0]]], ignore_index=True)
    with pytest.raises(figures.FigureContractError, match="9,000|duplicate"):
        figures.validate_case_table(duplicate)


def test_bootstrap_indices_are_deterministic_and_keep_full_season_slots() -> None:
    dates = figures.ordered_initializations(canonical_cases())
    within, slices = figures.year_stratified_circular_indices(
        dates, draws=32, block_length=13, seed=1234
    )
    within_again, slices_again = figures.year_stratified_circular_indices(
        dates, draws=32, block_length=13, seed=1234
    )
    assert np.array_equal(within, within_again)
    assert slices == slices_again
    source_year = np.asarray(dates.year)
    for draw in within:
        counts = {
            year: int(np.count_nonzero(source_year[draw] == year))
            for year in figures.EXPECTED_YEARS
        }
        assert counts == figures.EXPECTED_YEAR_COUNTS

    two_stage = figures.two_stage_circular_indices(
        dates, draws=32, block_length=13, seed=1234
    )
    two_stage_again = figures.two_stage_circular_indices(
        dates, draws=32, block_length=13, seed=1234
    )
    assert np.array_equal(two_stage, two_stage_again)
    assert two_stage.shape == (32, 100)
    slot_edges = (0, 35, 70, 100)
    for draw in two_stage:
        for left, right in zip(slot_edges[:-1], slot_edges[1:]):
            assert len(np.unique(source_year[draw[left:right]])) == 1


def test_effect_sign_convention_and_percentile_summary() -> None:
    baseline = np.full((4, 2), 10.0)
    candidate = np.full((4, 2), 9.0)
    indices = np.tile(np.arange(4), (20, 1))
    point, lower, upper = figures.summarize_effect(
        candidate, baseline, indices, "rmse_skill_pct"
    )
    assert point == pytest.approx(10.0)
    assert lower == pytest.approx(10.0)
    assert upper == pytest.approx(10.0)
    point, lower, upper = figures.summarize_effect(
        candidate + 0.2, baseline, indices, "acc_delta"
    )
    assert point == pytest.approx(-0.8)
    assert lower == pytest.approx(-0.8)
    assert upper == pytest.approx(-0.8)


def test_generated_headlines_reproduce_frozen_case_table() -> None:
    lead = pd.read_csv(OUTPUT / "tables/lead_summary.csv")
    pooled = lead.loc[lead.lead_week.astype(str).eq("ALL")].set_index("method")
    assert pooled.loc["raw_fuxi", "rmse_mm_day"] == pytest.approx(
        5.7226039746839135, abs=1.0e-12
    )
    assert pooled.loc["selected_adapter", "rmse_mm_day"] == pytest.approx(
        5.274947867479156, abs=1.0e-12
    )
    assert pooled.loc["selected_adapter", "acc"] - pooled.loc[
        "raw_fuxi", "acc"
    ] == pytest.approx(0.08241317448771163, abs=1.0e-12)

    effects = pd.read_csv(OUTPUT / "tables/paired_block_bootstrap_effects.csv")
    pooled_skill = effects.loc[
        effects.scope_type.eq("pooled")
        & effects.baseline.eq("raw_fuxi")
        & effects.metric.eq("rmse_skill_pct")
    ].iloc[0]
    assert pooled_skill.effect == pytest.approx(7.822594559839069, abs=1.0e-12)
    assert pooled_skill.ci_lower_2p5 == pytest.approx(6.421588066505529, abs=1.0e-12)
    assert pooled_skill.ci_upper_97p5 == pytest.approx(9.204564289794472, abs=1.0e-12)
    assert "two-stage" in pooled_skill.interval


def test_generated_extreme_diagnostic_matches_prediction_cube() -> None:
    threshold = pd.read_csv(OUTPUT / "tables/threshold_metrics.csv")
    selected_20 = threshold.loc[
        threshold.method.eq("selected_adapter") & threshold.threshold_mm_day.eq(20.0)
    ].iloc[0]
    raw_20 = threshold.loc[
        threshold.method.eq("raw_fuxi") & threshold.threshold_mm_day.eq(20.0)
    ].iloc[0]
    assert selected_20.ets == pytest.approx(0.16678881566901543, abs=1.0e-12)
    assert raw_20.ets == pytest.approx(0.1690947747213639, abs=1.0e-12)

    intensity = pd.read_csv(OUTPUT / "tables/intensity_strata_metrics.csv")
    selected_extreme = intensity.loc[
        intensity.method.eq("selected_adapter") & intensity.stratum.eq("extreme_ge20")
    ].iloc[0]
    assert selected_extreme.rmse_mm_day == pytest.approx(
        17.554402138103036, abs=1.0e-12
    )
    assert selected_extreme.bias_mm_day == pytest.approx(
        -14.38805524248896, abs=1.0e-12
    )


def test_posthoc_static_weight_discrepancy_is_quantified() -> None:
    weight = pd.read_csv(OUTPUT / "tables/weighting_diagnostic.csv").iloc[0]
    assert weight.static_area_weight_sum_km2_case_lead == pytest.approx(
        1_924_372_232.8794148, abs=1.0e-6
    )
    assert weight.audit_effective_area_sum_km2_case_lead == pytest.approx(
        1_924_278_857.2153347, abs=1.0e-6
    )
    assert weight.static_relative_excess_pct == pytest.approx(
        0.004852501690699285, abs=1.0e-14
    )
    assert int(weight.case_leads_with_fractional_coverage_difference) == 8
    assert int(weight.total_case_leads) == 600


def test_generated_package_has_exact_five_figure_pairs_and_valid_hashes() -> None:
    manifest = figures.validate_generated_package(OUTPUT, verify_sources=False)
    sources = manifest["source_files"]
    assert set(sources) == figures.EXPECTED_SOURCE_RECORDS
    for name, record in sources.items():
        path = Path(str(record["path"]))
        if name == "boundary_loader":
            path = HISTORICAL_BOUNDARY_LOADER
        elif not path.is_absolute():
            path = CLEAN_ROOT / path
        if record["hash_kind"] == "sha256_tree":
            actual = figures.sha256_tree(path)
        else:
            assert record["hash_kind"] == "sha256_file"
            actual = figures.sha256_file(path)
        assert actual == record["sha256"], name
    assert manifest["figure_count"] == 5
    expected = {
        f"{stem}.{suffix}" for stem in figures.FIGURE_STEMS for suffix in ("png", "pdf")
    }
    recorded_figures = {
        relative
        for relative in manifest["output_files"]
        if Path(relative).suffix in {".png", ".pdf"}
    }
    assert recorded_figures == expected
    for relative in expected:
        assert (OUTPUT / relative).stat().st_size > 20_000


def test_manifest_and_captions_keep_scope_and_alignment_separate() -> None:
    manifest = json.loads((OUTPUT / "MANIFEST.json").read_text(encoding="utf-8"))
    contract = manifest["evaluation_contract"]
    assert contract["weekly_alignment"] == "W1 init+0..6 through W6 init+35..41"
    assert contract["independent_2025_claim"] is False
    assert manifest["scientific_status"].startswith("2022-2024 development")
    captions = (OUTPUT / "CAPTIONS.md").read_text(encoding="utf-8")
    readme = (OUTPUT / "README.md").read_text(encoding="utf-8")
    assert "not an untouched independent test" in captions
    assert "Keep separate from INDIA-S2S-BENCH" in readme
    assert "no pixel-wise significance" in captions
    assert "fractional weekly-coverage" in captions
    assert "0.0049%" in captions


def test_hash_verifier_rejects_tampering(tmp_path: Path) -> None:
    path = tmp_path / "source.txt"
    path.write_text("frozen\n", encoding="utf-8")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    figures._verify_recorded_hash(path, expected)
    path.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(figures.FigureContractError, match="hash mismatch"):
        figures._verify_recorded_hash(path, expected)


def test_overwrite_guard_requires_owned_narrow_package(tmp_path: Path) -> None:
    bundle = SimpleNamespace(
        audit=AUDIT.resolve(),
        run=figures.DEFAULT_RUN.resolve(),
        boundary_path=figures.DEFAULT_BOUNDARY.resolve(),
    )
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    with pytest.raises(figures.FigureContractError, match="without this package"):
        figures.safe_output_target(unrelated, bundle, overwrite=True)

    owned = tmp_path / "owned"
    owned.mkdir()
    (owned / "MANIFEST.json").write_text(
        json.dumps(
            {
                "schema_name": "fuxi_imd_ccai_neural_adapter_figures",
                "schema_version": 1,
            }
        ),
        encoding="utf-8",
    )
    assert figures.safe_output_target(owned, bundle, overwrite=True) == owned.resolve()
    with pytest.raises(figures.FigureContractError, match="protected"):
        figures.safe_output_target(figures.CLEAN_ROOT, bundle, overwrite=True)
    with pytest.raises(figures.FigureContractError, match="nested inside"):
        figures.safe_output_target(AUDIT / "nested_package", bundle, overwrite=False)


def test_package_validator_rejects_unrecorded_extra_file(tmp_path: Path) -> None:
    copied = tmp_path / "package"
    shutil.copytree(OUTPUT, copied)
    (copied / "unrecorded-extra.txt").write_text("extra\n", encoding="utf-8")
    with pytest.raises(figures.FigureContractError, match="file set changed"):
        figures.validate_generated_package(copied, verify_sources=False)
