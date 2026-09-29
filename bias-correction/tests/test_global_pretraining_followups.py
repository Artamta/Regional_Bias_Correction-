"""Contracts for artifact-only global-pretraining follow-up diagnostics."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


HERE = Path(__file__).resolve().parents[1]
NEURAL_SRC = HERE.parent / "neural_adapter" / "src"
for path in (HERE / "src", HERE / "evaluate", NEURAL_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_global_pretraining_followups as followup  # noqa: E402


def _dates(*, include_2020: bool = False) -> np.ndarray:
    dates = np.concatenate(
        (
            np.datetime64("2018-06-01", "D") + np.arange(35) * np.timedelta64(3, "D"),
            np.datetime64("2019-06-01", "D") + np.arange(35) * np.timedelta64(3, "D"),
        )
    )
    if include_2020:
        dates[-1] = np.datetime64("2020-06-01", "D")
    return dates.astype("datetime64[D]")


def _synthetic_source(tmp_path: Path, *, include_2020: bool = False) -> Path:
    run = tmp_path / "source" / "full_synthetic"
    (run / "metrics").mkdir(parents=True)
    (run / "models").mkdir()
    shape = (70, 6, 27, 27)
    weights = np.zeros((27, 27), dtype=np.float64)
    weights.flat[:171] = np.linspace(1.0, 2.0, 171)
    support = weights > 0.0

    baseline = np.full(shape, np.nan, dtype=np.float32)
    spatial = np.linspace(0.5, 4.0, 171, dtype=np.float32)
    case = np.arange(70, dtype=np.float32)[:, None, None]
    lead = np.arange(6, dtype=np.float32)[None, :, None]
    baseline[..., support] = spatial[None, None] + 0.002 * case + 0.03 * lead
    truth = baseline.copy()
    truth[..., support] += (
        0.15
        + 0.02
        * np.sin(np.linspace(0.0, 3.0 * np.pi, 171, dtype=np.float32))[None, None]
    )
    climatology = baseline.copy()
    climatology[..., support] *= np.float32(0.85)
    raw = baseline.copy()
    raw[..., support] *= np.float32(1.05)
    residual = np.zeros(shape, dtype=np.float32)

    np.savez_compressed(
        run / "metrics" / "validation_outputs.npz",
        initializations=_dates(include_2020=include_2020),
        truth=truth,
        climatology=climatology,
        area_weights=weights,
        raw_fuxi=raw,
        log_bias=baseline,
        scratch=baseline,
        global_pretrained=baseline,
        scratch_standardized_residual=residual,
        global_pretrained_standardized_residual=residual,
    )
    np.savez_compressed(
        run / "models" / "log_bias_anchor.npz",
        lead_month_residual=np.zeros((6, 12, 27, 27), dtype=np.float32),
        shrinkage=np.float32(10.0),
        target_scale=np.ones(6, dtype=np.float32),
    )
    required = [
        "metrics/validation_outputs.npz",
        "models/log_bias_anchor.npz",
    ]
    for configuration in ("scratch", "global_pretrained"):
        for seed in followup.SEEDS:
            relative = f"models/{configuration}/seed_{seed}/validation_residual.npy"
            path = run / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, residual)
            required.append(relative)

    artifacts = {
        relative: followup.sha256_file(run / relative) for relative in required
    }
    manifest = {
        "schema_version": 1,
        "status": "complete_validation_only",
        "mode": "full",
        "smoke": False,
        "scientific_eligible": True,
        "test_predictions_created": False,
        "experiment": "matched_scratch_vs_global_pretraining",
        "split": {
            "train_initialization_years": list(range(2002, 2018)),
            "validation_initialization_years": [2018, 2019],
            "sealed_initialization_years": list(range(2020, 2026)),
            "later_year_predictions_created": False,
            "later_year_metrics_computed": False,
        },
        "training": {"seeds": [42, 43, 44]},
        "data": {
            "loaded_initialization_years": list(range(2002, 2020)),
            "maximum_loaded_initialization_year": 2019,
            "sealed_initialization_years_opened": False,
            "full_split_counts": {"train": 560, "validation": 70, "test": 0},
            "effective_validation_cases": 70,
            "support_cells": 171,
            "initialization_date_max": "2019-09-11",
            "verification_target_date_max": "2019-10-22",
            "observation_date_max": "2019-12-31",
        },
        "artifacts": artifacts,
    }
    (run / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return run


def test_anchor_mean_projection_is_nonnegative_and_closes_each_case_lead() -> None:
    baseline = np.asarray(
        [
            [
                [[0.0, 1.0], [2.0, np.nan]],
                [[1.0, 3.0], [0.5, np.nan]],
            ],
            [
                [[4.0, 0.0], [1.0, np.nan]],
                [[2.0, 2.5], [3.0, np.nan]],
            ],
        ],
        dtype=np.float32,
    )
    residual = np.asarray(
        [
            [
                [[0.4, -0.1], [0.2, 999.0]],
                [[-0.3, 0.2], [0.1, 999.0]],
            ],
            [
                [[0.1, 0.2], [-0.2, 999.0]],
                [[0.3, -0.4], [0.2, 999.0]],
            ],
        ],
        dtype=np.float32,
    )
    weights = np.asarray([[1.0, 2.0], [3.0, 0.0]], dtype=np.float64)
    valid = np.broadcast_to(weights > 0.0, baseline.shape)

    result = followup.preserve_anchor_area_mean(
        baseline,
        residual,
        np.asarray([0.7, 1.2], dtype=np.float32),
        weights,
        valid_mask=valid,
    )

    assert np.isfinite(result.prediction[valid]).all()
    assert np.all(result.prediction[valid] >= 0.0)
    assert float(result.absolute_closure_error.max()) <= followup.MEAN_CLOSURE_TOLERANCE
    np.testing.assert_allclose(
        result.projected_mean,
        result.anchor_mean,
        rtol=0.0,
        atol=followup.MEAN_CLOSURE_TOLERANCE,
    )


def test_zero_residual_projection_is_exact_anchor_identity() -> None:
    baseline = np.asarray([[[[0.0, 1.0], [4.0, np.nan]]]], dtype=np.float32)
    residual = np.asarray([[[[0.0, 0.0], [0.0, 999.0]]]], dtype=np.float32)
    weights = np.asarray([[1.0, 2.0], [3.0, 0.0]], dtype=np.float64)
    valid = np.broadcast_to(weights > 0.0, baseline.shape)

    result = followup.preserve_anchor_area_mean(
        baseline,
        residual,
        np.ones(1, dtype=np.float32),
        weights,
        valid_mask=valid,
    )

    assert result.log_offset.item() == 0.0
    assert result.zero_residual_identity.item()
    assert np.array_equal(result.prediction[valid], baseline[valid])
    assert np.array_equal(
        result.adjusted_standardized_residual[valid], np.zeros(3, dtype=np.float32)
    )


def test_year_stratified_plan_is_deterministic_circular_and_never_crosses_year() -> (
    None
):
    dates = np.concatenate(
        (
            np.datetime64("2018-06-01", "D") + np.arange(5) * np.timedelta64(3, "D"),
            np.datetime64("2019-06-01", "D") + np.arange(5) * np.timedelta64(3, "D"),
        )
    )
    first = followup.year_stratified_circular_block_indices(
        dates,
        draws=100,
        block_length=3,
        seed=9,
        expected_cases_per_year=5,
    )
    second = followup.year_stratified_circular_block_indices(
        dates,
        draws=100,
        block_length=3,
        seed=9,
        expected_cases_per_year=5,
    )

    np.testing.assert_array_equal(first.indices, second.indices)
    years = pd.DatetimeIndex(dates).year.to_numpy()
    for year in (2018, 2019):
        segment = first.indices[:, first.year_slices[year]]
        assert np.all(years[segment] == year)
        positions = first.year_positions[year]
        lookup = np.full(len(dates), -1, dtype=np.int64)
        lookup[positions] = np.arange(len(positions))
        local = lookup[segment]
        for start in range(0, local.shape[1], 3):
            block = local[:, start : min(start + 3, local.shape[1])]
            if block.shape[1] > 1:
                assert np.all(np.diff(block, axis=1) % len(positions) == 1)
    assert np.any((first.indices[:, :-1] == 4) & (first.indices[:, 1:] == 0)) or np.any(
        (first.indices[:, :-1] == 9) & (first.indices[:, 1:] == 5)
    )


def test_circular_plan_has_approximately_unit_per_date_multiplicity() -> None:
    dates = np.concatenate(
        (
            np.datetime64("2018-06-01", "D") + np.arange(35),
            np.datetime64("2019-06-01", "D") + np.arange(35),
        )
    )
    plan = followup.year_stratified_circular_block_indices(
        dates,
        draws=10_000,
        block_length=13,
        seed=20_260_822,
    )

    assert plan.mean_multiplicity.mean() == pytest.approx(1.0, abs=1.0e-12)
    assert plan.maximum_absolute_multiplicity_deviation < 0.06
    diagnostic = followup.bootstrap_date_multiplicity(plan, dates)
    assert len(diagnostic) == 70
    assert diagnostic.groupby("year").size().to_dict() == {2018: 35, 2019: 35}


def test_paired_bootstrap_recovers_constant_effect_with_all_leads_attached() -> None:
    dates = np.concatenate(
        (
            np.datetime64("2018-06-01", "D") + np.arange(4),
            np.datetime64("2019-06-01", "D") + np.arange(4),
        )
    ).astype("datetime64[D]")
    rows = []
    values = {
        "candidate": {"rmse": 1.0, "mae": 0.5, "bias": 0.2, "acc": 0.6},
        "reference": {"rmse": 2.0, "mae": 1.0, "bias": 0.5, "acc": 0.4},
    }
    for configuration, metrics in values.items():
        for date in dates:
            for lead in range(1, 7):
                rows.append(
                    {
                        "configuration": configuration,
                        "member": "ensemble",
                        "case_id": date,
                        "lead": lead,
                        "region": "india",
                        "season": "ALL",
                        **metrics,
                    }
                )
    case_metrics = pd.DataFrame(rows)
    plan = followup.year_stratified_circular_block_indices(
        dates,
        draws=50,
        block_length=2,
        seed=5,
        expected_cases_per_year=4,
    )

    effects = followup.paired_bootstrap_effects(
        case_metrics,
        dates,
        plan,
        (followup.ComparisonSpec("candidate", "reference", "ensemble", "ensemble"),),
    )

    assert set(effects.scope) == {
        "W1-W6",
        "W1",
        "W2",
        "W3",
        "W4",
        "W5",
        "W6",
        "2018",
        "2019",
    }
    expected = {"rmse": 1.0, "mae": 0.5, "bias": 0.3, "acc": 0.2}
    for metric, value in expected.items():
        selected = effects.loc[effects.metric.eq(metric)]
        np.testing.assert_allclose(selected.effect, value)
        np.testing.assert_allclose(selected.bootstrap_mean_effect, value)
        np.testing.assert_allclose(selected.ci_lower_2p5, value)
        np.testing.assert_allclose(selected.ci_upper_97p5, value)
        assert selected.probability_favourable.eq(1.0).all()
    pooled = effects.loc[effects.scope.eq("W1-W6")]
    assert pooled.case_leads.eq(8 * 6).all()


def test_source_validation_rejects_hash_mismatch_and_2020_dates(tmp_path: Path) -> None:
    valid = _synthetic_source(tmp_path / "valid")
    bundle = followup.load_source_bundle(valid)
    assert bundle.initializations.shape == (70,)

    residual_path = valid / "models" / "scratch" / "seed_42" / "validation_residual.npy"
    residual = np.load(residual_path)
    residual.flat[0] = 1.0
    np.save(residual_path, residual)
    with pytest.raises(followup.FollowupContractError, match="hash mismatch"):
        followup.load_source_bundle(valid)

    later = _synthetic_source(tmp_path / "later", include_2020=True)
    with pytest.raises(followup.FollowupContractError, match="35 starts"):
        followup.load_source_bundle(later)


def test_source_root_requires_exactly_one_completed_full_child(tmp_path: Path) -> None:
    execution_root = tmp_path / "execution"
    execution_root.mkdir()
    with pytest.raises(followup.FollowupContractError, match="found 0"):
        followup.resolve_source_run(execution_root)

    completed = execution_root / "full_one"
    completed.mkdir()
    (completed / "manifest.json").write_text(
        json.dumps(
            {
                "status": "complete_validation_only",
                "mode": "full",
                "smoke": False,
            }
        ),
        encoding="utf-8",
    )
    failed = execution_root / "full_failed.failed"
    failed.mkdir()
    (failed / "manifest.json").write_text(
        json.dumps({"status": "failed", "mode": "full", "smoke": False}),
        encoding="utf-8",
    )
    assert followup.resolve_source_run(execution_root) == completed.resolve()

    second = execution_root / "full_two"
    second.mkdir()
    (second / "manifest.json").write_text(
        json.dumps(
            {
                "status": "complete_validation_only",
                "mode": "full",
                "smoke": False,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(followup.FollowupContractError, match="found 2"):
        followup.resolve_source_run(execution_root)


def test_alternate_bootstrap_settings_require_explicit_noncanonical_label(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        followup.FollowupContractError, match="explicit --noncanonical-smoke"
    ):
        followup.run_analysis(
            tmp_path / "unused_source",
            tmp_path / "result",
            bootstrap_draws=8,
            block_length=2,
            bootstrap_seed=17,
        )
    with pytest.raises(
        followup.FollowupContractError, match="output name must contain"
    ):
        followup.run_analysis(
            tmp_path / "unused_source",
            tmp_path / "result",
            bootstrap_draws=8,
            block_length=2,
            bootstrap_seed=17,
            noncanonical_smoke=True,
        )


def test_full_artifact_only_run_publishes_atomically_without_touching_source(
    tmp_path: Path,
) -> None:
    source = _synthetic_source(tmp_path)
    source_manifest = source / "manifest.json"
    before = followup.sha256_file(source_manifest)
    output = tmp_path / "followup_noncanonical_smoke_result"

    published = followup.run_analysis(
        source,
        output,
        bootstrap_draws=8,
        block_length=2,
        bootstrap_seed=17,
        noncanonical_smoke=True,
    )

    assert published == output.resolve()
    assert followup.sha256_file(source_manifest) == before
    assert not list(tmp_path.glob(".followup_noncanonical_smoke_result.staging-*"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete_posthoc_validation_only_noncanonical_smoke"
    assert manifest["canonical_artifact"] is False
    assert manifest["execution_tier"] == "explicit_noncanonical_smoke"
    assert manifest["scientific_eligible"] is False
    assert manifest["raw_archives_opened"] is False
    assert manifest["later_year_predictions_created"] is False
    assert manifest["data"]["initialization_years"] == [2018, 2019]
    assert manifest["bootstrap"]["year_boundary_crossed"] is False
    assert manifest["bootstrap"]["method"] == (
        "paired year-stratified circular moving blocks"
    )
    assert manifest["bootstrap"]["mean_date_multiplicity"] == pytest.approx(1.0)
    assert manifest["projection"]["exact_zero_residual_identity"] is True
    for relative, expected_hash in manifest["artifacts"].items():
        assert followup.sha256_file(output / relative) == expected_hash
    diagnostics = pd.read_csv(
        output / "metrics" / "area_mean_projection_diagnostics.csv"
    )
    assert diagnostics.absolute_closure_error.max() <= followup.MEAN_CLOSURE_TOLERANCE
    assert diagnostics.zero_residual_identity.all()


def test_cpu_launcher_uses_an_account_accessible_safe_partition() -> None:
    launcher = (
        Path(__file__).resolve().parents[1]
        / "slurm"
        / "evaluate_global_pretraining_followups.sbatch"
    ).read_text(encoding="utf-8")

    assert "#SBATCH --partition=gpu\n" in launcher
    assert "#SBATCH --partition=iiser\n" not in launcher
    assert "#SBATCH --gres=gpu" not in launcher
    assert "#SBATCH --exclude=cn2,cn3,cn4,cn15,cn16,cn17\n" in launcher
