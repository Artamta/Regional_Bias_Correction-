from __future__ import annotations

from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "evaluate"
    / "evaluate_locked_physical_hindcast.py"
)
SPEC = importlib.util.spec_from_file_location(
    "evaluate_locked_physical_hindcast", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
locked = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = locked
SPEC.loader.exec_module(locked)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _synthetic_confirmation(root: Path, *, qualifying: str | None) -> None:
    (root / "metrics").mkdir(parents=True)
    (root / "models").mkdir()
    (root / "normalization.json").write_text("{}\n", encoding="utf-8")
    np.savez_compressed(
        root / "models" / "log_bias_anchor.npz",
        lead_month_residual=np.zeros((6, 12, 27, 27), dtype=np.float32),
        shrinkage=np.float32(10.0),
        target_scale=np.ones(6, dtype=np.float32),
    )
    candidates = [
        asdict(locked.sweep.CANDIDATE_BY_NAME[name])
        for name in locked.EXPECTED_CONFIGURATIONS
    ]
    (root / "code").mkdir()
    for frozen, live in locked.frozen_live_code_contract(root):
        frozen.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(live, frozen)
    rows = []
    ordered = (
        [qualifying]
        + [name for name in locked.EXPECTED_CONFIGURATIONS if name != qualifying]
        if qualifying is not None
        else list(locked.EXPECTED_CONFIGURATIONS)
    )
    for rank, name in enumerate(ordered, start=1):
        rows.append(
            {
                "configuration": name,
                "rank": rank,
                "qualifies": bool(name == qualifying),
            }
        )
    pd.DataFrame(rows).to_csv(
        root / "metrics" / "ranked_configurations.csv", index=False
    )
    (root / "metrics" / "validation_case_metrics.csv").write_text(
        "configuration,case_id,lead,rmse\n", encoding="utf-8"
    )
    for candidate in candidates:
        name = candidate["name"]
        for seed in locked.EXPECTED_SEEDS:
            checkpoint = root / "models" / name / f"seed_{seed}" / "checkpoints" / "best.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(f"{name}-{seed}".encode("ascii"))
            record_path = checkpoint.parents[1] / "run_record.json"
            _write_json(
                record_path,
                {
                    "status": "complete",
                    "candidate": candidate,
                    "seed": seed,
                    "checkpoint": str(checkpoint.relative_to(root)),
                    "checkpoint_sha256": locked.sha256_file(checkpoint),
                    "best_epoch_zero_based": 3,
                    "best_validation_loss": 0.5,
                },
            )
    manifest = {
            "status": "complete",
            "smoke": False,
            "training_mode": "full",
            "test_predictions_created": False,
            "train_years": list(locked.TRAIN_YEARS),
            "validation_years": list(locked.VALIDATION_YEARS),
            "split_counts": {"train": 560, "validation": 70, "test": 70},
            "seeds": list(locked.EXPECTED_SEEDS),
            "reference_configuration": locked.EXPECTED_REFERENCE,
            "loss_coefficients": locked.sweep.LOSS_COEFFICIENTS,
            "lead_weights": list(locked.sweep.LEAD_WEIGHTS),
            "minimum_loss_improvement": locked.EXPECTED_MINIMUM_LOSS_IMPROVEMENT,
            "candidates": candidates,
            "physical_predictors_loaded": True,
            "artifacts": {},
        }
    manifest["artifacts"] = {
        str(path.relative_to(root)): locked.sha256_file(path)
        for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    _write_json(root / "manifest.json", manifest)


def test_freeze_is_validation_only_and_hash_locks_checkpoints(tmp_path: Path) -> None:
    sweep_root = tmp_path / "confirm"
    _synthetic_confirmation(sweep_root, qualifying="physical_tcwv")
    selection_path = locked.freeze_validation_selection(
        sweep_root, tmp_path / "selection.json"
    )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    assert selection["selected_configuration"] == "physical_tcwv"
    assert selection["test_data_accessed_by_freeze"] is False
    assert selection["test_metrics_used_for_selection"] is False
    assert [item["seed"] for item in selection["checkpoints"]] == [42, 43, 44]
    locked.validate_frozen_selection(selection_path)

    checkpoint = sweep_root / selection["checkpoints"][0]["relative_path"]
    checkpoint.write_bytes(b"tampered")
    with pytest.raises(locked.EvaluationContractError, match="checkpoint"):
        locked.validate_frozen_selection(selection_path)


def test_freeze_falls_back_to_predeclared_control_when_no_model_qualifies(
    tmp_path: Path,
) -> None:
    sweep_root = tmp_path / "confirm"
    _synthetic_confirmation(sweep_root, qualifying=None)
    selection_path = locked.freeze_validation_selection(
        sweep_root, tmp_path / "selection.json"
    )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    assert selection["selected_configuration"] == locked.EXPECTED_REFERENCE
    assert "retain" in selection["decision"]


def test_validation_recomputes_the_frozen_selection_rule(tmp_path: Path) -> None:
    sweep_root = tmp_path / "confirm"
    _synthetic_confirmation(sweep_root, qualifying="physical_tcwv")
    selection_path = locked.freeze_validation_selection(
        sweep_root, tmp_path / "selection.json"
    )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["selected_configuration"] = "physical_control"
    selection["selected_candidate"] = asdict(
        locked.sweep.CANDIDATE_BY_NAME["physical_control"]
    )
    selection_path.write_text(json.dumps(selection), encoding="utf-8")

    with pytest.raises(locked.EvaluationContractError, match="predeclared"):
        locked.validate_frozen_selection(selection_path)


def test_selection_rejects_noncanonical_candidate_payload(tmp_path: Path) -> None:
    sweep_root = tmp_path / "confirm"
    _synthetic_confirmation(sweep_root, qualifying="physical_tcwv")
    selection_path = locked.freeze_validation_selection(
        sweep_root, tmp_path / "selection.json"
    )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["selected_candidate"]["dropout"] = 0.999
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    with pytest.raises(locked.EvaluationContractError, match="candidate"):
        locked.validate_frozen_selection(selection_path)


def test_confirmation_rejects_noncanonical_run_record(tmp_path: Path) -> None:
    sweep_root = tmp_path / "confirm"
    _synthetic_confirmation(sweep_root, qualifying="physical_tcwv")
    record = sweep_root / "models" / "physical_tcwv" / "seed_42" / "run_record.json"
    payload = json.loads(record.read_text(encoding="utf-8"))
    payload["candidate"]["dropout"] = 0.999
    _write_json(record, payload)
    with pytest.raises(locked.EvaluationContractError, match="run-record candidate"):
        locked.freeze_validation_selection(sweep_root, tmp_path / "selection.json")


def test_confirmation_rejects_frozen_training_code_drift(tmp_path: Path) -> None:
    sweep_root = tmp_path / "confirm"
    _synthetic_confirmation(sweep_root, qualifying="physical_tcwv")
    frozen, _ = next(iter(locked.frozen_live_code_contract(sweep_root)))
    frozen.write_text(
        frozen.read_text(encoding="utf-8") + "\n# synthetic post-run drift\n",
        encoding="utf-8",
    )
    manifest_path = sweep_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    relative = str(frozen.relative_to(sweep_root))
    manifest["artifacts"][relative] = locked.sha256_file(frozen)
    _write_json(manifest_path, manifest)

    with pytest.raises(
        locked.EvaluationContractError, match="live implementation differs"
    ):
        locked.freeze_validation_selection(sweep_root, tmp_path / "selection.json")


def test_forecast_preflight_failure_occurs_before_target_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sweep_root = tmp_path / "confirm"
    _synthetic_confirmation(sweep_root, qualifying="physical_tcwv")
    selection_path = locked.freeze_validation_selection(
        sweep_root, tmp_path / "selection.json"
    )
    target_called = False

    def fail_preflight(*args: object, **kwargs: object) -> None:
        raise RuntimeError("forecast-only gate")

    def forbidden_target(*args: object, **kwargs: object) -> None:
        nonlocal target_called
        target_called = True
        raise AssertionError("target loader must not run")

    monkeypatch.setattr(locked, "forecast_only_preflight", fail_preflight)
    monkeypatch.setattr(locked, "load_frozen_test_data", forbidden_target)
    output = tmp_path / "evaluation"
    args = SimpleNamespace(
        selection_manifest=selection_path,
        output=output,
        physical_cache=None,
        device="cuda:0",
        batch_size=32,
        bootstrap_replicates=2000,
        bootstrap_block_length=13,
        bootstrap_seed=9,
        fdr_q=0.05,
        india_boundary=locked.DEFAULT_INDIA_BOUNDARY,
    )
    with pytest.raises(RuntimeError, match="forecast-only gate"):
        locked.run_evaluation(args)
    assert not target_called
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["test_target_accessed"] is False
    assert manifest["test_years"] == [2020, 2021]
    assert manifest["evaluation_role"] == (
        "exploratory_reused_hindcast_evaluation"
    )
    assert manifest["evaluation_scope"] == (
        "2020-2021 exploratory/reused hindcast test; not independent confirmation"
    )
    assert manifest["genuine_independent_test"] is False
    assert manifest["reused_test_period"] is True
    assert manifest["selection_locked_before_target_access"] is True
    assert manifest["selection_locked_before_test"] is True
    assert manifest["model_selection_locked"] is True
    assert manifest["test_used_for_selection"] is False
    assert manifest["parameter_updates"] == 0
    assert manifest["selected_configuration"] == "physical_tcwv"


def test_auto_or_cpu_device_is_refused_before_cuda_probe() -> None:
    with pytest.raises(locked.EvaluationContractError, match="explicit CUDA"):
        locked._explicit_cuda_device("auto")
    with pytest.raises(locked.EvaluationContractError, match="explicit CUDA"):
        locked._explicit_cuda_device("cpu")


def test_locked_physical_cache_contract_is_exploratory_v2() -> None:
    import fuxi_physical_postselection_cache as postselection

    assert locked.EXPECTED_POSTSELECTION_PHYSICAL_CACHE.name.endswith(
        "exploratory_v2.npz"
    )
    assert locked.EXPECTED_POSTSELECTION_PHYSICAL_CACHE_SHA256 == (
        "1474b504bd72d155ae5876c34e7f376a1ed3e5880c8da5ceb6ca9f49458f764b"
    )
    assert postselection.DEFAULT_POSTSELECTION_CACHE.resolve() == (
        locked.EXPECTED_POSTSELECTION_PHYSICAL_CACHE
    )
    assert postselection.CACHE_SCHEMA_NAME == (
        locked.EXPECTED_POSTSELECTION_PHYSICAL_SCHEMA_NAME
    )
    assert postselection.CACHE_SCHEMA_VERSION == (
        locked.EXPECTED_POSTSELECTION_PHYSICAL_SCHEMA_VERSION
    )
    assert postselection.EVALUATION_ROLE == locked.EVALUATION_ROLE


def test_negative_bootstrap_seed_is_refused_before_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluation_called = False

    def forbidden_evaluation(*args: object, **kwargs: object) -> None:
        nonlocal evaluation_called
        evaluation_called = True

    monkeypatch.setattr(locked, "run_evaluation", forbidden_evaluation)
    with pytest.raises(ValueError, match="bootstrap-seed"):
        locked.main(
            [
                "evaluate",
                "selection.json",
                "evaluation",
                "--bootstrap-seed",
                "-1",
            ]
        )
    assert not evaluation_called


def _test_dates() -> np.ndarray:
    first = pd.date_range("2020-06-02", periods=35, freq="4D")
    second = pd.date_range("2021-06-02", periods=35, freq="4D")
    return np.asarray([*first, *second], dtype="datetime64[D]")


def test_two_stage_moving_blocks_are_deterministic_and_retain_cases() -> None:
    dates = _test_dates()
    first = locked.two_stage_moving_block_draws(
        dates, replicates=40, block_length=13, seed=9
    )
    second = locked.two_stage_moving_block_draws(
        dates, replicates=40, block_length=13, seed=9
    )
    assert np.array_equal(first, second)
    assert first.shape == (40, 70)
    assert np.all((first >= 0) & (first < 70))


def _case_metrics() -> pd.DataFrame:
    rows = []
    for method in locked.METHODS:
        for case_index, date in enumerate(_test_dates()):
            for lead in range(1, 7):
                if method == "raw_fuxi":
                    rmse, acc, bias = 4.0 + lead / 10, 0.20, 1.0
                elif method == "log_bias":
                    rmse, acc, bias = 3.8 + lead / 10, 0.23, 0.8
                else:
                    rmse, acc, bias = 3.0 + lead / 10, 0.35, 0.2
                rows.append(
                    {
                        "method": method,
                        "case_id": date,
                        "lead": lead,
                        "rmse": rmse + case_index * 1.0e-4,
                        "acc": acc + case_index * 1.0e-5,
                        "bias": bias + case_index * 1.0e-5,
                    }
                )
    return pd.DataFrame(rows)


def test_paired_bootstrap_keeps_all_leads_and_detects_clear_improvement() -> None:
    draws = locked.two_stage_moving_block_draws(
        _test_dates(), replicates=200, block_length=13, seed=11
    )
    result = locked.paired_metric_bootstrap(_case_metrics(), draws)
    assert set(result.scope) == {
        "W1",
        "W2",
        "W3",
        "W4",
        "W5",
        "W6",
        "ALL_WEEKS",
    }
    assert len(result) == 2 * 3 * 7
    raw = result.loc[result.baseline.eq("raw_fuxi")]
    assert (raw.effect_positive_is_better > 0.0).all()
    assert raw.bootstrap_supported_improvement.all()
    assert "centered_block_null_two_sided_p" in raw
    assert "two_sided_p" not in raw


def test_benjamini_hochberg_is_monotone_and_nan_safe() -> None:
    p = np.asarray([0.001, 0.02, 0.5, np.nan])
    q = locked.benjamini_hochberg(p)
    assert np.isnan(q[-1])
    assert np.all(q[:3] >= p[:3])
    assert q[0] <= q[1] <= q[2]


class _Forecast:
    pass


def _normalization() -> dict:
    offsets = locked.experiment.OFFSETS_DAYS
    input_channels = [
        "log_fuxi_mean",
        "log_fuxi_spread",
        "log_imd_calendar_climatology",
        "latitude",
        "longitude",
        "season_sin",
        "season_cos",
        "lead_week",
        "support",
        "explicit_log_fuxi_minus_imd_climatology",
        "fuxi_t2m_weekly",
        *[f"imd_climatology_offset_{offset:+d}d" for offset in offsets],
        *[
            f"fuxi_minus_imd_climatology_offset_{offset:+d}d"
            for offset in offsets
        ],
        *locked.PHYSICAL_NAMES,
    ]
    result = {"input_channels": input_channels}
    for name in (
        "log_fuxi_mean",
        "log_fuxi_spread",
        "log_imd_climatology",
        "explicit_log_fuxi_anomaly",
        "fuxi_t2m_weekly",
        *locked.PHYSICAL_NAMES,
    ):
        result[name] = {
            "mean_by_lead": [0.0] * 6,
            "std_by_lead": [1.0] * 6,
        }
    return result


def test_frozen_feature_builder_applies_saved_stats_without_refitting() -> None:
    forecast = _Forecast()
    forecast.initializations = np.asarray(["2020-06-02", "2021-06-02"], dtype="datetime64[D]")
    forecast.valid_dates = locked.sweep.base.derive_valid_dates(
        forecast.initializations
    )
    forecast.ensemble_mean = np.full((2, 6, 27, 27), 3.0, dtype=np.float32)
    forecast.ensemble_spread = np.ones((2, 6, 27, 27), dtype=np.float32)
    forecast.latitude = locked.EXPECTED_LATITUDE.copy()
    forecast.longitude = locked.EXPECTED_LONGITUDE.copy()
    weights = np.zeros((27, 27), dtype=np.float64)
    weights.reshape(-1)[:171] = 1.0
    support = weights > 0.0
    climatology_daily = np.full((366, 27, 27), np.nan, dtype=np.float32)
    climatology_daily[:, support] = 2.0
    weekly_climatology = np.full((2, 6, 27, 27), np.nan, dtype=np.float32)
    weekly_climatology[..., support] = 2.0
    t2m = np.full((2, 6, 27, 27), 280.0, dtype=np.float32)
    physical = {
        name: np.full((2, 6, 27, 27), index + 1.0, dtype=np.float32)
        for index, name in enumerate(locked.PHYSICAL_NAMES)
    }
    features = locked.build_frozen_test_features(
        forecast,
        weekly_climatology,
        climatology_daily,
        t2m,
        physical,
        weights,
        _normalization(),
    )
    assert features.shape == (2, 6, 38, 27, 27)
    assert np.allclose(features[:, :, 0], np.log1p(3.0))
    assert np.all(features[:, :, 2, ~support] == 0.0)
    assert np.allclose(features[:, :, -1], 9.0)


def test_spatial_field_bootstrap_uses_centred_null_bh_support() -> None:
    truth = np.zeros((70, 6, 27, 27), dtype=np.float32)
    raw = np.full_like(truth, 2.0)
    corrected = np.full_like(truth, 1.0)
    weights = np.ones((27, 27), dtype=np.float64)
    draws = locked.two_stage_moving_block_draws(
        _test_dates(), replicates=200, block_length=13, seed=13
    )
    result = locked.spatial_statistics(
        truth, raw, corrected, weights, draws, fdr_q=0.05
    )
    assert np.allclose(result["rmse_skill_pct"], 50.0)
    assert result["bootstrap_supported_rmse_improvement_fdr"].all()
    assert "mse_difference_centered_block_null_p" in result
