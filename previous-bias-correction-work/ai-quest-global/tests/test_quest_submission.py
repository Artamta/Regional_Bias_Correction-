"""Focused tests for the offline AI Weather Quest submission preview."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import predict
import quest_submission as submission


FINGERPRINT = "a" * 64
ISSUE_DATE = "2020-01-02"  # Thursday; historical on purpose.
CHECKPOINT_BYTES = b"offline-test-checkpoint"
CHECKPOINT_SHA256 = hashlib.sha256(CHECKPOINT_BYTES).hexdigest()


def _model_probabilities() -> np.ndarray:
    values = np.zeros((1, 2, 5, 121, 240), dtype=np.float32)
    values[:, :, 2] = 1.0
    return values


def _prepared_batch(source: Path) -> predict.PreparedBatch:
    return predict.PreparedBatch(
        features=np.zeros((1, 2, 18, 1, 1), dtype=np.float32),
        p0=np.zeros((1, 2, 5, 1, 1), dtype=np.float32),
        target=None,
        latitude=predict.LATITUDE.copy(),
        longitude=predict.LONGITUDE.copy(),
        land_fraction=None,
        init_dates=(ISSUE_DATE,),
        source=source,
    )


def _install_fake_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    prepared_fingerprint: str = FINGERPRINT,
    checkpoint_fingerprint: str = FINGERPRINT,
) -> tuple[Path, Path]:
    case = tmp_path / "prepared.npz"
    np.savez(case, cache_contract_sha256=np.asarray(prepared_fingerprint))
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(CHECKPOINT_BYTES)

    def load_prepared(path: object, *, init_date: str) -> predict.PreparedBatch:
        assert Path(path) == case.resolve()
        assert init_date == ISSUE_DATE
        return _prepared_batch(case.resolve())

    monkeypatch.setattr(submission.predict, "load_prepared", load_prepared)
    monkeypatch.setattr(
        submission.predict,
        "load_checkpoint_model",
        lambda *args, **kwargs: (
            object(),
            {
                "normalization": {
                    "cache_contract_sha256": checkpoint_fingerprint,
                    "provenance": {
                        "official_era5_target": False,
                        "official_ai_weather_quest_validation": False,
                        "purpose": "exploratory_pretraining",
                        "notice": "not official Quest validation",
                        "fuxi_permission": "written FuXi permission required",
                    },
                },
                "metadata": {"model_name": "TPProbUNet"},
            },
        ),
    )
    monkeypatch.setattr(
        submission.predict,
        "run_model",
        lambda *args, **kwargs: _model_probabilities().copy(),
    )
    return case, checkpoint


def _build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    calibration_json: Path | None = None,
) -> submission.SubmissionBundle:
    case, checkpoint = _install_fake_inference(tmp_path, monkeypatch)
    return submission.build_offline_submission_bundle(
        case,
        checkpoint,
        tmp_path / "preview",
        init_date=ISSUE_DATE,
        team="TEAM",
        model="MODEL",
        originating_centre="TEAMzz_01",
        expver="TEAM_01",
        calibration_json=calibration_json,
        allow_historical=True,
        now_utc=datetime(2026, 8, 15, 12, tzinfo=timezone.utc),
    )


@pytest.mark.parametrize(
    ("period", "bounds", "start_days", "end_days"),
    [
        (1, "[18.0,25.0]", 18, 25),
        (2, "[25.0,32.0]", 25, 32),
    ],
)
def test_precipitation_dataarray_matches_ai_wq_329_contract(
    period: int, bounds: str, start_days: int, end_days: int
) -> None:
    probabilities = np.full((5, 121, 240), 0.2, dtype=np.float32)

    data = submission.create_precipitation_dataarray(
        probabilities,
        init_date="20260813",
        period=period,
        team="TEAM",
        model="MODEL",
        originating_centre="TEAMzz_01",
        expver="TEAM_01",
    )

    assert data.name is None
    assert data.dims == ("quintile", "latitude", "longitude")
    assert data.shape == (5, 121, 240)
    assert data.dtype == np.dtype(np.float64)
    np.testing.assert_array_equal(data.quintile, [0.2, 0.4, 0.6, 0.8, 1.0])
    np.testing.assert_array_equal(data.latitude, np.arange(90.0, -91.0, -1.5))
    np.testing.assert_array_equal(data.longitude, np.arange(0.0, 360.0, 1.5))
    issue = np.datetime64("2026-08-13T00:00:00", "ns")
    assert data.forecast_issue_date == issue
    assert data.forecast_period_start == issue + np.timedelta64(start_days, "D")
    assert data.forecast_period_end == issue + np.timedelta64(end_days, "D")
    assert np.isnan(data.height.item())
    assert data.attrs == {
        "standard_name": "Total precipitation probability",
        "cell_methods": "time: sum (interval: 24 hours)",
        "units": "1",
        "coordinates": "latitude longitude",
        "description": f"pr prediction from TEAM using MODEL for forecasting period {period}",
        "Conventions": "CF-1.6",
        "forecast_period_bounds_units": "days into forecast",
        "forecast_period_bounds": bounds,
        "shortName": "tp",
        "originating_centre": "TEAMzz_01",
        "expver": "TEAM_01",
        "teamname": "TEAM",
        "modelname": "MODEL",
    }
    assert data.latitude.attrs == {
        "units": "degrees_north",
        "long_name": "latitude",
        "standard_name": "latitude",
        "axis": "X",
    }
    assert data.longitude.attrs == {
        "units": "degrees_east",
        "long_name": "longitude",
        "standard_name": "longitude",
        "axis": "Y",
    }


def test_wrong_day_is_rejected_before_creating_outputs(tmp_path: Path) -> None:
    output = tmp_path / "preview"
    with pytest.raises(ValueError, match="not a Thursday"):
        submission.build_offline_submission_bundle(
            tmp_path / "unused.npz",
            tmp_path / "unused.pt",
            output,
            init_date="2026-08-12",
            team="TEAM",
            model="MODEL",
            originating_centre="ORIGIN",
            expver="EXPVER",
        )
    assert not output.exists()


def test_out_of_window_requires_explicit_historical_flag(tmp_path: Path) -> None:
    output = tmp_path / "preview"
    with pytest.raises(ValueError, match="--allow-historical"):
        submission.build_offline_submission_bundle(
            tmp_path / "unused.npz",
            tmp_path / "unused.pt",
            output,
            init_date=ISSUE_DATE,
            team="TEAM",
            model="MODEL",
            originating_centre="ORIGIN",
            expver="EXPVER",
            now_utc=datetime(2026, 8, 15, tzinfo=timezone.utc),
        )
    assert not output.exists()


def test_checkpoint_cache_fingerprint_mismatch_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case, checkpoint = _install_fake_inference(
        tmp_path,
        monkeypatch,
        prepared_fingerprint="a" * 64,
        checkpoint_fingerprint="b" * 64,
    )
    model_called = False

    def unexpected_model(*args: object, **kwargs: object) -> np.ndarray:
        nonlocal model_called
        model_called = True
        return _model_probabilities()

    monkeypatch.setattr(submission.predict, "run_model", unexpected_model)

    with pytest.raises(ValueError, match="cache_contract_sha256 differs"):
        submission.build_offline_submission_bundle(
            case,
            checkpoint,
            tmp_path / "preview",
            init_date=ISSUE_DATE,
            team="TEAM",
            model="MODEL",
            originating_centre="ORIGIN",
            expver="EXPVER",
            allow_historical=True,
        )
    assert model_called is False
    assert not (tmp_path / "preview").exists()


def test_calibration_for_a_different_checkpoint_is_rejected_before_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case, checkpoint = _install_fake_inference(tmp_path, monkeypatch)
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "schema_version": submission.CALIBRATION_SCHEMA_VERSION,
                "method": submission.CALIBRATION_METHOD,
                "alpha_scope": submission.CALIBRATION_ALPHA_SCOPE,
                "alphas": {"model": 0.25},
                "checkpoint_cache_contract_sha256": FINGERPRINT,
                "checkpoint_sha256": "b" * 64,
            }
        ),
        encoding="utf-8",
    )
    model_called = False

    def unexpected_model(*args: object, **kwargs: object) -> np.ndarray:
        nonlocal model_called
        model_called = True
        return _model_probabilities()

    monkeypatch.setattr(submission.predict, "run_model", unexpected_model)
    with pytest.raises(ValueError, match="checkpoint_sha256 differs"):
        submission.build_offline_submission_bundle(
            case,
            checkpoint,
            tmp_path / "preview",
            init_date=ISSUE_DATE,
            team="TEAM",
            model="MODEL",
            originating_centre="ORIGIN",
            expver="EXPVER",
            calibration_json=calibration,
            allow_historical=True,
        )
    assert model_called is False
    assert not (tmp_path / "preview").exists()


def test_unknown_calibration_method_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case, checkpoint = _install_fake_inference(tmp_path, monkeypatch)
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "schema_version": submission.CALIBRATION_SCHEMA_VERSION,
                "method": "temperature scaling",
                "alpha_scope": submission.CALIBRATION_ALPHA_SCOPE,
                "alphas": {"model": 0.25},
                "checkpoint_cache_contract_sha256": FINGERPRINT,
                "checkpoint_sha256": CHECKPOINT_SHA256,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Calibration method"):
        submission.build_offline_submission_bundle(
            case,
            checkpoint,
            tmp_path / "preview",
            init_date=ISSUE_DATE,
            team="TEAM",
            model="MODEL",
            originating_centre="ORIGIN",
            expver="EXPVER",
            calibration_json=calibration,
            allow_historical=True,
        )
    assert not (tmp_path / "preview").exists()


def test_calibrated_bundle_is_normalized_and_round_trips_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "schema_version": submission.CALIBRATION_SCHEMA_VERSION,
                "method": submission.CALIBRATION_METHOD,
                "alpha_scope": submission.CALIBRATION_ALPHA_SCOPE,
                "alphas": {"model": 0.25},
                "checkpoint_cache_contract_sha256": FINGERPRINT,
                "checkpoint_sha256": CHECKPOINT_SHA256,
                "fit_years": [2019],
                "status": "exploratory validation fit",
            }
        ),
        encoding="utf-8",
    )

    bundle = _build(tmp_path, monkeypatch, calibration_json=calibration)

    assert [path.name for path in bundle.paths] == [
        "pr_20200102_p1_TEAM_MODEL.nc",
        "pr_20200102_p2_TEAM_MODEL.nc",
        "pr_20200102_TEAM_MODEL_manifest.json",
    ]
    expected = np.full((5, 121, 240), 0.15, dtype=np.float32)
    expected[2] = np.float32(0.4)
    expected = expected.astype(np.float64)
    for period, path in enumerate(bundle.paths[:2], start=1):
        with xr.open_dataarray(path) as opened:
            values = opened.values
            assert opened.dims == ("quintile", "latitude", "longitude")
            assert opened.dtype == np.dtype(np.float64)
            assert opened.attrs["forecast_period_bounds"] == (
                "[18.0,25.0]" if period == 1 else "[25.0,32.0]"
            )
        np.testing.assert_array_equal(values, expected)
        np.testing.assert_allclose(values.sum(axis=0), 1.0, rtol=0.0, atol=1.0e-7)

    manifest = json.loads(bundle.manifest.read_text(encoding="utf-8"))
    assert manifest["ai_wq_package_contract"] == "3.29"
    assert manifest["upload_performed"] is False
    assert manifest["offline_only"] is True
    assert manifest["historical_or_out_of_window_preview"] is True
    assert manifest["official_era5_target"] is False
    assert manifest["fuxi_permission_notice"] == "written FuXi permission required"
    assert manifest["competition_use_authorized"] is False
    assert manifest["submission_ready"] is False
    assert manifest[
        "preview_not_accepted_without_registered_package_ids_and_current_window"
    ] is True
    assert manifest["calibration"]["applied"] is True
    assert manifest["calibration"]["alpha_model"] == pytest.approx(0.25)
    for record, path in zip(manifest["files"], bundle.paths[:2], strict=True):
        assert record["filename"] == path.name
        assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert record["maximum_probability_sum_error"] <= 1.0e-7


def test_publish_failure_restores_previous_bundle_and_cleans_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _build(tmp_path, monkeypatch)
    previous = {path: path.read_bytes() for path in bundle.paths}
    real_replace = submission.os.replace
    stage_publish_calls = 0

    def fail_second_stage_publish(source: object, destination: object) -> None:
        nonlocal stage_publish_calls
        if ".stage." in Path(source).name:
            stage_publish_calls += 1
            if stage_publish_calls == 2:
                raise OSError("simulated publish failure")
        real_replace(source, destination)

    monkeypatch.setattr(submission.os, "replace", fail_second_stage_publish)

    with pytest.raises(OSError, match="simulated publish failure"):
        submission.build_offline_submission_bundle(
            tmp_path / "prepared.npz",
            tmp_path / "best.pt",
            tmp_path / "preview",
            init_date=ISSUE_DATE,
            team="TEAM",
            model="MODEL",
            originating_centre="TEAMzz_01",
            expver="TEAM_01",
            allow_historical=True,
        )

    assert stage_publish_calls == 2
    assert {path: path.read_bytes() for path in bundle.paths} == previous
    assert sorted(path.name for path in (tmp_path / "preview").iterdir()) == sorted(
        path.name for path in bundle.paths
    )


@pytest.mark.parametrize("unsafe", ["../TEAM", "TEAM/MODEL", ".hidden", "bad name"])
def test_filename_components_cannot_escape_output_directory(unsafe: str) -> None:
    probabilities = np.full((5, 121, 240), 0.2, dtype=np.float32)
    with pytest.raises(ValueError, match="filename component|ASCII"):
        submission.create_precipitation_dataarray(
            probabilities,
            init_date="20260813",
            period=1,
            team=unsafe,
            model="MODEL",
            originating_centre="ORIGIN",
            expver="EXPVER",
        )
