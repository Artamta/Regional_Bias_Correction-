from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace
import sys
import types

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

import precip_contract as contract
import select_precip_candidate
from train_precip import TrainConfig, config_sha256
from debias import (
    DebiasPlusPlus,
    SeasonalDebiasPlusPlus,
    apply_uniform_gamma,
    fit_uniform_gamma,
    isotonic_cdf,
)
from precip_model import PrecipQuestAdapter, trainable_parameter_count
import submit_precip_issue
import build_precip_leader_handoff
import predict_precip_issue
import build_precip_climatology_proxy


def test_frozen_split_is_disjoint_17_train_2_validation_1_test_year():
    split = contract.SPLIT
    assert split.train_years == tuple(range(2002, 2019))
    assert split.validation_years == (2019, 2020)
    assert split.test_years == (2021,)
    all_years = split.train_years + split.validation_years + split.test_years
    assert len(all_years) == len(set(all_years)) == 20
    assert set(all_years) == set(range(2002, 2022))


def test_smoke_configuration_is_distinct_and_cannot_enter_selection(tmp_path):
    full = TrainConfig(smoke_cases=None)
    smoke = TrainConfig(smoke_cases=2)
    assert config_sha256(full, "a" * 64) != config_sha256(smoke, "a" * 64)

    run = tmp_path / "smoke"
    (run / "checkpoints").mkdir(parents=True)
    summary = {
        "status": "smoke_complete",
        "smoke_only": True,
        "configuration_sha256": "smoke-hash",
    }
    (run / "summary.json").write_text(json.dumps(summary))
    torch.save(
        {
            "configuration_sha256": "smoke-hash",
            "train_config": {
                "variant": "full",
                "width": 16,
                "seed": 42,
                "smoke_cases": 2,
            },
        },
        run / "checkpoints" / "best.pt",
    )
    with pytest.raises(ValueError, match="not selectable"):
        select_precip_candidate._neural_metadata(run)


def test_era5_week_alignment_uses_exact_28_post_start_accumulations():
    times = pd.date_range("2020-01-01", periods=40, freq="6h")
    values = np.arange(40, dtype=np.float32)[:, None, None]
    field = xr.DataArray(values, dims=("time", "lat", "lon"), coords={"time": times})
    field.attrs["units"] = "m"
    result = contract.aggregate_era5_six_hour_week(field, "2020-01-01")
    assert result.attrs["accumulation_count"] == 28
    np.testing.assert_allclose(result.item(), np.arange(1, 29).sum() * 1000.0)


def test_climatology_is_exactly_100_samples_and_strictly_prior():
    start = pd.Timestamp("2021-03-01")
    dates = contract.climatology_sample_starts(start)
    assert len(dates) == 100
    assert len(set(dates)) == 100
    assert max(dates) < start
    fields = np.arange(100, dtype=np.float32)[:, None, None]
    bounds = contract.climatology_boundaries(fields)
    np.testing.assert_allclose(bounds[:, 0, 0], np.quantile(np.arange(100), [0.2, 0.4, 0.6, 0.8]))


def test_official_climatology_loader_normalizes_grid_and_validates_order(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(predict_precip_issue, "GRID_SHAPE", (3, 4))
    path = tmp_path / "climatology.nc"
    values = np.arange(4, dtype=np.float32)[:, None, None] + np.zeros(
        (4, 3, 4), dtype=np.float32
    )
    xr.DataArray(
        values[:, ::-1],
        dims=("quantile", "latitude", "longitude"),
        coords={
            "quantile": [0.2, 0.4, 0.6, 0.8],
            "latitude": [-90.0, 0.0, 90.0],
            "longitude": [0.0, 1.5, 3.0, 4.5],
        },
    ).to_netcdf(path)
    result = predict_precip_issue._official_climatology_boundaries(path)
    assert result.shape == (4, 3, 4)
    np.testing.assert_array_equal(result[:, 0, 0], np.arange(4, dtype=np.float32))

    invalid = tmp_path / "invalid.nc"
    bad = values.copy()
    bad[2] = -1.0
    xr.DataArray(
        bad,
        dims=("quintile", "lat", "lon"),
        coords={
            "quintile": [0.2, 0.4, 0.6, 0.8],
            "lat": [90.0, 0.0, -90.0],
            "lon": [0.0, 1.5, 3.0, 4.5],
        },
    ).to_netcdf(invalid)
    with pytest.raises(ValueError, match="invalid ordered boundaries"):
        predict_precip_issue._official_climatology_boundaries(invalid)


def test_lagged_proxy_climatology_has_20_years_and_100_unique_samples():
    starts = build_precip_climatology_proxy.proxy_sample_starts(
        pd.Timestamp("2026-08-31"), 2023
    )
    assert len(starts) == len(set(starts)) == 100
    assert min(item.year for item in starts) == 2004
    assert max(item.year for item in starts) == 2023


def test_lagged_proxy_clips_tiny_negative_precipitation_artifacts():
    values = np.asarray([[-1.0e-4, 0.0, 2.0]], dtype=np.float32)
    result = build_precip_climatology_proxy.nonnegative_precipitation(values)
    np.testing.assert_array_equal(result, np.asarray([[0.0, 0.0, 2.0]], dtype=np.float32))


def test_boundary_equality_enters_upper_category_and_q80_zero_is_arid():
    boundaries = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32)[:, None, None]
    category, valid = contract.observation_categories(
        np.asarray([[2.0]], dtype=np.float32), boundaries
    )
    assert valid.item()
    assert category.item() == 2
    collapsed = np.ones((4, 1, 1), dtype=np.float32)
    category, valid = contract.observation_categories(
        np.asarray([[1.0]], dtype=np.float32), collapsed
    )
    assert valid.item()
    assert category.item() == 4
    arid = np.zeros((4, 1, 1), dtype=np.float32)
    category, valid = contract.observation_categories(
        np.asarray([[0.0]], dtype=np.float32), arid
    )
    assert not valid.item()
    assert category.item() == -1


def test_counts_are_unsmoothed_and_jeffreys_anchor_is_positive(monkeypatch):
    monkeypatch.setattr(contract, "N_MEMBERS", 51)
    members = np.arange(51, dtype=np.float32)[:, None, None]
    boundaries = np.asarray([10, 20, 30, 40], dtype=np.float32)[:, None, None]
    counts = contract.member_category_counts(members, boundaries)
    np.testing.assert_array_equal(counts[:, 0, 0], [10, 10, 10, 10, 11])
    p0 = contract.jeffreys_probabilities(counts)
    assert np.all(p0 > 0)
    np.testing.assert_allclose(p0.sum(axis=0), 1.0, atol=1e-7)


def test_fuxi_weekly_reduction_sums_tp_and_means_physics(monkeypatch):
    monkeypatch.setattr(contract, "GRID_SHAPE", (1, 1))
    daily = {
        name: np.ones((51, 42, 1, 1), dtype=np.float32)
        for name in ("tp", *contract.PHYSICAL_VARIABLES)
    }
    context, targets = contract.weekly_fuxi_fields(daily)
    np.testing.assert_allclose(context[:, :5, 0, 0], np.log1p(7 * 24))
    np.testing.assert_allclose(context[:, 5::2, 0, 0], 1.0)
    np.testing.assert_allclose(context[:, 6::2, 0, 0], 0.0)
    np.testing.assert_allclose(targets[:, :, 0, 0], 7 * 24)


def test_train_land_normalization_ignores_nontraining_and_ocean(monkeypatch):
    monkeypatch.setattr(contract, "GRID_SHAPE", (2, 2))
    land = np.asarray([[1, 0], [1, 0]], dtype=np.float32)
    first = np.zeros((6, 15, 2, 2), dtype=np.float32)
    second = np.full_like(first, 2.0)
    # Ocean values would dominate if the mask were not enforced.
    first[:, :, :, 1] = 1000.0
    second[:, :, :, 1] = 1000.0
    mean, std = contract.fit_train_land_normalization([first, second], land)
    np.testing.assert_allclose(mean, 1.0)
    np.testing.assert_allclose(std, 1.0)


def test_observation_coverage_cannot_change_inputs(monkeypatch):
    monkeypatch.setattr(contract, "GRID_SHAPE", (2, 3))
    context = np.zeros((6, 15, 2, 3), dtype=np.float32)
    target_quantiles = np.ones((2, 5, 2, 3), dtype=np.float32)
    p0 = np.full((2, 5, 2, 3), 0.2, dtype=np.float32)
    land = np.ones((2, 3), dtype=np.float32)
    mean = np.zeros((6, 15), dtype=np.float32)
    std = np.ones((6, 15), dtype=np.float32)
    before = contract.build_model_inputs(
        context, target_quantiles, p0, "2018-01-01", land, mean, std
    )
    # target_valid is intentionally absent from the feature API.
    target_valid = np.zeros((2, 2, 3), dtype=bool)
    target_valid[:] = True
    after = contract.build_model_inputs(
        context, target_quantiles, p0, "2018-01-01", land, mean, std
    )
    assert target_valid.all()
    np.testing.assert_array_equal(before[0], after[0])
    np.testing.assert_array_equal(before[1], after[1])


def test_isotonic_projection_is_valid_and_least_squares_for_simple_pool():
    raw = np.asarray([0.8, 0.2, 0.6, 0.5], dtype=np.float32)[:, None]
    projected = isotonic_cdf(raw)[:, 0]
    np.testing.assert_allclose(projected, [0.5, 0.5, 0.55, 0.55], atol=1e-7)
    assert np.all(np.diff(projected) >= 0)


def test_debias_and_calibration_return_valid_probabilities():
    rng = np.random.default_rng(12)
    raw = rng.dirichlet(np.ones(5), size=(8, 2, 2, 3)).transpose(0, 1, 4, 2, 3)
    target = rng.integers(0, 5, size=(8, 2, 2, 3), dtype=np.int8)
    valid = np.ones_like(target, dtype=bool)
    debias = DebiasPlusPlus.fit(raw, target, valid)
    corrected = debias.predict(raw)
    assert np.all(corrected >= 0)
    np.testing.assert_allclose(corrected.sum(axis=2), 1.0, atol=1e-6)
    gamma = fit_uniform_gamma(corrected, target, valid, np.ones((2, 3)))
    assert gamma.shape == (2,)
    assert np.all((gamma >= 0) & (gamma <= 1))
    calibrated = apply_uniform_gamma(corrected, gamma)
    np.testing.assert_allclose(calibrated.sum(axis=2), 1.0, atol=1e-6)


def test_seasonal_debias_uses_cyclic_training_windows():
    model = SeasonalDebiasPlusPlus.empty(1, 1)
    probability = np.full((2, 5, 1, 1), 0.2, dtype=np.float32)
    valid = np.ones((2, 1, 1), dtype=bool)
    dry = np.zeros((2, 1, 1), dtype=np.int8)
    wet = np.full((2, 1, 1), 4, dtype=np.int8)
    model.add_case(probability, dry, valid, np.asarray([365, 365]))
    model.add_case(probability, wet, valid, np.asarray([180, 180]))
    inputs = np.stack([probability, probability])
    corrected = model.predict(
        inputs,
        np.asarray([[0, 0], [180, 180]], dtype=np.int16),
        np.asarray([14, 14], dtype=np.int16),
    )
    assert corrected[0, 0, 0, 0, 0] > corrected[1, 0, 0, 0, 0]
    np.testing.assert_allclose(corrected.sum(axis=2), 1.0, atol=1e-6)


def _model_inputs(batch: int = 1):
    torch.manual_seed(7)
    context = torch.randn(batch, 6, 23, 8, 12)
    target = torch.randn(batch, 2, 18, 8, 12)
    p0 = torch.softmax(torch.randn(batch, 2, 5, 8, 12), dim=2)
    return context, target, p0


def test_model_contract_parameter_budget_and_identity_at_initialization():
    model = PrecipQuestAdapter(width=16, dropout=0.0, attention_dropout=0.0)
    context, target, p0 = _model_inputs()
    with torch.no_grad():
        result = model(context, target, p0)
    torch.testing.assert_close(result, p0, atol=2e-7, rtol=2e-7)
    assert 145_000 <= trainable_parameter_count(model) <= 160_000


def test_both_targets_receive_gradients_from_all_six_context_weeks():
    model = PrecipQuestAdapter(width=4, dropout=0.0, attention_dropout=0.0)
    with torch.no_grad():
        model.correction_head.weight.fill_(0.1)
    context, target, p0 = _model_inputs()
    context.requires_grad_()
    for lead in range(2):
        model.zero_grad(set_to_none=True)
        if context.grad is not None:
            context.grad.zero_()
        model(context, target, p0)[:, lead, 0].sum().backward(retain_graph=True)
        per_week = context.grad.abs().sum(dim=(0, 2, 3, 4))
        assert torch.all(per_week > 0), (lead, per_week)


def test_checkpoint_reload_is_exact():
    model = PrecipQuestAdapter(width=4, dropout=0.0, attention_dropout=0.0)
    context, target, p0 = _model_inputs()
    with torch.no_grad():
        model.correction_head.bias.copy_(torch.arange(5) / 10)
        expected = model(context, target, p0)
    payload = copy.deepcopy(model.state_dict())
    restored = PrecipQuestAdapter(width=4, dropout=0.0, attention_dropout=0.0)
    restored.load_state_dict(payload, strict=True)
    with torch.no_grad():
        actual = restored(context, target, p0)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_submission_uses_package_native_constructor_and_serializes_exactly(
    tmp_path, monkeypatch
):
    evidence = tmp_path / "permission.txt"
    evidence.write_text("FuXi competition permission granted")
    digest = hashlib.sha256(evidence.read_bytes()).hexdigest()
    permission = tmp_path / "permission.json"
    permission.write_text(
        json.dumps(
            {
                "status": "granted",
                "model": "FuXi-S2S",
                "covers_competition_submission": True,
                "covers_derived_probabilities": True,
                "written_evidence_path": str(evidence),
                "written_evidence_sha256": digest,
            }
        )
    )
    forecast = tmp_path / "forecast.npz"
    np.savez_compressed(
        forecast,
        probabilities=np.full((2, 5, 121, 240), 0.2, dtype=np.float32),
        issue_date=np.asarray("20260813"),
    )
    calls = []

    def create(variable, issue, period, team, model, password):
        calls.append((variable, issue, period, team, model, password))
        return xr.DataArray(
            np.empty((5, 121, 240), dtype=np.float32),
            dims=("quintile", "latitude", "longitude"),
            coords={
                "quintile": np.arange(1, 6),
                "latitude": np.linspace(90, -90, 121),
                "longitude": np.arange(240) * 1.5,
            },
        )

    package = types.ModuleType("AI_WQ_package")
    package.forecast_submission = SimpleNamespace(
        AI_WQ_create_empty_dataarray=create,
        AI_WQ_forecast_submission=lambda *args: True,
    )
    monkeypatch.setitem(sys.modules, "AI_WQ_package", package)
    monkeypatch.setattr(submit_precip_issue, "version", lambda _name: "3.29")
    monkeypatch.setenv("AI_WQ_PASSWORD", "secret")
    output = tmp_path / "bundle"
    args = SimpleNamespace(
        issue_date="20260813",
        permission_record=permission,
        password_env="AI_WQ_PASSWORD",
        team="team",
        model="model",
        forecast=forecast,
        output_dir=output,
        confirm_submit=None,
    )
    submit_precip_issue.run(args)
    assert [item[2] for item in calls] == [1, 2]
    manifest = json.loads((output / "submission_manifest.json").read_text())
    assert manifest["ai_wq_package_version"] == "3.29"
    assert len(manifest["files"]) == 2


def test_leader_handoff_is_explicitly_non_official_and_exact(tmp_path):
    forecast = tmp_path / "forecast.npz"
    probability = np.full((2, 5, 121, 240), 0.2, dtype=np.float32)
    np.savez_compressed(
        forecast,
        probabilities=probability,
        issue_date=np.asarray("20260813"),
        winner=np.asarray("calibrated_neural_tp_only_w16_seed42"),
    )
    forecast_hash = hashlib.sha256(forecast.read_bytes()).hexdigest()
    prediction_manifest = tmp_path / "forecast.manifest.json"
    prediction_manifest.write_text(
        json.dumps(
            {
                "output_sha256": forecast_hash,
                "climatology": {"kind": "public_era5_lagged_proxy"},
            }
        )
    )
    output = tmp_path / "handoff"
    build_precip_leader_handoff.run(
        SimpleNamespace(
            forecast=forecast,
            prediction_manifest=prediction_manifest,
            output_dir=output,
            allow_proxy=True,
        )
    )
    manifest = json.loads((output / "leader_handoff_manifest.json").read_text())
    assert manifest["submission_ready"] is False
    assert len(manifest["files"]) == 2
    for period in (1, 2):
        path = output / f"pr_20260813_p{period}_probabilities_only.nc"
        with xr.open_dataarray(path) as data:
            np.testing.assert_array_equal(data.values, probability[period - 1])
            assert "not_official_ai_wq_template" in data.attrs["status"]
