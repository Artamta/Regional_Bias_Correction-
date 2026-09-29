"""Tests for leakage-safe physical predictors in the compact sweep."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


HERE = Path(__file__).resolve().parents[1]
NEURAL_SRC = HERE.parent / "neural_adapter" / "src"
for path in (HERE, NEURAL_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import fuxi_imd_compact_validation_sweep as sweep  # noqa: E402
from fuxi_adapter.validation_sweep_models import (  # noqa: E402
    FixedCapacityPhysicalTemporalUNet,
)


def _synthetic_inputs() -> tuple[
    np.ndarray,
    dict[str, object],
    SimpleNamespace,
    SimpleNamespace,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    initializations = np.asarray(
        [
            "2002-06-01",
            "2003-06-01",
            "2018-06-01",
            "2020-06-01",
            "2021-06-01",
        ],
        dtype="datetime64[D]",
    )
    latitude = np.linspace(39.0, 0.0, 27)
    longitude = np.linspace(60.0, 99.0, 27)
    forecast = SimpleNamespace(
        initializations=initializations,
        latitude=latitude,
        longitude=longitude,
        ensemble_mean=np.zeros((5, 6, 27, 27), dtype=np.float32),
    )
    features = np.zeros((5, 6, 29, 27, 27), dtype=np.float32)
    normalization: dict[str, object] = {
        "input_channels": [f"base_{index}" for index in range(29)],
        "spatial_context": {
            "full_domain_channels": ["base_0"],
            "support_limited_channels": [],
        },
    }
    train = np.asarray([0, 1], dtype=np.int64)
    validation = np.asarray([2], dtype=np.int64)
    weights = np.ones((27, 27), dtype=np.float64)

    case = np.arange(3, dtype=np.float32)[:, None, None, None]
    lead = np.arange(6, dtype=np.float32)[None, :, None, None]
    row = np.arange(27, dtype=np.float32)[None, None, :, None]
    column = np.arange(27, dtype=np.float32)[None, None, None, :]
    base_field = case + 0.2 * lead + 0.01 * row + 0.001 * column
    # Reverse cache order to verify alignment is by initialization, not row.
    cache_order = np.asarray([2, 0, 1])
    fields = {
        name: (base_field + float(index))[cache_order].astype(np.float32)
        for index, name in enumerate(sweep.PHYSICAL_CACHE_FIELD_NAMES)
    }
    predictors = SimpleNamespace(
        initializations=initializations[:3][cache_order],
        latitude=latitude,
        longitude=longitude,
        feature_fields=fields,
        source_fingerprint={"synthetic": True},
        cache_path="/tmp/synthetic_physical.npz",
        cache_sha256="synthetic-sha256",
    )
    return (
        features,
        normalization,
        predictors,
        forecast,
        train,
        validation,
        weights,
    )


def test_physical_features_align_normalize_and_zero_quarantined_rows() -> None:
    inputs = _synthetic_inputs()

    features, normalization, diagnostics = sweep.append_physical_predictors(
        *inputs
    )

    assert features.shape == (5, 6, 38, 27, 27)
    assert normalization["input_channels"][-9:] == list(
        sweep.PHYSICAL_PREDICTOR_NAMES
    )
    assert np.isfinite(features[:3]).all()
    assert np.count_nonzero(features[3:, :, 29:]) == 0
    assert normalization["fuxi_physical_predictors"][
        "cache_initialization_count"
    ] == 3
    assert normalization["fuxi_physical_predictors"][
        "cache_latest_initialization"
    ] == "2018-06-01"
    assert {row["split"] for row in diagnostics["statistics"]} == {
        "train",
        "validation",
    }


def test_physical_normalization_is_unchanged_by_validation_values() -> None:
    inputs = _synthetic_inputs()
    original, original_stats, _ = sweep.append_physical_predictors(*inputs)
    predictors = inputs[2]
    fields = {name: values.copy() for name, values in predictors.feature_fields.items()}
    validation_cache_row = int(
        np.flatnonzero(predictors.initializations == np.datetime64("2018-06-01"))[0]
    )
    for values in fields.values():
        values[validation_cache_row] += np.float32(1_000_000.0)
    changed_predictors = SimpleNamespace(**{**vars(predictors), "feature_fields": fields})
    changed_inputs = (
        inputs[0],
        inputs[1],
        changed_predictors,
        *inputs[3:],
    )

    changed, changed_stats, _ = sweep.append_physical_predictors(*changed_inputs)

    for name in sweep.PHYSICAL_PREDICTOR_NAMES:
        assert changed_stats[name] == original_stats[name]
    np.testing.assert_array_equal(changed[:2], original[:2])
    assert not np.array_equal(changed[2, :, 29:], original[2, :, 29:])


def test_physical_cache_rejects_any_quarantined_initialization() -> None:
    inputs = _synthetic_inputs()
    predictors = inputs[2]
    bad_initializations = predictors.initializations.copy()
    bad_initializations[0] = np.datetime64("2020-06-01")
    bad = SimpleNamespace(**{**vars(predictors), "initializations": bad_initializations})

    with pytest.raises(sweep.base.DataContractError, match="2020"):
        sweep.append_physical_predictors(
            inputs[0], inputs[1], bad, *inputs[3:]
        )


def test_physical_candidate_bank_is_fixed_capacity_and_conditionally_required() -> None:
    names = (
        "physical_control",
        "physical_tcwv",
        "physical_moisture_circulation",
        "physical_full_compact",
    )
    candidates = sweep.selected_candidates(",".join(names))

    assert sweep.needs_physical_predictors(candidates)
    assert candidates[0].physical_predictors == ()
    assert candidates[1].physical_predictors == ("tcwv_mean",)
    assert len(candidates[2].physical_predictors) == 6
    assert len(candidates[3].physical_predictors) == 9

    models = [
        sweep.build_model(candidate, 38, np.ones(6, dtype=np.float32))
        for candidate in candidates[1:]
    ]
    assert all(isinstance(model, FixedCapacityPhysicalTemporalUNet) for model in models)
    assert [int(model.active_physical_mask.sum()) for model in models] == [1, 6, 9]
    assert len(
        {
            sum(parameter.numel() for parameter in model.parameters())
            for model in models
        }
    ) == 1
    assert not sweep.needs_physical_predictors((candidates[0],))

