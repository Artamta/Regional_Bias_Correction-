"""Focused tests for validation-fitted scalar uniform blending."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import calibrate_uniform_blend as calibration
from predict import LATITUDE, LONGITUDE, PreparedBatch


def test_analytic_alpha_has_known_interior_optimum() -> None:
    probabilities = np.zeros((2, 1, 5, 1, 1), dtype=np.float32)
    probabilities[:, :, 0] = 1.0
    target = np.asarray([[[[0]]], [[[4]]]], dtype=np.int8)

    alpha = calibration.fit_uniform_blend_alpha(
        probabilities, target, np.ones((1, 1), dtype=np.float32)
    )

    assert alpha == pytest.approx(1.0 / 6.0)
    epsilon = 1.0e-2
    optimum = calibration.aggregate_rps(
        calibration.blend_probabilities(probabilities, alpha),
        target,
        np.ones((1, 1), dtype=np.float32),
    ).rps
    assert optimum < calibration.aggregate_rps(
        calibration.blend_probabilities(probabilities, alpha - epsilon),
        target,
        np.ones((1, 1), dtype=np.float32),
    ).rps
    assert optimum < calibration.aggregate_rps(
        calibration.blend_probabilities(probabilities, alpha + epsilon),
        target,
        np.ones((1, 1), dtype=np.float32),
    ).rps


def test_uniform_blend_stays_positive_and_normalized() -> None:
    probabilities = np.zeros((1, 2, 5, 2, 3), dtype=np.float32)
    probabilities[:, :, 3] = 1.0

    blended = calibration.blend_probabilities(probabilities, 0.25)

    assert blended.dtype == np.float32
    assert float(blended.min()) > 0.0
    np.testing.assert_allclose(blended.sum(axis=2), 1.0, rtol=0.0, atol=1.0e-7)
    np.testing.assert_allclose(blended[:, :, 3], 0.4, rtol=0.0, atol=1.0e-7)
    np.testing.assert_allclose(blended[:, :, 0], 0.15, rtol=0.0, atol=1.0e-7)


def test_aggregate_rps_is_ratio_weighted_not_mean_of_case_scores() -> None:
    probabilities = np.full((2, 1, 5, 1, 2), 0.2, dtype=np.float32)
    probabilities[0, 0, :, 0, 0] = 0.0
    probabilities[0, 0, 0, 0, 0] = 1.0
    target = np.asarray([[[[0, -1]]], [[[0, 0]]]], dtype=np.int8)

    result = calibration.aggregate_rps(
        probabilities, target, np.ones((1, 2), dtype=np.float32)
    )

    assert result.valid_cells == 3
    assert result.denominator == pytest.approx(3.0)
    assert result.rps == pytest.approx(0.8)
    assert result.rps != pytest.approx((0.0 + 1.2) / 2.0)


def _prepared_for_year(source: Path, year: int) -> PreparedBatch:
    features = np.zeros((1, 2, 18, 121, 240), dtype=np.float32)
    p0 = np.full((1, 2, 5, 121, 240), 0.2, dtype=np.float32)
    target = np.full((1, 2, 121, 240), 2, dtype=np.int8)
    return PreparedBatch(
        features=features,
        p0=p0,
        target=target,
        latitude=LATITUDE.copy(),
        longitude=LONGITUDE.copy(),
        land_fraction=np.ones((121, 240), dtype=np.float32),
        init_dates=(f"{year}-01-02",),
        source=source,
    )


def test_run_calibration_writes_frozen_split_results(tmp_path, monkeypatch) -> None:
    source = tmp_path / "synthetic.npz"
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"exact-checkpoint-for-calibration")
    monkeypatch.setattr(calibration, "_resolve_case_files", lambda _: [source])
    monkeypatch.setattr(calibration, "_cache_contract_fingerprint", lambda _: None)
    monkeypatch.setattr(
        calibration,
        "load_prepared",
        lambda path, *, require_target, years, thursday_only: _prepared_for_year(
            path, int(years[0])
        ),
    )
    monkeypatch.setattr(
        calibration,
        "load_checkpoint_model",
        lambda *args, **kwargs: (
            object(),
            {"metadata": {"model_name": "TPProbUNet"}},
        ),
    )

    def informative_model(_model, features, _p0, *, device):
        del _model, _p0, device
        result = np.full((features.shape[0], 2, 5, 121, 240), 0.1, dtype=np.float32)
        result[:, :, 2] = 0.6
        return result

    monkeypatch.setattr(calibration, "run_model", informative_model)
    output = tmp_path / "calibration"

    artifacts = calibration.run_calibration(
        [source],
        checkpoint,
        output,
        fit_years=[2019],
        evaluation_years=[2020],
        weighting="area-land",
    )

    for path in artifacts.paths:
        assert path.is_file() and path.stat().st_size > 0
    payload = json.loads(artifacts.calibration_json.read_text(encoding="utf-8"))
    assert payload["fit_years"] == [2019]
    assert payload["evaluation_years"] == [2020]
    assert payload["alphas"]["p0"] == pytest.approx(0.0)
    assert payload["alphas"]["model"] == pytest.approx(1.0)
    assert payload["checkpoint_model_name"] == "TPProbUNet"
    assert payload["checkpoint_sha256"] == hashlib.sha256(
        checkpoint.read_bytes()
    ).hexdigest()
    assert payload["results"]["evaluation"]["model_calibrated"]["alpha"] \
        == pytest.approx(payload["alphas"]["model"])
    with artifacts.summary_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 30
    assert {row["split"] for row in rows} == {"fit", "evaluation"}


def test_fit_and_evaluation_years_must_be_disjoint(tmp_path) -> None:
    with pytest.raises(ValueError, match="must be disjoint"):
        calibration.run_calibration(
            [tmp_path / "unused.npz"],
            tmp_path / "checkpoint.pt",
            tmp_path / "output",
            fit_years=[2020],
            evaluation_years=[2020],
        )
