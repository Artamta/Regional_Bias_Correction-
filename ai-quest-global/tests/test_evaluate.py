"""Synthetic end-to-end tests for publication evaluation diagnostics."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

import evaluate
from predict import LATITUDE, LONGITUDE, PreparedBatch


def _synthetic_prepared(source: Path) -> PreparedBatch:
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
        init_dates=("2020-01-02",),
        source=source,
    )


def test_publication_diagnostics_and_legacy_return_api(tmp_path, monkeypatch) -> None:
    source = tmp_path / "synthetic.npz"
    prepared = _synthetic_prepared(source)

    monkeypatch.setattr(evaluate, "_resolve_case_files", lambda _: [source])
    monkeypatch.setattr(evaluate, "load_prepared", lambda *args, **kwargs: prepared)
    monkeypatch.setattr(
        evaluate,
        "load_checkpoint_model",
        lambda *args, **kwargs: (object(), {}),
    )

    def perfect_model(_model, features, _anchor, *, device):
        del _model, device
        probabilities = np.zeros(
            (features.shape[0], 2, 5, 121, 240), dtype=np.float32
        )
        probabilities[:, :, 2] = 1.0
        return probabilities

    monkeypatch.setattr(evaluate, "run_model", perfect_model)
    output_directory = tmp_path / "evaluation"

    legacy_outputs = evaluate.evaluate_cases(
        [source],
        tmp_path / "checkpoint.pt",
        output_directory,
        weighting="area-land",
    )

    assert legacy_outputs == (
        output_directory / "evaluation_by_case.csv",
        output_directory / "evaluation_summary.csv",
        output_directory / "evaluation_by_lead.png",
    )
    for output in legacy_outputs:
        assert output.is_file() and output.stat().st_size > 0
    for name in ("reliability.png", "spatial_rps_improvement.png"):
        output = output_directory / name
        assert output.is_file() and output.stat().st_size > 0

    india_path = output_directory / "india_summary.csv"
    with india_path.open(newline="", encoding="utf-8") as handle:
        india_rows = list(csv.DictReader(handle))
    assert len(india_rows) == 2
    for lead, row in enumerate(india_rows, start=1):
        assert int(row["lead"]) == lead
        assert float(row["raw_p0_rps"]) == pytest.approx(0.4)
        assert float(row["model_rps"]) == pytest.approx(0.0)
        assert float(row["model_vs_raw_skill"]) == pytest.approx(1.0)
        assert int(row["n_cases"]) == 1
        assert row["latitude_bounds"] == "5N-40N"
        assert row["longitude_bounds"] == "65E-100E"


def test_spatial_rps_field_sign_convention() -> None:
    target = np.full((121, 240), 2, dtype=np.int8)
    raw = np.full((5, 121, 240), 0.2, dtype=np.float32)
    perfect = np.zeros_like(raw)
    perfect[2] = 1.0

    raw_rps = evaluate._rps_field(raw, target)
    model_rps = evaluate._rps_field(perfect, target)

    np.testing.assert_allclose(raw_rps, 0.4, rtol=0.0, atol=1.0e-7)
    np.testing.assert_allclose(model_rps, 0.0, rtol=0.0, atol=1.0e-7)
    assert np.all(raw_rps - model_rps > 0.0)


def test_evaluation_rejects_checkpoint_cache_contract_mismatch(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "synthetic.npz"
    prepared = _synthetic_prepared(source)
    monkeypatch.setattr(evaluate, "_resolve_case_files", lambda _: [source])
    monkeypatch.setattr(evaluate, "load_prepared", lambda *args, **kwargs: prepared)
    monkeypatch.setattr(
        evaluate, "_cache_contract_fingerprint", lambda _path: "cache-contract"
    )
    monkeypatch.setattr(
        evaluate,
        "load_checkpoint_model",
        lambda *args, **kwargs: (
            object(),
            {"normalization": {"cache_contract_sha256": "other-contract"}},
        ),
    )

    with pytest.raises(ValueError, match="cache_contract_sha256 differs"):
        evaluate.evaluate_cases(
            [source],
            tmp_path / "checkpoint.pt",
            tmp_path / "evaluation",
            weighting="area",
        )
