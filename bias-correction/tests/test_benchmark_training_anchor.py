from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


CONTRACT_PATH = (
    Path(__file__).resolve().parents[2]
    / "studies"
    / "fuxi_imd_adapter_benchmark_v1"
    / "training_anchor_contract.py"
)
SPEC = importlib.util.spec_from_file_location("training_anchor_contract", CONTRACT_PATH)
assert SPEC is not None and SPEC.loader is not None
anchor_contract = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(anchor_contract)


def test_benchmark_loads_raw_identity_reconstruction_contract(tmp_path: Path) -> None:
    models = tmp_path / "models"
    models.mkdir()
    scale = np.linspace(0.5, 1.0, 6, dtype=np.float32)
    np.savez_compressed(
        models / "training_anchor_contract.npz",
        anchor_kind=np.asarray("raw_fuxi"),
        target_scale=scale,
        fitted_target_years=np.arange(2002, 2018, dtype=np.int16),
    )
    raw = np.full((2, 6, 2, 2), 3.0, dtype=np.float32)
    log_bias = np.full_like(raw, 2.0)

    kind, baseline, loaded_scale = anchor_contract.load_neural_reconstruction_contract(
        tmp_path,
        {"training_anchor": "raw_fuxi"},
        raw,
        log_bias,
    )

    assert kind == "raw_fuxi"
    assert np.array_equal(baseline, raw)
    assert not np.array_equal(baseline, log_bias)
    assert np.array_equal(loaded_scale, scale)


def test_benchmark_preserves_legacy_log_bias_reconstruction_contract(
    tmp_path: Path,
) -> None:
    models = tmp_path / "models"
    models.mkdir()
    scale = np.linspace(0.6, 1.1, 6, dtype=np.float32)
    np.savez_compressed(
        models / "log_bias_anchor.npz",
        lead_month_residual=np.zeros((6, 12, 1, 1), dtype=np.float32),
        shrinkage=np.asarray(10.0),
        target_scale=scale,
    )
    raw = np.full((2, 6, 2, 2), 3.0, dtype=np.float32)
    log_bias = np.full_like(raw, 2.0)

    kind, baseline, loaded_scale = anchor_contract.load_neural_reconstruction_contract(
        tmp_path,
        {},
        raw,
        log_bias,
    )

    assert kind == "log_bias"
    assert np.array_equal(baseline, log_bias)
    assert not np.array_equal(baseline, raw)
    assert np.array_equal(loaded_scale, scale)


def test_benchmark_rejects_non_training_year_raw_scale(tmp_path: Path) -> None:
    models = tmp_path / "models"
    models.mkdir()
    np.savez_compressed(
        models / "training_anchor_contract.npz",
        anchor_kind=np.asarray("raw_fuxi"),
        target_scale=np.ones(6, dtype=np.float32),
        fitted_target_years=np.arange(2003, 2019, dtype=np.int16),
    )
    raw = np.ones((1, 6, 1, 1), dtype=np.float32)

    with pytest.raises(ValueError, match="2002-2017"):
        anchor_contract.load_neural_reconstruction_contract(
            tmp_path,
            {"training_anchor": "raw_fuxi"},
            raw,
            raw,
        )
