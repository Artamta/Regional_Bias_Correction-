"""Executable smoke test for the synthetic multi-variable prototype."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from multivariable import MULTIVARIABLE_FEATURE_NAMES
from prototype_multivariable import (
    run_prototype,
    synthetic_anchor,
    synthetic_features,
)
from predict import load_checkpoint_model


def test_synthetic_features_embed_each_variables_own_anchor() -> None:
    anchor = synthetic_anchor()
    features = synthetic_features(anchor, "2026-08-15")

    assert features.shape == (2, 38, 121, 240)
    for variable_index, start in enumerate((0, 10, 20)):
        np.testing.assert_allclose(
            features[:, start : start + 5],
            np.log(anchor[variable_index]),
            rtol=0.0,
            atol=1.0e-7,
        )
    lead_flag = MULTIVARIABLE_FEATURE_NAMES.index("lead_flag")
    np.testing.assert_array_equal(features[:, lead_flag, 0, 0], [-1.0, 1.0])


def test_run_prototype_writes_six_identity_forecasts_and_manifest(
    tmp_path: Path,
) -> None:
    artifacts = run_prototype(
        tmp_path,
        init_date="20260815",
        device="cpu",
        base_channels=1,
        attention_heads=1,
    )

    assert len(artifacts.forecasts) == 6
    assert all(path.is_file() for path in artifacts.forecasts)
    expected_anchor = synthetic_anchor()
    variables = ("pr", "tas", "mslp")
    for variable_index, variable in enumerate(variables):
        for lead_index in range(2):
            path = artifacts.forecasts[2 * variable_index + lead_index]
            with xr.open_dataset(path) as dataset:
                np.testing.assert_allclose(
                    dataset[variable],
                    expected_anchor[variable_index, lead_index],
                    rtol=0.0,
                    atol=1.0e-7,
                )
                assert "not uploaded" in dataset.attrs["submission_status"]

    checkpoint = torch.load(
        artifacts.checkpoint, map_location="cpu", weights_only=False
    )
    assert checkpoint["model_config"]["in_channels"] == 38
    assert checkpoint["metadata"]["trained"] is False
    assert checkpoint["metadata"]["submission_ready"] is False
    restored, _ = load_checkpoint_model(
        artifacts.checkpoint, device="cpu", in_channels=38
    )
    restored_features = torch.from_numpy(
        synthetic_features(expected_anchor, "2026-08-15")[None]
    )
    restored_anchor = torch.from_numpy(expected_anchor[None])
    with torch.inference_mode():
        restored_probabilities = restored(restored_features, restored_anchor)
    torch.testing.assert_close(
        restored_probabilities,
        restored_anchor,
        rtol=0.0,
        atol=1.0e-7,
    )
    manifest = json.loads(artifacts.manifest.read_text(encoding="utf-8"))
    assert manifest["status"] == "synthetic_untrained_contract_prototype"
    assert manifest["init_date"] == "2026-08-15"
    assert manifest["feature_count"] == 38
    assert manifest["maximum_anchor_identity_error"] <= 1.0e-6
    assert manifest["submission_ready"] is False
    assert len(manifest["forecast_files"]) == 6
