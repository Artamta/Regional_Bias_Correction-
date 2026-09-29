from __future__ import annotations

from pathlib import Path
import csv
import sys
import types

import numpy as np
import pytest
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import predict  # noqa: E402
import evaluate  # noqa: E402


def _two_lead_probabilities() -> np.ndarray:
    probabilities = np.empty(
        (
            predict.N_LEADS,
            predict.N_QUINTILES,
            predict.N_LATITUDE,
            predict.N_LONGITUDE,
        ),
        dtype=np.float32,
    )
    probabilities[0] = np.asarray([0.10, 0.15, 0.20, 0.25, 0.30], dtype=np.float32)[
        :, None, None
    ]
    probabilities[1] = np.asarray([0.30, 0.25, 0.20, 0.15, 0.10], dtype=np.float32)[
        :, None, None
    ]
    return probabilities


def test_export_two_local_lead_files_with_exact_grid(tmp_path: Path) -> None:
    paths = predict.export_two_leads(
        _two_lead_probabilities(),
        tmp_path,
        prefix="pr_2026-08-11",
        init_date="2026-08-11",
    )

    assert [path.name for path in paths] == [
        "pr_2026-08-11_p1.nc",
        "pr_2026-08-11_p2.nc",
    ]
    for lead, path in enumerate(paths, start=1):
        assert path.is_file()
        with xr.open_dataset(path) as dataset:
            assert set(dataset.data_vars) == {"pr"}
            assert dataset["pr"].dims == ("quintile", "latitude", "longitude")
            assert dataset.sizes == {"quintile": 5, "latitude": 121, "longitude": 240}
            np.testing.assert_allclose(dataset["quintile"], [0.2, 0.4, 0.6, 0.8, 1.0])
            np.testing.assert_allclose(dataset["latitude"], np.linspace(90.0, -90.0, 121))
            np.testing.assert_allclose(dataset["longitude"], np.arange(240) * 1.5)
            values = dataset["pr"].values
            assert np.isfinite(values).all()
            assert float(values.min()) >= 0.0
            assert float(values.max()) <= 1.0
            np.testing.assert_allclose(values.sum(axis=0), 1.0, atol=1.0e-6)
            assert dataset.attrs["lead"] == lead
            assert dataset.attrs["initialization_date"] == "2026-08-11"
            assert "standard_name" not in dataset["pr"].attrs
            assert "cell_methods" not in dataset["pr"].attrs
            assert dataset["pr"].attrs["forecast_period_aggregation"] == "sum"
            assert dataset["pr"].attrs["shortName"] == "tp"
            expected_start = np.datetime64("2026-08-11") + np.timedelta64(
                18 if lead == 1 else 25, "D"
            )
            expected_end = np.datetime64("2026-08-11") + np.timedelta64(
                25 if lead == 1 else 32, "D"
            )
            assert dataset["forecast_issue_date"].values == np.datetime64("2026-08-11")
            assert dataset["forecast_period_start"].values == expected_start
            assert dataset["forecast_period_end"].values == expected_end
            assert "not uploaded" in dataset.attrs["submission_status"]
            assert "permission" in dataset.attrs["permission_notice"].lower()


@pytest.mark.parametrize(
    ("variable", "aggregation", "short_name"),
    [
        ("pr", "sum", "tp"),
        ("tas", "mean", "2t"),
        ("mslp", "mean", "msl"),
    ],
)
def test_write_probability_netcdf_uses_variable_metadata(
    tmp_path: Path,
    variable: str,
    aggregation: str,
    short_name: str,
) -> None:
    path = predict.write_probability_netcdf(
        _two_lead_probabilities()[0],
        tmp_path / f"{variable}.nc",
        lead=1,
        init_date="2026-08-11",
        variable=variable,
    )

    with xr.open_dataset(path) as dataset:
        assert set(dataset.data_vars) == {variable}
        assert "standard_name" not in dataset[variable].attrs
        assert "cell_methods" not in dataset[variable].attrs
        assert dataset[variable].attrs["forecast_period_aggregation"] == aggregation
        assert dataset[variable].attrs["shortName"] == short_name
        assert variable in path.name
        assert "not uploaded" in dataset.attrs["submission_status"]
        assert "permission" in dataset.attrs["permission_notice"].lower()


def test_export_all_variables_writes_six_local_files(tmp_path: Path) -> None:
    probabilities = np.stack(
        [_two_lead_probabilities()] * len(predict.CANONICAL_VARIABLES), axis=0
    )

    paths = predict.export_all_variables(
        probabilities,
        tmp_path,
        init_date="2026-08-11",
    )

    assert [path.name for path in paths] == [
        "pr_2026-08-11_p1.nc",
        "pr_2026-08-11_p2.nc",
        "tas_2026-08-11_p1.nc",
        "tas_2026-08-11_p2.nc",
        "mslp_2026-08-11_p1.nc",
        "mslp_2026-08-11_p2.nc",
    ]
    for variable, variable_paths in zip(
        predict.CANONICAL_VARIABLES,
        (paths[0:2], paths[2:4], paths[4:6]),
        strict=True,
    ):
        for path in variable_paths:
            assert path.is_file()
            with xr.open_dataset(path) as dataset:
                assert set(dataset.data_vars) == {variable}
                assert dataset[variable].shape == (5, 121, 240)
                np.testing.assert_allclose(
                    dataset[variable].values.sum(axis=0), 1.0, atol=1.0e-6
                )
                assert dataset.attrs["model"] == "QuestTemporalProbUNet"
                assert "not uploaded" in dataset.attrs["submission_status"]


def test_export_all_variables_validates_shape_and_variables(tmp_path: Path) -> None:
    wrong_shape = np.full((2, 2, 5, 121, 240), 0.2, dtype=np.float32)
    with pytest.raises(ValueError, match=r"\(3, 2, 5, 121, 240\)"):
        predict.export_all_variables(
            wrong_shape,
            tmp_path,
            init_date="2026-08-11",
        )

    valid = np.full((3, 2, 5, 121, 240), 0.2, dtype=np.float32)
    with pytest.raises(ValueError, match="variable must be one of"):
        predict.export_all_variables(
            valid,
            tmp_path,
            init_date="2026-08-11",
            variables=("pr", "tas", "wind"),
        )

    assert list(tmp_path.iterdir()) == []


def test_export_all_variables_rolls_back_publish_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    probabilities = np.stack(
        [_two_lead_probabilities()] * len(predict.CANONICAL_VARIABLES), axis=0
    )
    real_replace = predict.os.replace
    stage_publish_calls = 0

    def fail_third_stage_publish(source: object, destination: object) -> None:
        nonlocal stage_publish_calls
        if ".stage.nc" in Path(source).name:
            stage_publish_calls += 1
            if stage_publish_calls == 3:
                raise OSError("simulated publish failure")
        real_replace(source, destination)

    monkeypatch.setattr(predict.os, "replace", fail_third_stage_publish)

    with pytest.raises(OSError, match="simulated publish failure"):
        predict.export_all_variables(
            probabilities,
            tmp_path,
            init_date="2026-08-11",
        )

    assert stage_publish_calls == 3
    assert list(tmp_path.iterdir()) == []


def test_probability_validation_rejects_nonfinite_and_unnormalized() -> None:
    cube = _two_lead_probabilities()[0]
    bad = cube.copy()
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        predict.validate_probability_cube(bad)

    bad = cube * 0.5
    with pytest.raises(ValueError, match="sum to one"):
        predict.validate_probability_cube(bad)


def test_load_prepared_npz_accepts_one_case_batch_contract(tmp_path: Path) -> None:
    features = np.zeros((1, 2, 18, 121, 240), dtype=np.float32)
    p0 = np.full((1, 2, 5, 121, 240), 0.2, dtype=np.float32)
    target = np.zeros((1, 2, 121, 240), dtype=np.int8)
    target[:, :, 0, 0] = -1
    case_path = tmp_path / "case.npz"
    np.savez_compressed(
        case_path,
        features=features,
        p0=p0,
        target=target,
        latitude=predict.LATITUDE,
        longitude=predict.LONGITUDE,
        init_dates=np.asarray(["2026-08-11"]),
    )

    prepared = predict.load_prepared_npz(case_path, require_target=True)

    assert prepared.features.shape == features.shape
    assert prepared.p0.shape == p0.shape
    assert prepared.target is not None
    assert prepared.target.shape == target.shape
    assert prepared.n_cases == 1
    assert prepared.init_dates == ("2026-08-11",)


def test_zero_correction_checkpoint_runs_both_leads(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from model import TPProbUNet

    case_path = tmp_path / "prediction_case.npz"
    p0 = np.full((1, 2, 5, 121, 240), 0.2, dtype=np.float32)
    np.savez_compressed(
        case_path,
        features=np.zeros((1, 2, 18, 121, 240), dtype=np.float32),
        p0=p0,
        target=np.zeros((1, 2, 121, 240), dtype=np.int8),
        latitude=predict.LATITUDE,
        longitude=predict.LONGITUDE,
        land_fraction=np.ones((121, 240), dtype=np.float32),
        init_dates=np.asarray(["2026-08-11"]),
    )
    checkpoint_path = tmp_path / "best.pt"
    network = TPProbUNet(in_channels=18, base_channels=1, dropout=0.0)
    torch.save(
        {
            "model_state": network.state_dict(),
            "model_config": {"in_channels": 18, "base_channels": 1, "dropout": 0.0},
            "normalization": {
                "tp_quantile_mean": [[0.0] * 5] * 2,
                "tp_quantile_std": [[1.0] * 5] * 2,
                "feature_names": [f"feature_{index}" for index in range(18)],
                "provenance": {
                    "official_ai_weather_quest_validation": False,
                    "notice": "Exploratory unit-test target; not official validation.",
                },
            },
            "metadata": {"fuxi_competition_use": "written_permission_required"},
        },
        checkpoint_path,
    )

    outputs = predict.predict_case(case_path, checkpoint_path, tmp_path / "out", device="cpu")

    assert [path.name for path in outputs] == [
        "pr_2026-08-11_p1.nc",
        "pr_2026-08-11_p2.nc",
    ]
    for path in outputs:
        with xr.open_dataset(path) as dataset:
            np.testing.assert_allclose(dataset["pr"], 0.2, atol=1.0e-6)
            assert dataset.attrs["experiment_notice"].startswith("Exploratory")

    reports = evaluate.evaluate_cases(
        [case_path],
        checkpoint_path,
        tmp_path / "evaluation",
        device="cpu",
        years=[2026],
    )
    assert all(path.is_file() and path.stat().st_size > 0 for path in reports)
    with reports[1].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6  # 2 leads x uniform/p0/model
    assert {row["system"] for row in rows} == {"uniform", "p0", "model"}
    np.testing.assert_allclose([float(row["rpss"]) for row in rows], 0.0, atol=1.0e-7)


def test_checkpoint_loader_restores_optional_tp_period_attention(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    from model import TPProbUNet

    original = TPProbUNet(
        in_channels=18,
        base_channels=1,
        dropout=0.0,
        period_attention=True,
        attention_heads=1,
        attention_dropout=0.0,
    )
    checkpoint_path = tmp_path / "attention.pt"
    torch.save(
        {
            "model_state": original.state_dict(),
            "model_config": {
                "in_channels": 18,
                "base_channels": 1,
                "dropout": 0.0,
                "period_attention": True,
                "attention_heads": 1,
                "attention_dropout": 0.0,
            },
            "metadata": {"model_name": "TPProbUNet"},
        },
        checkpoint_path,
    )

    restored, _ = predict.load_checkpoint_model(
        checkpoint_path, device="cpu", in_channels=18
    )

    assert isinstance(restored, TPProbUNet)
    assert restored.period_attention_enabled is True
    assert restored.attention_heads == 1
    for key, expected in original.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], expected)


def test_standard_prediction_rejects_multivariable_checkpoint_contract(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    from model import QuestTemporalProbUNet

    case_path = tmp_path / "single_variable_case.npz"
    np.savez_compressed(
        case_path,
        features=np.zeros((1, 2, 18, 121, 240), dtype=np.float32),
        p0=np.full((1, 2, 5, 121, 240), 0.2, dtype=np.float32),
        latitude=predict.LATITUDE,
        longitude=predict.LONGITUDE,
        init_dates=np.asarray(["2026-08-11"]),
    )
    network = QuestTemporalProbUNet(
        in_channels=18,
        variables=("pr", "tas", "mslp"),
        base_channels=1,
        dropout=0.0,
        attention_heads=1,
        attention_dropout=0.0,
    )
    checkpoint_path = tmp_path / "joint.pt"
    torch.save(
        {
            "model_name": "QuestTemporalProbUNet",
            "model_state": network.state_dict(),
            "model_config": {
                "in_channels": 18,
                "variables": ["pr", "tas", "mslp"],
                "base_channels": 1,
                "dropout": 0.0,
                "attention_heads": 1,
                "attention_dropout": 0.0,
            },
        },
        checkpoint_path,
    )

    with pytest.raises(
        ValueError,
        match=r"precipitation-only.*QuestTemporalProbUNet requires.*variable",
    ):
        predict.predict_case(
            case_path,
            checkpoint_path,
            tmp_path / "out",
            device="cpu",
        )

    assert not (tmp_path / "out").exists()


def test_multivariable_checkpoint_name_is_not_guessed_from_config(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    checkpoint_path = tmp_path / "unnamed_joint.pt"
    torch.save(
        {
            "model_state": {},
            "model_config": {
                "in_channels": 18,
                "variables": ["pr", "tas", "mslp"],
            },
        },
        checkpoint_path,
    )

    with pytest.raises(ValueError, match="must explicitly declare model_name"):
        predict.load_checkpoint_model(
            checkpoint_path,
            device="cpu",
            in_channels=18,
        )


def test_normalize_probabilities_handles_roundoff_but_not_negative_mass() -> None:
    values = np.asarray([0.2, 0.2, 0.2, 0.2, 0.200001], dtype=np.float64)[:, None, None]
    normalized = predict.normalize_probabilities(values, category_axis=0)
    np.testing.assert_allclose(normalized.sum(axis=0), 1.0, atol=1.0e-7)

    values[0, 0, 0] = -0.1
    with pytest.raises(ValueError, match="negative"):
        predict.normalize_probabilities(values, category_axis=0)


def test_project_zarr_loader_selects_init_year_and_thursday(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store_path = tmp_path / "prepared.zarr"
    store_path.mkdir()
    group = {
        "features": np.zeros((2, 2, 18, 121, 240), dtype=np.float16),
        "p0": np.full((2, 2, 5, 121, 240), 0.2, dtype=np.float16),
        "target": np.zeros((2, 2, 121, 240), dtype=np.int8),
        "latitude": predict.LATITUDE.astype(np.float64),
        "longitude": predict.LONGITUDE.astype(np.float64),
        "land_fraction": np.ones((121, 240), dtype=np.float32),
        "init_yyyymmdd": np.asarray([20200806, 20200809], dtype=np.int32),
        "case_complete": np.asarray([True, True]),
        "feature_complete": np.asarray([True, True]),
    }

    class FakeGroup(dict):
        attrs = {"status": "complete"}

    fake_group = FakeGroup(group)
    fake_zarr = types.SimpleNamespace(open_group=lambda path, mode: fake_group)
    monkeypatch.setitem(sys.modules, "zarr", fake_zarr)

    assert predict.load_prepared(store_path).n_cases == 2
    with pytest.raises(ValueError, match="requires --init-date"):
        predict.predict_case(store_path, tmp_path / "missing.pt", tmp_path / "out")

    prediction = predict.load_prepared(store_path, init_date="2020-08-09")
    assert prediction.n_cases == 1
    assert prediction.init_dates == ("2020-08-09",)

    thursdays = predict.load_prepared(
        store_path,
        require_target=True,
        years=[2020],
        thursday_only=True,
    )
    assert thursdays.n_cases == 1
    assert thursdays.init_dates == ("2020-08-06",)
    assert thursdays.target is not None
