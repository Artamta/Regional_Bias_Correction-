"""Contract tests for global patch pretraining and matched India transfer."""

from __future__ import annotations

import copy
from dataclasses import replace
from datetime import datetime, timedelta
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import zarr

import fuxi_global_patch_pretraining as global_pretraining
import fuxi_imd_global_pretraining_comparison as india_comparison


ROOT = Path(__file__).resolve().parents[1]
LATITUDE = np.linspace(90.0, -90.0, global_pretraining.N_LAT)
LONGITUDE = np.arange(0.0, 360.0, 1.5)


def _contract(
    *,
    mode: str = "full",
    fit_years: tuple[int, ...] = (2015,),
    validation_years: tuple[int, ...] = (2016, 2017),
    seed: int = 42,
) -> global_pretraining.GlobalPatchContract:
    return global_pretraining.GlobalPatchContract(
        mode=mode,
        cache_root="/tmp/not-opened-by-unit-tests",
        fit_years=fit_years,
        validation_years=validation_years,
        seed=seed,
        batch_size=2,
        epochs=1,
        patience=1,
    )


def _metadata(
    year: int, dates: tuple[int, ...]
) -> global_pretraining.YearMetadata:
    return global_pretraining.YearMetadata(
        year=year,
        path=f"/synthetic/{year}.zarr",
        init_yyyymmdd=dates,
        latitude=tuple(float(value) for value in LATITUDE),
        longitude=tuple(float(value) for value in LONGITUDE),
        attrs={
            "fuxi_source": "synthetic-fuxi",
            "imerg_source": "synthetic-imerg",
            "completed_utc": "2026-08-20T00:00:00+00:00",
        },
        metadata_sha256=f"{year:064d}"[-64:],
    )


def _preprocessing() -> global_pretraining.GlobalPreprocessing:
    calendar_shape = (
        global_pretraining.N_WEEK,
        12,
        global_pretraining.N_LAT,
        global_pretraining.N_LON,
    )
    return global_pretraining.GlobalPreprocessing(
        log_bias=np.zeros(calendar_shape, dtype=np.float32),
        climatology=np.ones(calendar_shape, dtype=np.float32),
        support=np.ones(
            (global_pretraining.N_LAT, global_pretraining.N_LON), dtype=bool
        ),
        feature_mean=np.zeros((4, global_pretraining.N_WEEK), dtype=np.float32),
        feature_std=np.ones((4, global_pretraining.N_WEEK), dtype=np.float32),
        target_scale=np.ones(global_pretraining.N_WEEK, dtype=np.float32),
        fit_content_sha256="c" * 64,
    )


def _write_annual_cache(
    root: Path,
    year: int,
    *,
    dynamic_dtype: str = "f2",
) -> Path:
    path = root / f"{year}.zarr"
    group = zarr.open_group(str(path), mode="w")
    group.attrs.update(
        {
            "status": "complete",
            "stat_names": list(global_pretraining.STAT_NAMES),
            "lead_windows": [
                "D1-7",
                "D8-14",
                "D15-21",
                "D22-28",
                "D29-35",
                "D36-42",
            ],
            "stored_units": "mm day-1 weekly mean rate",
        }
    )
    cases = 104
    group.create_dataset(
        "dynamic",
        shape=(cases, 6, 8, 121, 240),
        chunks=(1, 1, 8, 121, 240),
        dtype=dynamic_dtype,
        fill_value=np.nan,
    )
    group.create_dataset(
        "truth",
        shape=(cases, 6, 121, 240),
        chunks=(1, 1, 121, 240),
        dtype="f4",
        fill_value=np.nan,
    )
    group.create_dataset(
        "observation_fraction",
        shape=(cases, 6, 121, 240),
        chunks=(1, 1, 121, 240),
        dtype="f2",
        fill_value=np.nan,
    )
    group.create_dataset("case_complete", shape=(cases,), chunks=(cases,), dtype="bool")
    group.create_dataset("init_yyyymmdd", shape=(cases,), chunks=(cases,), dtype="i4")
    group.create_dataset("lat", shape=(121,), chunks=(121,), dtype="f8")
    group.create_dataset("lon", shape=(240,), chunks=(240,), dtype="f8")
    dates = pd.date_range(f"{year}-01-01", periods=cases, freq="3D")
    group["case_complete"][:] = True
    group["init_yyyymmdd"][:] = np.asarray(
        [int(value.strftime("%Y%m%d")) for value in dates], dtype=np.int32
    )
    group["lat"][:] = LATITUDE
    group["lon"][:] = LONGITUDE
    return path


def test_annual_cache_contract_checks_schema_units_grid_and_forbidden_year(
    tmp_path: Path,
) -> None:
    _write_annual_cache(tmp_path, 2002)
    metadata = global_pretraining.read_year_metadata(tmp_path, 2002)
    assert len(metadata.init_yyyymmdd) == 104
    assert metadata.latitude[0] == 90.0 and metadata.latitude[-1] == -90.0
    assert metadata.longitude[0] == 0.0 and metadata.longitude[-1] == 358.5

    group = zarr.open_group(str(tmp_path / "2002.zarr"), mode="a")
    group.attrs["stored_units"] = "incorrect"
    with pytest.raises(global_pretraining.DataContractError, match="units"):
        global_pretraining.read_year_metadata(tmp_path, 2002)

    _write_annual_cache(tmp_path, 2003, dynamic_dtype="f4")
    with pytest.raises(global_pretraining.DataContractError, match="dtype"):
        global_pretraining.read_year_metadata(tmp_path, 2003)

    with pytest.raises(global_pretraining.DataContractError, match=r"2018\+"):
        global_pretraining.read_year_metadata(tmp_path, 2018)


def test_india_array_fingerprint_supports_datetime64_and_binds_metadata() -> None:
    dates = np.asarray(["2018-06-01", "2018-06-04"], dtype="datetime64[D]")
    same = dates.copy()
    changed = dates.copy()
    changed[1] += np.timedelta64(1, "D")

    fingerprint = india_comparison.sha256_array(dates)
    assert len(fingerprint) == 64
    assert fingerprint == india_comparison.sha256_array(same)
    assert fingerprint != india_comparison.sha256_array(changed)
    assert fingerprint != india_comparison.sha256_array(dates.reshape(1, 2))
    assert fingerprint != india_comparison.sha256_array(dates.astype("datetime64[s]"))


def test_india_smoke_validation_subset_is_balanced_across_both_years() -> None:
    initializations = np.asarray(
        [
            *pd.date_range("2018-06-01", periods=35, freq="3D"),
            *pd.date_range("2019-06-01", periods=35, freq="3D"),
        ],
        dtype="datetime64[D]",
    )
    indices = np.arange(70, dtype=np.int64)
    selected = india_comparison.smoke_validation_indices(
        initializations, indices, cases_per_year=8
    )
    selected_years = pd.DatetimeIndex(initializations[selected]).year

    assert selected.tolist() == [*range(8), *range(35, 43)]
    assert selected_years.value_counts().to_dict() == {2018: 8, 2019: 8}


def test_date_level_purge_blocks_truth_crossing_both_evidence_boundaries() -> None:
    contract = _contract()
    metadata = {
        2015: _metadata(2015, (20151120, 20151121)),
        2016: _metadata(2016, (20160101,)),
        2017: _metadata(2017, (20171120, 20171121)),
    }
    fit, validation, fit_purged, validation_purged = (
        global_pretraining.build_case_references(contract, metadata)
    )

    assert tuple(item.init_yyyymmdd for item in fit) == (20151120,)
    assert tuple(item.init_yyyymmdd for item in validation) == (20160101, 20171120)
    assert fit_purged == 1
    assert validation_purged == 1
    assert max(
        datetime.strptime(str(item.init_yyyymmdd), "%Y%m%d") + timedelta(days=41)
        for item in fit
    ) < datetime(2016, 1, 1)
    assert max(
        datetime.strptime(str(item.init_yyyymmdd), "%Y%m%d") + timedelta(days=41)
        for item in validation
    ) < datetime(2018, 1, 1)


def test_patch_schedule_is_seed_stable_and_smoke_exercises_all_boundaries() -> None:
    cases = tuple(
        global_pretraining.CaseReference("fit", 2002, "synthetic", index, 20020101 + index)
        for index in range(4)
    )
    contract = _contract(
        mode="smoke", fit_years=(2002,), validation_years=(2003,), seed=42
    )
    np.random.seed(999)
    first = global_pretraining.build_patch_schedule(
        cases, LATITUDE, contract, split="fit"
    )
    np.random.seed(1)
    second = global_pretraining.build_patch_schedule(
        cases, LATITUDE, replace(contract, seed=44), split="fit"
    )

    assert np.array_equal(
        global_pretraining.patch_schedule_array(first),
        global_pretraining.patch_schedule_array(second),
    )
    diagnostics = global_pretraining.patch_schedule_diagnostics(first)
    assert diagnostics["longitude_wraps"] >= 2
    assert diagnostics["north_edge_patches"] >= 1
    assert diagnostics["south_edge_patches"] >= 1
    assert all(0 <= item.latitude_start <= 94 for item in first)
    assert all(0 <= item.longitude_start < 240 for item in first)


def test_patch_extraction_wraps_only_longitude_and_preserves_polar_edges() -> None:
    values = (
        np.arange(121, dtype=np.int64)[:, None] * 1000
        + np.arange(240, dtype=np.int64)[None, :]
    )
    north = global_pretraining._extract_patch(values, 0, 230)
    south = global_pretraining._extract_patch(values, 94, 230)
    expected_longitudes = np.r_[np.arange(230, 240), np.arange(17)]

    assert north.shape == (27, 27)
    assert np.array_equal(north[0] % 1000, expected_longitudes)
    assert np.array_equal(north[:, 0] // 1000, np.arange(27))
    assert np.array_equal(south[:, 0] // 1000, np.arange(94, 121))
    with pytest.raises(ValueError, match="latitude"):
        global_pretraining._extract_patch(values, -1, 0)
    with pytest.raises(ValueError, match="latitude"):
        global_pretraining._extract_patch(values, 95, 0)
    with pytest.raises(ValueError, match="longitude"):
        global_pretraining._extract_patch(values, 0, 240)


def test_patch_features_match_india_order_and_pole_weights_are_zero() -> None:
    case = global_pretraining.CaseReference("fit", 2002, "synthetic", 0, 20020601)
    schedule = (
        global_pretraining.PatchReference("fit", case, 0, 0, 230),
        global_pretraining.PatchReference("fit", case, 1, 94, 230),
    )
    contract = _contract(
        mode="smoke", fit_years=(2002,), validation_years=(2003,)
    )
    dataset = global_pretraining.GlobalPatchDataset(
        schedule, _preprocessing(), LATITUDE, contract
    )

    shape = (6, 121, 240)
    fields = global_pretraining.CaseFields(
        mean=np.full(shape, 2.0, dtype=np.float32),
        spread=np.full(shape, 0.5, dtype=np.float32),
        truth=np.full(shape, 3.0, dtype=np.float32),
        fraction=np.ones(shape, dtype=np.float32),
    )

    class Reader:
        def read_case(self, reference):
            assert reference == case
            return fields

    dataset.reader = Reader()
    north_features, north_target, north_weights = dataset[0]
    south_features, _, south_weights = dataset[1]

    assert north_features.shape == (6, 11, 27, 27)
    assert north_target.shape == north_weights.shape == (6, 27, 27)
    assert tuple(global_pretraining.FEATURE_NAMES) == india_comparison.GLOBAL_FEATURE_NAMES
    assert torch.equal(
        north_features[:, 3],
        torch.from_numpy(
            np.broadcast_to(
                np.linspace(1.0, -1.0, 27, dtype=np.float32)[:, None],
                (6, 27, 27),
            ).copy()
        ),
    )
    assert torch.equal(
        north_features[:, 4],
        torch.from_numpy(
            np.broadcast_to(
                np.linspace(-1.0, 1.0, 27, dtype=np.float32)[None, :],
                (6, 27, 27),
            ).copy()
        ),
    )
    assert torch.count_nonzero(north_features[:, 10]).item() == 0
    assert torch.count_nonzero(south_features[:, 10]).item() == 0
    assert torch.count_nonzero(north_weights[:, 0]).item() == 0
    assert torch.count_nonzero(south_weights[:, -1]).item() == 0
    assert torch.isfinite(north_target).all()


def _checkpoint_payload(seed: int = 42) -> dict:
    contract = _contract(
        mode="smoke", fit_years=(2002,), validation_years=(2003,), seed=seed
    )
    global_pretraining.set_deterministic_seed(seed)
    model = global_pretraining.build_model(contract)
    state = copy.deepcopy(model.state_dict())
    state["backbone.residual_head.weight"].fill_(3.0)
    state["backbone.residual_head.bias"].fill_(2.0)
    return {
        "schema_version": 1,
        "stage": "global_patch_pretraining",
        "model_state_dict": state,
        "architecture": global_pretraining._architecture_payload(contract),
        "feature_names": list(global_pretraining.FEATURE_NAMES),
        "contract": contract.payload(),
        "contract_sha256": contract.sha256,
        "seed": seed,
        "best_epoch": 1,
        "best_validation_loss": 0.9,
        "epoch_zero_validation_loss": 1.0,
        "source_snapshots": {
            "fuxi_global_patch_pretraining": {
                "source_path": "/synthetic/fuxi_global_patch_pretraining.py",
                "source_sha256": "e" * 64,
                "snapshot_path": "code/fuxi_global_patch_pretraining.py",
                "snapshot_sha256": "e" * 64,
            }
        },
        "selected_case_content_sha256": {
            "fit": "c" * 64,
            "validation": "d" * 64,
        },
        "transfer": {
            "reset_keys": list(global_pretraining.TRANSFER_RESET_KEYS),
            "zero_pretraining_channels": [global_pretraining.FEATURE_NAMES[10]],
        },
    }


def test_transfer_is_strict_restores_t2m_slice_and_resets_to_exact_noop(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "best.pt"
    torch.save(_checkpoint_payload(), checkpoint)
    source = india_comparison.load_checkpoint_source(checkpoint, expected_seed=42)

    torch.manual_seed(42)
    scratch = india_comparison.build_model()
    scratch_state = copy.deepcopy(scratch.state_dict())
    destination = india_comparison.build_model()
    audit = india_comparison.transfer_global_initialization(
        destination, source, scratch_state
    )
    transferred = destination.state_dict()
    source_state = source.payload["model_state_dict"]

    assert audit["reset_head_parameters"] == 17
    assert audit["restored_scratch_parameters"] == 144
    for key in india_comparison.HEAD_KEYS:
        assert torch.count_nonzero(transferred[key]).item() == 0
    first = india_comparison.FIRST_CONV_KEY
    assert torch.equal(
        transferred[first][:, india_comparison.T2M_CHANNEL_INDEX],
        scratch_state[first][:, india_comparison.T2M_CHANNEL_INDEX],
    )
    retained = [index for index in range(11) if index != 10]
    assert torch.equal(transferred[first][:, retained], source_state[first][:, retained])
    destination.eval()
    with torch.inference_mode():
        residual = destination(torch.randn(2, 6, 29, 27, 27))
    assert torch.equal(residual, torch.zeros_like(residual))


def test_incompatible_transfer_fails_before_mutating_destination(tmp_path: Path) -> None:
    payload = _checkpoint_payload()
    checkpoint = tmp_path / "best.pt"
    torch.save(payload, checkpoint)
    source = india_comparison.load_checkpoint_source(checkpoint, expected_seed=42)
    broken_payload = copy.deepcopy(source.payload)
    key = next(iter(broken_payload["model_state_dict"]))
    broken_payload["model_state_dict"][key] = torch.zeros(1)
    broken = replace(source, payload=broken_payload)
    destination = india_comparison.build_model()
    before = copy.deepcopy(destination.state_dict())

    with pytest.raises(ValueError, match="shape"):
        india_comparison.transfer_global_initialization(destination, broken, before)
    assert all(
        torch.equal(destination.state_dict()[name], value)
        for name, value in before.items()
    )


def test_full_checkpoint_set_rejects_a_heterogeneous_pretraining_recipe(
    tmp_path: Path,
) -> None:
    paths = []
    for seed in (42, 43, 44):
        payload = _checkpoint_payload(seed)
        payload["contract"]["mode"] = "full"
        payload["contract"].update(
            copy.deepcopy(india_comparison.CANONICAL_GLOBAL_RECIPES["full"])
        )
        payload["contract"]["seed"] = seed
        if seed == 44:
            payload["contract"]["patch_seed"] += 1
        payload["contract_sha256"] = global_pretraining.canonical_sha256(
            payload["contract"]
        )
        payload["normalization"] = {"same": True}
        payload["target_scale"] = np.ones(6, dtype=np.float32)
        run = tmp_path / f"seed_{seed}"
        path = run / "checkpoints" / "best.pt"
        path.parent.mkdir(parents=True)
        torch.save(payload, path)
        years = list(range(2002, 2018))
        manifest = {
            "status": "complete",
            "smoke": False,
            "scientific_eligible": False,
            "test_predictions_created": False,
            "contract_sha256": payload["contract_sha256"],
            "selection": {
                "best_epoch": 1,
                "selected_not_worse_than_epoch_zero": True,
            },
            "source_provenance": {
                "opened_years": years,
                "hard_blocked_years": "2018 and later",
                "annual": {
                    str(year): {"metadata_sha256": f"{year:064d}"[-64:]}
                    for year in years
                },
                "selected_case_content_sha256": payload[
                    "selected_case_content_sha256"
                ],
            },
            "code": payload["source_snapshots"],
            "splits": {
                "fit_schedule_sha256": "a" * 64,
                "validation_schedule_sha256": "b" * 64,
                "fit_date_bounds": {"target_date_max": "2015-12-31"},
                "validation_date_bounds": {"target_date_max": "2017-12-31"},
            },
            "artifacts": {
                "checkpoints/best.pt": global_pretraining.sha256_file(path)
            },
        }
        (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        paths.append(path)

    with pytest.raises(ValueError, match="canonical full recipe"):
        india_comparison.resolve_checkpoint_sources(
            paths, (42, 43, 44), expected_mode="full"
        )


def _promotion_inputs() -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    def pooled_row(
        configuration: str,
        member: str,
        rmse: float,
        mae: float,
        bias: float,
        acc: float,
    ) -> dict:
        row = {
            "configuration": configuration,
            "member": member,
            "pooled_rmse": rmse,
            "pooled_mae": mae,
            "pooled_bias": bias,
            "pooled_acc": acc,
            "2018_rmse": rmse + 0.1,
            "2019_rmse": rmse - 0.1,
        }
        for lead in range(1, 7):
            row[f"W{lead}_rmse"] = rmse
            row[f"W{lead}_acc"] = acc
        return row

    pooled = pd.DataFrame(
        [
            pooled_row("log_bias", "deterministic", 6.0, 4.0, -0.50, 0.20),
            pooled_row("scratch", "ensemble", 5.0, 3.0, -0.40, 0.30),
            pooled_row("global_pretrained", "ensemble", 4.8, 2.9, -0.42, 0.301),
        ]
    )
    scores = []
    for configuration, loss in (("scratch", 0.50), ("global_pretrained", 0.49)):
        row = {
            "configuration": configuration,
            "member": "ensemble",
            "composite_loss": loss,
        }
        for lead in range(1, 7):
            row[f"W{lead}_composite_loss"] = loss
        scores.append(row)
    records = [
        {
            "configuration": configuration,
            "seed": seed,
            "best_validation_loss": loss,
            "initialization": (
                {"source_best_epoch": 1}
                if configuration == "global_pretrained"
                else {"kind": "scratch"}
            ),
        }
        for seed in (42, 43, 44)
        for configuration, loss in (("scratch", 0.50), ("global_pretrained", 0.49))
    ]
    return pooled, pd.DataFrame(scores), records


def test_promotion_gate_promotes_only_when_scientific_and_every_guard_passes() -> None:
    pooled, scores, records = _promotion_inputs()
    promoted = india_comparison.build_promotion_gate(
        pooled, scores, records, scientific_eligible=True
    )
    smoke = india_comparison.build_promotion_gate(
        pooled, scores, records, scientific_eligible=False
    )
    failed_pooled = pooled.copy()
    failed_pooled.loc[
        failed_pooled.configuration.eq("global_pretrained"), "2019_rmse"
    ] = 5.1
    failed = india_comparison.build_promotion_gate(
        failed_pooled, scores, records, scientific_eligible=True
    )

    assert promoted["global_pretraining_qualifies"] is True
    assert promoted["selected_system"] == "global_pretrained"
    assert smoke["global_pretraining_qualifies"] is False
    assert smoke["selected_system"] == "none_smoke_is_non_scientific"
    assert failed["conditions"]["rmse_improves_in_2019"] is False
    assert failed["selected_system"] == "scratch"


def test_pretraining_cannot_promote_when_both_neural_models_lose_to_anchor() -> None:
    pooled, scores, records = _promotion_inputs()
    anchor = pooled.configuration.eq("log_bias")
    pooled.loc[anchor, ["pooled_rmse", "2018_rmse", "2019_rmse"]] = [
        4.0,
        4.1,
        3.9,
    ]
    pooled.loc[anchor, "pooled_mae"] = 2.5
    result = india_comparison.build_promotion_gate(
        pooled, scores, records, scientific_eligible=True
    )

    assert result["conditions"]["rmse_improves_in_2018"] is True
    assert (
        result["global_pretrained_anchor_conditions"][
            "pooled_rmse_improves_anchor"
        ]
        is False
    )
    assert result["global_pretraining_qualifies"] is False
    assert result["selected_system"] == "log_bias"


def test_smoke_manifest_is_atomic_explicitly_non_scientific_and_boundary_audited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _contract(
        mode="smoke", fit_years=(2002,), validation_years=(2003,)
    )
    metadata = {
        2002: _metadata(2002, (20020101, 20020601)),
        2003: _metadata(2003, (20030101, 20030601)),
    }
    preprocessing = _preprocessing()
    monkeypatch.setattr(
        global_pretraining, "load_allowed_metadata", lambda selected: metadata
    )
    monkeypatch.setattr(
        global_pretraining,
        "fit_global_preprocessing",
        lambda cases, latitude, selected: preprocessing,
    )

    def fake_train(model, fit_data, validation_data, selected, **kwargs):
        del fit_data, validation_data, selected, kwargs
        state = global_pretraining._cpu_state_dict(model)
        history = [
            {
                "epoch": 0,
                "train_loss": None,
                "validation_loss": 1.0,
                "learning_rate": 2.0e-4,
                "selected": True,
                "fallback_anchor": True,
            }
        ]
        return state, history, 0, 1.0, 1.0

    monkeypatch.setattr(global_pretraining, "train_model", fake_train)
    monkeypatch.setattr(
        global_pretraining,
        "fingerprint_case_content",
        lambda references: "d" * 64,
    )
    output_root = tmp_path / "runs"
    output = global_pretraining.run_pretraining(
        contract,
        output_root=output_root,
        run_name="smoke_seed42",
        device_name="cpu",
        num_workers=0,
    )

    assert output == output_root / "smoke_seed42"
    assert not list(output_root.glob(".smoke_seed42.partial-*"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["smoke"] is True
    assert manifest["scientific_eligible"] is False
    assert manifest["test_predictions_created"] is False
    assert manifest["source_provenance"]["opened_years"] == [2002, 2003]
    assert manifest["splits"]["fit_date_bounds"] == {
        "initialization_date_min": "2002-01-01",
        "initialization_date_max": "2002-06-01",
        "target_date_min": "2002-01-01",
        "target_date_max": "2002-07-12",
    }
    assert manifest["splits"]["validation_date_bounds"]["target_date_max"] == (
        "2003-07-12"
    )
    for split in ("fit", "validation"):
        diagnostics = manifest["splits"][f"{split}_schedule_diagnostics"]
        assert diagnostics["longitude_wraps"] == 2
        assert diagnostics["north_edge_patches"] == 1
        assert diagnostics["south_edge_patches"] == 1
    assert (output / "checkpoints" / "best.pt").is_file()


def test_gpu_launchers_encode_blacklists_cuda_probe_and_manual_smoke_gate() -> None:
    smoke = (ROOT / "slurm" / "run_global_pretrain_india_smoke.sbatch").read_text(
        encoding="utf-8"
    )
    full = (ROOT / "slurm" / "run_global_pretrain_india_full.sbatch").read_text(
        encoding="utf-8"
    )
    seed_array = (
        ROOT / "slurm" / "run_global_pretrain_seed_array.sbatch"
    ).read_text(encoding="utf-8")
    comparison = (
        ROOT / "slurm" / "run_global_pretrain_india_compare.sbatch"
    ).read_text(encoding="utf-8")

    assert "#SBATCH --exclude=cn2,cn3,cn4,cn15,cn16,cn17\n" in smoke
    assert "#SBATCH --partition=gpu_prio\n" in full
    for launcher in (full, seed_array, comparison):
        assert "#SBATCH --partition=gpu_prio\n" in launcher
        assert "#SBATCH --exclude=cn2,cn3,cn4,cn15,cn16,cn17\n" in launcher
        assert "nvidia-smi --query-gpu=" in launcher
    assert "nvidia-smi --query-gpu=" in smoke and "nvidia-smi --query-gpu=" in full
    assert "--mode smoke" in smoke and "--device cuda" in smoke
    assert "COMPLETED_SMOKE_MANIFEST" in full
    assert 'm["status"]=="complete_smoke"' in full
    assert "SEEDS=(42 43 44)" in full
    assert "#SBATCH --array=0-2%3\n" in seed_array
    assert "SEEDS=(42 43 44)" in seed_array
    assert "SEED=${SEEDS[${SLURM_ARRAY_TASK_ID}]}" in seed_array
    assert "RUN_DIR=${PRETRAIN_ROOT}/${RUN_NAME}" in seed_array
    assert "--mode full" in seed_array
    assert "COMPLETED_SMOKE_MANIFEST" in seed_array
    assert "--dependency=afterok:ARRAY_JOB_ID" in comparison
    assert "for SEED in 42 43 44" in comparison
    assert "--pretrained-checkpoint" in comparison
    assert "--full" in comparison
