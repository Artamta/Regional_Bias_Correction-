#!/usr/bin/env python3
"""Run the aligned FuXi bridge without opening verification targets.

The command applies the single validation-selected calibration checkpoint to
the 2020--2024 operational FuXi ensemble.  It writes compact adjustment fields
and ensemble means; corrected members remain exactly reconstructible from the
immutable raw archive plus those adjustments.  Verification truth is outside
this module's access path.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import xarray as xr

import fuxi_allseason_ensemble_calibration as calibration
import india_s2s_probabilistic_bridge as bridge
from fuxi_ensemble_calibration_core import EnsembleLocationSpreadCalibrator
from project_paths import PROJECT_ROOT


EXPERIMENT = "india_s2s_probabilistic_bridge_inference_v1"
TRAIN_CONTEXT_YEARS = tuple(range(2002, 2018))
FORECAST_YEARS = bridge.YEARS
EXPECTED_MEMBER_COUNT = 50
EXPECTED_GRID_SHAPE = (27, 27)
EXPECTED_FIELD_SHAPE = (6, *EXPECTED_GRID_SHAPE)
SMOKE_CASE_COUNT = 5
ALLOWED_SHARD_KEYS = {
    "initializations",
    "scoreable_for_truth",
    "delta_log_location",
    "log_spread",
    "raw_ensemble_mean",
    "corrected_ensemble_mean",
    "latitude",
    "longitude",
    "member_count",
    "selected_seed",
}


@dataclass(frozen=True)
class FrozenParent:
    """Validated parent inputs required for forecast-only inference."""

    manifest_path: Path
    run_root: Path
    manifest: Mapping[str, Any]
    receipt: Mapping[str, Any]
    checkpoint: Path
    support_artifact: Path
    normalization_mean: np.ndarray
    normalization_std: np.ndarray


@dataclass(frozen=True)
class ContextInputs:
    """Training-only fields needed to reproduce the seven context channels."""

    daily_climatology: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    support: np.ndarray
    observation_fraction: np.ndarray
    weights: np.ndarray
    normalization_mean: np.ndarray
    normalization_std: np.ndarray


def _resolve_child(root: Path, relative: str) -> Path:
    candidate_relative = Path(relative)
    if candidate_relative.is_absolute():
        raise bridge.BridgeContractError(f"parent artifact path is absolute: {relative}")
    candidate = (root / candidate_relative).resolve()
    if root.resolve() not in candidate.parents:
        raise bridge.BridgeContractError(f"parent artifact escapes run: {relative}")
    return candidate


def _require_file_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise bridge.BridgeContractError(f"{label} is missing: {path}")
    observed = bridge.sha256_file(path)
    if observed != expected:
        raise bridge.BridgeContractError(
            f"{label} hash differs: {observed} != {expected}"
        )


def _path_mentions_sealed_2025(path: Path) -> bool:
    return any("2025" in component for component in Path(path).parts)


def load_frozen_parent(path: Path) -> FrozenParent:
    """Run the parent gate and bind all preprocessing/model inputs."""

    receipt = bridge.validate_parent_manifest(path)
    manifest_path = Path(receipt["parent_manifest"]).resolve()
    run_root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(receipt["selected_seed"]) != 43:
        raise bridge.BridgeContractError("canonical bridge requires selected seed 43")

    model_contract = manifest.get("training", {}).get("model", {})
    expected_model = {
        "class": "EnsembleLocationSpreadCalibrator",
        "context_channels": 7,
        "member_hidden_channels": 8,
        "backbone_channels": 24,
        "dropout": 0.05,
        "max_abs_log_spread": 2.0,
        "permutation_invariant_member_axis": True,
    }
    if model_contract != expected_model:
        raise bridge.BridgeContractError(
            f"parent model contract differs: {model_contract!r}"
        )

    normalization = manifest.get("normalization", {})
    mean = np.asarray(
        normalization.get("climatology_log1p_mean_by_lead", []), dtype=np.float32
    )
    std = np.asarray(
        normalization.get("climatology_log1p_std_by_lead", []), dtype=np.float32
    )
    if mean.shape != (6,) or std.shape != (6,) or not np.isfinite(mean).all():
        raise bridge.BridgeContractError("parent normalization means are invalid")
    if not np.isfinite(std).all() or np.any(std <= 0.0):
        raise bridge.BridgeContractError("parent normalization standard deviations are invalid")
    train_count = int(manifest.get("split_counts_selected", {}).get("train", -1))
    fit_indices = normalization.get("fit_indices", [])
    retained_train = manifest.get("retained_initializations", {}).get("train", [])
    if train_count != 1652 or len(fit_indices) != train_count or len(retained_train) != train_count:
        raise bridge.BridgeContractError("parent normalization is not bound to 1,652 training starts")

    checkpoint = _resolve_child(run_root, str(receipt["selected_checkpoint"]))
    _require_file_hash(
        checkpoint, str(receipt["selected_checkpoint_sha256"]), "selected checkpoint"
    )
    evaluation = manifest.get("evaluation", {})
    support_relative = evaluation.get("scoring_support_artifact")
    support_hash = evaluation.get("scoring_support_sha256")
    if not isinstance(support_relative, str) or not isinstance(support_hash, str):
        raise bridge.BridgeContractError("parent scoring support is absent")
    support_artifact = _resolve_child(run_root, support_relative)
    _require_file_hash(support_artifact, support_hash, "parent scoring support")
    if manifest.get("artifact_sha256", {}).get(support_relative) != support_hash:
        raise bridge.BridgeContractError("parent scoring support is not artifact-bound")

    # The live functions used below must be byte-identical to the frozen parent
    # snapshot.  Refuse inference after any preprocessing/model-code drift.
    live_sources = {
        "fuxi_allseason_ensemble_calibration.py": Path(calibration.__file__).resolve(),
        "fuxi_ensemble_calibration_core.py": (
            PROJECT_ROOT / "src" / "fuxi_ensemble_calibration_core.py"
        ).resolve(),
    }
    snapshot_hashes = manifest.get("source_snapshot_sha256", {})
    for name, live_path in live_sources.items():
        matches = [value for key, value in snapshot_hashes.items() if Path(key).name == name]
        if len(matches) != 1:
            raise bridge.BridgeContractError(f"parent snapshot for {name} is ambiguous")
        _require_file_hash(live_path, matches[0], f"live {name}")

    observation_stores = [Path(item) for item in manifest.get("observation_stores", [])]
    expected_prefix = [calibration.IMD_DAILY / f"{year}.zarr" for year in TRAIN_CONTEXT_YEARS]
    if observation_stores[: len(expected_prefix)] != expected_prefix:
        raise bridge.BridgeContractError("parent training IMD store list differs")
    if any(_path_mentions_sealed_2025(path) for path in observation_stores):
        raise bridge.BridgeContractError("parent observation list contains sealed 2025")

    return FrozenParent(
        manifest_path=manifest_path,
        run_root=run_root,
        manifest=manifest,
        receipt=receipt,
        checkpoint=checkpoint,
        support_artifact=support_artifact,
        normalization_mean=mean,
        normalization_std=std,
    )


def load_support(parent: FrozenParent) -> ContextInputs:
    """Load the parent's small, hash-bound spatial support artifact."""

    with np.load(parent.support_artifact, allow_pickle=False) as archive:
        required = {
            "latitude",
            "longitude",
            "observation_fraction",
            "support_mask",
            "scoring_weight_km2_fraction",
        }
        if not required.issubset(archive.files):
            raise bridge.BridgeContractError("parent scoring support keys differ")
        latitude = np.asarray(archive["latitude"], dtype=np.float64)
        longitude = np.asarray(archive["longitude"], dtype=np.float64)
        fraction = np.asarray(archive["observation_fraction"], dtype=np.float32)
        support = np.asarray(archive["support_mask"], dtype=bool)
        weights = np.asarray(archive["scoring_weight_km2_fraction"], dtype=np.float64)
    if latitude.shape != (27,) or longitude.shape != (27,):
        raise bridge.BridgeContractError("parent scoring grid is not 27 x 27")
    if fraction.shape != EXPECTED_GRID_SHAPE or support.shape != EXPECTED_GRID_SHAPE:
        raise bridge.BridgeContractError("parent support shape differs")
    if weights.shape != EXPECTED_GRID_SHAPE or np.count_nonzero(support) != 171:
        raise bridge.BridgeContractError("parent support-cell contract differs")
    if not np.array_equal(support, weights > 0.0):
        raise bridge.BridgeContractError("parent support mask differs from positive weights")
    return ContextInputs(
        daily_climatology=np.empty((0, *EXPECTED_GRID_SHAPE), dtype=np.float32),
        latitude=latitude,
        longitude=longitude,
        support=support,
        observation_fraction=fraction,
        weights=weights,
        normalization_mean=parent.normalization_mean.copy(),
        normalization_std=parent.normalization_std.copy(),
    )


def _weekly_climatology(
    initializations: np.ndarray, daily_climatology: np.ndarray
) -> np.ndarray:
    valid_dates = bridge.target_valid_dates(initializations)
    positions = calibration.calendar_positions(valid_dates)
    return np.mean(
        np.asarray(daily_climatology, dtype=np.float32)[positions],
        axis=2,
        dtype=np.float64,
    ).astype(np.float32)


def verify_parent_normalization(parent: FrozenParent, context: ContextInputs) -> None:
    """Recompute the frozen lead moments from its exact training starts."""

    train_initializations = np.asarray(
        parent.manifest["retained_initializations"]["train"], dtype="datetime64[D]"
    )
    weekly = _weekly_climatology(train_initializations, context.daily_climatology)
    log_weekly = np.log1p(weekly).astype(np.float32)
    indices = np.arange(train_initializations.size, dtype=np.int64)
    mean, std = calibration.weighted_lead_moments(log_weekly, indices, context.weights)
    if not np.allclose(mean, context.normalization_mean, rtol=0.0, atol=5.0e-7):
        raise bridge.BridgeContractError("recomputed climatology means differ from parent")
    if not np.allclose(std, context.normalization_std, rtol=0.0, atol=5.0e-7):
        raise bridge.BridgeContractError("recomputed climatology stds differ from parent")


def build_training_context(parent: FrozenParent) -> tuple[ContextInputs, list[dict[str, Any]]]:
    """Build the 2002--2017 daily climatology without opening test truth."""

    base = load_support(parent)
    all_dates: list[np.ndarray] = []
    all_values: list[np.ndarray] = []
    sources: list[dict[str, Any]] = []
    for year in TRAIN_CONTEXT_YEARS:
        store = calibration.IMD_DAILY / f"{year}.zarr"
        zmetadata = store / ".zmetadata"
        if not zmetadata.is_file():
            raise FileNotFoundError(zmetadata)
        with xr.open_zarr(store, consolidated=True) as dataset:
            if dataset.attrs.get("source") != "imd" or dataset.attrs.get("units") != "mm day-1":
                raise bridge.BridgeContractError(f"unexpected training IMD metadata: {store}")
            if not np.array_equal(dataset.latitude.values, base.latitude):
                raise bridge.BridgeContractError(f"training IMD latitude differs: {store}")
            if not np.array_equal(dataset.longitude.values, base.longitude):
                raise bridge.BridgeContractError(f"training IMD longitude differs: {store}")
            dates = np.asarray(dataset.time.values, dtype="datetime64[D]")
            values = np.asarray(dataset.observation.load().values, dtype=np.float32)
            fraction = calibration.collapse_fraction(dataset, EXPECTED_GRID_SHAPE)
        if dates.size not in (365, 366) or values.shape != (dates.size, 27, 27):
            raise bridge.BridgeContractError(f"incomplete training IMD year: {store}")
        if not np.all(np.diff(dates) == np.timedelta64(1, "D")):
            raise bridge.BridgeContractError(f"non-daily training IMD calendar: {store}")
        if not np.allclose(
            fraction, base.observation_fraction, rtol=0.0, atol=1.0e-7, equal_nan=True
        ):
            raise bridge.BridgeContractError(f"training IMD support differs: {store}")
        if not np.isfinite(values[:, base.support]).all() or np.any(values[:, base.support] < 0.0):
            raise bridge.BridgeContractError(f"invalid supported training IMD values: {store}")
        all_dates.append(dates)
        all_values.append(values)
        sources.append(
            {
                "year": year,
                "store": str(store),
                "zmetadata_sha256": bridge.sha256_file(zmetadata),
            }
        )
    dates = np.concatenate(all_dates)
    values = np.concatenate(all_values)
    daily_climatology = calibration.build_training_climatology(
        dates, values, base.support
    )
    context = ContextInputs(
        daily_climatology=daily_climatology,
        latitude=base.latitude,
        longitude=base.longitude,
        support=base.support,
        observation_fraction=base.observation_fraction,
        weights=base.weights,
        normalization_mean=base.normalization_mean,
        normalization_std=base.normalization_std,
    )
    verify_parent_normalization(parent, context)
    return context, sources


def _atomic_savez(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _save_context_artifact(
    output: Path,
    context: ContextInputs,
    sources: Sequence[Mapping[str, Any]],
    parent: FrozenParent,
) -> None:
    artifact = output / "context" / "training_context.npz"
    _atomic_savez(
        artifact,
        daily_climatology=context.daily_climatology,
        latitude=context.latitude,
        longitude=context.longitude,
        support=context.support,
        observation_fraction=context.observation_fraction,
        weights=context.weights,
        normalization_mean=context.normalization_mean,
        normalization_std=context.normalization_std,
    )
    bridge.write_json(
        output / "context" / "receipt.json",
        {
            "artifact": str(artifact.relative_to(output)),
            "artifact_sha256": bridge.sha256_file(artifact),
            "parent_manifest_sha256": parent.receipt["parent_manifest_sha256"],
            "training_years": list(TRAIN_CONTEXT_YEARS),
            "training_source_zmetadata": list(sources),
            "normalization_mean": context.normalization_mean.tolist(),
            "normalization_std": context.normalization_std.tolist(),
            "normalization_recomputed_equal": True,
            "target_day_offsets": list(bridge.TARGET_DAY_OFFSETS),
            "verification_truth_opened": False,
            "sealed_2025_target_opened": False,
        },
    )


def _load_context_artifact(output: Path, parent: FrozenParent) -> ContextInputs:
    artifact = output / "context" / "training_context.npz"
    receipt_path = output / "context" / "receipt.json"
    if artifact.exists() != receipt_path.exists():
        raise bridge.BridgeContractError("partial context artifact cannot be resumed")
    if not artifact.exists():
        context, sources = build_training_context(parent)
        _save_context_artifact(output, context, sources, parent)
        return context
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("parent_manifest_sha256") != parent.receipt["parent_manifest_sha256"]:
        raise bridge.BridgeContractError("context parent hash differs on resume")
    _require_file_hash(artifact, receipt.get("artifact_sha256", ""), "context artifact")
    if receipt.get("training_years") != list(TRAIN_CONTEXT_YEARS):
        raise bridge.BridgeContractError("context training years differ on resume")
    if receipt.get("verification_truth_opened") is not False or receipt.get(
        "sealed_2025_target_opened"
    ) is not False:
        raise bridge.BridgeContractError("context receipt reports verification target access")
    with np.load(artifact, allow_pickle=False) as archive:
        context = ContextInputs(
            daily_climatology=np.asarray(archive["daily_climatology"], dtype=np.float32),
            latitude=np.asarray(archive["latitude"], dtype=np.float64),
            longitude=np.asarray(archive["longitude"], dtype=np.float64),
            support=np.asarray(archive["support"], dtype=bool),
            observation_fraction=np.asarray(archive["observation_fraction"], dtype=np.float32),
            weights=np.asarray(archive["weights"], dtype=np.float64),
            normalization_mean=np.asarray(archive["normalization_mean"], dtype=np.float32),
            normalization_std=np.asarray(archive["normalization_std"], dtype=np.float32),
        )
    if context.daily_climatology.shape != (366, 27, 27):
        raise bridge.BridgeContractError("context climatology shape differs")
    if not np.array_equal(context.normalization_mean, parent.normalization_mean):
        raise bridge.BridgeContractError("context normalization mean differs from parent")
    if not np.array_equal(context.normalization_std, parent.normalization_std):
        raise bridge.BridgeContractError("context normalization std differs from parent")
    return context


def build_frozen_context(
    initializations: np.ndarray, context: ContextInputs
) -> calibration.ContextBundle:
    """Build the exact seven parent context channels for new starts."""

    starts = np.asarray(initializations, dtype="datetime64[D]")
    if starts.ndim != 1 or starts.size == 0:
        raise bridge.BridgeContractError("context initializations must be non-empty")
    weekly = _weekly_climatology(starts, context.daily_climatology)
    normalized = (np.log1p(weekly) - context.normalization_mean[None, :, None, None]) / context.normalization_std[
        None, :, None, None
    ]
    normalized = np.where(
        context.support[None, None] & np.isfinite(normalized), normalized, 0.0
    ).astype(np.float32)
    latitude = context.latitude.astype(np.float32)
    longitude = context.longitude.astype(np.float32)
    lat_scaled = 2.0 * (latitude - latitude.min()) / (latitude.max() - latitude.min()) - 1.0
    lon_scaled = 2.0 * (longitude - longitude.min()) / (longitude.max() - longitude.min()) - 1.0
    midpoints = bridge.target_valid_dates(starts)[:, :, 3]
    midpoint_index = pd.DatetimeIndex(midpoints.reshape(-1))
    day_of_year = (midpoint_index.dayofyear.to_numpy() - 1).reshape(starts.size, 6)
    angle = 2.0 * np.pi * day_of_year / 365.2425
    return calibration.ContextBundle(
        normalized_climatology=normalized,
        climatology_mean_by_lead=context.normalization_mean.copy(),
        climatology_std_by_lead=context.normalization_std.copy(),
        latitude_scaled=lat_scaled,
        longitude_scaled=lon_scaled,
        season_sin=np.sin(angle).astype(np.float32),
        season_cos=np.cos(angle).astype(np.float32),
        lead_scaled=np.linspace(-1.0, 1.0, 6, dtype=np.float32),
        support=context.support.copy(),
    )


def _selected_validation_crps(receipt: Mapping[str, Any]) -> float:
    selected_seed = int(receipt["selected_seed"])
    selected_records = [
        record
        for record in receipt["candidate_validation_crps"]
        if int(record["seed"]) == selected_seed
    ]
    if len(selected_records) != 1:
        raise bridge.BridgeContractError("selected seed has no unique validation-CRPS record")
    return float(selected_records[0]["best_validation_crps"])


def load_selected_model(
    parent: FrozenParent, device: torch.device
) -> EnsembleLocationSpreadCalibrator:
    """Load only the parent-selected seed-43 state dictionary."""

    checkpoint = torch.load(parent.checkpoint, map_location=device, weights_only=False)
    expected = {
        "experiment": calibration.EXPERIMENT,
        "configuration": calibration.PRIMARY_CONFIGURATION,
        "seed": 43,
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise bridge.BridgeContractError(
                f"checkpoint {key} differs: {checkpoint.get(key)!r} != {value!r}"
            )
    expected_crps = _selected_validation_crps(parent.receipt)
    if not np.isclose(float(checkpoint.get("validation_crps", np.nan)), expected_crps):
        raise bridge.BridgeContractError("checkpoint validation CRPS differs from selection")
    contract = parent.manifest["training"]["model"]
    model = EnsembleLocationSpreadCalibrator(
        context_channels=int(contract["context_channels"]),
        member_hidden_channels=int(contract["member_hidden_channels"]),
        backbone_channels=int(contract["backbone_channels"]),
        mode=calibration.PRIMARY_CONFIGURATION,
        dropout=float(contract["dropout"]),
        max_abs_log_spread=float(contract["max_abs_log_spread"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model


def reconstruct_corrected_members(
    raw_members: np.ndarray,
    delta_log_location: np.ndarray,
    log_spread: np.ndarray,
) -> np.ndarray:
    """Reconstruct the exact 50-member forecast represented by a shard."""

    raw = np.asarray(raw_members, dtype=np.float32)
    if raw.ndim != 5 or raw.shape[1:] != (50, 6, 27, 27):
        raise bridge.BridgeContractError(
            f"raw operational members must be [case,50,6,27,27], found {raw.shape}"
        )
    return calibration.apply_affine_log_calibration(
        raw,
        np.asarray(delta_log_location, dtype=np.float32),
        np.exp(np.clip(np.asarray(log_spread, dtype=np.float32), -2.0, 2.0)).astype(
            np.float32
        ),
    )


def _add_benchmark_source_path() -> None:
    path = str(bridge.BENCHMARK_CODE_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)


def _forecast_loader() -> Any:
    _add_benchmark_source_path()
    return bridge.build_benchmark_loader()


def _forecast_record(loader: Any, year: int) -> dict[str, Any]:
    if year not in FORECAST_YEARS:
        raise bridge.BridgeContractError(f"forecast year is outside 2020-2024: {year}")
    frame = loader.list_forecasts(
        model="fuxi_s2s",
        variable="tp",
        year=year,
        grid="common_1p5",
        experiment_id=bridge.FUXI_EXPERIMENT_ID,
    )
    if len(frame) != 1:
        raise bridge.BridgeContractError(f"expected one FuXi forecast record for {year}")
    record = frame.iloc[0].to_dict()
    manifest_path = Path(record["manifest"])
    store = Path(record["store"])
    zmetadata = store / ".zmetadata"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("year", -1)) != year or manifest.get("status") != "complete":
        raise bridge.BridgeContractError(f"forecast manifest is invalid for {year}")
    if manifest.get("experiment_id") != bridge.FUXI_EXPERIMENT_ID:
        raise bridge.BridgeContractError(f"forecast experiment differs for {year}")
    expected_zmetadata = str(record["zmetadata_sha256"])
    _require_file_hash(zmetadata, expected_zmetadata, f"FuXi {year} .zmetadata")
    if manifest.get("zmetadata_sha256") != expected_zmetadata:
        raise bridge.BridgeContractError(f"forecast manifest/catalog hash differs for {year}")
    return {
        "year": year,
        "store": str(store),
        "manifest": str(manifest_path),
        "manifest_sha256": bridge.sha256_file(manifest_path),
        "zmetadata_sha256": expected_zmetadata,
    }


def validate_forecast_dataset(
    dataset: xr.Dataset,
    *,
    year: int,
    canonical_initializations: np.ndarray,
    context: ContextInputs,
) -> None:
    """Validate the raw 50-member weekly operational archive contract."""

    if year not in FORECAST_YEARS:
        raise bridge.BridgeContractError("dataset year escapes the forecast contract")
    attrs = dataset.attrs
    expected_attrs = {
        "model": "fuxi_s2s",
        "variable": "tp",
        "grid_id": "common_1p5",
        "experiment_id": bridge.FUXI_EXPERIMENT_ID,
        "distribution_representation": "members",
    }
    for key, value in expected_attrs.items():
        if attrs.get(key) != value:
            raise bridge.BridgeContractError(f"FuXi {year} {key} differs")
    if "forecast_weekly_mean" not in dataset or "member_available" not in dataset:
        raise bridge.BridgeContractError(f"FuXi {year} member fields are absent")
    variable = dataset["forecast_weekly_mean"]
    if variable.dims != ("init", "member", "lead_week", "latitude", "longitude"):
        raise bridge.BridgeContractError(f"FuXi {year} weekly-member dimensions differ")
    expected_count = bridge.EXPECTED_FORECAST_COUNTS[year]
    if variable.shape != (expected_count, 50, 6, 27, 27):
        raise bridge.BridgeContractError(f"FuXi {year} weekly-member shape differs")
    if variable.attrs.get("units") != "mm day-1":
        raise bridge.BridgeContractError(f"FuXi {year} units differ")
    if not np.array_equal(dataset.member.values, np.arange(50)):
        raise bridge.BridgeContractError(f"FuXi {year} member labels differ")
    if not np.array_equal(dataset.lead_week.values, np.arange(1, 7)):
        raise bridge.BridgeContractError(f"FuXi {year} lead-week labels differ")
    if not np.array_equal(dataset.latitude.values, context.latitude):
        raise bridge.BridgeContractError(f"FuXi {year} latitude differs")
    if not np.array_equal(dataset.longitude.values, context.longitude):
        raise bridge.BridgeContractError(f"FuXi {year} longitude differs")
    observed_init = np.asarray(dataset.init.values, dtype="datetime64[D]")
    if not np.array_equal(observed_init, canonical_initializations):
        raise bridge.BridgeContractError(f"FuXi {year} initializations differ")
    available = np.asarray(dataset.member_available.load().values, dtype=bool)
    if available.shape != (expected_count, 50) or not available.all():
        raise bridge.BridgeContractError(f"FuXi {year} does not have all 50 members")


@torch.no_grad()
def infer_adjustments(
    model: EnsembleLocationSpreadCalibrator,
    raw_members: np.ndarray,
    context_fields: np.ndarray,
    *,
    device: torch.device,
    use_amp: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Infer compact adjustment fields for one in-memory batch."""

    members = torch.as_tensor(raw_members, dtype=torch.float32, device=device)
    context = torch.as_tensor(context_fields, dtype=torch.float32, device=device)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
        enabled=use_amp and device.type == "cuda",
    ):
        parameters = model.predict_parameters(members, context=context)
    delta = parameters.delta_log_location.detach().float().cpu().numpy().astype(np.float32)
    log_spread = parameters.log_spread.detach().float().cpu().numpy().astype(np.float32)
    expected = (raw_members.shape[0], *EXPECTED_FIELD_SHAPE)
    if delta.shape != expected or log_spread.shape != expected:
        raise bridge.BridgeContractError(
            f"model adjustment shapes differ: {delta.shape}, {log_spread.shape}"
        )
    if not np.isfinite(delta).all() or not np.isfinite(log_spread).all():
        raise bridge.BridgeContractError("model adjustments are non-finite")
    if np.any(np.abs(log_spread) > 2.00001):
        raise bridge.BridgeContractError("model log-spread exceeds the frozen bound")
    return delta, log_spread


def _selection_receipt(
    all_initializations: np.ndarray, selected: np.ndarray
) -> dict[str, Any]:
    scoreable = bridge.scoring_mask(selected)
    if np.any(bridge.target_valid_dates(selected[scoreable])[:, -1, -1] > bridge.LAST_ALLOWED_TARGET_DATE):
        raise bridge.BridgeContractError("scoreable selection crosses the 2025 firewall")
    return {
        "archive_count": int(all_initializations.size),
        "selected_count": int(selected.size),
        "selected_dates_sha256": bridge.initialization_dates_sha256(selected),
        "scoreable_count": int(np.count_nonzero(scoreable)),
        "forecast_only_count": int(np.count_nonzero(~scoreable)),
        "scoreable_dates_sha256": bridge.initialization_dates_sha256(selected[scoreable]),
        "first": np.datetime_as_string(selected[0], unit="D"),
        "last": np.datetime_as_string(selected[-1], unit="D"),
        "verification_truth_opened": False,
        "sealed_2025_target_opened": False,
    }


def select_initializations(
    initializations: np.ndarray, *, smoke: bool, max_cases: int | None
) -> np.ndarray:
    """Return a deterministic bounded cross-archive inference cohort."""

    values = np.asarray(initializations, dtype="datetime64[D]")
    if values.size != 517:
        raise bridge.BridgeContractError("canonical inference archive must contain 517 starts")
    if smoke and max_cases is not None:
        raise ValueError("use either --smoke or --max-cases, not both")
    count = SMOKE_CASE_COUNT if smoke else (values.size if max_cases is None else max_cases)
    if count < 1 or count > values.size:
        raise ValueError("max cases must be in 1..517")
    if count == values.size:
        return values.copy()
    indices = np.linspace(0, values.size - 1, count, dtype=np.int64)
    if np.unique(indices).size != count:
        raise RuntimeError("bounded selection produced duplicate cases")
    return values[indices]


def _content_digest_update(digest: Any, values: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(values)
    digest.update(memoryview(contiguous).cast("B"))


def _infer_year(
    *,
    year: int,
    selected_initializations: np.ndarray,
    canonical_year_initializations: np.ndarray,
    loader: Any,
    model: EnsembleLocationSpreadCalibrator,
    context_inputs: ContextInputs,
    device: torch.device,
    batch_size: int,
    use_amp: bool,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    source = _forecast_record(loader, year)
    dataset = loader.open_forecast_dataset(
        model="fuxi_s2s",
        variable="tp",
        year=year,
        grid="common_1p5",
        experiment_id=bridge.FUXI_EXPERIMENT_ID,
    )
    try:
        validate_forecast_dataset(
            dataset,
            year=year,
            canonical_initializations=canonical_year_initializations,
            context=context_inputs,
        )
        archive_initializations = np.asarray(dataset.init.values, dtype="datetime64[D]")
        positions = np.searchsorted(archive_initializations, selected_initializations)
        if np.any(positions >= archive_initializations.size) or not np.array_equal(
            archive_initializations[positions], selected_initializations
        ):
            raise bridge.BridgeContractError(f"selected FuXi {year} starts are absent")
        frozen_context = build_frozen_context(selected_initializations, context_inputs)
        count = selected_initializations.size
        delta = np.empty((count, *EXPECTED_FIELD_SHAPE), dtype=np.float32)
        log_spread = np.empty_like(delta)
        raw_mean = np.empty_like(delta)
        corrected_mean = np.empty_like(delta)
        raw_digest = hashlib.sha256()
        variable = dataset["forecast_weekly_mean"]
        for begin in range(0, count, batch_size):
            end = min(begin + batch_size, count)
            batch_positions = positions[begin:end]
            raw = np.asarray(variable.isel(init=batch_positions).load().values, dtype=np.float32)
            expected = (end - begin, 50, 6, 27, 27)
            if raw.shape != expected or not np.isfinite(raw).all() or np.any(raw < 0.0):
                raise bridge.BridgeContractError(f"invalid FuXi {year} raw batch {begin}:{end}")
            _content_digest_update(raw_digest, raw)
            fields = np.stack(
                [calibration.context_for_case(frozen_context, index) for index in range(begin, end)]
            )
            batch_delta, batch_log_spread = infer_adjustments(
                model, raw, fields, device=device, use_amp=use_amp
            )
            corrected = reconstruct_corrected_members(raw, batch_delta, batch_log_spread)
            delta[begin:end] = batch_delta
            log_spread[begin:end] = batch_log_spread
            raw_mean[begin:end] = raw.mean(axis=1, dtype=np.float64).astype(np.float32)
            corrected_mean[begin:end] = corrected.mean(axis=1, dtype=np.float64).astype(np.float32)
        arrays = {
            "initializations": selected_initializations.astype("datetime64[D]"),
            "scoreable_for_truth": bridge.scoring_mask(selected_initializations),
            "delta_log_location": delta,
            "log_spread": log_spread,
            "raw_ensemble_mean": raw_mean,
            "corrected_ensemble_mean": corrected_mean,
            "latitude": context_inputs.latitude,
            "longitude": context_inputs.longitude,
            "member_count": np.asarray(EXPECTED_MEMBER_COUNT, dtype=np.int16),
            "selected_seed": np.asarray(43, dtype=np.int16),
        }
        source.update(
            {
                "archive_initialization_count": int(archive_initializations.size),
                "archive_initialization_dates_sha256": bridge.initialization_dates_sha256(
                    archive_initializations
                ),
                "selection_indices": positions.tolist(),
                "selected_initialization_dates_sha256": bridge.initialization_dates_sha256(
                    selected_initializations
                ),
                "raw_weekly_members_sha256": raw_digest.hexdigest(),
                "raw_member_shape": [count, 50, 6, 27, 27],
            }
        )
        return arrays, source
    finally:
        dataset.close()


def validate_adjustment_shard(
    shard: Path,
    receipt: Mapping[str, Any],
    *,
    expected_initializations: np.ndarray,
    expected_latitude: np.ndarray,
    expected_longitude: np.ndarray,
    checkpoint_sha256: str,
) -> None:
    """Validate one completed shard before resuming or publishing."""

    _require_file_hash(shard, str(receipt.get("artifact_sha256", "")), "inference shard")
    if receipt.get("checkpoint_sha256") != checkpoint_sha256:
        raise bridge.BridgeContractError("shard checkpoint hash differs")
    if receipt.get("verification_truth_opened") is not False or receipt.get(
        "sealed_2025_target_opened"
    ) is not False:
        raise bridge.BridgeContractError("shard reports verification target access")
    scoreable = bridge.scoring_mask(expected_initializations)
    expected_years = set(pd.DatetimeIndex(expected_initializations).year)
    if len(expected_years) != 1 or int(receipt.get("year", -1)) != next(iter(expected_years)):
        raise bridge.BridgeContractError("shard receipt year differs")
    expected_receipt = {
        "case_count": int(expected_initializations.size),
        "scoreable_case_count": int(np.count_nonzero(scoreable)),
        "selected_seed": 43,
        "member_count": 50,
    }
    for key, value in expected_receipt.items():
        if receipt.get(key) != value:
            raise bridge.BridgeContractError(f"shard receipt {key} differs")
    raw_source = receipt.get("raw_source", {})
    if raw_source.get("selected_initialization_dates_sha256") != bridge.initialization_dates_sha256(
        expected_initializations
    ):
        raise bridge.BridgeContractError("shard raw-source initialization hash differs")
    if raw_source.get("raw_member_shape") != [
        int(expected_initializations.size),
        50,
        6,
        27,
        27,
    ]:
        raise bridge.BridgeContractError("shard raw-source member shape differs")
    raw_digest = raw_source.get("raw_weekly_members_sha256")
    if not isinstance(raw_digest, str) or len(raw_digest) != 64:
        raise bridge.BridgeContractError("shard raw-source content hash is invalid")
    selection_indices = raw_source.get("selection_indices", [])
    if len(selection_indices) != expected_initializations.size or len(set(selection_indices)) != len(
        selection_indices
    ):
        raise bridge.BridgeContractError("shard raw-source selection indices differ")
    with np.load(shard, allow_pickle=False) as archive:
        if set(archive.files) != ALLOWED_SHARD_KEYS:
            raise bridge.BridgeContractError(f"inference shard keys differ: {archive.files}")
        initializations = np.asarray(archive["initializations"], dtype="datetime64[D]")
        if not np.array_equal(initializations, expected_initializations):
            raise bridge.BridgeContractError("inference shard initializations differ")
        count = initializations.size
        expected_shape = (count, *EXPECTED_FIELD_SHAPE)
        for key in (
            "delta_log_location",
            "log_spread",
            "raw_ensemble_mean",
            "corrected_ensemble_mean",
        ):
            values = np.asarray(archive[key])
            if values.shape != expected_shape or values.dtype != np.float32:
                raise bridge.BridgeContractError(f"inference shard {key} shape/dtype differs")
            if not np.isfinite(values).all():
                raise bridge.BridgeContractError(f"inference shard {key} is non-finite")
        if np.any(archive["raw_ensemble_mean"] < 0.0) or np.any(
            archive["corrected_ensemble_mean"] < 0.0
        ):
            raise bridge.BridgeContractError("inference shard has negative ensemble means")
        if np.any(np.abs(archive["log_spread"]) > 2.00001):
            raise bridge.BridgeContractError("inference shard log-spread exceeds its bound")
        if not np.array_equal(
            np.asarray(archive["scoreable_for_truth"], dtype=bool),
            scoreable,
        ):
            raise bridge.BridgeContractError("inference shard scoreable mask differs")
        if not np.array_equal(archive["latitude"], expected_latitude) or not np.array_equal(
            archive["longitude"], expected_longitude
        ):
            raise bridge.BridgeContractError("inference shard grid differs")
        if int(archive["member_count"]) != 50 or int(archive["selected_seed"]) != 43:
            raise bridge.BridgeContractError("inference shard member/seed contract differs")


def _device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _source_snapshot(output: Path) -> dict[str, str]:
    sources = {
        "code/src/india_s2s_probabilistic_bridge_run.py": Path(__file__).resolve(),
        "code/src/india_s2s_probabilistic_bridge.py": Path(bridge.__file__).resolve(),
        "code/src/fuxi_allseason_ensemble_calibration.py": Path(
            calibration.__file__
        ).resolve(),
        "code/src/fuxi_ensemble_calibration_core.py": PROJECT_ROOT
        / "src"
        / "fuxi_ensemble_calibration_core.py",
        "code/slurm/run_india_s2s_probabilistic_bridge_inference.sbatch": PROJECT_ROOT
        / "slurm"
        / "run_india_s2s_probabilistic_bridge_inference.sbatch",
        "code/plan/INDIA_S2S_PROBABILISTIC_BRIDGE_20260826.md": PROJECT_ROOT
        / "plan"
        / "INDIA_S2S_PROBABILISTIC_BRIDGE_20260826.md",
    }
    hashes: dict[str, str] = {}
    for relative, source in sources.items():
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        hashes[str(destination.relative_to(output))] = bridge.sha256_file(destination)
    return hashes


def _artifact_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): bridge.sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.name not in {"manifest.json", "failure.json"}
        and ".tmp-" not in path.name
    }


def _initialize_or_validate_run(
    output: Path,
    *,
    parent: FrozenParent,
    full_cohort: Mapping[str, Any],
    selection: Mapping[str, Any],
    execution_contract: Mapping[str, Any],
    source_hashes: Mapping[str, str] | None = None,
) -> dict[str, str]:
    contract_path = output / "run_contract.json"
    expected = {
        "experiment": EXPERIMENT,
        "parent_manifest_sha256": parent.receipt["parent_manifest_sha256"],
        "selected_checkpoint_sha256": parent.receipt["selected_checkpoint_sha256"],
        "selected_seed": 43,
        "archive_dates_sha256": full_cohort["forecast_only"]["dates_sha256"],
        "selected_dates_sha256": selection["selected_dates_sha256"],
        "selected_count": selection["selected_count"],
        "forecast_years": list(FORECAST_YEARS),
        "target_day_offsets": list(bridge.TARGET_DAY_OFFSETS),
        "execution_contract": dict(execution_contract),
        "verification_truth_opened": False,
        "sealed_2025_target_opened": False,
    }
    if contract_path.is_file():
        observed = json.loads(contract_path.read_text(encoding="utf-8"))
        for key, value in expected.items():
            if observed.get(key) != value:
                raise bridge.BridgeContractError(f"resume run contract {key} differs")
        hashes = observed.get("source_snapshot_sha256", {})
        for relative, digest in hashes.items():
            _require_file_hash(output / relative, digest, "resume source snapshot")
        return dict(hashes)
    if any(output.iterdir()):
        raise bridge.BridgeContractError("non-empty output lacks run_contract.json")
    hashes = dict(source_hashes or _source_snapshot(output))
    expected["source_snapshot_sha256"] = hashes
    bridge.write_json(contract_path, expected)
    return hashes


def _validate_complete_run(
    output: Path,
    *,
    parent: FrozenParent,
    full_cohort: Mapping[str, Any],
    selected: np.ndarray,
    selection: Mapping[str, Any],
    execution_contract: Mapping[str, Any],
    source_hashes: Mapping[str, str],
) -> Mapping[str, Any] | None:
    """Return an immutable complete manifest after full artifact validation."""

    manifest_path = output / "manifest.json"
    if not manifest_path.is_file():
        return None
    if (output / "failure.json").exists():
        raise bridge.BridgeContractError("complete output also contains failure.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    full = selected.size == 517
    expected_scalars = {
        "experiment": EXPERIMENT,
        "status": "complete" if full else "smoke_complete",
        "mode": "full" if full else "bounded_smoke",
        "output_path": str(output),
        "selected_checkpoint_sha256": parent.receipt["selected_checkpoint_sha256"],
        "verification_truth_opened": False,
        "sealed_2025_target_opened": False,
    }
    for key, value in expected_scalars.items():
        if manifest.get(key) != value:
            raise bridge.BridgeContractError(f"complete manifest {key} differs")
    if manifest.get("parent_receipt") != parent.receipt:
        raise bridge.BridgeContractError("complete manifest parent receipt differs")
    if manifest.get("inference_selection") != selection:
        raise bridge.BridgeContractError("complete manifest selection differs")
    expected_cohort = {
        "forecast_only_count": full_cohort["forecast_only"]["count"],
        "scoring_count": full_cohort["scoring"]["count"],
        "forecast_only_dates_sha256": full_cohort["forecast_only"]["dates_sha256"],
        "scoring_dates_sha256": full_cohort["scoring"]["dates_sha256"],
    }
    if manifest.get("full_cohort_contract") != expected_cohort:
        raise bridge.BridgeContractError("complete manifest cohort contract differs")
    for key, value in execution_contract.items():
        if manifest.get("execution", {}).get(key) != value:
            raise bridge.BridgeContractError(f"complete manifest execution {key} differs")
    if manifest.get("source_snapshot_sha256") != dict(source_hashes):
        raise bridge.BridgeContractError("complete manifest source snapshot differs")
    observation_access = manifest.get("observation_access", {})
    if observation_access.get("training_context_years") != list(TRAIN_CONTEXT_YEARS):
        raise bridge.BridgeContractError("complete manifest context years differ")
    if observation_access.get("verification_truth_opened") is not False or observation_access.get(
        "2025_observation_opened"
    ) is not False or observation_access.get("sealed_2025_target_opened") is not False:
        raise bridge.BridgeContractError("complete manifest reports verification target access")

    expected_hashes = manifest.get("artifact_sha256", {})
    if not isinstance(expected_hashes, Mapping) or expected_hashes != _artifact_hashes(output):
        raise bridge.BridgeContractError("complete manifest artifact hashes differ")
    required_artifacts = {
        "run_contract.json",
        "context/training_context.npz",
        "context/receipt.json",
        *source_hashes.keys(),
    }
    if not required_artifacts.issubset(expected_hashes):
        raise bridge.BridgeContractError("complete manifest omits required context/source artifacts")

    context = _load_context_artifact(output, parent)
    receipts = manifest.get("inference_shards", [])
    if not isinstance(receipts, list):
        raise bridge.BridgeContractError("complete manifest shard receipts are invalid")
    selected_year_values = pd.DatetimeIndex(selected).year.to_numpy()
    expected_years = [
        year for year in FORECAST_YEARS if np.any(selected_year_values == year)
    ]
    if [int(item.get("year", -1)) for item in receipts] != expected_years:
        raise bridge.BridgeContractError("complete manifest shard years differ")
    for receipt in receipts:
        year = int(receipt["year"])
        selected_year = selected[selected_year_values == year]
        shard = output / "inference" / f"adjustments_{year}.npz"
        receipt_path = output / "inference" / f"adjustments_{year}.receipt.json"
        if str(shard.relative_to(output)) not in expected_hashes or str(
            receipt_path.relative_to(output)
        ) not in expected_hashes:
            raise bridge.BridgeContractError(f"complete manifest omits {year} inference artifacts")
        on_disk_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if on_disk_receipt != receipt:
            raise bridge.BridgeContractError(f"complete manifest/receipt differ for {year}")
        if receipt.get("execution_contract") != dict(execution_contract):
            raise bridge.BridgeContractError(f"complete shard execution differs for {year}")
        validate_adjustment_shard(
            shard,
            receipt,
            expected_initializations=selected_year,
            expected_latitude=context.latitude,
            expected_longitude=context.longitude,
            checkpoint_sha256=str(parent.receipt["selected_checkpoint_sha256"]),
        )
    if sum(int(item["case_count"]) for item in receipts) != selected.size:
        raise bridge.BridgeContractError("complete manifest inferred-case count differs")
    if sum(int(item["scoreable_case_count"]) for item in receipts) != selection[
        "scoreable_count"
    ]:
        raise bridge.BridgeContractError("complete manifest scoreable-case count differs")
    return manifest


def run_inference(args: argparse.Namespace) -> Mapping[str, Any]:
    """Execute or safely resume compact 2020--2024 forecast inference."""

    output = Path(args.output).resolve()
    if bridge.DEFAULT_OUTPUT_ROOT.resolve() not in output.parents:
        raise bridge.BridgeContractError("inference output must be under the resultsv3 bridge root")
    parent = load_frozen_parent(Path(args.parent_manifest))
    canonical = bridge.load_canonical_initializations()
    full_cohort = bridge.build_cohort_receipt(canonical, strict=True)
    selected = select_initializations(canonical, smoke=args.smoke, max_cases=args.max_cases)
    selection = _selection_receipt(canonical, selected)
    device = _device(args.device)
    use_amp = not args.no_amp and device.type == "cuda"
    execution_contract = {
        "requested_device": args.device,
        "resolved_device": str(device),
        "automatic_mixed_precision": use_amp,
        "batch_size": args.batch_size,
    }
    output.mkdir(parents=True, exist_ok=True)
    source_hashes = _initialize_or_validate_run(
        output,
        parent=parent,
        full_cohort=full_cohort,
        selection=selection,
        execution_contract=execution_contract,
    )
    complete = _validate_complete_run(
        output,
        parent=parent,
        full_cohort=full_cohort,
        selected=selected,
        selection=selection,
        execution_contract=execution_contract,
        source_hashes=source_hashes,
    )
    if complete is not None:
        return complete
    context = _load_context_artifact(output, parent)
    model = load_selected_model(parent, device)
    loader = _forecast_loader()
    receipts: list[dict[str, Any]] = []
    for year in FORECAST_YEARS:
        canonical_year = canonical[
            pd.DatetimeIndex(canonical).year.to_numpy() == year
        ]
        selected_year = selected[pd.DatetimeIndex(selected).year.to_numpy() == year]
        if selected_year.size == 0:
            continue
        shard = output / "inference" / f"adjustments_{year}.npz"
        receipt_path = output / "inference" / f"adjustments_{year}.receipt.json"
        if shard.exists() != receipt_path.exists():
            raise bridge.BridgeContractError(f"partial inference shard for {year}")
        if shard.exists():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            validate_adjustment_shard(
                shard,
                receipt,
                expected_initializations=selected_year,
                expected_latitude=context.latitude,
                expected_longitude=context.longitude,
                checkpoint_sha256=str(parent.receipt["selected_checkpoint_sha256"]),
            )
            receipts.append(receipt)
            continue
        arrays, raw_source = _infer_year(
            year=year,
            selected_initializations=selected_year,
            canonical_year_initializations=canonical_year,
            loader=loader,
            model=model,
            context_inputs=context,
            device=device,
            batch_size=args.batch_size,
            use_amp=use_amp,
        )
        _atomic_savez(shard, **arrays)
        receipt = {
            "year": year,
            "artifact": str(shard.relative_to(output)),
            "artifact_sha256": bridge.sha256_file(shard),
            "case_count": int(selected_year.size),
            "scoreable_case_count": int(np.count_nonzero(arrays["scoreable_for_truth"])),
            "checkpoint_sha256": parent.receipt["selected_checkpoint_sha256"],
            "selected_seed": 43,
            "member_count": 50,
            "execution_contract": execution_contract,
            "raw_source": raw_source,
            "reconstruction": "apply rank-preserving affine transform in log1p space using delta_log_location and exp(log_spread)",
            "verification_truth_opened": False,
            "sealed_2025_target_opened": False,
        }
        bridge.write_json(receipt_path, receipt)
        validate_adjustment_shard(
            shard,
            receipt,
            expected_initializations=selected_year,
            expected_latitude=context.latitude,
            expected_longitude=context.longitude,
            checkpoint_sha256=str(parent.receipt["selected_checkpoint_sha256"]),
        )
        receipts.append(receipt)
        print(f"PASS: completed compact FuXi inference shard for {year}", flush=True)
    inferred_count = sum(int(item["case_count"]) for item in receipts)
    scoreable_count = sum(int(item["scoreable_case_count"]) for item in receipts)
    if inferred_count != selected.size or scoreable_count != selection["scoreable_count"]:
        raise bridge.BridgeContractError("completed shard counts differ from selection")
    failure = output / "failure.json"
    if failure.is_file():
        failure.unlink()
    manifest: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "status": "complete" if selected.size == 517 else "smoke_complete",
        "mode": "full" if selected.size == 517 else "bounded_smoke",
        "scientific_status": (
            "forecast-only inference artifact; benchmark scoring has not run"
            if selected.size == 517
            else "non-scientific bounded inference smoke test"
        ),
        "created_utc": bridge.utc_now(),
        "output_path": str(output),
        "parent_receipt": parent.receipt,
        "full_cohort_contract": {
            "forecast_only_count": full_cohort["forecast_only"]["count"],
            "scoring_count": full_cohort["scoring"]["count"],
            "forecast_only_dates_sha256": full_cohort["forecast_only"]["dates_sha256"],
            "scoring_dates_sha256": full_cohort["scoring"]["dates_sha256"],
        },
        "inference_selection": selection,
        "inference_shards": receipts,
        "storage_contract": {
            "stored": [
                "delta_log_location",
                "log_spread",
                "raw_ensemble_mean",
                "corrected_ensemble_mean",
            ],
            "corrected_members_stored": False,
            "corrected_members_exactly_reconstructible": True,
            "raw_member_archive_content_bound_by_sha256": True,
            "operational_member_count": 50,
            "training_member_count": 51,
            "permutation_invariant_member_axis": True,
        },
        "observation_access": {
            "training_context_years": list(TRAIN_CONTEXT_YEARS),
            "verification_truth_opened": False,
            "2025_observation_opened": False,
            "sealed_2025_target_opened": False,
        },
        "execution": {
            **execution_contract,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "xarray": xr.__version__,
            "zarr": importlib.metadata.version("zarr"),
        },
        "source_snapshot_sha256": source_hashes,
        "selected_checkpoint_sha256": parent.receipt["selected_checkpoint_sha256"],
        "verification_truth_opened": False,
        "sealed_2025_target_opened": False,
    }
    manifest["artifact_sha256"] = _artifact_hashes(output)
    bridge.write_json(output / "manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply the aligned selected FuXi calibration without opening truth."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    inference = subparsers.add_parser("inference", help="write compact forecast-only shards")
    inference.add_argument("--parent-manifest", type=Path, required=True)
    inference.add_argument("--output", type=Path, required=True)
    inference.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    inference.add_argument("--batch-size", type=int, default=4)
    inference.add_argument(
        "--smoke",
        action="store_true",
        help=f"run a deterministic {SMOKE_CASE_COUNT}-case cross-year plumbing gate",
    )
    inference.add_argument(
        "--max-cases",
        type=int,
        help="deterministically select 1..517 cases across the archive",
    )
    inference.add_argument("--no-amp", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    try:
        manifest = run_inference(args)
    except Exception as error:
        output = Path(args.output).resolve()
        # A published manifest makes the run immutable, even when a later
        # invocation supplies incompatible arguments or detects corruption.
        if output.exists() and not (output / "manifest.json").exists():
            bridge.write_json(
                output / "failure.json",
                {
                    "experiment": EXPERIMENT,
                    "status": "failed",
                    "failed_utc": bridge.utc_now(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "verification_truth_opened": False,
                    "sealed_2025_target_opened": False,
                },
            )
        raise
    print(
        f"PASS: {manifest['mode']} bridge inference published at {Path(args.output).resolve()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
