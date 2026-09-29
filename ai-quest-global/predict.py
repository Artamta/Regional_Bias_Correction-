#!/usr/bin/env python3
"""Run the local two-lead probability model and export NetCDF files.

This module is deliberately offline-only: it reads a case from either a
prepared ``.npz`` artifact or the project's Zarr cache plus a local PyTorch
checkpoint, then writes two local NetCDF files.  It contains no submission,
upload, authentication, or other network code.

IMPORTANT FUXI PERMISSION NOTICE
--------------------------------
The FuXi-S2S authors' published competition-use notice says their model may not
be used in a competition without the authors' permission.  Producing files with
this script does not grant that permission.  Obtain and retain written
permission before using FuXi-derived inputs in a competition submission.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence
import uuid

import numpy as np
import xarray as xr


N_LEADS = 2
N_QUINTILES = 5
N_LATITUDE = 121
N_LONGITUDE = 240
QUINTILES = np.asarray([0.2, 0.4, 0.6, 0.8, 1.0], dtype=np.float32)
LATITUDE = np.linspace(90.0, -90.0, N_LATITUDE, dtype=np.float32)
LONGITUDE = np.arange(N_LONGITUDE, dtype=np.float32) * np.float32(1.5)
CANONICAL_VARIABLES = ("pr", "tas", "mslp")

VARIABLE_METADATA: Mapping[str, Mapping[str, str]] = {
    "pr": {
        "display_name": "precipitation",
        "long_name": "forecast probability for precipitation quintile",
        "forecast_period_aggregation": "sum",
        "shortName": "tp",
    },
    "tas": {
        "display_name": "near-surface (2m) temperature",
        "long_name": "forecast probability for near-surface temperature quintile",
        "forecast_period_aggregation": "mean",
        "shortName": "2t",
    },
    "mslp": {
        "display_name": "mean sea-level pressure",
        "long_name": "forecast probability for mean sea-level pressure quintile",
        "forecast_period_aggregation": "mean",
        "shortName": "msl",
    },
}

FUXI_PERMISSION_WARNING = (
    "FuXi-S2S competition-use warning: the model authors' published notice "
    "requires their permission for competition use. Local export does not "
    "grant permission; obtain written authorization before submission."
)


def _variable_metadata(variable: str) -> Mapping[str, str]:
    """Return metadata for one explicitly supported AI Quest grid variable."""

    try:
        return VARIABLE_METADATA[variable]
    except (KeyError, TypeError) as exc:
        supported = ", ".join(CANONICAL_VARIABLES)
        raise ValueError(f"variable must be one of: {supported}; got {variable!r}") from exc


@dataclass(frozen=True)
class PreparedBatch:
    """Validated arrays loaded from a prepared NPZ or Zarr artifact."""

    features: np.ndarray
    p0: np.ndarray
    target: np.ndarray | None
    latitude: np.ndarray
    longitude: np.ndarray
    land_fraction: np.ndarray | None
    init_dates: tuple[str, ...]
    source: Path

    @property
    def n_cases(self) -> int:
        return int(self.features.shape[0])


class NoMatchingPreparedCases(ValueError):
    """Raised when valid filters select no cases from a prepared artifact."""


def _as_numpy(value: Any, *, dtype: np.dtype[Any] | type | None = None) -> np.ndarray:
    """Convert NumPy-like or CPU torch values without importing torch here."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=dtype)


def normalize_probabilities(
    probabilities: np.ndarray,
    *,
    category_axis: int = -3,
    negative_tolerance: float = 1.0e-7,
) -> np.ndarray:
    """Return finite, non-negative probabilities normalized over quintiles.

    This helper is intended for already probabilistic values, not logits.  A
    materially negative value or a grid cell with zero total mass is rejected.
    Tiny negative round-off is clipped before normalization.
    """

    values = np.asarray(probabilities, dtype=np.float64)
    axis = category_axis if category_axis >= 0 else values.ndim + category_axis
    if axis < 0 or axis >= values.ndim:
        raise ValueError(f"Invalid category axis {category_axis} for shape {values.shape}")
    if values.shape[axis] != N_QUINTILES:
        raise ValueError(
            f"Expected {N_QUINTILES} quintiles on axis {category_axis}; got {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError("Probability values must all be finite")
    minimum = float(values.min())
    if minimum < -negative_tolerance:
        raise ValueError(f"Probability values cannot be negative (minimum={minimum:g})")

    values = np.maximum(values, 0.0)
    total = values.sum(axis=axis, keepdims=True)
    if np.any(total <= 0.0):
        raise ValueError("Every grid cell must have positive total probability mass")
    return (values / total).astype(np.float32, copy=False)


def validate_probability_cube(
    probabilities: np.ndarray,
    *,
    require_normalized: bool = True,
    sum_tolerance: float = 2.0e-4,
) -> np.ndarray:
    """Validate and safely canonicalize one ``[5, 121, 240]`` forecast."""

    values = np.asarray(probabilities)
    expected = (N_QUINTILES, N_LATITUDE, N_LONGITUDE)
    if values.shape != expected:
        raise ValueError(f"Expected probability shape {expected}; got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Probability values must all be finite")
    if float(values.min()) < -1.0e-7 or float(values.max()) > 1.0 + 1.0e-7:
        raise ValueError("Probabilities must lie in [0, 1]")
    totals = np.asarray(values, dtype=np.float64).sum(axis=0)
    maximum_error = float(np.max(np.abs(totals - 1.0)))
    if require_normalized and maximum_error > sum_tolerance:
        raise ValueError(
            "Quintile probabilities must sum to one at every grid cell "
            f"(maximum absolute error={maximum_error:g})"
        )
    return normalize_probabilities(values, category_axis=0)


def _first_present(archive: Mapping[str, Any], names: Sequence[str]) -> Any | None:
    for name in names:
        if name in archive:
            return archive[name]
    return None


def _canonical_date(value: Any) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, np.datetime64):
        text = np.datetime_as_string(value, unit="D")
    elif isinstance(value, (int, np.integer)):
        text = f"{int(value):08d}"
    else:
        text = str(value)
    if len(text) == 8 and text.isdigit():
        text = f"{text[:4]}-{text[4:6]}-{text[6:]}"
    elif len(text) >= 10 and text[4] == "-" and text[7] == "-":
        text = text[:10]
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError as exc:
        raise ValueError(f"initialization dates must be YYYY-MM-DD/YYYMMDD; got {value!r}") from exc


def canonical_date(value: Any) -> str:
    """Return the canonical ``YYYY-MM-DD`` form accepted by local exports."""

    return _canonical_date(value)


def _decode_dates(value: Any | None, n_cases: int) -> tuple[str, ...]:
    if value is None:
        return tuple("" for _ in range(n_cases))
    raw = np.asarray(value).reshape(-1)
    if len(raw) != n_cases:
        raise ValueError(f"init_dates has length {len(raw)} but there are {n_cases} cases")
    dates: list[str] = []
    for item in raw:
        dates.append(_canonical_date(item))
    return tuple(dates)


def _validate_grid(latitude: np.ndarray, longitude: np.ndarray) -> None:
    if latitude.shape != LATITUDE.shape or not np.allclose(latitude, LATITUDE, atol=1.0e-6):
        raise ValueError("latitude must be the 121-point 90 to -90 grid in descending order")
    if longitude.shape != LONGITUDE.shape or not np.allclose(
        longitude, LONGITUDE, atol=1.0e-6
    ):
        raise ValueError("longitude must be the 240-point 0 to 358.5 grid")


def _prepared_batch_from_arrays(
    source: Path,
    *,
    features_raw: Any,
    p0_raw: Any,
    target_raw: Any | None,
    latitude_raw: Any | None,
    longitude_raw: Any | None,
    land_raw: Any | None,
    dates_raw: Any | None,
    require_target: bool,
) -> PreparedBatch:
    features = np.asarray(features_raw, dtype=np.float32)
    p0 = np.asarray(p0_raw, dtype=np.float32)
    if features.ndim == 4:
        features = features[None, ...]
    if p0.ndim == 4:
        p0 = p0[None, ...]
    if features.ndim != 5 or features.shape[1] != N_LEADS:
        raise ValueError(
            "features must have shape [case, 2, channel, 121, 240]; "
            f"got {features.shape}"
        )
    if features.shape[0] == 0:
        raise NoMatchingPreparedCases(f"No prepared cases were selected from {source}")
    if features.shape[-2:] != (N_LATITUDE, N_LONGITUDE):
        raise ValueError(f"features uses the wrong spatial grid: {features.shape[-2:]}")
    expected_p0 = (
        features.shape[0],
        N_LEADS,
        N_QUINTILES,
        N_LATITUDE,
        N_LONGITUDE,
    )
    if p0.shape != expected_p0:
        raise ValueError(f"p0 must have shape {expected_p0}; got {p0.shape}")
    if not np.isfinite(features).all():
        raise ValueError("features must all be finite; encode invalid truth only as target=-1")

    p0_totals = np.asarray(p0, dtype=np.float64).sum(axis=2)
    if (
        not np.isfinite(p0).all()
        or float(p0.min()) < -1.0e-7
        or float(p0.max()) > 1.0 + 1.0e-7
        # The prepared Zarr deliberately stores p0 as float16; allow its
        # quantization error, then renormalize exactly below.
        or float(np.max(np.abs(p0_totals - 1.0))) > 2.0e-3
    ):
        raise ValueError("p0 must be finite [0,1] probabilities summing to one over quintiles")
    p0 = normalize_probabilities(p0, category_axis=2)

    latitude = (
        LATITUDE.copy()
        if latitude_raw is None
        else np.asarray(latitude_raw, dtype=np.float32).copy()
    )
    longitude = (
        LONGITUDE.copy()
        if longitude_raw is None
        else np.asarray(longitude_raw, dtype=np.float32).copy()
    )
    _validate_grid(latitude, longitude)

    target: np.ndarray | None = None
    if target_raw is not None:
        target = np.asarray(target_raw)
        if target.ndim == 3:
            target = target[None, ...]
        expected_target = (
            features.shape[0],
            N_LEADS,
            N_LATITUDE,
            N_LONGITUDE,
        )
        if target.shape != expected_target:
            raise ValueError(f"target must have shape {expected_target}; got {target.shape}")
        if not np.isfinite(target).all() or not np.equal(target, np.round(target)).all():
            raise ValueError("target must contain integer categories -1 or 0..4")
        target = target.astype(np.int8, copy=True)
        if int(target.min()) < -1 or int(target.max()) >= N_QUINTILES:
            raise ValueError("target must contain only -1 (invalid) or categories 0..4")
    elif require_target:
        raise ValueError("Prepared artifact must contain 'target' for evaluation")

    land_fraction: np.ndarray | None = None
    if land_raw is not None:
        land_fraction = np.asarray(land_raw, dtype=np.float32)
        if land_fraction.shape != (N_LATITUDE, N_LONGITUDE):
            raise ValueError(
                "land_fraction must have shape "
                f"{(N_LATITUDE, N_LONGITUDE)}; got {land_fraction.shape}"
            )
        if not np.isfinite(land_fraction).all() or (
            float(land_fraction.min()) < 0.0 or float(land_fraction.max()) > 1.0
        ):
            raise ValueError("land_fraction must be finite and in [0, 1]")
        land_fraction = land_fraction.copy()

    return PreparedBatch(
        features=np.ascontiguousarray(features),
        p0=np.ascontiguousarray(p0),
        target=target,
        latitude=latitude,
        longitude=longitude,
        land_fraction=land_fraction,
        init_dates=_decode_dates(dates_raw, features.shape[0]),
        source=source,
    )


def load_prepared_npz(path: str | Path, *, require_target: bool = False) -> PreparedBatch:
    """Load and validate the project's prepared-batch NPZ contract."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Prepared NPZ not found: {source}")
    with np.load(source, allow_pickle=False) as archive:
        features_raw = _first_present(archive, ("features", "x"))
        p0_raw = _first_present(archive, ("p0", "raw_probabilities"))
        if features_raw is None or p0_raw is None:
            raise ValueError("Prepared NPZ must contain 'features' and 'p0' arrays")
        return _prepared_batch_from_arrays(
            source,
            features_raw=features_raw,
            p0_raw=p0_raw,
            target_raw=_first_present(
                archive, ("target", "targets", "y", "obs_category", "observation_category")
            ),
            latitude_raw=_first_present(archive, ("latitude", "lat")),
            longitude_raw=_first_present(archive, ("longitude", "lon")),
            land_raw=_first_present(archive, ("land_fraction", "landfrac")),
            dates_raw=_first_present(archive, ("init_dates", "init_date")),
            require_target=require_target,
        )


def is_zarr_store(path: str | Path) -> bool:
    candidate = Path(path).expanduser()
    return candidate.is_dir() and (
        candidate.suffix.lower() == ".zarr"
        or (candidate / ".zgroup").exists()
        or (candidate / "zarr.json").exists()
    )


def _selected_case_indices(
    init_dates: Sequence[str],
    *,
    init_date: str | None = None,
    years: Sequence[int] | None = None,
    thursday_only: bool = False,
) -> np.ndarray:
    available = np.asarray(init_dates, dtype="U10")
    mask = np.ones(len(available), dtype=bool)
    if init_date is not None:
        requested = _canonical_date(init_date)
        mask &= available == requested
    if years is not None:
        requested_years = {int(year) for year in years}
        if not requested_years or any(year < 1 or year > 9999 for year in requested_years):
            raise ValueError("years must contain valid calendar years")
        mask &= np.asarray(
            [bool(value) and date.fromisoformat(value).year in requested_years for value in available]
        )
    if thursday_only:
        mask &= np.asarray(
            [bool(value) and date.fromisoformat(value).weekday() == 3 for value in available]
        )
    return np.flatnonzero(mask)


def _take_zarr_cases(array: Any, indices: np.ndarray) -> np.ndarray:
    try:
        return np.asarray(array.oindex[indices.tolist(), ...])
    except (AttributeError, IndexError, TypeError):
        return np.stack([np.asarray(array[int(index)]) for index in indices], axis=0)


def load_prepared_zarr(
    path: str | Path,
    *,
    require_target: bool = False,
    init_date: str | None = None,
    years: Sequence[int] | None = None,
    thursday_only: bool = False,
) -> PreparedBatch:
    """Read selected cases directly from the cache built by ``prepare_data.py``."""

    source = Path(path).expanduser().resolve()
    if not is_zarr_store(source):
        raise FileNotFoundError(f"Prepared Zarr store not found: {source}")
    try:
        import zarr
    except Exception as exc:  # pragma: no cover - depends on the environment's zarr build
        raise RuntimeError("A compatible zarr installation is required to read the cache") from exc

    group = zarr.open_group(str(source), mode="r")
    status = str(group.attrs.get("status", ""))
    if status and status != "complete":
        raise RuntimeError(f"Prepared cache status is {status!r}, not 'complete'")
    required = ("features", "p0", "latitude", "longitude", "land_fraction", "init_yyyymmdd")
    missing = [name for name in required if name not in group]
    if require_target and "target" not in group:
        missing.append("target")
    if missing:
        raise ValueError(f"Prepared Zarr cache is missing arrays: {missing}")

    raw_dates = np.asarray(group["init_yyyymmdd"][:])
    init_dates = _decode_dates(raw_dates, len(raw_dates))
    indices = _selected_case_indices(
        init_dates,
        init_date=init_date,
        years=years,
        thursday_only=thursday_only,
    )
    if len(indices) == 0:
        filters = f"init_date={init_date!r}, years={years}, thursday_only={thursday_only}"
        raise NoMatchingPreparedCases(f"No prepared cases in {source} match {filters}")
    if init_date is not None and len(indices) != 1:
        raise ValueError(f"Initialization {init_date} occurs {len(indices)} times in {source}")
    for flag_name in ("case_complete", "feature_complete"):
        if flag_name in group and not _take_zarr_cases(group[flag_name], indices).astype(bool).all():
            raise RuntimeError(f"Selected Zarr cases include incomplete {flag_name} entries")

    return _prepared_batch_from_arrays(
        source,
        features_raw=_take_zarr_cases(group["features"], indices),
        p0_raw=_take_zarr_cases(group["p0"], indices),
        target_raw=_take_zarr_cases(group["target"], indices) if "target" in group else None,
        latitude_raw=np.asarray(group["latitude"][:]),
        longitude_raw=np.asarray(group["longitude"][:]),
        land_raw=np.asarray(group["land_fraction"][:]),
        dates_raw=np.asarray([init_dates[int(index)] for index in indices]),
        require_target=require_target,
    )


def _select_batch(
    batch: PreparedBatch,
    *,
    init_date: str | None,
    years: Sequence[int] | None,
    thursday_only: bool,
) -> PreparedBatch:
    # Undated legacy NPZs remain usable when no exact date was requested.
    if not any(batch.init_dates):
        if init_date is not None or years is not None or thursday_only:
            raise ValueError("Cannot apply initialization-date filters to an undated NPZ")
        return batch
    indices = _selected_case_indices(
        batch.init_dates,
        init_date=init_date,
        years=years,
        thursday_only=thursday_only,
    )
    if len(indices) == 0:
        raise NoMatchingPreparedCases(f"No cases in {batch.source} match the requested filters")
    if init_date is not None and len(indices) != 1:
        raise ValueError(f"Initialization {init_date} occurs {len(indices)} times in {batch.source}")
    return PreparedBatch(
        features=np.ascontiguousarray(batch.features[indices]),
        p0=np.ascontiguousarray(batch.p0[indices]),
        target=None if batch.target is None else np.ascontiguousarray(batch.target[indices]),
        latitude=batch.latitude,
        longitude=batch.longitude,
        land_fraction=batch.land_fraction,
        init_dates=tuple(batch.init_dates[int(index)] for index in indices),
        source=batch.source,
    )


def load_prepared(
    path: str | Path,
    *,
    require_target: bool = False,
    init_date: str | None = None,
    years: Sequence[int] | None = None,
    thursday_only: bool = False,
) -> PreparedBatch:
    """Load selected cases from either a prepared NPZ file or project Zarr cache."""

    source = Path(path).expanduser().resolve()
    if is_zarr_store(source):
        return load_prepared_zarr(
            source,
            require_target=require_target,
            init_date=init_date,
            years=years,
            thursday_only=thursday_only,
        )
    if source.suffix.lower() != ".npz":
        raise ValueError(f"Prepared input must be an NPZ file or Zarr store: {source}")
    batch = load_prepared_npz(source, require_target=require_target)
    return _select_batch(
        batch,
        init_date=init_date,
        years=years,
        thursday_only=thursday_only,
    )


def _import_model_class(model_name: str) -> type:
    try:
        from .model import QuestTemporalProbUNet, TPProbUNet  # type: ignore
    except (ImportError, ValueError):
        from model import QuestTemporalProbUNet, TPProbUNet  # type: ignore
    classes = {
        "TPProbUNet": TPProbUNet,
        "QuestTemporalProbUNet": QuestTemporalProbUNet,
    }
    if model_name not in classes:
        raise ValueError(
            "checkpoint model_name must be TPProbUNet or QuestTemporalProbUNet; "
            f"got {model_name!r}"
        )
    return classes[model_name]


def _torch_load(path: Path, device: str) -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - dependency error is environment-specific
        raise RuntimeError("PyTorch is required to run a checkpoint") from exc

    # weights_only is both safer and sufficient for the project's tensor/dict
    # checkpoint contract. Older supported PyTorch releases lack this argument.
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # pragma: no cover - used only on older PyTorch
        return torch.load(path, map_location=device)


def load_checkpoint_model(
    checkpoint_path: str | Path,
    *,
    device: str = "cpu",
    in_channels: int = 18,
) -> tuple[Any, Mapping[str, Any]]:
    """Instantiate the declared project model and strictly restore its state."""

    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = _torch_load(path, device)
    if not isinstance(payload, Mapping):
        raise ValueError("Checkpoint must be a mapping containing model_state")

    raw_config = payload.get("model_config", {})
    if raw_config is None:
        raw_config = {}
    if not isinstance(raw_config, Mapping):
        raise ValueError("checkpoint model_config must be a mapping")
    raw_model_name = payload.get("model_name")
    metadata = payload.get("metadata", {})
    if raw_model_name is None and isinstance(metadata, Mapping):
        raw_model_name = metadata.get("model_name")
    if raw_model_name is None:
        if "variables" in raw_config:
            raise ValueError(
                "A multivariable checkpoint must explicitly declare "
                "model_name='QuestTemporalProbUNet'"
            )
        # Backward compatibility for precipitation-only checkpoints predating
        # the model_name field.  Attention settings alone are intentionally
        # not used to guess a different model family.
        raw_model_name = "TPProbUNet"
    model_name = str(raw_model_name)
    if model_name == "QuestTemporalProbUNet":
        allowed = {
            "in_channels",
            "variables",
            "base_channels",
            "dropout",
            "attention_heads",
            "attention_dropout",
        }
    else:
        allowed = {
            "in_channels",
            "base_channels",
            "dropout",
            "period_attention",
            "attention_heads",
            "attention_dropout",
        }
    model_config = {key: raw_config[key] for key in allowed if key in raw_config}
    if "variables" in model_config:
        variables = model_config["variables"]
        if isinstance(variables, str) or not isinstance(variables, Sequence):
            raise ValueError("checkpoint variables must be a sequence of names")
        model_config["variables"] = tuple(str(variable) for variable in variables)
    model_config.setdefault("in_channels", in_channels)
    if int(model_config["in_channels"]) != int(in_channels):
        raise ValueError(
            "Checkpoint/model input mismatch: checkpoint expects "
            f"{model_config['in_channels']} channels, prepared case has {in_channels}"
        )

    state = _first_present(
        payload, ("model_state", "model_state_dict", "state_dict")
    )
    if state is None and payload and all(hasattr(value, "shape") for value in payload.values()):
        state = payload
    if not isinstance(state, Mapping):
        raise ValueError("Checkpoint does not contain a model_state mapping")

    # DataParallel and torch.compile may prefix otherwise identical state keys.
    clean_state: dict[str, Any] = {}
    for key, value in state.items():
        clean_key = str(key)
        for prefix in ("module.", "_orig_mod."):
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix) :]
        clean_state[clean_key] = value

    model_class = _import_model_class(model_name)
    model = model_class(**model_config)
    model.load_state_dict(clean_state, strict=True)
    model.to(device)
    model.eval()
    return model, payload


def run_model(
    model: Any,
    features: np.ndarray,
    p0: np.ndarray,
    *,
    device: str = "cpu",
) -> np.ndarray:
    """Run one or more cases and return ``[case,2,5,121,240]`` probabilities."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for model inference") from exc

    if type(model).__name__ == "QuestTemporalProbUNet":
        raise ValueError(
            "The standard prepared prediction contract is precipitation-only: "
            "p0 has shape [case, period, 5, latitude, longitude]. "
            "QuestTemporalProbUNet requires a multivariable anchor with shape "
            "[case, variable, period, 5, latitude, longitude], which predict.py "
            "cannot load. Use the dedicated multivariable prototype/workflow."
        )

    features_np = np.asarray(features, dtype=np.float32)
    p0_np = np.asarray(p0, dtype=np.float32)
    if features_np.ndim != 5 or p0_np.ndim != 5:
        raise ValueError("features and p0 must include case and lead dimensions")
    expected = (
        features_np.shape[0],
        N_LEADS,
        N_QUINTILES,
        N_LATITUDE,
        N_LONGITUDE,
    )
    if p0_np.shape != expected:
        raise ValueError(f"p0 must have shape {expected}; got {p0_np.shape}")

    with torch.inference_mode():
        output = model(
            torch.from_numpy(np.ascontiguousarray(features_np)).to(device),
            torch.from_numpy(np.ascontiguousarray(p0_np)).to(device),
        )
    if isinstance(output, Mapping):
        output = _first_present(output, ("probabilities", "probs", "prediction"))
    elif isinstance(output, (tuple, list)):
        output = output[0]
    if output is None:
        raise ValueError("Model did not return probabilities")
    probabilities = _as_numpy(output, dtype=np.float32)
    if probabilities.shape != expected:
        raise ValueError(f"Model returned shape {probabilities.shape}; expected {expected}")

    validated = np.empty(expected, dtype=np.float32)
    for case_index in range(expected[0]):
        for lead_index in range(N_LEADS):
            validated[case_index, lead_index] = validate_probability_cube(
                probabilities[case_index, lead_index]
            )
    return validated


def write_probability_netcdf(
    probabilities: np.ndarray,
    path: str | Path,
    *,
    lead: int,
    init_date: str = "",
    model_name: str = "TPProbUNet",
    variable: str = "pr",
    experiment_notice: str | None = None,
) -> Path:
    """Write one validated lead as a self-describing local NetCDF file."""

    if lead not in (1, 2):
        raise ValueError("lead must be 1 or 2")
    metadata = _variable_metadata(variable)
    values = validate_probability_cube(probabilities)
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    data = xr.DataArray(
        values,
        name=variable,
        dims=("quintile", "latitude", "longitude"),
        coords={
            "quintile": QUINTILES,
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
        },
        attrs={
            "long_name": metadata["long_name"],
            # This is a categorical probability, for which there is no CF
            # standard name.  ``cell_methods`` is also deliberately omitted:
            # the field has no time dimension over which a CF method applies.
            "forecast_period_aggregation": metadata["forecast_period_aggregation"],
            "units": "1",
            "shortName": metadata["shortName"],
            "valid_min": np.float32(0.0),
            "valid_max": np.float32(1.0),
        },
    )
    dataset = data.to_dataset()
    dataset["quintile"].attrs.update(
        {
            "long_name": "upper cumulative probability of climatological quintile",
            "units": "1",
        }
    )
    dataset["latitude"].attrs.update({"standard_name": "latitude", "units": "degrees_north"})
    dataset["longitude"].attrs.update(
        {"standard_name": "longitude", "units": "degrees_east"}
    )
    dataset.attrs.update(
        {
            "title": (
                f"Local global {metadata['display_name']} "
                "quintile-probability forecast"
            ),
            "lead": int(lead),
            "initialization_date": init_date,
            "model": model_name,
            "permission_notice": FUXI_PERMISSION_WARNING,
            "submission_status": "local output only; not uploaded",
        }
    )
    if experiment_notice:
        dataset.attrs["experiment_notice"] = str(experiment_notice)
    if init_date:
        issue = np.datetime64(_canonical_date(init_date), "ns")
        start_offsets = (18, 25)
        end_offsets = (25, 32)
        dataset = dataset.assign_coords(
            forecast_issue_date=issue,
            forecast_period_start=issue + np.timedelta64(start_offsets[lead - 1], "D"),
            forecast_period_end=issue + np.timedelta64(end_offsets[lead - 1], "D"),
        )
        dataset["forecast_issue_date"].attrs["long_name"] = "forecast initialization date"
        dataset["forecast_period_start"].attrs.update(
            {"long_name": "inclusive forecast period start"}
        )
        dataset["forecast_period_end"].attrs.update(
            {"long_name": "exclusive forecast period end"}
        )
    dataset.to_netcdf(destination)
    return destination


def export_two_leads(
    probabilities: np.ndarray,
    output_dir: str | Path,
    *,
    prefix: str,
    init_date: str = "",
    model_name: str = "TPProbUNet",
    variable: str = "pr",
    experiment_notice: str | None = None,
) -> tuple[Path, Path]:
    """Export exactly two files named ``<prefix>_p1.nc`` and ``<prefix>_p2.nc``."""

    values = np.asarray(probabilities)
    expected = (N_LEADS, N_QUINTILES, N_LATITUDE, N_LONGITUDE)
    if values.shape != expected:
        raise ValueError(f"Expected both leads with shape {expected}; got {values.shape}")
    if not prefix or Path(prefix).name != prefix:
        raise ValueError("prefix must be a non-empty filename component, not a path")
    _variable_metadata(variable)
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    paths = tuple(
        write_probability_netcdf(
            values[lead_index],
            directory / f"{prefix}_p{lead_index + 1}.nc",
            lead=lead_index + 1,
            init_date=init_date,
            model_name=model_name,
            variable=variable,
            experiment_notice=experiment_notice,
        )
        for lead_index in range(N_LEADS)
    )
    return paths  # type: ignore[return-value]


def export_all_variables(
    probabilities: np.ndarray,
    output_dir: str | Path,
    *,
    init_date: str,
    variables: Sequence[str] = CANONICAL_VARIABLES,
    model_name: str = "QuestTemporalProbUNet",
) -> tuple[Path, ...]:
    """Export two local NetCDFs for each requested gridded variable.

    ``probabilities`` must use ``[variable, lead, quintile, latitude,
    longitude]`` order.  All six files are staged and checked before they are
    published with sibling ``os.replace`` operations.  If publication fails,
    files already replaced by this call are rolled back.
    """

    if isinstance(variables, str):
        raise ValueError("variables must be a sequence of canonical variable names")
    variable_names = tuple(variables)
    if not variable_names:
        raise ValueError("variables must contain at least one variable")
    if len(set(variable_names)) != len(variable_names):
        raise ValueError("variables must not contain duplicates")
    for variable in variable_names:
        _variable_metadata(variable)

    values = np.asarray(probabilities)
    expected = (
        len(variable_names),
        N_LEADS,
        N_QUINTILES,
        N_LATITUDE,
        N_LONGITUDE,
    )
    if values.shape != expected:
        raise ValueError(
            "Expected variable probabilities with shape "
            f"{expected}; got {values.shape}"
        )

    validated = np.empty(expected, dtype=np.float32)
    for variable_index in range(len(variable_names)):
        for lead_index in range(N_LEADS):
            validated[variable_index, lead_index] = validate_probability_cube(
                values[variable_index, lead_index]
            )

    canonical_init_date = _canonical_date(init_date)
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    transaction_id = uuid.uuid4().hex
    specifications = [
        (
            variable_index,
            variable,
            lead_index,
            directory / f"{variable}_{canonical_init_date}_p{lead_index + 1}.nc",
        )
        for variable_index, variable in enumerate(variable_names)
        for lead_index in range(N_LEADS)
    ]
    final_paths = tuple(specification[3] for specification in specifications)
    stage_paths = tuple(
        final_path.with_name(
            f".{final_path.name}.{transaction_id}.stage.nc"
        )
        for final_path in final_paths
    )
    backup_paths: dict[Path, Path] = {}
    published: list[Path] = []

    def cleanup(paths: Sequence[Path]) -> None:
        for cleanup_path in paths:
            try:
                cleanup_path.unlink()
            except FileNotFoundError:
                pass

    try:
        for stage_path, (variable_index, variable, lead_index, _) in zip(
            stage_paths, specifications, strict=True
        ):
            write_probability_netcdf(
                validated[variable_index, lead_index],
                stage_path,
                lead=lead_index + 1,
                init_date=canonical_init_date,
                model_name=model_name,
                variable=variable,
            )
            # Validate the serialized file, not just the in-memory array, before
            # making any final filename visible.
            with xr.open_dataset(stage_path) as staged:
                if set(staged.data_vars) != {variable}:
                    raise ValueError(
                        f"Staged export {stage_path} has unexpected variables "
                        f"{tuple(staged.data_vars)}"
                    )
                validate_probability_cube(staged[variable].values)
                if int(staged.attrs.get("lead", -1)) != lead_index + 1:
                    raise ValueError(f"Staged export {stage_path} has the wrong lead")
                if staged.attrs.get("initialization_date") != canonical_init_date:
                    raise ValueError(
                        f"Staged export {stage_path} has the wrong initialization date"
                    )

        # Preserve any previous complete set so an I/O failure during the
        # sequence of atomic renames can restore it.
        for final_path in final_paths:
            if final_path.exists():
                backup_path = final_path.with_name(
                    f".{final_path.name}.{transaction_id}.backup.nc"
                )
                os.replace(final_path, backup_path)
                backup_paths[final_path] = backup_path

        for stage_path, final_path in zip(stage_paths, final_paths, strict=True):
            os.replace(stage_path, final_path)
            published.append(final_path)
    except BaseException as exc:
        cleanup(tuple(reversed(published)))
        rollback_failures: list[str] = []
        for final_path, backup_path in reversed(tuple(backup_paths.items())):
            try:
                os.replace(backup_path, final_path)
            except OSError as rollback_exc:
                rollback_failures.append(
                    f"{backup_path.name} -> {final_path.name}: {rollback_exc}"
                )
        cleanup(stage_paths)
        if rollback_failures:
            details = "; ".join(rollback_failures)
            raise RuntimeError(
                f"Multivariable export failed and rollback was incomplete: {details}"
            ) from exc
        raise
    else:
        cleanup(tuple(backup_paths.values()))
        return final_paths
    finally:
        cleanup(stage_paths)


def predict_case(
    case_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    device: str = "cpu",
    prefix: str | None = None,
    init_date: str | None = None,
) -> tuple[Path, Path]:
    """Load one prepared case, run both leads, and export both local files."""

    if is_zarr_store(case_path) and init_date is None:
        raise ValueError("Prediction from a Zarr cache requires --init-date YYYY-MM-DD")
    batch = load_prepared(case_path, init_date=init_date)
    if batch.n_cases != 1:
        raise ValueError(
            f"Prediction requires exactly one case; found {batch.n_cases}. "
            "Pass --init-date to select a case or use a one-case NPZ."
        )
    model, payload = load_checkpoint_model(
        checkpoint_path,
        device=device,
        in_channels=int(batch.features.shape[2]),
    )
    probabilities = run_model(model, batch.features, batch.p0, device=device)[0]
    selected_init_date = batch.init_dates[0]
    if prefix is None:
        prefix = f"pr_{selected_init_date}" if selected_init_date else batch.source.stem
    metadata = payload.get("metadata", {}) if isinstance(payload, Mapping) else {}
    model_name = "TPProbUNet"
    if isinstance(metadata, Mapping) and metadata.get("model_name"):
        model_name = str(metadata["model_name"])
    experiment_notice = None
    normalization = payload.get("normalization", {}) if isinstance(payload, Mapping) else {}
    if isinstance(normalization, Mapping):
        provenance = normalization.get("provenance", {})
        if isinstance(provenance, Mapping) and not provenance.get(
            "official_ai_weather_quest_validation", True
        ):
            experiment_notice = str(
                provenance.get(
                    "notice",
                    "Exploratory training data; not official Quest validation.",
                )
            )
    return export_two_leads(
        probabilities,
        output_dir,
        prefix=prefix,
        init_date=selected_init_date,
        model_name=model_name,
        experiment_notice=experiment_notice,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a local two-lead precipitation probability forecast.",
        epilog=FUXI_PERMISSION_WARNING,
    )
    parser.add_argument(
        "--case",
        required=True,
        type=Path,
        help="Prepared one-case NPZ file or project Zarr cache",
    )
    parser.add_argument(
        "--init-date",
        default=None,
        help="Initialization YYYY-MM-DD (required for Zarr; optional NPZ selector)",
    )
    parser.add_argument("--checkpoint", required=True, type=Path, help="Local model checkpoint")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for two NetCDFs")
    parser.add_argument(
        "--device",
        default="cpu",
        help="PyTorch device, for example cpu or cuda (default: cpu)",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Output filename prefix (default: pr_<init-date>)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(FUXI_PERMISSION_WARNING, file=sys.stderr)
    outputs = predict_case(
        args.case,
        args.checkpoint,
        args.output_dir,
        device=args.device,
        prefix=args.prefix,
        init_date=args.init_date,
    )
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
