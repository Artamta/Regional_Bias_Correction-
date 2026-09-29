#!/usr/bin/env python3
"""Build an offline AI Weather Quest precipitation submission preview.

This module intentionally has no upload, authentication, FTP, ECBox, HTTP, or
``AI_WQ_forecast_submission`` call.  It mirrors the global precipitation
DataArray produced by AI-WQ-package 3.29, writes both forecast periods plus a
manifest as one local transaction, and reopens every NetCDF before publishing
it.

Example (the selected prepared case must itself be a Thursday)::

    python quest_submission.py \
      --case /path/to/prepared.zarr \
      --checkpoint /path/to/best.pt \
      --init-date 20260813 \
      --team TEAM_ID --model MODEL_ID \
      --originating-centre ORIGIN_ID --expver EXPVER_ID \
      --calibration-json /path/to/uniform_blend_calibration.json \
      --output-dir /path/to/preview

Historical dates are refused unless ``--allow-historical`` is supplied.  That
flag only permits an offline preview; it never relaxes the Thursday rule and
never makes a historical artifact submit-able.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
import uuid

import numpy as np
import xarray as xr

try:
    from . import predict  # type: ignore
except (ImportError, ValueError):
    import predict  # type: ignore


AI_WQ_PACKAGE_CONTRACT = "3.29"
VARIABLE = "pr"
CALIBRATION_SCHEMA_VERSION = 2
CALIBRATION_METHOD = "P_cal=(1-alpha)*uniform+alpha*P"
CALIBRATION_ALPHA_SCOPE = "one scalar per forecast source across both lead periods"
PERIOD_BOUNDS: Mapping[int, tuple[float, float]] = {
    1: (18.0, 25.0),
    2: (25.0, 32.0),
}
_SAFE_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_PREVIEW_NOTICE = (
    "preview_not_accepted_without_registered package IDs/current window; "
    "use the current AI-WQ-package for official validation and upload"
)


@dataclass(frozen=True)
class SubmissionBundle:
    """Paths published by :func:`build_offline_submission_bundle`."""

    period_1: Path
    period_2: Path
    manifest: Path

    @property
    def paths(self) -> tuple[Path, Path, Path]:
        return (self.period_1, self.period_2, self.manifest)


def _filename_component(value: str, *, name: str) -> str:
    text = str(value)
    if not _SAFE_COMPONENT.fullmatch(text):
        raise ValueError(
            f"{name} must be 1-64 ASCII letters, digits, underscores, or "
            "hyphens, beginning with a letter or digit"
        )
    if text in {".", ".."} or Path(text).name != text:
        raise ValueError(f"{name} must be a safe filename component")
    return text


def _issue_date(value: str | date) -> date:
    if isinstance(value, datetime):
        parsed = value.date()
    elif isinstance(value, date):
        parsed = value
    else:
        text = str(value)
        if len(text) == 8 and text.isdigit():
            text = f"{text[:4]}-{text[4:6]}-{text[6:]}"
        try:
            parsed = date.fromisoformat(text)
        except ValueError as exc:
            raise ValueError("init_date must be YYYYMMDD or YYYY-MM-DD") from exc
    if parsed.weekday() != 3:
        raise ValueError(
            f"Forecast initialization {parsed.isoformat()} is not a Thursday"
        )
    return parsed


def _as_utc(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _inside_submission_window(issue: date, now_utc: datetime) -> bool:
    start = datetime.combine(issue, time.min, tzinfo=timezone.utc)
    return start <= now_utc < start + timedelta(days=4)


def _validate_window(
    issue: date,
    *,
    allow_historical: bool,
    now_utc: datetime | None,
) -> tuple[datetime, bool]:
    checked_at = _as_utc(now_utc)
    inside = _inside_submission_window(issue, checked_at)
    if not inside and not allow_historical:
        end = issue + timedelta(days=3)
        raise ValueError(
            "Forecast initialization is outside its Thursday-Sunday UTC "
            f"submission window ({issue.isoformat()} through {end.isoformat()}). "
            "Use --allow-historical only to make a clearly non-submittable "
            "offline preview."
        )
    return checked_at, inside


def _canonical_fingerprint(value: Any, *, source: str) -> str:
    fingerprint = str(value).strip().lower()
    if not _SHA256.fullmatch(fingerprint):
        raise ValueError(f"{source} cache_contract_sha256 must be 64 hexadecimal characters")
    return fingerprint


def _prepared_contract_fingerprint(path: Path) -> str:
    source = path.expanduser().resolve()
    if predict.is_zarr_store(source):
        try:
            import zarr
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise RuntimeError("zarr is required to verify prepared-cache provenance") from exc
        value = zarr.open_group(str(source), mode="r").attrs.get(
            "cache_contract_sha256"
        )
    elif source.is_file() and source.suffix.lower() == ".npz":
        with np.load(source, allow_pickle=False) as archive:
            value = (
                np.asarray(archive["cache_contract_sha256"]).item()
                if "cache_contract_sha256" in archive
                else None
            )
    else:
        raise FileNotFoundError(f"Prepared NPZ/Zarr artifact not found: {source}")
    if value in (None, ""):
        raise ValueError("Prepared artifact does not declare cache_contract_sha256")
    return _canonical_fingerprint(value, source="prepared artifact")


def _checkpoint_contract_fingerprint(payload: Mapping[str, Any]) -> str:
    normalization = payload.get("normalization")
    if not isinstance(normalization, Mapping):
        raise ValueError("Checkpoint normalization metadata is missing")
    value = normalization.get("cache_contract_sha256")
    if value in (None, ""):
        raise ValueError("Checkpoint does not declare cache_contract_sha256")
    return _canonical_fingerprint(value, source="checkpoint")


def _verify_contract_fingerprint(path: Path, payload: Mapping[str, Any]) -> str:
    prepared = _prepared_contract_fingerprint(path)
    checkpoint = _checkpoint_contract_fingerprint(payload)
    if prepared != checkpoint:
        raise ValueError(
            "Checkpoint/prepared provenance mismatch: cache_contract_sha256 differs"
        )
    return prepared


def _load_model_alpha(
    path: Path,
    *,
    fingerprint: str,
    checkpoint_sha256: str,
) -> tuple[float, Mapping[str, Any]]:
    source = path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Calibration JSON not found: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"Calibration file is not valid UTF-8 JSON: {source}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Calibration JSON root must be an object")
    if payload.get("schema_version") != CALIBRATION_SCHEMA_VERSION:
        raise ValueError(
            f"Calibration schema_version must be {CALIBRATION_SCHEMA_VERSION}"
        )
    if payload.get("method") != CALIBRATION_METHOD:
        raise ValueError(
            f"Calibration method must be exactly {CALIBRATION_METHOD!r}"
        )
    if payload.get("alpha_scope") != CALIBRATION_ALPHA_SCOPE:
        raise ValueError(
            f"Calibration alpha_scope must be exactly {CALIBRATION_ALPHA_SCOPE!r}"
        )
    alphas = payload.get("alphas")
    if not isinstance(alphas, Mapping) or "model" not in alphas:
        raise ValueError("Calibration JSON must contain alphas.model")
    try:
        alpha = float(alphas["model"])
    except (TypeError, ValueError) as exc:
        raise ValueError("Calibration alphas.model must be numeric") from exc
    if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("Calibration alphas.model must be finite and in [0, 1]")
    calibration_fingerprint = payload.get("checkpoint_cache_contract_sha256")
    if calibration_fingerprint in (None, ""):
        raise ValueError("Calibration JSON does not declare checkpoint_cache_contract_sha256")
    calibration_fingerprint = _canonical_fingerprint(
        calibration_fingerprint, source="calibration"
    )
    if calibration_fingerprint != fingerprint:
        raise ValueError(
            "Calibration/checkpoint provenance mismatch: cache_contract_sha256 differs"
        )
    calibration_checkpoint_sha256 = payload.get("checkpoint_sha256")
    if calibration_checkpoint_sha256 in (None, ""):
        raise ValueError("Calibration JSON does not declare checkpoint_sha256")
    calibration_checkpoint_sha256 = _canonical_fingerprint(
        calibration_checkpoint_sha256, source="calibration checkpoint"
    )
    if calibration_checkpoint_sha256 != checkpoint_sha256:
        raise ValueError(
            "Calibration/checkpoint provenance mismatch: checkpoint_sha256 differs"
        )
    return alpha, payload


def _uniform_blend(probabilities: np.ndarray, alpha: float) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    blended = (1.0 - alpha) / predict.N_QUINTILES + alpha * values
    return predict.normalize_probabilities(blended, category_axis=1)


def _official_attrs(
    *, period: int, team: str, model: str, originating_centre: str, expver: str
) -> dict[str, str]:
    start, end = PERIOD_BOUNDS[period]
    return {
        "standard_name": "Total precipitation probability",
        "cell_methods": "time: sum (interval: 24 hours)",
        "units": "1",
        "coordinates": "latitude longitude",
        "description": (
            f"pr prediction from {team} using {model} for forecasting period {period}"
        ),
        "Conventions": "CF-1.6",
        "forecast_period_bounds_units": "days into forecast",
        "forecast_period_bounds": f"[{start},{end}]",
        "shortName": "tp",
        "originating_centre": originating_centre,
        "expver": expver,
        "teamname": team,
        "modelname": model,
    }


def create_precipitation_dataarray(
    probabilities: np.ndarray,
    *,
    init_date: str | date,
    period: int,
    team: str,
    model: str,
    originating_centre: str,
    expver: str,
) -> xr.DataArray:
    """Create one AI-WQ-package 3.29-compatible precipitation DataArray."""

    if period not in PERIOD_BOUNDS:
        raise ValueError("period must be 1 or 2")
    issue = _issue_date(init_date)
    team = _filename_component(team, name="team")
    model = _filename_component(model, name="model")
    originating_centre = _filename_component(
        originating_centre, name="originating_centre"
    )
    expver = _filename_component(expver, name="expver")
    # AI-WQ-package 3.29 constructs its global template with np.empty(), whose
    # default dtype is float64.  Preserve that serialization contract.
    values = np.asarray(
        predict.validate_probability_cube(probabilities), dtype=np.float64
    )
    issue_time = np.datetime64(issue.isoformat() + "T00:00:00", "ns")
    start, end = PERIOD_BOUNDS[period]
    data = xr.DataArray(
        values,
        dims=("quintile", "latitude", "longitude"),
        coords={
            "quintile": np.arange(1, 6, dtype=np.float64) / 5.0,
            "latitude": (
                "latitude",
                np.arange(90.0, -91.0, -1.5, dtype=np.float64),
                {
                    "units": "degrees_north",
                    "long_name": "latitude",
                    "standard_name": "latitude",
                    "axis": "X",
                },
            ),
            "longitude": (
                "longitude",
                np.arange(0.0, 360.0, 1.5, dtype=np.float64),
                {
                    "units": "degrees_east",
                    "long_name": "longitude",
                    "standard_name": "longitude",
                    "axis": "Y",
                },
            ),
            "forecast_issue_date": issue_time,
            "forecast_period_start": issue_time
            + np.timedelta64(int(start * 24), "h"),
            "forecast_period_end": issue_time + np.timedelta64(int(end * 24), "h"),
            # The official precipitation template supplies ``height=None``;
            # after NetCDF serialization this is a scalar floating NaN.
            "height": np.float64(np.nan),
        },
        attrs=_official_attrs(
            period=period,
            team=team,
            model=model,
            originating_centre=originating_centre,
            expver=expver,
        ),
    )
    data.coords["forecast_issue_date"].attrs = {
        "standard_name": "forecast_issue_time",
        "long_name": "forecast issue time",
        "axis": "T",
    }
    data.coords["forecast_period_start"].attrs = {
        "long_name": "forecast period start",
        "axis": "T",
    }
    data.coords["forecast_period_end"].attrs = {
        "long_name": "forecast period end",
        "axis": "T",
    }
    return data


def _decoded_coordinates_attr(data: xr.DataArray) -> str | None:
    value = data.attrs.get("coordinates")
    if value is None:
        # xarray decodes CF ``coordinates`` into coordinate membership and
        # retains the original text in encoding rather than attrs.
        value = data.encoding.get("coordinates")
    return None if value is None else str(value)


def _validate_dataarray(
    data: xr.DataArray,
    expected_values: np.ndarray,
    *,
    issue: date,
    period: int,
    team: str,
    model: str,
    originating_centre: str,
    expver: str,
) -> None:
    expected_dims = ("quintile", "latitude", "longitude")
    if data.dims != expected_dims or data.shape != (5, 121, 240):
        raise ValueError(
            f"Serialized DataArray has dims/shape {data.dims}/{data.shape}, "
            f"expected {expected_dims}/(5, 121, 240)"
        )
    if data.dtype != np.dtype(np.float64):
        raise ValueError(f"Serialized probabilities must be float64, got {data.dtype}")
    if not np.array_equal(data.values, np.asarray(expected_values, dtype=np.float64)):
        raise ValueError("Serialized probabilities differ from the validated forecast")
    predict.validate_probability_cube(data.values)
    maximum_sum_error = float(
        np.max(np.abs(data.values.sum(axis=0, dtype=np.float64) - 1.0))
    )
    if maximum_sum_error > 1.0e-6:
        raise ValueError(
            "Serialized quintile probabilities do not sum to one within 1e-6 "
            f"(maximum absolute error={maximum_sum_error:g})"
        )
    np.testing.assert_array_equal(
        data.coords["quintile"].values,
        np.arange(1, 6, dtype=np.float64) / 5.0,
    )
    np.testing.assert_array_equal(
        data.coords["latitude"].values,
        np.arange(90.0, -91.0, -1.5, dtype=np.float64),
    )
    np.testing.assert_array_equal(
        data.coords["longitude"].values,
        np.arange(0.0, 360.0, 1.5, dtype=np.float64),
    )
    issue_time = np.datetime64(issue.isoformat() + "T00:00:00", "ns")
    start, end = PERIOD_BOUNDS[period]
    if data.coords["forecast_issue_date"].values != issue_time:
        raise ValueError("Serialized forecast_issue_date is incorrect")
    if data.coords["forecast_period_start"].values != (
        issue_time + np.timedelta64(int(start * 24), "h")
    ):
        raise ValueError("Serialized forecast_period_start is incorrect")
    if data.coords["forecast_period_end"].values != (
        issue_time + np.timedelta64(int(end * 24), "h")
    ):
        raise ValueError("Serialized forecast_period_end is incorrect")
    if "height" not in data.coords or not np.isnan(data.coords["height"].item()):
        raise ValueError("Serialized precipitation template must retain scalar height=None")

    expected_attrs = _official_attrs(
        period=period,
        team=team,
        model=model,
        originating_centre=originating_centre,
        expver=expver,
    )
    for name, value in expected_attrs.items():
        actual = _decoded_coordinates_attr(data) if name == "coordinates" else data.attrs.get(name)
        if actual != value:
            raise ValueError(
                f"Serialized AI-WQ attribute {name!r} is {actual!r}, expected {value!r}"
            )
    expected_coord_attrs = {
        "latitude": {
            "units": "degrees_north",
            "long_name": "latitude",
            "standard_name": "latitude",
            "axis": "X",
        },
        "longitude": {
            "units": "degrees_east",
            "long_name": "longitude",
            "standard_name": "longitude",
            "axis": "Y",
        },
        "forecast_issue_date": {
            "standard_name": "forecast_issue_time",
            "long_name": "forecast issue time",
            "axis": "T",
        },
        "forecast_period_start": {"long_name": "forecast period start", "axis": "T"},
        "forecast_period_end": {"long_name": "forecast period end", "axis": "T"},
    }
    for name, attrs in expected_coord_attrs.items():
        for attr_name, value in attrs.items():
            if data.coords[name].attrs.get(attr_name) != value:
                raise ValueError(
                    f"Serialized coordinate {name!r} attribute {attr_name!r} is incorrect"
                )


def _write_and_validate_netcdf(
    data: xr.DataArray,
    path: Path,
    expected_values: np.ndarray,
    *,
    issue: date,
    period: int,
    team: str,
    model: str,
    originating_centre: str,
    expver: str,
) -> None:
    data.to_netcdf(path)
    with xr.open_dataarray(path) as opened:
        decoded = opened.load()
    _validate_dataarray(
        decoded,
        expected_values,
        issue=issue,
        period=period,
        team=team,
        model=model,
        originating_centre=originating_centre,
        expver=expver,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cleanup(paths: Sequence[Path]) -> None:
    for path in paths:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _publish_transaction(staged: Sequence[tuple[Path, Path]]) -> None:
    transaction = uuid.uuid4().hex
    backups: dict[Path, Path] = {}
    published: list[Path] = []
    for _, final in staged:
        if final.exists() and not final.is_file():
            raise ValueError(f"Refusing to replace non-file output path: {final}")
    try:
        for _, final in staged:
            if final.exists():
                backup = final.with_name(f".{final.name}.{transaction}.backup")
                os.replace(final, backup)
                backups[final] = backup
        for stage, final in staged:
            os.replace(stage, final)
            published.append(final)
    except BaseException as exc:
        _cleanup(tuple(reversed(published)))
        rollback_errors: list[str] = []
        for final, backup in reversed(tuple(backups.items())):
            try:
                os.replace(backup, final)
            except OSError as rollback_error:
                rollback_errors.append(f"{backup.name} -> {final.name}: {rollback_error}")
        _cleanup(tuple(stage for stage, _ in staged))
        if rollback_errors:
            raise RuntimeError(
                "Submission preview failed and rollback was incomplete: "
                + "; ".join(rollback_errors)
            ) from exc
        raise
    else:
        _cleanup(tuple(backups.values()))


def build_offline_submission_bundle(
    case_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    init_date: str | date,
    team: str,
    model: str,
    originating_centre: str,
    expver: str,
    calibration_json: str | Path | None = None,
    device: str = "cpu",
    allow_historical: bool = False,
    now_utc: datetime | None = None,
) -> SubmissionBundle:
    """Run one prepared case and atomically publish an offline preview bundle."""

    issue = _issue_date(init_date)
    checked_at, inside_window = _validate_window(
        issue,
        allow_historical=allow_historical,
        now_utc=now_utc,
    )
    team = _filename_component(team, name="team")
    model = _filename_component(model, name="model")
    originating_centre = _filename_component(
        originating_centre, name="originating_centre"
    )
    expver = _filename_component(expver, name="expver")
    case = Path(case_path).expanduser().resolve()
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    canonical_init = issue.isoformat()

    batch = predict.load_prepared(case, init_date=canonical_init)
    if batch.n_cases != 1:
        raise ValueError(f"Submission preview requires exactly one prepared case; got {batch.n_cases}")
    if batch.init_dates != (canonical_init,):
        raise ValueError(
            f"Prepared case initialization {batch.init_dates!r} does not match {canonical_init}"
        )
    loaded_model, checkpoint_payload = predict.load_checkpoint_model(
        checkpoint,
        device=device,
        in_channels=int(batch.features.shape[2]),
    )
    if not isinstance(checkpoint_payload, Mapping):
        raise ValueError("Checkpoint payload must be a mapping")
    fingerprint = _verify_contract_fingerprint(case, checkpoint_payload)
    checkpoint_sha256 = _sha256_file(checkpoint)
    normalization = checkpoint_payload.get("normalization", {})
    provenance = (
        normalization.get("provenance", {})
        if isinstance(normalization, Mapping)
        else {}
    )
    if not isinstance(provenance, Mapping):
        provenance = {}
    official_era5_target = bool(provenance.get("official_era5_target", False))
    fuxi_permission_notice = provenance.get("fuxi_permission")
    if fuxi_permission_notice is not None:
        fuxi_permission_notice = str(fuxi_permission_notice)
    alpha = 1.0
    calibration_payload: Mapping[str, Any] | None = None
    calibration_source: Path | None = None
    if calibration_json is not None:
        calibration_source = Path(calibration_json).expanduser().resolve()
        alpha, calibration_payload = _load_model_alpha(
            calibration_source,
            fingerprint=fingerprint,
            checkpoint_sha256=checkpoint_sha256,
        )
    probabilities = predict.run_model(
        loaded_model, batch.features, batch.p0, device=device
    )[0]
    if calibration_source is not None:
        probabilities = _uniform_blend(probabilities, alpha)
    probabilities = np.stack(
        [predict.validate_probability_cube(probabilities[index]) for index in range(2)],
        axis=0,
    )

    issue_compact = issue.strftime("%Y%m%d")
    filenames = {
        period: f"pr_{issue_compact}_p{period}_{team}_{model}.nc"
        for period in (1, 2)
    }
    manifest_name = f"pr_{issue_compact}_{team}_{model}_manifest.json"
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    transaction = uuid.uuid4().hex
    final_paths = {period: destination / name for period, name in filenames.items()}
    stage_paths = {
        period: destination / f".{name}.{transaction}.stage.nc"
        for period, name in filenames.items()
    }
    final_manifest = destination / manifest_name
    stage_manifest = destination / f".{manifest_name}.{transaction}.stage.json"
    staged_all = tuple(stage_paths.values()) + (stage_manifest,)

    try:
        for period in (1, 2):
            data = create_precipitation_dataarray(
                probabilities[period - 1],
                init_date=issue,
                period=period,
                team=team,
                model=model,
                originating_centre=originating_centre,
                expver=expver,
            )
            _write_and_validate_netcdf(
                data,
                stage_paths[period],
                probabilities[period - 1],
                issue=issue,
                period=period,
                team=team,
                model=model,
                originating_centre=originating_centre,
                expver=expver,
            )

        files = []
        for period in (1, 2):
            values = probabilities[period - 1]
            maximum_sum_error = float(
                np.max(np.abs(values.astype(np.float64).sum(axis=0) - 1.0))
            )
            if maximum_sum_error > 1.0e-6:
                raise ValueError(
                    "Quintile probabilities do not sum to one within 1e-6 "
                    f"for period {period} (maximum error={maximum_sum_error:g})"
                )
            files.append(
                {
                    "period": period,
                    "filename": filenames[period],
                    "sha256": _sha256_file(stage_paths[period]),
                    "size_bytes": stage_paths[period].stat().st_size,
                    "forecast_period_bounds_days": list(PERIOD_BOUNDS[period]),
                    "probability_min": float(values.min()),
                    "probability_max": float(values.max()),
                    "maximum_probability_sum_error": maximum_sum_error,
                }
            )
        calibration_manifest: dict[str, Any] = {
            "applied": calibration_source is not None,
            "alpha_model": float(alpha),
            "path": None if calibration_source is None else str(calibration_source),
            "sha256": (
                None if calibration_source is None else _sha256_file(calibration_source)
            ),
        }
        if calibration_payload is not None:
            calibration_manifest["status"] = calibration_payload.get("status")
            calibration_manifest["fit_years"] = calibration_payload.get("fit_years")
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "kind": "ai_weather_quest_offline_submission_preview",
            "ai_wq_package_contract": AI_WQ_PACKAGE_CONTRACT,
            "variable": VARIABLE,
            "initialization_date": issue_compact,
            "initialization_is_thursday": True,
            "inside_current_submission_window_at_build": inside_window,
            "window_check_utc": checked_at.isoformat(),
            "historical_or_out_of_window_preview": not inside_window,
            "allow_historical_used": bool(allow_historical),
            "teamname": team,
            "modelname": model,
            "originating_centre": originating_centre,
            "expver": expver,
            "prepared_case": str(case),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "cache_contract_sha256": fingerprint,
            "checkpoint_cache_contract_sha256": fingerprint,
            "official_era5_target": official_era5_target,
            "fuxi_permission_notice": fuxi_permission_notice,
            "checkpoint_provenance": {
                "official_era5_target": official_era5_target,
                "official_ai_weather_quest_validation": bool(
                    provenance.get("official_ai_weather_quest_validation", False)
                ),
                "purpose": provenance.get("purpose"),
                "notice": provenance.get("notice"),
                "fuxi_permission": fuxi_permission_notice,
            },
            "calibration": calibration_manifest,
            "files": files,
            "offline_only": True,
            "upload_performed": False,
            "registered_identifiers_verified": False,
            "official_package_validation_performed": False,
            "competition_use_authorized": False,
            "submission_ready": False,
            "preview_not_accepted_without_registered_package_ids_and_current_window": True,
            "preview_notice": _PREVIEW_NOTICE,
        }
        encoded = (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
            "utf-8"
        )
        stage_manifest.write_bytes(encoded)
        decoded_manifest = json.loads(stage_manifest.read_text(encoding="utf-8"))
        if decoded_manifest != manifest:
            raise ValueError("Serialized manifest differs from the validated manifest")

        _publish_transaction(
            (
                (stage_paths[1], final_paths[1]),
                (stage_paths[2], final_paths[2]),
                (stage_manifest, final_manifest),
            )
        )
    except BaseException:
        _cleanup(staged_all)
        raise
    finally:
        _cleanup(staged_all)

    return SubmissionBundle(final_paths[1], final_paths[2], final_manifest)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build two AI-WQ 3.29-format precipitation NetCDF previews and a "
            "manifest. This command is offline-only and never uploads."
        ),
        epilog=(
            "The team/model/origin/expver values are not verified offline. "
            "Use registered values and the current AI-WQ-package for an actual submission."
        ),
    )
    parser.add_argument("--case", required=True, type=Path, help="Prepared NPZ or Zarr cache")
    parser.add_argument("--checkpoint", required=True, type=Path, help="TPProbUNet checkpoint")
    parser.add_argument("--init-date", required=True, help="Thursday YYYYMMDD or YYYY-MM-DD")
    parser.add_argument("--team", required=True, help="Registered team filename component")
    parser.add_argument("--model", required=True, help="Registered model filename component")
    parser.add_argument(
        "--originating-centre", required=True, help="Registered package-origin identifier"
    )
    parser.add_argument("--expver", required=True, help="Registered package expver identifier")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--calibration-json", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--allow-historical",
        action="store_true",
        help="Allow an out-of-window offline preview; it remains non-submittable",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    bundle = build_offline_submission_bundle(
        args.case,
        args.checkpoint,
        args.output_dir,
        init_date=args.init_date,
        team=args.team,
        model=args.model,
        originating_centre=args.originating_centre,
        expver=args.expver,
        calibration_json=args.calibration_json,
        device=args.device,
        allow_historical=args.allow_historical,
    )
    for path in bundle.paths:
        print(path)
    print("OFFLINE PREVIEW ONLY: upload_performed=false")


if __name__ == "__main__":
    main()
