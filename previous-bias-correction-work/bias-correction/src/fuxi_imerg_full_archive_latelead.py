#!/usr/bin/env python
"""Full-archive FuXi rainfall adapters for forecast Weeks 5 and 6.

The spatial and temporal models use the same data, predictors, loss, seeds,
and validation split. Both predict a residual around a training-only log-bias
forecast. Weeks 1--4 remain exactly equal to that baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr
from cartopy import crs as ccrs
from cartopy import feature as cfeature
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER


from project_paths import NEURAL_ADAPTER_SRC as NEURAL_SRC
from project_paths import PROJECT_ROOT as HERE
from project_paths import SOURCE_ROOT
if str(NEURAL_SRC) not in sys.path:
    sys.path.insert(0, str(NEURAL_SRC))

import fuxi_imerg_experiment as base  # noqa: E402
import fuxi_imerg_spatiotemporal as diagnostics  # noqa: E402
from fuxi_adapter.anchored import (  # noqa: E402
    fit_anchored_target_scale,
    reconstruct_anchored_precipitation,
    standardize_anchored_target,
)
from fuxi_adapter.baselines import (  # noqa: E402
    apply_log_bias_correction,
    fit_log_bias_correction,
)
from fuxi_adapter.models import ResidualUNet, TemporalAttentionUNet  # noqa: E402
from fuxi_adapter.training import predict, set_deterministic_seed  # noqa: E402
from fuxi_adapter.v3_training import (  # noqa: E402
    AnchoredSequenceDataset,
    train_anchored_model,
)


TRAIN_YEARS = tuple(range(2002, 2019))
VALIDATION_YEARS = (2019,)
TEST_YEARS = (2020, 2021)
SEEDS = (42, 43, 44)
ACTIVE_LEADS = (4, 5)  # zero-based W5 and W6
LEAD_WEIGHTS = (0.0, 0.0, 0.0, 0.0, 0.5, 0.5)
LOSS_COEFFICIENTS = {"smooth_l1": 0.75, "acc": 0.20, "bias": 0.05}
RESULTS_ROOT = HERE / "results" / "fuxi_imerg_full_archive_latelead"
DEFAULT_MEMBER_SUMMARY_CACHE = (
    HERE / "cache" / "fuxi_51member_weekly_summaries_v1.npz"
)
MEMBER_SUMMARY_CACHE_VERSION = 1
MEMBER_SUMMARY_FEATURE_NAMES = (
    "fuxi_tp_member_log_median",
    "fuxi_tp_member_log_iqr",
    "fuxi_tp_member_probability_ge_1mm_day",
    "fuxi_tp_member_probability_ge_10mm_day",
    "fuxi_t2m_member_spread_weekly",
)
MEMBER_SUMMARY_DEFINITIONS = {
    "fuxi_tp_member_log_median": (
        "q50 across 51 members after log1p of each member's complete "
        "seven-day mean TP in mm day-1"
    ),
    "fuxi_tp_member_log_iqr": (
        "q75-q25 across 51 members after log1p of each member's complete "
        "seven-day mean TP in mm day-1"
    ),
    "fuxi_tp_member_probability_ge_1mm_day": (
        "fraction of 51 members whose complete seven-day mean TP is at "
        "least 1 mm day-1"
    ),
    "fuxi_tp_member_probability_ge_10mm_day": (
        "fraction of 51 members whose complete seven-day mean TP is at "
        "least 10 mm day-1"
    ),
    "fuxi_t2m_member_spread_weekly": (
        "population standard deviation across 51 members after forming "
        "each member's complete seven-day mean T2M in K"
    ),
}


@dataclass(frozen=True)
class FuxiMemberSummaries:
    """Raw weekly FuXi ensemble summaries aligned to ``ForecastData``.

    All gridded arrays are unnormalized float32 fields with shape
    ``[initialization, lead_week, latitude, longitude]``.  Keeping this cache
    raw prevents a normalization fitted for one temporal split from being
    accidentally reused by another split.
    """

    initializations: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    t2m_weekly_mean: np.ndarray
    tp_member_log_median: np.ndarray
    tp_member_log_iqr: np.ndarray
    tp_member_probability_ge_1mm_day: np.ndarray
    tp_member_probability_ge_10mm_day: np.ndarray
    t2m_member_spread_weekly: np.ndarray
    source_fingerprint: str
    cache_path: str | None = None
    cache_sha256: str | None = None

    @property
    def feature_fields(self) -> Mapping[str, np.ndarray]:
        """Return the five member predictors in their frozen channel order."""

        return {
            "fuxi_tp_member_log_median": self.tp_member_log_median,
            "fuxi_tp_member_log_iqr": self.tp_member_log_iqr,
            "fuxi_tp_member_probability_ge_1mm_day": (
                self.tp_member_probability_ge_1mm_day
            ),
            "fuxi_tp_member_probability_ge_10mm_day": (
                self.tp_member_probability_ge_10mm_day
            ),
            "fuxi_t2m_member_spread_weekly": self.t2m_member_spread_weekly,
        }


class LateLeadSpatialUNet(ResidualUNet):
    """Spatial U-Net whose correction is identically zero before Week 5."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = super().forward(inputs)
        return torch.cat((torch.zeros_like(residual[:, :4]), residual[:, 4:]), dim=1)


class LateLeadTemporalUNet(TemporalAttentionUNet):
    """Temporal U-Net whose correction is identically zero before Week 5."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = super().forward(inputs)
        return torch.cat((torch.zeros_like(residual[:, :4]), residual[:, 4:]), dim=1)


def configure_contract() -> None:
    """Apply the expanded split to the tested native-archive loader."""

    base.TRAIN_YEARS = TRAIN_YEARS
    base.VALIDATION_YEARS = VALIDATION_YEARS
    base.TEST_YEARS = TEST_YEARS
    base.ALL_YEARS = TRAIN_YEARS + VALIDATION_YEARS + TEST_YEARS

    diagnostics.METHOD_LABELS.update(
        {
            "raw_fuxi": "Raw FuXi",
            "log_bias": "Log-bias",
            "spatial_unet": "Spatial W5–W6 adapter",
            "spatiotemporal_unet": "Temporal W5–W6 adapter",
            "lead_adaptive_hybrid": "Validation-selected adapter",
        }
    )
    diagnostics.PLOT_METHODS = (
        "raw_fuxi",
        "log_bias",
        "spatial_unet",
        "spatiotemporal_unet",
    )
    diagnostics.ALL_COMPARISONS = (
        ("log_bias", "raw_fuxi"),
        ("spatial_unet", "raw_fuxi"),
        ("spatial_unet", "log_bias"),
        ("spatiotemporal_unet", "raw_fuxi"),
        ("spatiotemporal_unet", "log_bias"),
        ("spatiotemporal_unet", "spatial_unet"),
        ("lead_adaptive_hybrid", "raw_fuxi"),
        ("lead_adaptive_hybrid", "log_bias"),
        ("lead_adaptive_hybrid", "spatial_unet"),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    root = Path(path)
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        digest.update(item.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        with item.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _member_summary_source_fingerprint(source_files: Sequence[str]) -> str:
    """Fingerprint the exact ordered native-shard inventory used by a cache."""

    digest = hashlib.sha256()
    for value in source_files:
        path = Path(value).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        stat = path.stat()
        record = f"{path}\0{stat.st_size}\0{stat.st_mtime_ns}\n"
        digest.update(record.encode("utf-8"))
    return digest.hexdigest()


def _summarize_member_fields(
    tp_hourly: np.ndarray,
    t2m_daily: np.ndarray,
) -> Mapping[str, np.ndarray]:
    """Reduce raw daily members to weekly means, then summarize members.

    This ordering is deliberate: ensemble quantiles of member-wise weekly
    rainfall are not interchangeable with quantiles computed separately for
    each day and averaged afterward.
    """

    tp_hourly = np.asarray(tp_hourly, dtype=np.float32)
    t2m_daily = np.asarray(t2m_daily, dtype=np.float32)
    if tp_hourly.ndim != 4 or tp_hourly.shape[:2] != (51, 42):
        raise base.DataContractError(
            f"unexpected TP member shape: {tp_hourly.shape}"
        )
    if t2m_daily.shape != tp_hourly.shape:
        raise base.DataContractError(
            f"T2M member shape {t2m_daily.shape} differs from TP {tp_hourly.shape}"
        )
    if (
        not np.isfinite(tp_hourly).all()
        or np.any(tp_hourly < 0.0)
        or not np.isfinite(t2m_daily).all()
    ):
        raise base.DataContractError("TP/T2M member fields contain invalid values")

    height, width = tp_hourly.shape[-2:]
    tp_member_weekly = (
        (tp_hourly * np.float32(24.0))
        .reshape(51, 6, 7, height, width)
        .mean(axis=2, dtype=np.float64)
        .astype(np.float32)
    )
    t2m_member_weekly = (
        t2m_daily.reshape(51, 6, 7, height, width)
        .mean(axis=2, dtype=np.float64)
        .astype(np.float32)
    )
    log_tp = np.log1p(tp_member_weekly).astype(np.float32)
    log_quantiles = np.quantile(
        log_tp,
        np.asarray([0.25, 0.50, 0.75], dtype=np.float64),
        axis=0,
        method="linear",
    ).astype(np.float32)
    result = {
        # Preserve the exact accumulation order used by load_t2m_weekly so
        # enabling member summaries cannot change the existing T2M channel.
        "t2m_weekly_mean": t2m_daily.reshape(
            51, 6, 7, height, width
        ).mean(axis=(0, 2), dtype=np.float64).astype(np.float32),
        "fuxi_tp_member_log_median": log_quantiles[1],
        "fuxi_tp_member_log_iqr": (log_quantiles[2] - log_quantiles[0]).astype(
            np.float32
        ),
        "fuxi_tp_member_probability_ge_1mm_day": np.mean(
            tp_member_weekly >= np.float32(1.0), axis=0, dtype=np.float32
        ),
        "fuxi_tp_member_probability_ge_10mm_day": np.mean(
            tp_member_weekly >= np.float32(10.0), axis=0, dtype=np.float32
        ),
        "fuxi_t2m_member_spread_weekly": t2m_member_weekly.std(
            axis=0, ddof=0, dtype=np.float64
        ).astype(np.float32),
    }
    expected_shape = (6, height, width)
    for name, values in result.items():
        if values.shape != expected_shape or values.dtype != np.float32:
            raise base.DataContractError(
                f"invalid {name} member-summary shape or dtype"
            )
        if not np.isfinite(values).all():
            raise base.DataContractError(f"{name} contains non-finite values")
    if np.any(result["fuxi_tp_member_log_iqr"] < 0.0):
        raise base.DataContractError("member log-IQR contains negative values")
    if np.any(result["fuxi_t2m_member_spread_weekly"] < 0.0):
        raise base.DataContractError("member T2M spread contains negative values")
    for name in (
        "fuxi_tp_member_probability_ge_1mm_day",
        "fuxi_tp_member_probability_ge_10mm_day",
    ):
        if np.any(result[name] < 0.0) or np.any(result[name] > 1.0):
            raise base.DataContractError(f"{name} lies outside [0, 1]")
    return result


def _validate_member_summaries(
    summaries: FuxiMemberSummaries,
    forecast: base.ForecastData,
) -> None:
    """Validate alignment and physical bounds against one forecast archive."""

    if not np.array_equal(
        np.asarray(summaries.initializations, dtype="datetime64[D]"),
        np.asarray(forecast.initializations, dtype="datetime64[D]"),
    ):
        raise base.DataContractError("member-summary initializations are misaligned")
    if not np.array_equal(summaries.latitude, forecast.latitude) or not np.array_equal(
        summaries.longitude, forecast.longitude
    ):
        raise base.DataContractError("member-summary grid is misaligned")
    expected_shape = forecast.ensemble_mean.shape
    fields = {"t2m_weekly_mean": summaries.t2m_weekly_mean}
    fields.update(summaries.feature_fields)
    for name, values in fields.items():
        if values.shape != expected_shape or values.dtype != np.float32:
            raise base.DataContractError(
                f"member-summary {name} shape or dtype is invalid"
            )
        if not np.isfinite(values).all():
            raise base.DataContractError(f"member-summary {name} is non-finite")
    for name in (
        "fuxi_tp_member_probability_ge_1mm_day",
        "fuxi_tp_member_probability_ge_10mm_day",
    ):
        values = summaries.feature_fields[name]
        if np.any(values < 0.0) or np.any(values > 1.0):
            raise base.DataContractError(f"member-summary {name} lies outside [0, 1]")
    if np.any(summaries.tp_member_log_iqr < 0.0):
        raise base.DataContractError("member-summary log-IQR is negative")
    if np.any(summaries.t2m_member_spread_weekly < 0.0):
        raise base.DataContractError("member-summary T2M spread is negative")


def validate_fuxi_member_summaries(
    summaries: FuxiMemberSummaries,
    forecast: base.ForecastData,
) -> None:
    """Public alignment check used before inserting member-summary features."""

    _validate_member_summaries(summaries, forecast)


def _member_summaries_from_cache(
    cache_path: Path,
    forecast: base.ForecastData,
    source_fingerprint: str,
) -> FuxiMemberSummaries:
    required = {
        "contract_version",
        "source_fingerprint",
        "initializations",
        "latitude",
        "longitude",
        "t2m_weekly_mean",
        *MEMBER_SUMMARY_FEATURE_NAMES,
    }
    try:
        with np.load(cache_path, allow_pickle=False) as cached:
            missing = required.difference(cached.files)
            if missing:
                raise base.DataContractError(
                    f"member-summary cache is missing fields: {sorted(missing)}"
                )
            version = int(np.asarray(cached["contract_version"]).item())
            cached_fingerprint = str(
                np.asarray(cached["source_fingerprint"]).item()
            )
            if version != MEMBER_SUMMARY_CACHE_VERSION:
                raise base.DataContractError(
                    f"member-summary cache version {version} is unsupported"
                )
            if cached_fingerprint != source_fingerprint:
                raise base.DataContractError(
                    "member-summary cache source fingerprint is stale"
                )
            summaries = FuxiMemberSummaries(
                initializations=np.asarray(
                    cached["initializations"], dtype="datetime64[D]"
                ).copy(),
                latitude=np.asarray(cached["latitude"], dtype=np.float64).copy(),
                longitude=np.asarray(cached["longitude"], dtype=np.float64).copy(),
                t2m_weekly_mean=np.asarray(
                    cached["t2m_weekly_mean"], dtype=np.float32
                ).copy(),
                tp_member_log_median=np.asarray(
                    cached["fuxi_tp_member_log_median"], dtype=np.float32
                ).copy(),
                tp_member_log_iqr=np.asarray(
                    cached["fuxi_tp_member_log_iqr"], dtype=np.float32
                ).copy(),
                tp_member_probability_ge_1mm_day=np.asarray(
                    cached["fuxi_tp_member_probability_ge_1mm_day"],
                    dtype=np.float32,
                ).copy(),
                tp_member_probability_ge_10mm_day=np.asarray(
                    cached["fuxi_tp_member_probability_ge_10mm_day"],
                    dtype=np.float32,
                ).copy(),
                t2m_member_spread_weekly=np.asarray(
                    cached["fuxi_t2m_member_spread_weekly"], dtype=np.float32
                ).copy(),
                source_fingerprint=source_fingerprint,
                cache_path=str(cache_path.resolve()),
                cache_sha256=sha256_file(cache_path),
            )
    except (OSError, ValueError) as error:
        raise base.DataContractError(
            f"cannot read member-summary cache {cache_path}: {error}"
        ) from error
    _validate_member_summaries(summaries, forecast)
    return summaries


def _write_member_summary_cache(
    cache_path: Path,
    summaries: FuxiMemberSummaries,
) -> str:
    cache_path = cache_path.resolve()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(
        f".{cache_path.name}.{os.getpid()}.temporary"
    )
    payload = {
        "contract_version": np.asarray(
            MEMBER_SUMMARY_CACHE_VERSION, dtype=np.int16
        ),
        "source_fingerprint": np.asarray(summaries.source_fingerprint),
        "initializations": np.asarray(
            summaries.initializations, dtype="datetime64[D]"
        ),
        "latitude": np.asarray(summaries.latitude, dtype=np.float64),
        "longitude": np.asarray(summaries.longitude, dtype=np.float64),
        "t2m_weekly_mean": summaries.t2m_weekly_mean,
        **summaries.feature_fields,
    }
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **payload)
        os.replace(temporary, cache_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return sha256_file(cache_path)


def load_fuxi_member_summaries(
    forecast: base.ForecastData,
    cache_path: Path | None = DEFAULT_MEMBER_SUMMARY_CACHE,
) -> FuxiMemberSummaries:
    """Load or derive raw weekly summaries from all 51 FuXi members.

    TP and T2M are read together once per native shard.  The shared cache is
    deliberately pre-normalization so every experiment must fit its own
    train-only transformation.
    """

    source_fingerprint = _member_summary_source_fingerprint(forecast.source_files)
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.is_file():
            return _member_summaries_from_cache(
                cache_path, forecast, source_fingerprint
            )

    by_initialization: dict[np.datetime64, Mapping[str, np.ndarray]] = {}
    expected_members = np.arange(51, dtype=np.int16)
    expected_leads = np.arange(1, 43, dtype=np.int16)
    for source in forecast.source_files:
        path = Path(source)
        with xr.open_dataset(path) as dataset:
            if dataset.sizes.get("member") != 51 or dataset.sizes.get("lead_day") != 42:
                raise base.DataContractError(f"unexpected member/lead dimensions in {path}")
            if not np.array_equal(dataset.member.values, expected_members):
                raise base.DataContractError(f"member labels differ in {path}")
            if not np.array_equal(dataset.lead_day.values, expected_leads):
                raise base.DataContractError(f"lead_day must be 1..42 in {path}")
            if dataset.tp.attrs.get("units") != "mm h-1":
                raise base.DataContractError(f"unexpected TP units in {path}")
            if dataset.t2m.attrs.get("units") != "K":
                raise base.DataContractError(f"unexpected T2M units in {path}")
            init = np.asarray(
                dataset.forecast_reference_time.values, dtype="datetime64[D]"
            ).reshape(-1)[0]
            if init in by_initialization:
                raise base.DataContractError(
                    f"duplicate member-summary initialization {init}"
                )
            tp = np.asarray(dataset.tp.load().values, dtype=np.float32)
            t2m = np.asarray(dataset.t2m.load().values, dtype=np.float32)
            if tp.shape != (51, 42, 27, 27) or t2m.shape != tp.shape:
                raise base.DataContractError(
                    f"unexpected TP/T2M member shape in {path}"
                )
            by_initialization[init] = _summarize_member_fields(tp, t2m)

    if len(by_initialization) != len(forecast.initializations):
        raise base.DataContractError(
            "member-summary initialization count differs from forecast"
        )
    missing = [
        init
        for init in np.asarray(forecast.initializations, dtype="datetime64[D]")
        if init not in by_initialization
    ]
    if missing:
        raise base.DataContractError(
            f"member summaries are missing {len(missing)} forecast initializations"
        )

    def stack(name: str) -> np.ndarray:
        values = np.stack(
            [by_initialization[init][name] for init in forecast.initializations]
        )
        return np.asarray(values, dtype=np.float32)

    summaries = FuxiMemberSummaries(
        initializations=np.asarray(
            forecast.initializations, dtype="datetime64[D]"
        ).copy(),
        latitude=np.asarray(forecast.latitude, dtype=np.float64).copy(),
        longitude=np.asarray(forecast.longitude, dtype=np.float64).copy(),
        t2m_weekly_mean=stack("t2m_weekly_mean"),
        tp_member_log_median=stack("fuxi_tp_member_log_median"),
        tp_member_log_iqr=stack("fuxi_tp_member_log_iqr"),
        tp_member_probability_ge_1mm_day=stack(
            "fuxi_tp_member_probability_ge_1mm_day"
        ),
        tp_member_probability_ge_10mm_day=stack(
            "fuxi_tp_member_probability_ge_10mm_day"
        ),
        t2m_member_spread_weekly=stack(
            "fuxi_t2m_member_spread_weekly"
        ),
        source_fingerprint=source_fingerprint,
        cache_path=str(Path(cache_path).resolve()) if cache_path is not None else None,
    )
    _validate_member_summaries(summaries, forecast)
    if cache_path is None:
        return summaries
    cache_sha256 = _write_member_summary_cache(Path(cache_path), summaries)
    return FuxiMemberSummaries(
        initializations=summaries.initializations,
        latitude=summaries.latitude,
        longitude=summaries.longitude,
        t2m_weekly_mean=summaries.t2m_weekly_mean,
        tp_member_log_median=summaries.tp_member_log_median,
        tp_member_log_iqr=summaries.tp_member_log_iqr,
        tp_member_probability_ge_1mm_day=(
            summaries.tp_member_probability_ge_1mm_day
        ),
        tp_member_probability_ge_10mm_day=(
            summaries.tp_member_probability_ge_10mm_day
        ),
        t2m_member_spread_weekly=summaries.t2m_member_spread_weekly,
        source_fingerprint=summaries.source_fingerprint,
        cache_path=summaries.cache_path,
        cache_sha256=cache_sha256,
    )


def load_t2m_weekly(forecast: base.ForecastData) -> np.ndarray:
    """Read weekly ensemble-mean T2M in the forecast initialization order."""

    fields: dict[np.datetime64, np.ndarray] = {}
    for source in forecast.source_files:
        path = Path(source)
        with xr.open_dataset(path) as dataset:
            if dataset.t2m.attrs.get("units") != "K":
                raise base.DataContractError(f"unexpected T2M units in {path}")
            init = np.asarray(
                dataset.forecast_reference_time.values, dtype="datetime64[D]"
            ).reshape(-1)[0]
            daily = np.asarray(dataset.t2m.load().values, dtype=np.float32)
            if daily.shape != (51, 42, 27, 27) or not np.isfinite(daily).all():
                raise base.DataContractError(f"invalid T2M field in {path}")
            weekly = daily.reshape(51, 6, 7, 27, 27).mean(
                axis=(0, 2), dtype=np.float64
            )
            fields[init] = weekly.astype(np.float32)
    if len(fields) != len(forecast.initializations):
        raise base.DataContractError("T2M initialization count differs from TP")
    result = np.stack([fields[init] for init in forecast.initializations])
    if result.shape != forecast.ensemble_mean.shape:
        raise base.DataContractError("weekly T2M shape differs from weekly TP")
    return result


def normalize_feature(
    values: np.ndarray,
    train_indices: np.ndarray,
    weights: np.ndarray,
    *,
    preserve_full_domain: bool = False,
) -> tuple[np.ndarray, Mapping[str, Sequence[float]]]:
    mean, std = base.weighted_lead_moments(values, train_indices, weights)
    normalized = (values - mean[None, :, None, None]) / std[None, :, None, None]
    support = weights > 0.0
    valid = (
        np.isfinite(normalized)
        if preserve_full_domain
        else support[None, None] & np.isfinite(normalized)
    )
    normalized = np.where(valid, normalized, 0.0).astype(np.float32)
    return normalized, {"mean_by_lead": mean.tolist(), "std_by_lead": std.tolist()}


def make_features(
    forecast: base.ForecastData,
    observations: base.ObservationData,
    weights: np.ndarray,
    train_indices: np.ndarray,
    t2m_weekly: np.ndarray,
    *,
    preserve_fuxi_context: bool = False,
) -> tuple[np.ndarray, Mapping[str, object]]:
    standard = base.make_neural_arrays(
        forecast,
        observations,
        weights,
        train_indices,
        preserve_fuxi_context=preserve_fuxi_context,
    )
    log_anomaly = (
        np.log1p(forecast.ensemble_mean)
        - np.log1p(observations.weekly_climatology)
    ).astype(np.float32)
    anomaly, anomaly_stats = normalize_feature(log_anomaly, train_indices, weights)
    t2m, t2m_stats = normalize_feature(
        t2m_weekly,
        train_indices,
        weights,
        preserve_full_domain=preserve_fuxi_context,
    )
    features = np.concatenate(
        (standard.inputs, anomaly[:, :, None], t2m[:, :, None]), axis=2
    ).astype(np.float32)
    if features.shape != (len(forecast.initializations), 6, 11, 27, 27):
        raise base.DataContractError(f"unexpected feature shape: {features.shape}")
    if not np.isfinite(features).all():
        raise base.DataContractError("neural features contain non-finite values")
    normalization = dict(standard.feature_stats)
    normalization["explicit_log_fuxi_anomaly"] = anomaly_stats
    normalization["fuxi_t2m_weekly"] = t2m_stats
    normalization["input_channels"] = [
        *standard.feature_stats["input_channels"],
        "explicit_log_fuxi_anomaly",
        "fuxi_t2m_weekly",
    ]
    normalization["spatial_context"] = {
        "enabled": bool(preserve_fuxi_context),
        "full_domain_channels": (
            ["log_fuxi_mean", "log_fuxi_spread", "fuxi_t2m_weekly"]
            if preserve_fuxi_context
            else []
        ),
        "support_limited_channels": [
            "log_imerg_climatology",
            "explicit_log_fuxi_anomaly",
        ],
        "normalization_fit": "training cases and positive target weights only",
        "target_and_loss_support": "positive target weights only",
    }
    if preserve_fuxi_context:
        support = weights > 0.0
        for channel_index, name in ((0, "log_fuxi_mean"), (1, "log_fuxi_spread"), (10, "fuxi_t2m_weekly")):
            outside = features[:, :, channel_index, ~support]
            if not np.isfinite(outside).all() or not np.any(outside != 0.0):
                raise base.DataContractError(
                    f"full-domain FuXi context is missing for {name}"
                )
        for channel_index, name in ((2, "log_imerg_climatology"), (9, "explicit_log_fuxi_anomaly")):
            if np.any(features[:, :, channel_index, ~support] != 0.0):
                raise base.DataContractError(
                    f"IMD-supported channel leaks outside support: {name}"
                )
    return features, normalization


def make_dataset(
    indices: np.ndarray,
    features: np.ndarray,
    target: np.ndarray,
    bias_baseline: np.ndarray,
    observations: base.ObservationData,
    support: np.ndarray,
) -> AnchoredSequenceDataset:
    valid = np.broadcast_to(
        support[None, None], (len(indices), 6, 27, 27)
    ).copy()
    return AnchoredSequenceDataset(
        features[indices],
        target[indices],
        bias_baseline[indices],
        observations.weekly_truth[indices],
        observations.weekly_climatology[indices],
        valid,
    )


def reconstruct(
    baseline: np.ndarray,
    residual: np.ndarray,
    target_scale: np.ndarray,
    support: np.ndarray,
) -> np.ndarray:
    valid = np.broadcast_to(support[None, None], baseline.shape)
    prediction = reconstruct_anchored_precipitation(
        baseline, residual, target_scale, valid_mask=valid
    )
    prediction[..., ~support] = np.nan
    return prediction


def train_candidate(
    name: str,
    features: np.ndarray,
    target: np.ndarray,
    target_scale: np.ndarray,
    bias_baseline: np.ndarray,
    forecast: base.ForecastData,
    observations: base.ObservationData,
    weights: np.ndarray,
    splits: Mapping[str, np.ndarray],
    output: Path,
    *,
    smoke: bool,
) -> tuple[np.ndarray, np.ndarray, Mapping[str, object]]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(min(8, os.cpu_count() or 1))
    train_indices = splits["train"][:32] if smoke else splits["train"]
    validation_indices = splits["validation"][:8] if smoke else splits["validation"]
    test_indices = splits["test"][:8] if smoke else splits["test"]
    seeds = SEEDS[:1] if smoke else SEEDS
    support = weights > 0.0
    train_data = make_dataset(
        train_indices, features, target, bias_baseline, observations, support
    )
    validation_data = make_dataset(
        validation_indices, features, target, bias_baseline, observations, support
    )

    validation_members = []
    test_members = []
    records = []
    parameter_count = None
    for seed in seeds:
        print(f"  {name}, seed {seed}", flush=True)
        set_deterministic_seed(seed)
        if name == "spatial":
            model = LateLeadSpatialUNet(
                in_channels=features.shape[2], base_channels=18, dropout=0.2
            )
        elif name == "temporal":
            model = LateLeadTemporalUNet(
                in_channels=features.shape[2],
                base_channels=16,
                dropout=0.2,
                max_leads=6,
            )
        else:
            raise ValueError(name)
        count = sum(parameter.numel() for parameter in model.parameters())
        if parameter_count is None:
            parameter_count = count
        elif parameter_count != count:
            raise RuntimeError("parameter count changed between seeds")
        run_directory = output / "models" / name / f"seed_{seed}"
        (run_directory / "logs").mkdir(parents=True, exist_ok=True)
        (run_directory / "checkpoints").mkdir(parents=True, exist_ok=True)
        result = train_anchored_model(
            model,
            train_data,
            validation_data,
            weights,
            target_scale,
            LEAD_WEIGHTS,
            LOSS_COEFFICIENTS,
            run_directory,
            seed=seed,
            device=device,
            batch_size=16,
            max_epochs=2 if smoke else 100,
            patience=1 if smoke else 15,
            learning_rate=3.0e-4,
            weight_decay=3.0e-4,
            smooth_l1_beta=1.0,
            num_workers=0,
            use_amp=True,
        )
        validation_residual = predict(
            model,
            features[validation_indices],
            device=device,
            batch_size=32,
            use_amp=True,
        )
        test_residual = predict(
            model,
            features[test_indices],
            device=device,
            batch_size=32,
            use_amp=True,
        )
        if not np.array_equal(validation_residual[:, :4], np.zeros_like(validation_residual[:, :4])):
            raise base.DataContractError(f"{name} changed an inactive validation lead")
        if not np.array_equal(test_residual[:, :4], np.zeros_like(test_residual[:, :4])):
            raise base.DataContractError(f"{name} changed an inactive test lead")
        validation_members.append(
            reconstruct(
                bias_baseline[validation_indices],
                validation_residual,
                target_scale,
                support,
            )
        )
        test_members.append(
            reconstruct(
                bias_baseline[test_indices], test_residual, target_scale, support
            )
        )
        records.append(
            {
                "seed": seed,
                "best_epoch": int(result.best_epoch),
                "best_validation_loss": float(result.best_validation_loss),
                "elapsed_seconds": float(result.elapsed_seconds),
                "checkpoint": str(
                    (run_directory / "checkpoints" / "best.pt").relative_to(output)
                ),
            }
        )
    validation_prediction = np.mean(
        validation_members, axis=0, dtype=np.float64
    ).astype(np.float32)
    test_prediction = np.mean(test_members, axis=0, dtype=np.float64).astype(np.float32)
    return validation_prediction, test_prediction, {
        "architecture": type(model).__name__,
        "device": device,
        "parameter_count": parameter_count,
        "input_channels": features.shape[2],
        "seeds": list(seeds),
        "train_case_count": len(train_indices),
        "validation_case_count": len(validation_indices),
        "test_case_count": len(test_indices),
        "active_leads": [5, 6],
        "batch_size": 16,
        "max_epochs": 2 if smoke else 100,
        "early_stopping_patience": 1 if smoke else 15,
        "learning_rate": 3.0e-4,
        "weight_decay": 3.0e-4,
        "dropout": 0.2,
        "loss_coefficients": LOSS_COEFFICIENTS,
        "runs": records,
    }


def write_prediction_store(
    path: Path,
    forecast: base.ForecastData,
    observations: base.ObservationData,
    test_indices: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    weights: np.ndarray,
    *,
    smoke: bool,
) -> None:
    dataset = xr.Dataset(
        {
            "prediction": (
                ("method", "init", "lead_week", "latitude", "longitude"),
                np.stack(
                    [predictions[method] for method in diagnostics.METHOD_ORDER]
                ).astype(np.float32),
            ),
            "truth_imerg": (
                ("init", "lead_week", "latitude", "longitude"),
                observations.weekly_truth[test_indices].astype(np.float32),
            ),
            "imerg_climatology": (
                ("init", "lead_week", "latitude", "longitude"),
                observations.weekly_climatology[test_indices].astype(np.float32),
            ),
            "area_weight_km2": (
                ("latitude", "longitude"), weights.astype(np.float64)
            ),
        },
        coords={
            "method": list(diagnostics.METHOD_ORDER),
            "init": forecast.initializations[test_indices].astype("datetime64[ns]"),
            "lead_week": np.arange(1, 7, dtype=np.int16),
            "latitude": forecast.latitude,
            "longitude": forecast.longitude,
        },
        attrs={
            "title": "Full-archive late-lead FuXi rainfall post-processing",
            "train_years": "2002-2018",
            "validation_years": "2019",
            "test_years": "2020-2021",
            "test_status": "exploratory; this test period was examined previously",
            "weekly_alignment": "W1 init+0..6 through W6 init+35..41",
            "early_lead_contract": "all learned methods equal log-bias at W1-W4",
            "units": "mm day-1",
            "smoke": smoke,
        },
    )
    chunk_cases = min(35, len(test_indices))
    dataset.to_zarr(
        path,
        mode="w",
        consolidated=True,
        encoding={
            "prediction": {"chunks": (1, chunk_cases, 1, 27, 27)},
            "truth_imerg": {"chunks": (chunk_cases, 1, 27, 27)},
            "imerg_climatology": {"chunks": (chunk_cases, 1, 27, 27)},
            "area_weight_km2": {"chunks": (27, 27)},
        },
    )


def plot_training_curves(output: Path, training: Mapping[str, Mapping[str, object]]) -> None:
    seeds = max(len(training["spatial"]["runs"]), len(training["temporal"]["runs"]))
    figure, axes = plt.subplots(2, seeds, figsize=(4.0 * seeds, 6.5), squeeze=False)
    for row, name in enumerate(("spatial", "temporal")):
        for column, record in enumerate(training[name]["runs"]):
            axis = axes[row, column]
            history_path = output.parents[1] / str(record["checkpoint"]).replace(
                "checkpoints/best.pt", "logs/training_history.csv"
            )
            history = pd.read_csv(history_path)
            axis.plot(history.epoch + 1, history.train_loss, color="#0072B2", label="Train")
            axis.plot(
                history.epoch + 1,
                history.validation_loss,
                color="#D55E00",
                label="Validation",
            )
            axis.scatter(
                record["best_epoch"] + 1,
                record["best_validation_loss"],
                marker="*",
                s=85,
                color="black",
                zorder=4,
            )
            axis.set_title(f"{name.title()} · seed {record['seed']}", fontsize=10)
            axis.set_xlabel("Epoch")
            axis.grid(alpha=0.2)
            axis.spines[["top", "right"]].set_visible(False)
        axes[row, 0].set_ylabel("Composite objective")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        ncol=2,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
    )
    figure.suptitle(
        "Capacity-matched late-lead adapters\n2002–2018 train; 2019 checkpoint selection",
        fontweight="semibold",
        y=0.995,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.84))
    diagnostics.save_figure(figure, output)


def plot_direct_added_value(intervals: pd.DataFrame, output: Path, *, smoke: bool) -> None:
    """Show only comparisons needed to separate calibration and learned skill."""

    rows = (
        ("log_bias", "raw_fuxi", "W1-W6", "Log-bias vs raw · W1–W6"),
        ("spatial_unet", "log_bias", "W1-W6", "Spatial vs log-bias · W1–W6"),
        ("spatiotemporal_unet", "log_bias", "W1-W6", "Temporal vs log-bias · W1–W6"),
        ("spatial_unet", "log_bias", "W5-W6", "Spatial vs log-bias · W5–W6"),
        ("spatiotemporal_unet", "log_bias", "W5-W6", "Temporal vs log-bias · W5–W6"),
        ("spatiotemporal_unet", "spatial_unet", "W5-W6", "Temporal vs spatial · W5–W6"),
    )
    y = np.arange(len(rows))[::-1]
    colors = ("#0072B2", "#009E73", "#CC79A7", "#009E73", "#CC79A7", "#D55E00")
    figure, axes = plt.subplots(1, 3, figsize=(12.8, 4.8), sharey=True)
    axis_specs = (
        ("acc", "ΔACC (model − baseline)"),
        ("rmse", "RMSE reduction (%)"),
        ("mae", "MAE reduction (%)"),
    )
    for axis, (metric, label) in zip(axes, axis_specs):
        for index, (method, baseline, scope, _) in enumerate(rows):
            value = interval_row(intervals, method, baseline, scope, metric)
            axis.hlines(y[index], value.ci_lower, value.ci_upper, color=colors[index], linewidth=1.8)
            axis.plot(value.effect, y[index], marker="D", color=colors[index], markersize=6)
        axis.axvline(0.0, color="0.4", linestyle="--", linewidth=1.0)
        axis.set_xlabel(label)
        axis.grid(axis="x", alpha=0.2)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", left=False)
    axes[0].set_yticks(y, [row[3] for row in rows], fontsize=9)
    context = "SMOKE CHECK" if smoke else "n=70; paired year + block-13 bootstrap, 2,000 draws"
    figure.suptitle(
        "What contributes skill beyond raw FuXi?\n" + context,
        fontsize=13,
        fontweight="semibold",
        y=0.98,
    )
    figure.text(
        0.66,
        0.025,
        "Positive values indicate improvement; bars are 95% paired bootstrap intervals.",
        ha="center",
        fontsize=8.5,
        color="0.35",
    )
    figure.subplots_adjust(left=0.30, right=0.985, top=0.80, bottom=0.15, wspace=0.26)
    diagnostics.save_figure(figure, output)


def late_spatial_metrics(
    predictions: Mapping[str, np.ndarray],
    truth: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    weights: np.ndarray,
) -> tuple[xr.Dataset, pd.DataFrame]:
    support = weights > 0.0
    methods = ("raw_fuxi", "log_bias", "spatial_unet", "spatiotemporal_unet")
    rmse = {}
    for method in methods:
        error = predictions[method][:, ACTIVE_LEADS] - truth[:, ACTIVE_LEADS]
        field = np.sqrt(np.mean(error.astype(np.float64) ** 2, axis=(0, 1)))
        field[~support] = np.nan
        rmse[method] = field
    pairs = (
        ("spatial_vs_log_bias", "spatial_unet", "log_bias"),
        ("temporal_vs_log_bias", "spatiotemporal_unet", "log_bias"),
        ("temporal_vs_spatial", "spatiotemporal_unet", "spatial_unet"),
    )
    reductions = np.stack([rmse[baseline] - rmse[method] for _, method, baseline in pairs])
    dataset = xr.Dataset(
        {
            "rmse_reduction": (
                ("comparison", "latitude", "longitude"), reductions.astype(np.float32)
            ),
            "area_weight_km2": (("latitude", "longitude"), weights),
        },
        coords={
            "comparison": [name for name, _, _ in pairs],
            "latitude": latitude,
            "longitude": longitude,
        },
        attrs={
            "lead_scope": "W5-W6",
            "positive_reduction": "model has lower local RMSE than baseline",
            "map_status": "descriptive; no pixel-wise significance inference",
        },
    )
    dataset.rmse_reduction.attrs["units"] = "mm day-1"
    rows = []
    for index, (name, method, baseline) in enumerate(pairs):
        values = reductions[index]
        improved = support & (values > 0.0)
        rows.append(
            {
                "comparison": name,
                "method": method,
                "baseline": baseline,
                "supported_cells": int(support.sum()),
                "cells_improved": int(improved.sum()),
                "area_fraction_improved_pct": float(
                    100.0 * weights[improved].sum() / weights[support].sum()
                ),
                "area_weighted_mean_rmse_reduction_mm_day": float(
                    np.sum(weights[support] * values[support]) / weights[support].sum()
                ),
            }
        )
    return dataset, pd.DataFrame(rows)


def plot_late_maps(dataset: xr.Dataset, summary: pd.DataFrame, output: Path) -> None:
    values = np.asarray(dataset.rmse_reduction.values, dtype=np.float64)
    finite = np.abs(values[np.isfinite(values)])
    limit = max(0.1, float(np.ceil(np.percentile(finite, 98) / 0.1) * 0.1))
    projection = ccrs.PlateCarree()
    figure, axes = plt.subplots(
        1, 3, figsize=(12.0, 4.3), subplot_kw={"projection": projection}
    )
    titles = (
        "Spatial vs log-bias",
        "Temporal vs log-bias",
        "Temporal vs spatial",
    )
    image = None
    for index, (axis, title) in enumerate(zip(axes, titles)):
        field = values[index]
        image = axis.pcolormesh(
            dataset.longitude,
            dataset.latitude,
            np.ma.masked_invalid(field),
            transform=projection,
            cmap="RdBu",
            vmin=-limit,
            vmax=limit,
            shading="nearest",
        )
        axis.coastlines(resolution="50m", linewidth=0.65, color="0.2")
        axis.add_feature(
            cfeature.BORDERS.with_scale("50m"), linewidth=0.45, edgecolor="0.25"
        )
        axis.set_extent(
            [
                float(dataset.longitude.min()) - 0.8,
                float(dataset.longitude.max()) + 0.8,
                float(dataset.latitude.min()) - 0.8,
                float(dataset.latitude.max()) + 0.8,
            ],
            crs=projection,
        )
        grid = axis.gridlines(
            draw_labels=True,
            linewidth=0.3,
            color="0.6",
            alpha=0.5,
            x_inline=False,
            y_inline=False,
        )
        grid.top_labels = False
        grid.right_labels = False
        grid.left_labels = index == 0
        grid.xformatter = LONGITUDE_FORMATTER
        grid.yformatter = LATITUDE_FORMATTER
        grid.xlabel_style = {"size": 7.5}
        grid.ylabel_style = {"size": 7.5}
        row = summary.iloc[index]
        axis.set_title(title, fontweight="semibold", fontsize=10.5)
        axis.text(
            0.98,
            0.03,
            f"{row.area_fraction_improved_pct:.1f}% of area improved",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none", "pad": 2},
        )
    assert image is not None
    color_axis = figure.add_axes([0.26, 0.15, 0.48, 0.035])
    colorbar = figure.colorbar(image, cax=color_axis, orientation="horizontal", extend="both")
    colorbar.set_label("Local RMSE reduction (mm day$^{-1}$; positive is better)")
    figure.suptitle(
        "Late-lead spatial added value (Weeks 5–6)\nIMERG, exploratory 2020–2021 test",
        fontsize=13,
        fontweight="semibold",
        y=0.98,
    )
    figure.text(
        0.5,
        0.025,
        "Pooled over test initializations and W5–W6; descriptive point estimates on 174 supported cells.",
        ha="center",
        fontsize=8.3,
        color="0.35",
    )
    figure.subplots_adjust(left=0.06, right=0.98, top=0.82, bottom=0.27, wspace=0.08)
    diagnostics.save_figure(figure, output)


def interval_row(
    intervals: pd.DataFrame,
    method: str,
    baseline: str,
    scope: str,
    metric: str,
) -> pd.Series:
    selected = intervals.loc[
        intervals.method.eq(method)
        & intervals.baseline.eq(baseline)
        & intervals.lead_scope.eq(scope)
        & intervals.metric.eq(metric)
    ]
    if len(selected) != 1:
        raise base.DataContractError(
            f"missing interval: {method} vs {baseline}, {scope}, {metric}"
        )
    return selected.iloc[0]


def write_results(
    output: Path,
    headline: pd.DataFrame,
    intervals: pd.DataFrame,
    validation_selection: pd.DataFrame,
    training: Mapping[str, Mapping[str, object]],
    *,
    smoke: bool,
) -> None:
    lines = [
        "# Full-archive late-lead FuXi–IMERG result",
        "",
        "> Smoke check only; do not interpret these scores."
        if smoke
        else "> Exploratory result: 2020–2021 had been examined before this experiment.",
        "",
        "## Frozen experiment",
        "",
        "- Train: 2002–2018 (595 JJAS initializations)",
        "- Validation: 2019 (35 initializations)",
        "- Test: 2020–2021 (70 initializations)",
        "- Target: IMERG Final V07B weekly rainfall",
        "- Learned corrections: Weeks 5–6 only; Weeks 1–4 equal log-bias exactly",
        "- Loss: 0.75 Smooth-L1 + 0.20 ACC + 0.05 bias",
        "- Inputs: FuXi TP mean/spread, IMERG climatology, explicit TP anomaly, T2M, coordinates/calendar/lead/support",
        "",
        "## Test headline",
        "",
        "| Method | ACC | RMSE | MAE | Bias | RMSE skill vs raw |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in headline.itertuples(index=False):
        lines.append(
            f"| {row.method_label} | {row.acc:.3f} | {row.rmse_mm_day:.3f} | "
            f"{row.mae_mm_day:.3f} | {row.bias_mm_day:+.3f} | "
            f"{row.rmse_skill_vs_raw:+.2f}% |"
        )
    lines.extend(["", "## Direct learned added value", ""])
    for scope in ("W1-W6", "W5-W6"):
        lines.extend(
            [
                f"### {scope}",
                "",
                "| Comparison | ΔACC | RMSE reduction | MAE reduction |",
                "|---|---:|---:|---:|",
            ]
        )
        for method, baseline, label in (
            ("spatial_unet", "log_bias", "Spatial vs log-bias"),
            ("spatiotemporal_unet", "log_bias", "Temporal vs log-bias"),
            ("spatiotemporal_unet", "spatial_unet", "Temporal vs spatial"),
        ):
            acc = interval_row(intervals, method, baseline, scope, "acc")
            rmse = interval_row(intervals, method, baseline, scope, "rmse")
            mae = interval_row(intervals, method, baseline, scope, "mae")
            lines.append(
                f"| {label} | {acc.effect:+.3f} [{acc.ci_lower:+.3f}, {acc.ci_upper:+.3f}] | "
                f"{rmse.effect:+.2f}% [{rmse.ci_lower:+.2f}, {rmse.ci_upper:+.2f}] | "
                f"{mae.effect:+.2f}% [{mae.ci_lower:+.2f}, {mae.ci_upper:+.2f}] |"
            )
        lines.append("")
    selected = validation_selection.set_index("lead_bin").loc["W5-W6"]
    lines.extend(
        [
            "## Validation decision",
            "",
            f"The fixed W5–W6 rule selected **{diagnostics.METHOD_LABELS[selected.selected_method]}** ",
            "using 2019 only. Temporal is selected only if ACC, RMSE, and MAE all improve.",
            "",
            "## Model sizes",
            "",
            f"- Spatial: {training['spatial']['parameter_count']:,} parameters",
            f"- Temporal: {training['temporal']['parameter_count']:,} parameters",
            "",
            "FuXi itself is unchanged. These are deterministic rainfall post-processors.",
        ]
    )
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def quick_source_inventory(forecast: base.ForecastData, output: Path) -> None:
    rows = []
    for source in forecast.source_files:
        path = Path(source)
        stat = path.stat()
        rows.append(
            {
                "path": str(path),
                "kind": "netcdf_shard",
                "size_bytes": stat.st_size,
                "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                "sha256": "not-computed-fast-run",
            }
        )
    pd.DataFrame(rows).to_csv(output / "source_inventory.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="short end-to-end check")
    args = parser.parse_args()
    configure_contract()
    started = time.monotonic()
    run_name = ("smoke" if args.smoke else "full") + "_" + datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")
    output = RESULTS_ROOT / run_name
    for directory in (
        output,
        output / "metrics",
        output / "figures",
        output / "models",
        output / "code",
    ):
        directory.mkdir(parents=True, exist_ok=False)

    print("Loading 2002–2021 FuXi TP...", flush=True)
    forecast = base.load_fuxi()
    print("Loading aligned IMERG and the 2002–2018 climatology...", flush=True)
    observations = base.load_imerg(forecast)
    weights = base.load_area_weights(forecast, observations.observation_fraction)
    splits = base.split_indices(forecast.initializations)
    expected = {"train": 595, "validation": 35, "test": 70}
    counts = {name: len(values) for name, values in splits.items()}
    if counts != expected:
        raise base.DataContractError(f"unexpected split counts: {counts}")
    support = weights > 0.0

    print("Loading weekly FuXi T2M...", flush=True)
    t2m_weekly = load_t2m_weekly(forecast)
    print("Building eleven-channel training arrays...", flush=True)
    features, normalization = make_features(
        forecast, observations, weights, splits["train"], t2m_weekly
    )
    (output / "normalization.json").write_text(
        json.dumps(normalization, indent=2) + "\n", encoding="utf-8"
    )

    print("Fitting the training-only log-bias anchor...", flush=True)
    correction = fit_log_bias_correction(
        forecast.ensemble_mean[splits["train"]],
        observations.weekly_truth[splits["train"]],
        forecast.initializations[splits["train"]],
        support,
        shrinkage=10.0,
    )
    bias_baseline = apply_log_bias_correction(
        forecast.ensemble_mean, forecast.initializations, correction
    )
    valid = np.broadcast_to(support[None, None], bias_baseline.shape)
    target_scale = fit_anchored_target_scale(
        observations.weekly_truth[splits["train"]],
        bias_baseline[splits["train"]],
        weights,
        split_name="train",
        valid_mask=valid[splits["train"]],
    )
    target = standardize_anchored_target(
        observations.weekly_truth,
        bias_baseline,
        target_scale,
        valid_mask=valid,
    )
    np.savez_compressed(
        output / "models" / "log_bias_anchor.npz",
        lead_month_residual=correction.lead_month_residual,
        shrinkage=np.float32(correction.shrinkage),
        target_scale=target_scale,
    )

    print("Training capacity-matched spatial control...", flush=True)
    spatial_validation, spatial_test, spatial_training = train_candidate(
        "spatial",
        features,
        target,
        target_scale,
        bias_baseline,
        forecast,
        observations,
        weights,
        splits,
        output,
        smoke=args.smoke,
    )
    print("Training temporal model...", flush=True)
    temporal_validation, temporal_test, temporal_training = train_candidate(
        "temporal",
        features,
        target,
        target_scale,
        bias_baseline,
        forecast,
        observations,
        weights,
        splits,
        output,
        smoke=args.smoke,
    )

    test_indices = splits["test"][:8] if args.smoke else splits["test"]
    validation_indices = splits["validation"][:8] if args.smoke else splits["validation"]
    raw_test = forecast.ensemble_mean[test_indices].copy()
    raw_test[..., ~support] = np.nan
    log_bias_test = bias_baseline[test_indices].copy()
    log_bias_test[..., ~support] = np.nan
    hybrid, validation_selection = diagnostics.select_lead_adaptive_hybrid(
        spatial_validation,
        temporal_validation,
        observations.weekly_truth[validation_indices],
        observations.weekly_climatology[validation_indices],
        forecast.initializations[validation_indices],
        weights,
        spatial_test,
        temporal_test,
    )
    predictions = {
        "raw_fuxi": raw_test,
        "log_bias": log_bias_test,
        "spatial_unet": spatial_test,
        "spatiotemporal_unet": temporal_test,
        "lead_adaptive_hybrid": hybrid,
    }
    for method in ("spatial_unet", "spatiotemporal_unet", "lead_adaptive_hybrid"):
        if not np.array_equal(
            predictions[method][:, :4], log_bias_test[:, :4], equal_nan=True
        ):
            raise base.DataContractError(f"{method} differs from log-bias before W5")

    print("Scoring and bootstrapping the test comparison...", flush=True)
    truth = observations.weekly_truth[test_indices]
    climatology = observations.weekly_climatology[test_indices]
    initializations = forecast.initializations[test_indices]
    case_metrics = diagnostics.evaluate_predictions(
        truth, climatology, predictions, initializations, weights
    )
    summary = diagnostics.summarize_by_lead(case_metrics)
    intervals = diagnostics.paired_intervals(
        case_metrics, initializations, smoke=args.smoke
    )
    headline = diagnostics.headline_table(case_metrics, intervals)
    yearly = diagnostics.yearly_table(case_metrics)
    spatial_dataset, spatial_summary = late_spatial_metrics(
        predictions, truth, forecast.latitude, forecast.longitude, weights
    )
    metrics = output / "metrics"
    case_metrics.to_csv(metrics / "case_metrics.csv", index=False)
    summary.to_csv(metrics / "summary_by_lead.csv", index=False)
    intervals.to_csv(metrics / "paired_skill.csv", index=False)
    headline.to_csv(metrics / "headline_metrics.csv", index=False)
    yearly.to_csv(metrics / "yearly_skill.csv", index=False)
    validation_selection.to_csv(metrics / "validation_lead_selection.csv", index=False)
    spatial_summary.to_csv(metrics / "late_spatial_summary.csv", index=False)
    spatial_dataset.to_netcdf(metrics / "late_spatial_rmse.nc")

    prediction_store = output / "predictions.zarr"
    write_prediction_store(
        prediction_store,
        forecast,
        observations,
        test_indices,
        predictions,
        weights,
        smoke=args.smoke,
    )
    with xr.open_zarr(prediction_store, consolidated=True) as stored:
        rebuilt = {
            method: np.asarray(
                stored.prediction.sel({"method": method}).load(), dtype=np.float32
            )
            for method in diagnostics.METHOD_ORDER
        }
    for method in diagnostics.METHOD_ORDER:
        if not np.array_equal(rebuilt[method], predictions[method], equal_nan=True):
            raise base.DataContractError(f"prediction round-trip failed for {method}")
    rebuilt_metrics = diagnostics.evaluate_predictions(
        truth, climatology, rebuilt, initializations, weights
    )
    pd.testing.assert_frame_equal(case_metrics, rebuilt_metrics, check_exact=True)

    print("Drawing meeting figures...", flush=True)
    training = {"spatial": spatial_training, "temporal": temporal_training}
    plot_training_curves(output / "figures" / "00_training_curves", training)
    diagnostics.plot_skill_by_lead(
        summary, output / "figures" / "01_skill_by_lead", smoke=args.smoke
    )
    plot_direct_added_value(
        intervals, output / "figures" / "02_direct_added_value", smoke=args.smoke
    )
    plot_late_maps(
        spatial_dataset, spatial_summary, output / "figures" / "03_late_spatial_added_value"
    )
    write_results(
        output,
        headline,
        intervals,
        validation_selection,
        training,
        smoke=args.smoke,
    )
    quick_source_inventory(forecast, output)

    sources = (
        Path(__file__),
        SOURCE_ROOT / "fuxi_imerg_experiment.py",
        SOURCE_ROOT / "fuxi_imerg_spatiotemporal.py",
        NEURAL_SRC / "fuxi_adapter" / "anchored.py",
        NEURAL_SRC / "fuxi_adapter" / "v3_training.py",
        NEURAL_SRC / "fuxi_adapter" / "models.py",
    )
    code_hashes = {}
    for source in sources:
        destination = output / "code" / source.name
        shutil.copy2(source, destination)
        code_hashes[source.name] = sha256_file(destination)

    elapsed = time.monotonic() - started
    manifest = {
        "status": "complete",
        "smoke": args.smoke,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed,
        "scientific_status": "exploratory; reused 2020-2021 test",
        "split_years": {
            "train": list(TRAIN_YEARS),
            "validation": list(VALIDATION_YEARS),
            "test": list(TEST_YEARS),
        },
        "split_counts": counts,
        "test_count_used": len(test_indices),
        "member_count": 51,
        "lead_weeks": 6,
        "support_cells": int(support.sum()),
        "active_leads": [5, 6],
        "inactive_lead_identity_verified": True,
        "prediction_store_roundtrip_verified": True,
        "features": normalization["input_channels"],
        "base_forecast": "training-only lead/month/cell log-bias FuXi",
        "loss_coefficients": LOSS_COEFFICIENTS,
        "lead_weights": list(LEAD_WEIGHTS),
        "climatology": "IMERG 2002-2018 equal-year centred 31-day calendar mean",
        "training": training,
        "validation_selection": validation_selection.to_dict(orient="records"),
        "bootstrap": {
            "method": "paired two-stage test-year and within-year block resampling",
            "resamples": 50 if args.smoke else diagnostics.BOOTSTRAP_SAMPLES,
            "block_length": 2 if args.smoke else diagnostics.BOOTSTRAP_BLOCK_LENGTH,
            "seed": 42,
        },
        "source_inventory": {
            "file": "source_inventory.csv",
            "fuxi_shard_count": len(forecast.source_files),
            "imerg_store_count": len(observations.source_stores),
            "content_hashes": "deferred to avoid a second 5.5-GiB read before meeting",
        },
        "code_sha256": code_hashes,
        "software": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "xarray": xr.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "artifacts": {},
    }
    for artifact in sorted(path for path in output.rglob("*") if path.is_file()):
        if prediction_store in artifact.parents:
            continue
        manifest["artifacts"][str(artifact.relative_to(output))] = sha256_file(artifact)
    manifest["artifacts"]["predictions.zarr"] = sha256_tree(prediction_store)
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )

    print("\n" + headline.to_string(index=False), flush=True)
    print(f"\nPASS: data contracts, early-lead identity, metrics, and Zarr round-trip", flush=True)
    print(f"Completed in {elapsed / 60.0:.1f} minutes", flush=True)
    print(output, flush=True)


if __name__ == "__main__":
    main()
