#!/usr/bin/env python
"""Build the isolated 2020--2021 FuXi physical-feature exploratory cache.

This is deliberately separate from the 2002--2019 development cache.  It
selects exactly 35 JJAS initializations in each of 2020 and 2021 and delegates
the field reduction to :mod:`fuxi_physical_feature_cache`, so clipping,
weekly/ensemble averaging, moisture-flux products, and the OLR sign convention
cannot drift between development and post-selection evaluation.

The builder reads FuXi predictors only.  It neither imports nor opens IMD (or
any other target) data.  One atomic part is produced per initialization before
a separate, strict finalization step publishes the 70-initialization cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import fuxi_physical_feature_cache as development

from project_paths import PROJECT_ROOT as HERE
DEFAULT_SOURCE_STORE = development.DEFAULT_SOURCE_STORE
DEFAULT_POSTSELECTION_CACHE = (
    HERE / "cache" / "fuxi_physical_weekly_2020_2021_jjas_exploratory_v2.npz"
)
DEFAULT_POSTSELECTION_PARTS_DIR = (
    HERE / "cache" / "fuxi_physical_weekly_2020_2021_jjas_exploratory_v2.parts"
)

CACHE_SCHEMA_NAME = "fuxi-physical-weekly-jjas-exploratory"
CACHE_SCHEMA_VERSION = 2
POSTSELECTION_YEARS = (2020, 2021)
POSTSELECTION_MONTHS = development.SCREENING_MONTHS
EXPECTED_INITIALIZATIONS_PER_YEAR = development.EXPECTED_INITIALIZATIONS_PER_YEAR
EXPECTED_INITIALIZATION_COUNT = (
    len(POSTSELECTION_YEARS) * EXPECTED_INITIALIZATIONS_PER_YEAR
)
SELECTION_DESCRIPTION = (
    "JJAS initializations, 2020-2021 inclusive; isolated exploratory reused-hindcast evaluation"
)
EVALUATION_ROLE = "exploratory_reused_hindcast_evaluation"
TARGET_DATA_ACCESS = "none"
REDUCTION_IMPLEMENTATION = "fuxi_physical_feature_cache._summarize_initialization"

# Aliases intentionally point at the development-cache contract.  The actual
# reduction below calls the development implementation directly as well.
PHYSICAL_FEATURE_NAMES = development.PHYSICAL_FEATURE_NAMES
PHYSICAL_FEATURE_DEFINITIONS = development.PHYSICAL_FEATURE_DEFINITIONS
PHYSICAL_FEATURE_UNITS = development.PHYSICAL_FEATURE_UNITS
PHYSICAL_TRANSFORMS = development.PHYSICAL_TRANSFORMS
EXPECTED_LATITUDE = development.EXPECTED_LATITUDE
EXPECTED_LONGITUDE = development.EXPECTED_LONGITUDE
EXPECTED_GRID_SHAPE = development.EXPECTED_GRID_SHAPE
LEAD_WEEK_COUNT = development.LEAD_WEEK_COUNT


class PostSelectionCacheContractError(ValueError):
    """Raised if post-selection source, scope, parts, or cache are unsafe."""


@dataclass(frozen=True)
class PostSelectionSourceContract:
    """Validated source identity and exact 2020--2021 extraction scope."""

    source_store: str
    source_fingerprint: str
    scope_fingerprint: str
    all_initializations: np.ndarray
    selected_source_indices: np.ndarray
    selected_initializations: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    latitude_slice: slice
    longitude_slice: slice
    channel_names: tuple[str, ...]


@dataclass(frozen=True)
class FuxiPostSelectionPhysicalPredictors:
    """Unnormalized weekly FuXi predictors for the frozen 2020--2021 scope."""

    initializations: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    tcwv_mean: np.ndarray
    tcwv_spread: np.ndarray
    q850_mean: np.ndarray
    u850_mean: np.ndarray
    v850_mean: np.ndarray
    z500_mean: np.ndarray
    msl_mean: np.ndarray
    olr_mean: np.ndarray
    q850_u850_flux_mean: np.ndarray
    q850_v850_flux_mean: np.ndarray
    source_fingerprint: str
    scope_fingerprint: str
    source_store: str
    cache_path: str | None = None
    cache_sha256: str | None = None

    @property
    def feature_fields(self) -> Mapping[str, np.ndarray]:
        """Return fields in exactly the development model-channel order."""

        return {name: getattr(self, name) for name in PHYSICAL_FEATURE_NAMES}


def _canonical_fingerprint(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _select_postselection_initializations(
    all_initializations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Select exactly 35 JJAS dates in 2020 and 35 in 2021."""

    initializations = np.asarray(all_initializations, dtype="datetime64[D]")
    if initializations.ndim != 1:
        raise PostSelectionCacheContractError(
            "source initialization coordinate is not 1-D"
        )
    if np.unique(initializations).size != len(initializations):
        raise PostSelectionCacheContractError("source initializations are not unique")
    if np.any(initializations[1:] <= initializations[:-1]):
        raise PostSelectionCacheContractError(
            "source initializations are not strictly increasing"
        )
    years, months = development._date_parts(initializations)
    selected_mask = np.isin(years, POSTSELECTION_YEARS) & np.isin(
        months, POSTSELECTION_MONTHS
    )
    selected_indices = np.flatnonzero(selected_mask).astype(np.int64)
    selected = initializations[selected_indices]
    selected_years, selected_months = development._date_parts(selected)
    counts = {
        year: int(np.count_nonzero(selected_years == year))
        for year in POSTSELECTION_YEARS
    }
    expected = {year: EXPECTED_INITIALIZATIONS_PER_YEAR for year in POSTSELECTION_YEARS}
    if counts != expected or len(selected) != EXPECTED_INITIALIZATION_COUNT:
        raise PostSelectionCacheContractError(
            "expected exactly 35 JJAS initializations in each of 2020 and 2021; "
            f"found {counts}"
        )
    if not np.all(np.isin(selected_months, POSTSELECTION_MONTHS)) or not np.all(
        np.isin(selected_years, POSTSELECTION_YEARS)
    ):
        raise PostSelectionCacheContractError(
            "post-selection scope contains a date outside 2020--2021 JJAS"
        )
    return selected_indices, selected


def _source_identity_fingerprint(
    source_store: Path,
    group: Any,
    all_initializations: np.ndarray,
    channel_names: Sequence[str],
) -> str:
    """Fingerprint source identity without coupling it to an evaluation scope."""

    forecast = group["forecast"]
    attrs = dict(group.attrs)
    payload = {
        "source_store": str(Path(source_store).resolve()),
        "schema_version": str(attrs.get("schema_version", "")),
        "status": str(attrs.get("status", "")),
        "archive_manifest_sha256": str(attrs.get("archive_manifest_sha256", "")),
        "archive_records_sha256": str(attrs.get("archive_records_sha256", "")),
        "completed_utc": str(attrs.get("completed_utc", "")),
        "forecast_shape": list(map(int, forecast.shape)),
        "forecast_chunks": list(map(int, forecast.chunks)),
        "forecast_dtype": np.dtype(forecast.dtype).str,
        "all_initializations": np.datetime_as_string(
            np.asarray(all_initializations, dtype="datetime64[D]"), unit="D"
        ).tolist(),
        "channel_names": list(channel_names),
        "latitude": np.asarray(group["lat"][:], dtype=np.float64).tolist(),
        "longitude": np.asarray(group["lon"][:], dtype=np.float64).tolist(),
    }
    return _canonical_fingerprint(payload)


def _scope_fingerprint(
    source_fingerprint: str,
    selected_source_indices: np.ndarray,
    selected_initializations: np.ndarray,
) -> str:
    """Fingerprint the immutable post-selection reduction contract."""

    payload = {
        "cache_schema_name": CACHE_SCHEMA_NAME,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "source_fingerprint": source_fingerprint,
        "selection": SELECTION_DESCRIPTION,
        "evaluation_role": EVALUATION_ROLE,
        "target_data_access": TARGET_DATA_ACCESS,
        "selected_source_indices": np.asarray(
            selected_source_indices, dtype=np.int64
        ).tolist(),
        "selected_initializations": np.datetime_as_string(
            np.asarray(selected_initializations, dtype="datetime64[D]"), unit="D"
        ).tolist(),
        "feature_names": list(PHYSICAL_FEATURE_NAMES),
        "feature_definitions": dict(PHYSICAL_FEATURE_DEFINITIONS),
        "feature_units": dict(PHYSICAL_FEATURE_UNITS),
        "physical_transforms": dict(PHYSICAL_TRANSFORMS),
        "reduction_implementation": REDUCTION_IMPLEMENTATION,
        "latitude": np.asarray(EXPECTED_LATITUDE, dtype=np.float64).tolist(),
        "longitude": np.asarray(EXPECTED_LONGITUDE, dtype=np.float64).tolist(),
    }
    return _canonical_fingerprint(payload)


def inspect_source(
    source_store: Path = DEFAULT_SOURCE_STORE,
) -> tuple[Any, PostSelectionSourceContract]:
    """Validate source metadata, then freeze the exact 70-date scope."""

    source_store = Path(source_store).resolve()
    # This validates the full archive shape, chunks, channels, completion
    # status, coordinate grids, and historical development subset without
    # reading forecast fields.
    group, source_contract = development.inspect_source(source_store)
    indices, selected = _select_postselection_initializations(
        source_contract.all_initializations
    )
    source_fingerprint = _source_identity_fingerprint(
        source_store,
        group,
        source_contract.all_initializations,
        source_contract.channel_names,
    )
    scope_fingerprint = _scope_fingerprint(source_fingerprint, indices, selected)
    return group, PostSelectionSourceContract(
        source_store=str(source_store),
        source_fingerprint=source_fingerprint,
        scope_fingerprint=scope_fingerprint,
        all_initializations=source_contract.all_initializations.copy(),
        selected_source_indices=indices,
        selected_initializations=selected,
        latitude=source_contract.latitude.copy(),
        longitude=source_contract.longitude.copy(),
        latitude_slice=source_contract.latitude_slice,
        longitude_slice=source_contract.longitude_slice,
        channel_names=source_contract.channel_names,
    )


def _validate_predictors(
    predictors: FuxiPostSelectionPhysicalPredictors,
    *,
    expected_initializations: np.ndarray | None = None,
    expected_source_fingerprint: str | None = None,
    expected_scope_fingerprint: str | None = None,
) -> None:
    initializations = np.asarray(predictors.initializations, dtype="datetime64[D]")
    if len(initializations) != EXPECTED_INITIALIZATION_COUNT:
        raise PostSelectionCacheContractError(
            f"post-selection cache must contain exactly "
            f"{EXPECTED_INITIALIZATION_COUNT} initializations"
        )
    if np.unique(initializations).size != EXPECTED_INITIALIZATION_COUNT or np.any(
        initializations[1:] <= initializations[:-1]
    ):
        raise PostSelectionCacheContractError(
            "post-selection initializations are duplicated or unsorted"
        )
    years, months = development._date_parts(initializations)
    counts = {
        year: int(np.count_nonzero(years == year)) for year in POSTSELECTION_YEARS
    }
    if counts != {
        year: EXPECTED_INITIALIZATIONS_PER_YEAR for year in POSTSELECTION_YEARS
    } or not np.all(np.isin(months, POSTSELECTION_MONTHS)):
        raise PostSelectionCacheContractError(
            "cache scope is not exactly 35 JJAS initializations in each of "
            "2020 and 2021"
        )
    if expected_initializations is not None and not np.array_equal(
        initializations,
        np.asarray(expected_initializations, dtype="datetime64[D]"),
    ):
        raise PostSelectionCacheContractError(
            "cache initializations differ from the frozen post-selection scope"
        )
    if not np.array_equal(predictors.latitude, EXPECTED_LATITUDE) or not np.array_equal(
        predictors.longitude, EXPECTED_LONGITUDE
    ):
        raise PostSelectionCacheContractError("post-selection grid is not exact")
    expected_shape = (
        EXPECTED_INITIALIZATION_COUNT,
        LEAD_WEEK_COUNT,
        *EXPECTED_GRID_SHAPE,
    )
    for name, values in predictors.feature_fields.items():
        if values.shape != expected_shape or values.dtype != np.float32:
            raise PostSelectionCacheContractError(
                f"{name} has shape/dtype {values.shape}/{values.dtype}; "
                f"expected {expected_shape}/float32"
            )
        if not np.isfinite(values).all():
            raise PostSelectionCacheContractError(f"{name} contains non-finite values")
    if np.any(predictors.tcwv_spread < 0.0) or np.any(predictors.olr_mean < 0.0):
        raise PostSelectionCacheContractError(
            "post-selection cache violates spread or OLR physical bounds"
        )
    if not predictors.source_fingerprint or not predictors.scope_fingerprint:
        raise PostSelectionCacheContractError(
            "post-selection cache is missing source/scope fingerprints"
        )
    if (
        expected_source_fingerprint is not None
        and predictors.source_fingerprint != expected_source_fingerprint
    ):
        raise PostSelectionCacheContractError("source fingerprint is stale")
    if (
        expected_scope_fingerprint is not None
        and predictors.scope_fingerprint != expected_scope_fingerprint
    ):
        raise PostSelectionCacheContractError("scope fingerprint is stale")


def validate_fuxi_postselection_physical_predictors(
    predictors: FuxiPostSelectionPhysicalPredictors,
    forecast: Any,
) -> None:
    """Require exact initialization and grid alignment with a FuXi archive."""

    forecast_initializations = np.asarray(
        forecast.initializations, dtype="datetime64[D]"
    )
    indices, expected = _select_postselection_initializations(forecast_initializations)
    del indices
    _validate_predictors(predictors, expected_initializations=expected)
    if not np.array_equal(
        np.asarray(forecast.latitude, dtype=np.float64), predictors.latitude
    ) or not np.array_equal(
        np.asarray(forecast.longitude, dtype=np.float64), predictors.longitude
    ):
        raise PostSelectionCacheContractError(
            "post-selection cache grid differs from the FuXi rainfall grid"
        )


def _part_path(parts_dir: Path, initialization: np.datetime64) -> Path:
    stamp = np.datetime_as_string(initialization, unit="D").replace("-", "")
    return Path(parts_dir) / f"{stamp}.npz"


def _part_payload(
    contract: PostSelectionSourceContract,
    source_index: int,
    initialization: np.datetime64,
    feature_fields: Mapping[str, np.ndarray],
) -> Mapping[str, Any]:
    return {
        "schema_name": np.asarray(CACHE_SCHEMA_NAME),
        "schema_version": np.asarray(CACHE_SCHEMA_VERSION, dtype=np.int16),
        "source_store": np.asarray(contract.source_store),
        "source_fingerprint": np.asarray(contract.source_fingerprint),
        "scope_fingerprint": np.asarray(contract.scope_fingerprint),
        "source_init_index": np.asarray(source_index, dtype=np.int32),
        "initialization": np.asarray(initialization, dtype="datetime64[D]"),
        "latitude": np.asarray(contract.latitude, dtype=np.float64),
        "longitude": np.asarray(contract.longitude, dtype=np.float64),
        "selection": np.asarray(SELECTION_DESCRIPTION),
        "evaluation_role": np.asarray(EVALUATION_ROLE),
        "target_data_access": np.asarray(TARGET_DATA_ACCESS),
        "normalization": np.asarray("none_raw_native_values"),
        "reduction_implementation": np.asarray(REDUCTION_IMPLEMENTATION),
        "physical_transforms_json": np.asarray(
            json.dumps(PHYSICAL_TRANSFORMS, sort_keys=True)
        ),
        **feature_fields,
    }


def _load_part(
    part_path: Path,
    contract: PostSelectionSourceContract,
    source_index: int,
    initialization: np.datetime64,
) -> Mapping[str, np.ndarray]:
    required = {
        "schema_name",
        "schema_version",
        "source_store",
        "source_fingerprint",
        "scope_fingerprint",
        "source_init_index",
        "initialization",
        "latitude",
        "longitude",
        "selection",
        "evaluation_role",
        "target_data_access",
        "normalization",
        "reduction_implementation",
        "physical_transforms_json",
        *PHYSICAL_FEATURE_NAMES,
    }
    try:
        with np.load(part_path, allow_pickle=False) as part:
            missing = required.difference(part.files)
            if missing:
                raise PostSelectionCacheContractError(
                    f"part {part_path} is missing fields: {sorted(missing)}"
                )
            scalar = lambda name: np.asarray(part[name]).item()
            if (
                str(scalar("schema_name")) != CACHE_SCHEMA_NAME
                or int(scalar("schema_version")) != CACHE_SCHEMA_VERSION
            ):
                raise PostSelectionCacheContractError(
                    f"part {part_path} schema differs"
                )
            if (
                str(scalar("source_store")) != contract.source_store
                or str(scalar("source_fingerprint")) != contract.source_fingerprint
                or str(scalar("scope_fingerprint")) != contract.scope_fingerprint
            ):
                raise PostSelectionCacheContractError(
                    f"part {part_path} source/scope is stale"
                )
            stored_init = np.asarray(
                part["initialization"], dtype="datetime64[D]"
            ).item()
            requested_init = np.asarray(initialization, dtype="datetime64[D]").item()
            if int(scalar("source_init_index")) != int(source_index) or (
                stored_init != requested_init
            ):
                raise PostSelectionCacheContractError(f"part {part_path} init differs")
            exact_scalars = {
                "selection": SELECTION_DESCRIPTION,
                "evaluation_role": EVALUATION_ROLE,
                "target_data_access": TARGET_DATA_ACCESS,
                "normalization": "none_raw_native_values",
                "reduction_implementation": REDUCTION_IMPLEMENTATION,
            }
            for name, expected in exact_scalars.items():
                if str(scalar(name)) != expected:
                    raise PostSelectionCacheContractError(
                        f"part {part_path} has invalid {name}"
                    )
            if json.loads(str(scalar("physical_transforms_json"))) != (
                PHYSICAL_TRANSFORMS
            ):
                raise PostSelectionCacheContractError(
                    f"part {part_path} physical transforms differ"
                )
            if not np.array_equal(part["latitude"], contract.latitude) or not (
                np.array_equal(part["longitude"], contract.longitude)
            ):
                raise PostSelectionCacheContractError(f"part {part_path} grid differs")
            fields = {
                name: np.asarray(part[name], dtype=np.float32).copy()
                for name in PHYSICAL_FEATURE_NAMES
            }
    except (OSError, ValueError) as error:
        if isinstance(error, PostSelectionCacheContractError):
            raise
        raise PostSelectionCacheContractError(
            f"cannot read post-selection part {part_path}: {error}"
        ) from error
    expected_shape = (LEAD_WEEK_COUNT, *EXPECTED_GRID_SHAPE)
    for name, values in fields.items():
        if values.shape != expected_shape or not np.isfinite(values).all():
            raise PostSelectionCacheContractError(
                f"part {part_path} has invalid {name}"
            )
    if np.any(fields["tcwv_spread"] < 0.0) or np.any(fields["olr_mean"] < 0.0):
        raise PostSelectionCacheContractError(
            f"part {part_path} violates physical bounds"
        )
    return fields


def build_part(
    group: Any,
    contract: PostSelectionSourceContract,
    source_index: int,
    initialization: np.datetime64,
    parts_dir: Path = DEFAULT_POSTSELECTION_PARTS_DIR,
) -> tuple[Path, bool]:
    """Build or safely reuse one atomic 2020--2021 predictor part."""

    initialization = np.datetime64(initialization, "D")
    matches = np.flatnonzero(contract.selected_initializations == initialization)
    if len(matches) != 1 or int(contract.selected_source_indices[matches[0]]) != int(
        source_index
    ):
        raise PostSelectionCacheContractError(
            "refusing to build an initialization outside the frozen scope"
        )
    part_path = _part_path(parts_dir, initialization).resolve()
    if part_path.is_file():
        try:
            _load_part(part_path, contract, source_index, initialization)
            return part_path, False
        except PostSelectionCacheContractError as error:
            print(f"rebuilding invalid part {part_path}: {error}", flush=True)
    if not bool(np.asarray(group["init_complete"][int(source_index)]).item()):
        raise PostSelectionCacheContractError(
            f"source initialization {initialization} is not marked complete"
        )
    # This direct call is the central no-drift guarantee: there is no copied
    # reduction code in this module.
    fields = development._summarize_initialization(
        group["forecast"],
        source_index,
        contract.channel_names,
        contract.latitude_slice,
        contract.longitude_slice,
    )
    development._atomic_npz(
        part_path,
        _part_payload(contract, source_index, initialization, fields),
    )
    _load_part(part_path, contract, source_index, initialization)
    return part_path, True


def _cache_payload(
    predictors: FuxiPostSelectionPhysicalPredictors,
) -> Mapping[str, Any]:
    return {
        "schema_name": np.asarray(CACHE_SCHEMA_NAME),
        "schema_version": np.asarray(CACHE_SCHEMA_VERSION, dtype=np.int16),
        "source_store": np.asarray(predictors.source_store),
        "source_fingerprint": np.asarray(predictors.source_fingerprint),
        "scope_fingerprint": np.asarray(predictors.scope_fingerprint),
        "initializations": np.asarray(
            predictors.initializations, dtype="datetime64[D]"
        ),
        "latitude": np.asarray(predictors.latitude, dtype=np.float64),
        "longitude": np.asarray(predictors.longitude, dtype=np.float64),
        "feature_names": np.asarray(PHYSICAL_FEATURE_NAMES),
        "feature_definitions_json": np.asarray(
            json.dumps(PHYSICAL_FEATURE_DEFINITIONS, sort_keys=True)
        ),
        "feature_units_json": np.asarray(
            json.dumps(PHYSICAL_FEATURE_UNITS, sort_keys=True)
        ),
        "physical_transforms_json": np.asarray(
            json.dumps(PHYSICAL_TRANSFORMS, sort_keys=True)
        ),
        "selection": np.asarray(SELECTION_DESCRIPTION),
        "evaluation_role": np.asarray(EVALUATION_ROLE),
        "target_data_access": np.asarray(TARGET_DATA_ACCESS),
        "normalization": np.asarray("none_raw_native_values"),
        "reduction_implementation": np.asarray(REDUCTION_IMPLEMENTATION),
        "weekly_reduction": np.asarray(
            "seven-day mean per member, then 51-member ensemble statistics"
        ),
        "moisture_flux_reduction": np.asarray(
            "daily per-member q850*wind products, then seven-day and " "51-member means"
        ),
        "olr_conversion": np.asarray("olr=-ttr"),
        **predictors.feature_fields,
    }


def _write_final_cache(
    path: Path, predictors: FuxiPostSelectionPhysicalPredictors
) -> str:
    _validate_predictors(predictors)
    development._atomic_npz(path, _cache_payload(predictors))
    return development.sha256_file(path)


def _load_final_cache(
    cache_path: Path,
    *,
    expected_initializations: np.ndarray | None = None,
    expected_source_fingerprint: str | None = None,
    expected_scope_fingerprint: str | None = None,
) -> FuxiPostSelectionPhysicalPredictors:
    cache_path = Path(cache_path).resolve()
    required = {
        "schema_name",
        "schema_version",
        "source_store",
        "source_fingerprint",
        "scope_fingerprint",
        "initializations",
        "latitude",
        "longitude",
        "feature_names",
        "feature_definitions_json",
        "feature_units_json",
        "physical_transforms_json",
        "selection",
        "evaluation_role",
        "target_data_access",
        "normalization",
        "reduction_implementation",
        "weekly_reduction",
        "moisture_flux_reduction",
        "olr_conversion",
        *PHYSICAL_FEATURE_NAMES,
    }
    try:
        with np.load(cache_path, allow_pickle=False) as cached:
            missing = required.difference(cached.files)
            if missing:
                raise PostSelectionCacheContractError(
                    f"post-selection cache is missing fields: {sorted(missing)}"
                )
            scalar = lambda name: np.asarray(cached[name]).item()
            if (
                str(scalar("schema_name")) != CACHE_SCHEMA_NAME
                or int(scalar("schema_version")) != CACHE_SCHEMA_VERSION
            ):
                raise PostSelectionCacheContractError(
                    "post-selection cache schema differs"
                )
            if tuple(np.asarray(cached["feature_names"]).astype(str)) != (
                PHYSICAL_FEATURE_NAMES
            ):
                raise PostSelectionCacheContractError("feature order has changed")
            exact_scalars = {
                "selection": SELECTION_DESCRIPTION,
                "evaluation_role": EVALUATION_ROLE,
                "target_data_access": TARGET_DATA_ACCESS,
                "normalization": "none_raw_native_values",
                "reduction_implementation": REDUCTION_IMPLEMENTATION,
                "olr_conversion": "olr=-ttr",
            }
            for name, expected in exact_scalars.items():
                if str(scalar(name)) != expected:
                    raise PostSelectionCacheContractError(
                        f"post-selection cache has invalid {name}"
                    )
            exact_json = {
                "feature_definitions_json": PHYSICAL_FEATURE_DEFINITIONS,
                "feature_units_json": PHYSICAL_FEATURE_UNITS,
                "physical_transforms_json": PHYSICAL_TRANSFORMS,
            }
            for name, expected in exact_json.items():
                if json.loads(str(scalar(name))) != expected:
                    raise PostSelectionCacheContractError(
                        f"post-selection cache has changed {name}"
                    )
            values = {
                name: np.asarray(cached[name], dtype=np.float32).copy()
                for name in PHYSICAL_FEATURE_NAMES
            }
            predictors = FuxiPostSelectionPhysicalPredictors(
                initializations=np.asarray(
                    cached["initializations"], dtype="datetime64[D]"
                ).copy(),
                latitude=np.asarray(cached["latitude"], dtype=np.float64).copy(),
                longitude=np.asarray(cached["longitude"], dtype=np.float64).copy(),
                source_fingerprint=str(scalar("source_fingerprint")),
                scope_fingerprint=str(scalar("scope_fingerprint")),
                source_store=str(scalar("source_store")),
                cache_path=str(cache_path),
                cache_sha256=development.sha256_file(cache_path),
                **values,
            )
    except (OSError, ValueError) as error:
        if isinstance(error, PostSelectionCacheContractError):
            raise
        raise PostSelectionCacheContractError(
            f"cannot read post-selection cache {cache_path}: {error}"
        ) from error
    _validate_predictors(
        predictors,
        expected_initializations=expected_initializations,
        expected_source_fingerprint=expected_source_fingerprint,
        expected_scope_fingerprint=expected_scope_fingerprint,
    )
    return predictors


def load_fuxi_postselection_physical_predictors(
    forecast: Any,
    cache_path: Path = DEFAULT_POSTSELECTION_CACHE,
    *,
    expected_source_fingerprint: str | None = None,
    expected_scope_fingerprint: str | None = None,
) -> FuxiPostSelectionPhysicalPredictors:
    """Load and align the isolated 2020--2021 FuXi-only predictor cache."""

    predictors = _load_final_cache(
        cache_path,
        expected_source_fingerprint=expected_source_fingerprint,
        expected_scope_fingerprint=expected_scope_fingerprint,
    )
    validate_fuxi_postselection_physical_predictors(predictors, forecast)
    return predictors


def finalize_cache(
    contract: PostSelectionSourceContract,
    parts_dir: Path = DEFAULT_POSTSELECTION_PARTS_DIR,
    output: Path = DEFAULT_POSTSELECTION_CACHE,
) -> tuple[Path, str]:
    """Validate all 70 atomic parts before publishing the final cache."""

    fields: dict[str, list[np.ndarray]] = {name: [] for name in PHYSICAL_FEATURE_NAMES}
    missing: list[str] = []
    for source_index, initialization in zip(
        contract.selected_source_indices,
        contract.selected_initializations,
        strict=True,
    ):
        part_path = _part_path(parts_dir, initialization)
        if not part_path.is_file():
            missing.append(np.datetime_as_string(initialization, unit="D"))
            continue
        part_fields = _load_part(part_path, contract, int(source_index), initialization)
        for name in PHYSICAL_FEATURE_NAMES:
            fields[name].append(part_fields[name])
    if missing:
        preview = ", ".join(missing[:12])
        suffix = "..." if len(missing) > 12 else ""
        raise PostSelectionCacheContractError(
            f"cannot finalize: {len(missing)} of "
            f"{EXPECTED_INITIALIZATION_COUNT} parts are missing "
            f"({preview}{suffix})"
        )
    predictors = FuxiPostSelectionPhysicalPredictors(
        initializations=contract.selected_initializations.copy(),
        latitude=contract.latitude.copy(),
        longitude=contract.longitude.copy(),
        source_fingerprint=contract.source_fingerprint,
        scope_fingerprint=contract.scope_fingerprint,
        source_store=contract.source_store,
        **{
            name: np.asarray(np.stack(values), dtype=np.float32)
            for name, values in fields.items()
        },
    )
    _validate_predictors(
        predictors,
        expected_initializations=contract.selected_initializations,
        expected_source_fingerprint=contract.source_fingerprint,
        expected_scope_fingerprint=contract.scope_fingerprint,
    )
    output = Path(output).resolve()
    checksum = _write_final_cache(output, predictors)
    _load_final_cache(
        output,
        expected_initializations=contract.selected_initializations,
        expected_source_fingerprint=contract.source_fingerprint,
        expected_scope_fingerprint=contract.scope_fingerprint,
    )
    return output, checksum


def _selected_task_records(
    contract: PostSelectionSourceContract,
    *,
    task_index: int | None,
    task_count: int | None,
    initialization: str | None,
) -> list[tuple[int, np.datetime64]]:
    records = [
        (int(index), np.datetime64(initialization_, "D"))
        for index, initialization_ in zip(
            contract.selected_source_indices,
            contract.selected_initializations,
            strict=True,
        )
    ]
    if initialization is not None:
        if task_index is not None or task_count is not None:
            raise PostSelectionCacheContractError(
                "--init cannot be combined with --task-index/--task-count"
            )
        requested = np.datetime64(initialization, "D")
        records = [record for record in records if record[1] == requested]
        if len(records) != 1:
            raise PostSelectionCacheContractError(
                f"{initialization} is not a 2020--2021 JJAS FuXi initialization"
            )
        return records
    if task_index is None or task_count is None:
        raise PostSelectionCacheContractError(
            "build requires either --init or both --task-index and --task-count"
        )
    if task_count != EXPECTED_INITIALIZATION_COUNT:
        raise PostSelectionCacheContractError(
            f"post-selection array must use exactly "
            f"{EXPECTED_INITIALIZATION_COUNT} tasks"
        )
    if task_index < 0 or task_index >= task_count:
        raise PostSelectionCacheContractError(
            f"task index {task_index} is outside [0, {task_count})"
        )
    selected = records[task_index::task_count]
    if len(selected) != 1:
        raise PostSelectionCacheContractError(
            "each post-selection array task must resolve to exactly one initialization"
        )
    return selected


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-store", type=Path, default=DEFAULT_SOURCE_STORE)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("inventory", help="validate source metadata and scope")

    build = subparsers.add_parser("build", help="build one atomic predictor part")
    build.add_argument(
        "--parts-dir", type=Path, default=DEFAULT_POSTSELECTION_PARTS_DIR
    )
    build.add_argument("--task-index", type=int)
    build.add_argument("--task-count", type=int)
    build.add_argument("--init")

    finalize = subparsers.add_parser(
        "finalize", help="validate 70 parts and publish the final cache"
    )
    finalize.add_argument(
        "--parts-dir", type=Path, default=DEFAULT_POSTSELECTION_PARTS_DIR
    )
    finalize.add_argument("--output", type=Path, default=DEFAULT_POSTSELECTION_CACHE)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    group, contract = inspect_source(args.source_store)
    if args.command == "inventory":
        print(
            json.dumps(
                {
                    "source_store": contract.source_store,
                    "source_fingerprint": contract.source_fingerprint,
                    "scope_fingerprint": contract.scope_fingerprint,
                    "selection": SELECTION_DESCRIPTION,
                    "evaluation_role": EVALUATION_ROLE,
                    "target_data_access": TARGET_DATA_ACCESS,
                    "selected_initialization_count": len(
                        contract.selected_initializations
                    ),
                    "first_initialization": np.datetime_as_string(
                        contract.selected_initializations[0], unit="D"
                    ),
                    "last_initialization": np.datetime_as_string(
                        contract.selected_initializations[-1], unit="D"
                    ),
                    "array_tasks": EXPECTED_INITIALIZATION_COUNT,
                    "reduction_implementation": REDUCTION_IMPLEMENTATION,
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        return
    if args.command == "build":
        records = _selected_task_records(
            contract,
            task_index=args.task_index,
            task_count=args.task_count,
            initialization=args.init,
        )
        for source_index, initialization in records:
            part_path, created = build_part(
                group,
                contract,
                source_index,
                initialization,
                args.parts_dir,
            )
            print(
                f"{'built' if created else 'reused'} "
                f"{np.datetime_as_string(initialization, unit='D')}: {part_path}",
                flush=True,
            )
        return
    output, checksum = finalize_cache(contract, args.parts_dir, args.output)
    print(f"published {output}", flush=True)
    print(f"sha256={checksum}", flush=True)


if __name__ == "__main__":
    main()
