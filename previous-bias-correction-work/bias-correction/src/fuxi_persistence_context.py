"""Leakage-safe persistence context for the all-season FuXi adapter.

This module is deliberately separate from the frozen ensemble-calibration and
PBC implementations.  It adapts the exact issue-time rainfall windows produced
by :func:`fuxi_pbc_core.build_daily_issue_time_lags` into optional neural
context channels without changing either frozen source.

Three experiment arms share one explicit context schema:

``base_42k``
    The exact seven-channel context used by the frozen adapter.
``zero_lag_45k``
    The base context followed by four identically zero control channels.
``persistence_lag12_45k``
    The base context followed by normalized one- and two-week rainfall lags and
    their independent availability masks.

Lag normalization is fitted separately for each lag using only the supplied
training indices and the frozen spatial weights.  Missing and unsupported lag
values are represented by normalized zero plus an availability value of zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence, TypeAlias, cast

import numpy as np
import torch
from torch.utils.data import Dataset

import fuxi_allseason_ensemble_calibration as base
from fuxi_pbc_core import IssueTimeLags


BASE_ARM = "base_42k"
ZERO_LAG_ARM = "zero_lag_45k"
PERSISTENCE_LAG_ARM = "persistence_lag12_45k"
ARMS = (BASE_ARM, ZERO_LAG_ARM, PERSISTENCE_LAG_ARM)

ContextArm: TypeAlias = Literal[
    "base_42k",
    "zero_lag_45k",
    "persistence_lag12_45k",
]

BASE_CONTEXT_CHANNEL_NAMES = (
    "training_normalized_climatology",
    "verification_day_of_year_sin",
    "verification_day_of_year_cos",
    "latitude_scaled",
    "longitude_scaled",
    "lead_scaled",
    "scoring_support",
)
LAG_CONTEXT_CHANNEL_NAMES = (
    "lag_1week_log1p_z",
    "lag_2week_log1p_z",
    "lag_1week_available",
    "lag_2week_available",
)
PERSISTENCE_CONTEXT_CHANNEL_NAMES = (
    *BASE_CONTEXT_CHANNEL_NAMES,
    *LAG_CONTEXT_CHANNEL_NAMES,
)

BASE_CONTEXT_CHANNEL_COUNT = len(BASE_CONTEXT_CHANNEL_NAMES)
PERSISTENCE_CONTEXT_CHANNEL_COUNT = len(PERSISTENCE_CONTEXT_CHANNEL_NAMES)
LEAD_COUNT = 6
LAG_WEEKS = (1, 2)


class PersistenceContextError(ValueError):
    """Raised when persistence context violates an array or timing contract."""


@dataclass(frozen=True)
class PersistenceContextBundle:
    """Preprocessed context shared by the three persistence experiment arms.

    ``normalized_lag_log1p`` has shape ``[case,2,latitude,longitude]`` and is
    already zero-filled wherever a lag is temporally unavailable or outside the
    fixed scoring support.  ``lag_available`` intentionally retains independent
    one- and two-week flags instead of using :attr:`IssueTimeLags.available`,
    which collapses the two flags into one case-level value.
    """

    base_context: base.ContextBundle
    normalized_lag_log1p: np.ndarray
    lag_available: np.ndarray
    lag_log1p_mean_by_lag: np.ndarray
    lag_log1p_std_by_lag: np.ndarray
    normalization_fit_indices: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.base_context, base.ContextBundle):
            raise PersistenceContextError("base_context must be a ContextBundle")
        normalized = np.asarray(self.normalized_lag_log1p)
        available = np.asarray(self.lag_available)
        if normalized.ndim != 4 or normalized.shape[1] != len(LAG_WEEKS):
            raise PersistenceContextError(
                "normalized lag fields must have shape [case,2,latitude,longitude]"
            )
        case_count = int(normalized.shape[0])
        support = np.asarray(self.base_context.support, dtype=bool)
        if support.ndim != 2 or not np.any(support):
            raise PersistenceContextError("base context has an invalid support mask")
        spatial_shape = (int(support.shape[0]), int(support.shape[1]))
        if normalized.shape != (case_count, len(LAG_WEEKS), *spatial_shape):
            raise PersistenceContextError("normalized lag fields do not match base support")
        if available.shape != (case_count, len(LAG_WEEKS)):
            raise PersistenceContextError("per-lag availability is not aligned with cases")
        if available.dtype != np.bool_:
            raise PersistenceContextError("per-lag availability must use boolean dtype")
        means = np.asarray(self.lag_log1p_mean_by_lag)
        stds = np.asarray(self.lag_log1p_std_by_lag)
        if (
            means.shape != (len(LAG_WEEKS),)
            or stds.shape != (len(LAG_WEEKS),)
            or not np.isfinite(means).all()
            or not np.isfinite(stds).all()
            or np.any(stds <= 0.0)
        ):
            raise PersistenceContextError("bundle contains invalid lag normalization statistics")
        if not np.isfinite(normalized).all():
            raise PersistenceContextError("normalized lag fields must be finite")
        usable = available[..., None, None] & support[None, None]
        if np.any(normalized[~usable] != 0.0):
            raise PersistenceContextError(
                "missing and unsupported normalized lag fields must be exactly zero"
            )
        _validate_fit_indices(
            np.asarray(self.normalization_fit_indices),
            case_count,
            label="normalization fit indices",
        )
        _validate_base_context(self.base_context, case_count, spatial_shape)

    @property
    def case_count(self) -> int:
        return int(self.normalized_lag_log1p.shape[0])

    @property
    def spatial_shape(self) -> tuple[int, int]:
        shape = self.base_context.support.shape
        return int(shape[0]), int(shape[1])


def _readonly_copy(values: np.ndarray, *, dtype: Any) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True)
    result.flags.writeable = False
    return result


def _validate_arm(arm: str) -> ContextArm:
    if arm not in ARMS:
        raise PersistenceContextError(
            f"unknown persistence-context arm {arm!r}; expected one of {ARMS}"
        )
    return cast(ContextArm, arm)


def _validate_fit_indices(
    indices: np.ndarray,
    case_count: int,
    *,
    label: str = "normalization fit indices",
) -> np.ndarray:
    raw = np.asarray(indices)
    if raw.ndim != 1 or raw.size == 0 or not np.issubdtype(raw.dtype, np.integer):
        raise PersistenceContextError(
            f"{label} must be a non-empty one-dimensional integer array"
        )
    selected = raw.astype(np.int64, copy=False)
    if np.any(selected < 0) or np.any(selected >= case_count):
        raise PersistenceContextError(f"{label} contain an out-of-range value")
    if np.unique(selected).size != selected.size:
        raise PersistenceContextError(f"{label} must be unique")
    if selected.size > 1 and np.any(np.diff(selected) <= 0):
        raise PersistenceContextError(
            f"{label} must be strictly increasing"
        )
    return selected.copy()


def _validate_base_context(
    context: base.ContextBundle,
    case_count: int,
    spatial_shape: tuple[int, int],
) -> None:
    if not isinstance(context, base.ContextBundle):
        raise PersistenceContextError("base_context must be a ContextBundle")
    height, width = spatial_shape
    expected_climatology = (case_count, LEAD_COUNT, height, width)
    if context.normalized_climatology.shape != expected_climatology:
        raise PersistenceContextError(
            "base normalized climatology is not aligned with issue-time lags"
        )
    expected_season = (case_count, LEAD_COUNT)
    if (
        context.season_sin.shape != expected_season
        or context.season_cos.shape != expected_season
    ):
        raise PersistenceContextError("base seasonal context is misaligned")
    if context.latitude_scaled.shape != (height,):
        raise PersistenceContextError("base latitude context is misaligned")
    if context.longitude_scaled.shape != (width,):
        raise PersistenceContextError("base longitude context is misaligned")
    if context.lead_scaled.shape != (LEAD_COUNT,):
        raise PersistenceContextError("base lead context is misaligned")
    if context.support.shape != spatial_shape:
        raise PersistenceContextError("base support is misaligned")


def validate_issue_time_lags(
    initializations: np.ndarray,
    lags: IssueTimeLags,
    support: np.ndarray,
) -> np.ndarray:
    """Validate exact completed lag windows and return per-lag availability.

    Lag one must cover ``issue-7`` through ``issue-1`` and lag two must cover
    ``issue-14`` through ``issue-8``.  Available windows must contain finite,
    nonnegative rainfall on the fixed support.  Missing windows must retain the
    canonical ``source_index=-1``, ``NaT`` bounds, and ``NaN`` values so an
    accidental fallback cannot masquerade as an observation.
    """

    raw_starts = np.asarray(initializations)
    if not np.issubdtype(raw_starts.dtype, np.datetime64):
        raise PersistenceContextError("initializations must use a datetime64 dtype")
    starts = raw_starts.astype("datetime64[D]")
    if (
        starts.ndim != 1
        or starts.size == 0
        or np.isnat(starts).any()
        or np.unique(starts).size != starts.size
    ):
        raise PersistenceContextError(
            "initializations must be a non-empty unique datetime64 array without NaT"
        )
    if starts.size > 1 and np.any(np.diff(starts) <= np.timedelta64(0, "D")):
        raise PersistenceContextError("initializations must be strictly increasing")

    mask = np.asarray(support, dtype=bool)
    if mask.ndim != 2 or not np.any(mask):
        raise PersistenceContextError("support must be a non-empty two-dimensional mask")
    if not isinstance(lags, IssueTimeLags):
        raise PersistenceContextError("lags must be an IssueTimeLags instance")

    values = np.asarray(lags.values)
    raw_sources = np.asarray(lags.source_indices)
    raw_window_start = np.asarray(lags.window_start)
    raw_window_end = np.asarray(lags.window_end)
    expected_values_shape = (starts.size, len(LAG_WEEKS), *mask.shape)
    expected_metadata_shape = (starts.size, len(LAG_WEEKS))
    if values.shape != expected_values_shape or not np.issubdtype(
        values.dtype, np.floating
    ):
        raise PersistenceContextError(
            f"lag values must have floating shape {expected_values_shape}, got {values.shape}"
        )
    if raw_sources.shape != expected_metadata_shape or not np.issubdtype(
        raw_sources.dtype, np.integer
    ):
        raise PersistenceContextError(
            "lag source indices must be a two-column integer array aligned with cases"
        )
    if (
        raw_window_start.shape != expected_metadata_shape
        or raw_window_end.shape != expected_metadata_shape
        or not np.issubdtype(raw_window_start.dtype, np.datetime64)
        or not np.issubdtype(raw_window_end.dtype, np.datetime64)
    ):
        raise PersistenceContextError(
            "lag window bounds must be two-column datetime64 arrays aligned with cases"
        )

    sources = raw_sources.astype(np.int64, copy=False)
    if np.any(sources < -1):
        raise PersistenceContextError("missing lag source indices must be exactly -1")
    available = sources >= 0
    window_start = raw_window_start.astype("datetime64[D]")
    window_end = raw_window_end.astype("datetime64[D]")
    lag_offsets = (7 * np.asarray(LAG_WEEKS)).astype("timedelta64[D]")
    expected_start = starts[:, None] - lag_offsets[None]
    expected_end = expected_start + np.timedelta64(6, "D")

    if np.any(window_start[available] != expected_start[available]):
        raise PersistenceContextError("available lag window has an incorrect start date")
    if np.any(window_end[available] != expected_end[available]):
        raise PersistenceContextError("available lag window has an incorrect end date")
    issue_grid = np.broadcast_to(starts[:, None], expected_metadata_shape)
    if np.any(window_end[available] >= issue_grid[available]):
        raise PersistenceContextError("available lag window reaches forecast issuance")

    missing = ~available
    if np.any(~np.isnat(window_start[missing])) or np.any(
        ~np.isnat(window_end[missing])
    ):
        raise PersistenceContextError("missing lag window must retain NaT bounds")

    supported_available = values[available][..., mask]
    if supported_available.size and (
        not np.isfinite(supported_available).all()
        or np.any(supported_available < 0.0)
    ):
        raise PersistenceContextError(
            "available lag rainfall must be finite and nonnegative on support"
        )
    supported_missing = values[missing][..., mask]
    if supported_missing.size and not np.isnan(supported_missing).all():
        raise PersistenceContextError(
            "missing lag rainfall must remain NaN on support before preprocessing"
        )
    return available.copy()


def build_persistence_context_bundle(
    base_context: base.ContextBundle,
    initializations: np.ndarray,
    lags: IssueTimeLags,
    train_indices: np.ndarray,
    weights: np.ndarray,
) -> PersistenceContextBundle:
    """Fit train-only lag normalization and construct a reusable context bundle.

    Each lag is transformed with ``log1p`` and standardized using a separate
    area-and-case-weighted mean and population standard deviation.  Only finite,
    supported values from ``train_indices`` enter those statistics.  No
    validation or later value is used to fit preprocessing.
    """

    raw_starts = np.asarray(initializations)
    if not np.issubdtype(raw_starts.dtype, np.datetime64):
        raise PersistenceContextError("initializations must use a datetime64 dtype")
    starts = raw_starts.astype("datetime64[D]")
    if starts.ndim != 1:
        raise PersistenceContextError("initializations must be one-dimensional")

    if not isinstance(base_context, base.ContextBundle):
        raise PersistenceContextError("base_context must be a ContextBundle")
    mask = np.asarray(base_context.support, dtype=bool)
    if mask.ndim != 2 or not np.any(mask):
        raise PersistenceContextError("base context has an invalid support mask")
    _validate_base_context(base_context, starts.size, mask.shape)

    spatial_weights = np.asarray(weights, dtype=np.float64)
    if spatial_weights.shape != mask.shape:
        raise PersistenceContextError("normalization weights do not match support")
    if not np.isfinite(spatial_weights).all() or np.any(spatial_weights < 0.0):
        raise PersistenceContextError("normalization weights must be finite and nonnegative")
    if not np.array_equal(spatial_weights > 0.0, mask):
        raise PersistenceContextError(
            "positive normalization weights must match the base support exactly"
        )

    available = validate_issue_time_lags(starts, lags, mask)
    selected = _validate_fit_indices(train_indices, starts.size)
    for column, lag_week in enumerate(LAG_WEEKS):
        if not np.any(available[selected, column]):
            raise PersistenceContextError(
                f"training split has no available lag-{lag_week} observations"
            )

    lag_values = np.asarray(lags.values, dtype=np.float32)
    usable = available[..., None, None] & mask[None, None]
    log_lags = np.full(lag_values.shape, np.nan, dtype=np.float32)
    np.log1p(lag_values, out=log_lags, where=usable)
    means, stds = base.weighted_lead_moments(
        log_lags,
        selected,
        spatial_weights,
    )
    means = np.asarray(means, dtype=np.float32)
    stds = np.asarray(stds, dtype=np.float32)
    if (
        means.shape != (len(LAG_WEEKS),)
        or stds.shape != (len(LAG_WEEKS),)
        or not np.isfinite(means).all()
        or not np.isfinite(stds).all()
        or np.any(stds <= 0.0)
    ):
        raise PersistenceContextError("lag normalization produced invalid statistics")

    normalized = (log_lags - means[None, :, None, None]) / stds[
        None, :, None, None
    ]
    normalized = np.where(usable & np.isfinite(normalized), normalized, 0.0).astype(
        np.float32
    )
    if normalized.shape != lag_values.shape or not np.isfinite(normalized).all():
        raise PersistenceContextError("normalized lag fields are invalid")

    return PersistenceContextBundle(
        base_context=base_context,
        normalized_lag_log1p=_readonly_copy(normalized, dtype=np.float32),
        lag_available=_readonly_copy(available, dtype=np.bool_),
        lag_log1p_mean_by_lag=_readonly_copy(means, dtype=np.float32),
        lag_log1p_std_by_lag=_readonly_copy(stds, dtype=np.float32),
        normalization_fit_indices=_readonly_copy(selected, dtype=np.int64),
    )


def context_channels_for_arm(arm: str) -> int:
    """Return the exact neural context width for an experiment arm."""

    selected = _validate_arm(arm)
    return (
        BASE_CONTEXT_CHANNEL_COUNT
        if selected == BASE_ARM
        else PERSISTENCE_CONTEXT_CHANNEL_COUNT
    )


def context_channel_names_for_arm(arm: str) -> tuple[str, ...]:
    """Return the fixed, ordered semantic channel names for an arm."""

    selected = _validate_arm(arm)
    return (
        BASE_CONTEXT_CHANNEL_NAMES
        if selected == BASE_ARM
        else PERSISTENCE_CONTEXT_CHANNEL_NAMES
    )


def context_for_case(
    bundle: PersistenceContextBundle,
    case_index: int,
    arm: str,
) -> np.ndarray:
    """Materialize one arm's context as ``[lead,channel,latitude,longitude]``."""

    selected_arm = _validate_arm(arm)
    if not isinstance(bundle, PersistenceContextBundle):
        raise PersistenceContextError("bundle must be a PersistenceContextBundle")
    if isinstance(case_index, (bool, np.bool_)) or not isinstance(
        case_index, (int, np.integer)
    ):
        raise PersistenceContextError("case_index must be an integer")
    index = int(case_index)
    if index < 0 or index >= bundle.case_count:
        raise PersistenceContextError("case_index is out of range")

    base_fields = base.context_for_case(bundle.base_context, index)
    expected_base_shape = (
        LEAD_COUNT,
        BASE_CONTEXT_CHANNEL_COUNT,
        *bundle.spatial_shape,
    )
    if base_fields.shape != expected_base_shape or not np.isfinite(base_fields).all():
        raise PersistenceContextError("base context materialization is invalid")
    if selected_arm == BASE_ARM:
        return base_fields

    if selected_arm == ZERO_LAG_ARM:
        lag_fields = np.zeros(
            (LEAD_COUNT, len(LAG_CONTEXT_CHANNEL_NAMES), *bundle.spatial_shape),
            dtype=np.float32,
        )
    else:
        normalized = bundle.normalized_lag_log1p[index]
        spatial_available = (
            bundle.lag_available[index, :, None, None]
            & np.asarray(bundle.base_context.support, dtype=bool)[None]
        ).astype(np.float32)
        extras = np.concatenate((normalized, spatial_available), axis=0).astype(
            np.float32,
            copy=False,
        )
        lag_fields = np.broadcast_to(
            extras[None],
            (LEAD_COUNT, len(LAG_CONTEXT_CHANNEL_NAMES), *bundle.spatial_shape),
        )

    result = np.concatenate((base_fields, lag_fields), axis=1).astype(
        np.float32,
        copy=False,
    )
    expected_shape = (
        LEAD_COUNT,
        PERSISTENCE_CONTEXT_CHANNEL_COUNT,
        *bundle.spatial_shape,
    )
    if result.shape != expected_shape or not np.isfinite(result).all():
        raise PersistenceContextError("expanded persistence context is invalid")
    return np.ascontiguousarray(result)


class PersistenceCaseDataset(
    Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
):
    """Lazy case dataset that materializes only the selected arm's context."""

    def __init__(
        self,
        members: np.ndarray,
        truth: np.ndarray,
        context: PersistenceContextBundle,
        indices: Sequence[int] | np.ndarray,
        arm: str,
    ) -> None:
        selected_arm = _validate_arm(arm)
        if not isinstance(context, PersistenceContextBundle):
            raise PersistenceContextError("context must be a PersistenceContextBundle")
        member_values = np.asarray(members)
        truth_values = np.asarray(truth)
        expected_field_shape = (LEAD_COUNT, *context.spatial_shape)
        if member_values.ndim != 5 or member_values.shape[0] != context.case_count:
            raise PersistenceContextError(
                "members must have shape [case,member,lead,latitude,longitude]"
            )
        if tuple(member_values.shape[2:]) != expected_field_shape:
            raise PersistenceContextError("member fields do not match persistence context")
        if truth_values.shape != (context.case_count, *expected_field_shape):
            raise PersistenceContextError(
                "truth must have shape [case,lead,latitude,longitude] aligned with context"
            )
        selected_indices = _validate_fit_indices(
            np.asarray(indices),
            context.case_count,
            label="dataset indices",
        )
        self.members = members
        self.truth = truth
        self.context = context
        self.indices = selected_indices
        self.arm: ContextArm = selected_arm

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(
        self, item: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(item, (bool, np.bool_)) or not isinstance(
            item, (int, np.integer)
        ):
            raise IndexError("dataset item must be an integer")
        position = int(item)
        if position < 0:
            position += len(self)
        if position < 0 or position >= len(self):
            raise IndexError("dataset item is out of range")
        case_index = int(self.indices[position])
        members = np.array(self.members[case_index], dtype=np.float32, copy=True)
        truth = np.array(self.truth[case_index], dtype=np.float32, copy=True)
        context = context_for_case(self.context, case_index, self.arm)
        return (
            torch.from_numpy(members),
            torch.from_numpy(context),
            torch.from_numpy(truth),
        )
