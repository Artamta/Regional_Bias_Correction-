"""Independent, benchmark-compatible scoring for the FuXi India S2S bridge.

The functions in this module deliberately do not import the benchmark
implementation.  They reproduce its frozen scientific contract while keeping
forecast-only climatology construction separate from truth-bearing scoring:

* corrected forecast climatology uses all 517 starts from 2020--2024;
* IMD scoring uses only the 505 starts ending no later than 2024-12-31;
* forecast and observation anomalies use separate climatologies; and
* spatial scores are calculated per initialization before arithmetic
  aggregation.

The accumulator and chunk scorer allow inference products to be processed in
bounded memory.  No function in this module opens data or a 2025 target.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


LEAD_WEEKS = (1, 2, 3, 4, 5, 6)
TARGET_DAY_OFFSETS = tuple(range(1, 43))
FORECAST_YEARS = (2020, 2021, 2022, 2023, 2024)
FORECAST_COUNTS = {2020: 105, 2021: 104, 2022: 104, 2023: 104, 2024: 100}
SCORING_COUNTS = {2020: 105, 2021: 104, 2022: 104, 2023: 104, 2024: 88}
FORECAST_INITIALIZATION_SHA256 = (
    "e95a7f88fdc0ca13b775e1c8708b510310e4a647ff2040e099f65db71baeef56"
)
SCORING_INITIALIZATION_SHA256 = (
    "41b67539c1d9196dfe5b598e0613a80a794dfea9da6ecbaae40cc102ebd66062"
)
RAW_IDENTITY_CASE_COUNT = 505 * len(LEAD_WEEKS) * 5
RAW_IDENTITY_CASE_IDS_SHA256 = (
    "623713c5d2b93dd0a8501086e1c9dd2d2596fa6017c2b1572f94ba1e0a709983"
)
LAST_ALLOWED_TARGET_DATE = np.datetime64("2024-12-31", "D")
REGION_ORDER = (
    "all_india",
    "northwest_india",
    "central_india",
    "south_peninsula",
    "east_northeast_india",
)
REGION_LABELS = {
    "all_india": "All India",
    "northwest_india": "Northwest India",
    "central_india": "Central India",
    "south_peninsula": "South Peninsula",
    "east_northeast_india": "East & Northeast India",
}


class ScoringContractError(ValueError):
    """Raised when an input crosses a frozen benchmark boundary."""


def _dates(values: Iterable[Any], *, name: str) -> np.ndarray:
    result = np.asarray(values, dtype="datetime64[D]")
    if result.ndim != 1 or result.size == 0:
        raise ScoringContractError(f"{name} must be a non-empty one-dimensional array")
    if np.isnat(result).any():
        raise ScoringContractError(f"{name} contains NaT")
    if np.unique(result).size != result.size:
        raise ScoringContractError(f"{name} contains duplicate dates")
    if np.any(np.diff(result) <= np.timedelta64(0, "D")):
        raise ScoringContractError(f"{name} must be strictly increasing")
    return result


def fixed_climatology_day(values: Iterable[Any]) -> np.ndarray:
    """Map month/day to the fixed leap-year-2000 index in ``1..366``."""

    dates = pd.DatetimeIndex(values)
    if dates.hasnans:
        raise ScoringContractError("climatology dates contain NaT")
    result = np.empty(len(dates), dtype=np.int16)
    for index, value in enumerate(dates):
        result[index] = pd.Timestamp(2000, value.month, value.day).dayofyear
    return result


def initialization_dates_sha256(values: Iterable[Any]) -> str:
    """Hash a strict date sequence with the benchmark's stable delimiter."""

    dates = _dates(values, name="initialization hash dates")
    payload = "".join(
        f"{np.datetime_as_string(value, unit='D')}\n" for value in dates
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def imd_target_dates(initializations: Iterable[Any]) -> np.ndarray:
    """Return exact end-labelled IMD target dates as ``[case, week, day]``.

    W1 is ``init+1`` through ``init+7`` and W6 is ``init+36`` through
    ``init+42``.  The initialization day is never part of the target.
    """

    starts = _dates(initializations, name="initializations")
    offsets = np.asarray(TARGET_DAY_OFFSETS, dtype="timedelta64[D]").reshape(1, 6, 7)
    return starts[:, None, None] + offsets


def scoreable_initialization_mask(
    initializations: Iterable[Any],
    *,
    last_allowed_target_date: np.datetime64 = LAST_ALLOWED_TARGET_DATE,
) -> np.ndarray:
    """Select starts whose final end-labelled IMD target remains unsealed."""

    targets = imd_target_dates(initializations)
    cutoff = np.datetime64(last_allowed_target_date, "D")
    return targets[:, -1, -1] <= cutoff


def valid_period_midpoints(initializations: Iterable[Any]) -> np.ndarray:
    """Return exact forecast-window midpoints with shape ``[case, 6]``."""

    starts = _dates(initializations, name="initializations").astype("datetime64[h]")
    # [init, init+7d) has midpoint init+3.5d; each later week adds seven days.
    hours = np.asarray([84 + 168 * week for week in range(6)], dtype="timedelta64[h]")
    return starts[:, None] + hours[None, :]


def valid_midpoint_jjas_mask(
    initializations: Iterable[Any], lead_week: int | None = None
) -> np.ndarray:
    """Return valid-period-midpoint JJAS membership, never init-month JJAS."""

    midpoint = valid_period_midpoints(initializations)
    months = pd.DatetimeIndex(midpoint.reshape(-1)).month.to_numpy().reshape(midpoint.shape)
    mask = np.isin(months, (6, 7, 8, 9))
    if lead_week is None:
        return mask
    if lead_week not in LEAD_WEEKS:
        raise ScoringContractError("lead_week must be in 1..6")
    return mask[:, lead_week - 1]


def _year_counts(values: np.ndarray) -> dict[int, int]:
    years = pd.DatetimeIndex(values).year.to_numpy()
    return {year: int(np.count_nonzero(years == year)) for year in FORECAST_YEARS}


def validate_bridge_cohorts(
    forecast_only_initializations: Iterable[Any],
    scoring_initializations: Iterable[Any],
    *,
    strict_counts: bool = True,
) -> dict[str, Any]:
    """Validate and index the distinct forecast-only and scoring cohorts.

    In strict mode the first cohort is the canonical 517 starts and the second
    is exactly its 505-member scoreable subset.  The returned ``score_indices``
    indexes forecast-only arrays without copying or reopening truth.
    """

    forecast = _dates(forecast_only_initializations, name="forecast-only initializations")
    scoring = _dates(scoring_initializations, name="scoring initializations")
    forecast_years = set(pd.DatetimeIndex(forecast).year)
    scoring_years = set(pd.DatetimeIndex(scoring).year)
    allowed = set(FORECAST_YEARS)
    if not forecast_years.issubset(allowed) or not scoring_years.issubset(allowed):
        raise ScoringContractError("bridge initializations must remain within 2020--2024")

    expected_scoring = forecast[scoreable_initialization_mask(forecast)]
    if not np.array_equal(scoring, expected_scoring):
        raise ScoringContractError(
            "scoring initializations must equal the forecast cohort filtered at init+42"
        )
    lookup = {date: index for index, date in enumerate(forecast)}
    score_indices = np.asarray([lookup[date] for date in scoring], dtype=np.int64)

    if strict_counts:
        if len(forecast) != 517 or _year_counts(forecast) != FORECAST_COUNTS:
            raise ScoringContractError("forecast-only cohort is not the canonical 517 starts")
        if len(scoring) != 505 or _year_counts(scoring) != SCORING_COUNTS:
            raise ScoringContractError("scoring cohort is not the canonical 505 starts")
        forecast_hash = initialization_dates_sha256(forecast)
        scoring_hash = initialization_dates_sha256(scoring)
        if forecast_hash != FORECAST_INITIALIZATION_SHA256:
            raise ScoringContractError("forecast-only cohort dates differ from the frozen hash")
        if scoring_hash != SCORING_INITIALIZATION_SHA256:
            raise ScoringContractError("scoring cohort dates differ from the frozen hash")
        jjas_counts = {
            lead: int(valid_midpoint_jjas_mask(scoring, lead).sum())
            for lead in LEAD_WEEKS
        }
        if set(jjas_counts.values()) != {169}:
            raise ScoringContractError(
                f"valid-midpoint JJAS must contain 169 cases per lead: {jjas_counts}"
            )
    else:
        forecast_hash = initialization_dates_sha256(forecast)
        scoring_hash = initialization_dates_sha256(scoring)
        jjas_counts = {
            lead: int(valid_midpoint_jjas_mask(scoring, lead).sum())
            for lead in LEAD_WEEKS
        }
    return {
        "forecast_only_count": int(len(forecast)),
        "forecast_only_year_counts": _year_counts(forecast),
        "forecast_only_dates_sha256": forecast_hash,
        "scoring_count": int(len(scoring)),
        "scoring_year_counts": _year_counts(scoring),
        "scoring_dates_sha256": scoring_hash,
        "score_indices": score_indices,
        "valid_midpoint_jjas_count_by_lead": jjas_counts,
        "last_scoring_target_date": np.datetime_as_string(
            imd_target_dates(scoring)[:, -1, -1].max(), unit="D"
        ),
        "sealed_2025_target_opened": False,
    }


@dataclass(frozen=True)
class WeeklyIMDTargets:
    """Weekly IMD truth, normal, and dynamic seven-day support."""

    truth: np.ndarray
    climatology: np.ndarray
    observation_support: np.ndarray
    target_dates: np.ndarray


def weekly_imd_targets(
    initializations: Iterable[Any],
    daily_dates: Iterable[Any],
    daily_values: np.ndarray,
    daily_observation_fraction: np.ndarray,
    imd_daily_climatology_1991_2019: np.ndarray,
    *,
    last_allowed_target_date: np.datetime64 = LAST_ALLOWED_TARGET_DATE,
    reject_daily_dates_after_cutoff: bool = True,
) -> WeeklyIMDTargets:
    """Construct coverage-weighted weekly IMD targets at exact +1..+42.

    Weekly observation support is the minimum of the seven daily fractions,
    matching the benchmark.  The IMD normal is averaged over the same seven
    end-labelled dates but is not coverage-weighted.
    """

    inits = _dates(initializations, name="scoring initializations")
    dates = _dates(daily_dates, name="IMD daily dates")
    cutoff = np.datetime64(last_allowed_target_date, "D")
    targets = imd_target_dates(inits)
    if targets[:, -1, -1].max() > cutoff:
        raise ScoringContractError("requested IMD target crosses the sealed 2025 boundary")
    if reject_daily_dates_after_cutoff and dates.max() > cutoff:
        raise ScoringContractError("IMD daily inputs include dates after the 2024 firewall")

    values = np.asarray(daily_values)
    fraction = np.asarray(daily_observation_fraction)
    if values.ndim < 2 or values.shape[0] != len(dates):
        raise ScoringContractError("daily_values must be [date, ...field dimensions]")
    field_shape = values.shape[1:]
    if fraction.shape == field_shape:
        fraction = np.broadcast_to(fraction, values.shape)
    if fraction.shape != values.shape:
        raise ScoringContractError("daily observation fractions do not match daily values")
    if not np.isfinite(fraction).all() or np.any((fraction < 0.0) | (fraction > 1.0)):
        raise ScoringContractError("daily observation fractions must be finite in [0, 1]")
    represented = fraction > 0.0
    if not np.isfinite(values[represented]).all():
        raise ScoringContractError("IMD values are non-finite on represented support")

    normal = np.asarray(imd_daily_climatology_1991_2019)
    if normal.shape != (366,) + field_shape:
        raise ScoringContractError("IMD 1991--2019 climatology must be [366, ...field]")

    index = pd.Index(dates)
    positions = index.get_indexer(targets.reshape(-1))
    if np.any(positions < 0):
        missing = targets.reshape(-1)[positions < 0][0]
        raise ScoringContractError(
            f"IMD target date is unavailable: {np.datetime_as_string(missing, unit='D')}"
        )
    shape = (len(inits), 6, 7) + field_shape
    selected_values = values[positions].reshape(shape).astype(np.float64, copy=False)
    selected_fraction = fraction[positions].reshape(shape).astype(np.float64, copy=False)
    denominator = selected_fraction.sum(axis=2, dtype=np.float64)
    numerator = (
        np.nan_to_num(selected_values, nan=0.0) * selected_fraction
    ).sum(axis=2, dtype=np.float64)
    truth = np.divide(
        numerator,
        denominator,
        out=np.full(denominator.shape, np.nan, dtype=np.float64),
        where=denominator > 0.0,
    )
    support = selected_fraction.min(axis=2)

    normal_days = fixed_climatology_day(targets.reshape(-1)) - 1
    selected_normal = normal[normal_days].reshape(shape).astype(np.float64, copy=False)
    finite_count = np.isfinite(selected_normal).sum(axis=2)
    weekly_normal = np.divide(
        np.nansum(selected_normal, axis=2, dtype=np.float64),
        finite_count,
        out=np.full(finite_count.shape, np.nan, dtype=np.float64),
        where=finite_count > 0,
    )
    return WeeklyIMDTargets(
        truth=truth.astype(np.float32),
        climatology=weekly_normal.astype(np.float32),
        observation_support=support.astype(np.float32),
        target_dates=targets,
    )


def circular_window_mean(
    values: np.ndarray,
    climatology_days: np.ndarray,
    *,
    half_width_days: int = 15,
    calendar_days: int = 366,
) -> tuple[np.ndarray, np.ndarray]:
    """Centered circular-window mean, matching ``india_s2s_climatology_v1``."""

    array = np.asarray(values)
    days = np.asarray(climatology_days, dtype=np.int64)
    if array.ndim < 2 or array.shape[0] != len(days):
        raise ScoringContractError("values and climatology-day dimensions disagree")
    if half_width_days < 0 or 2 * half_width_days + 1 > calendar_days:
        raise ScoringContractError("invalid circular climatology window")
    if np.any((days < 1) | (days > calendar_days)):
        raise ScoringContractError("climatology day lies outside the fixed calendar")
    if np.unique(days).size != len(days):
        raise ScoringContractError("a source year has multiple starts on one calendar day")
    if not np.isfinite(array).all():
        raise ScoringContractError("forecast climatology input is non-finite")

    daily = np.zeros((calendar_days,) + array.shape[1:], dtype=np.float64)
    counts = np.zeros(calendar_days, dtype=np.int16)
    daily[days - 1] = array.astype(np.float64, copy=False)
    counts[days - 1] = 1
    radius = half_width_days
    if radius:
        extended = np.concatenate((daily[-radius:], daily, daily[:radius]), axis=0)
        extended_counts = np.concatenate(
            (counts[-radius:], counts, counts[:radius]), axis=0
        )
    else:
        extended = daily
        extended_counts = counts
    leading = np.zeros((1,) + daily.shape[1:], dtype=np.float64)
    cumulative = np.concatenate((leading, np.cumsum(extended, axis=0)), axis=0)
    cumulative_counts = np.concatenate(
        (np.zeros(1, dtype=np.int32), np.cumsum(extended_counts, dtype=np.int32))
    )
    width = 2 * radius + 1
    window_sum = cumulative[width:] - cumulative[:-width]
    window_count = cumulative_counts[width:] - cumulative_counts[:-width]
    if window_sum.shape[0] != calendar_days:
        raise AssertionError("circular climatology returned the wrong calendar length")
    if np.any(window_count == 0):
        raise ScoringContractError("at least one 31-day climatology window is empty")
    reshape = (calendar_days,) + (1,) * (array.ndim - 1)
    mean = window_sum / window_count.reshape(reshape)
    return mean.astype(np.float32), window_count.astype(np.int16)


@dataclass(frozen=True)
class ForecastClimatology:
    """Equal-year forecast climatology and exact four-year LOYO fields."""

    training_years: tuple[int, ...]
    mean_by_year: np.ndarray
    mean: np.ndarray
    mean_loyo: np.ndarray
    initialization_count: np.ndarray
    initialization_counts_by_year: Mapping[int, int]

    def select_loyo(self, initializations: Iterable[Any]) -> np.ndarray:
        dates = _dates(initializations, name="climatology-selection initializations")
        years = pd.DatetimeIndex(dates).year.to_numpy()
        positions = {year: index for index, year in enumerate(self.training_years)}
        unknown = sorted(set(years).difference(positions))
        if unknown:
            raise ScoringContractError(f"LOYO selection contains unknown years: {unknown}")
        year_index = np.asarray([positions[int(year)] for year in years], dtype=np.int64)
        day_index = fixed_climatology_day(dates).astype(np.int64) - 1
        return self.mean_loyo[year_index, day_index]


class EqualYearForecastClimatologyAccumulator:
    """Streaming collector for the frozen 366-day equal-year estimator."""

    def __init__(
        self,
        *,
        training_years: Sequence[int] = FORECAST_YEARS,
        half_width_days: int = 15,
        expected_counts: Mapping[int, int] | None = None,
    ) -> None:
        years = tuple(int(year) for year in training_years)
        if len(years) < 2 or len(set(years)) != len(years):
            raise ScoringContractError("climatology requires at least two unique years")
        self.training_years = years
        self.half_width_days = int(half_width_days)
        self.expected_counts = (
            {int(year): int(count) for year, count in expected_counts.items()}
            if expected_counts is not None
            else None
        )
        if self.expected_counts is not None and set(self.expected_counts) != set(years):
            raise ScoringContractError(
                "expected climatology counts must cover every training year exactly"
            )
        self._fields: dict[int, dict[int, np.ndarray]] = {year: {} for year in years}
        self._field_shape: tuple[int, ...] | None = None

    def update(self, initializations: Iterable[Any], ensemble_means: np.ndarray) -> None:
        dates = _dates(initializations, name="forecast climatology chunk")
        fields = np.asarray(ensemble_means)
        if fields.ndim < 2 or fields.shape[0] != len(dates):
            raise ScoringContractError("ensemble means must be [case, ...field dimensions]")
        if not np.isfinite(fields).all():
            raise ScoringContractError("forecast climatology fields contain non-finite values")
        if self._field_shape is None:
            self._field_shape = fields.shape[1:]
        elif fields.shape[1:] != self._field_shape:
            raise ScoringContractError("forecast climatology field shape changed across chunks")

        years = pd.DatetimeIndex(dates).year.to_numpy()
        days = fixed_climatology_day(dates)
        for date, year, day, field in zip(dates, years, days, fields, strict=True):
            if int(year) not in self._fields:
                raise ScoringContractError(f"forecast climatology year is outside contract: {year}")
            yearly = self._fields[int(year)]
            if int(day) in yearly:
                raise ScoringContractError(
                    "duplicate forecast climatology date/calendar day: "
                    f"{np.datetime_as_string(date, unit='D')}"
                )
            yearly[int(day)] = np.asarray(field, dtype=np.float32).copy()

    def finalize(self) -> ForecastClimatology:
        if self._field_shape is None:
            raise ScoringContractError("forecast climatology accumulator is empty")
        yearly_fields: list[np.ndarray] = []
        yearly_counts: list[np.ndarray] = []
        counts_by_year: dict[int, int] = {}
        for year in self.training_years:
            records = self._fields[year]
            if not records:
                raise ScoringContractError(f"forecast climatology year {year} is empty")
            counts_by_year[year] = len(records)
            if self.expected_counts is not None and len(records) != self.expected_counts[year]:
                raise ScoringContractError(
                    f"forecast climatology year {year} has {len(records)} starts; "
                    f"expected {self.expected_counts[year]}"
                )
            days = np.asarray(sorted(records), dtype=np.int16)
            values = np.stack([records[int(day)] for day in days])
            smoothed, counts = circular_window_mean(
                values,
                days,
                half_width_days=self.half_width_days,
                calendar_days=366,
            )
            yearly_fields.append(smoothed)
            yearly_counts.append(counts)
        by_year = np.stack(yearly_fields).astype(np.float32)
        total = by_year.sum(axis=0, dtype=np.float64)
        loyo = np.empty_like(by_year, dtype=np.float32)
        divisor = len(self.training_years) - 1
        for index in range(len(self.training_years)):
            loyo[index] = (
                (total - by_year[index].astype(np.float64)) / divisor
            ).astype(np.float32)
        return ForecastClimatology(
            training_years=self.training_years,
            mean_by_year=by_year,
            mean=by_year.mean(axis=0, dtype=np.float64).astype(np.float32),
            mean_loyo=loyo,
            initialization_count=np.stack(yearly_counts),
            initialization_counts_by_year=counts_by_year,
        )


def build_equal_year_forecast_climatology(
    initializations: Iterable[Any],
    ensemble_means: np.ndarray,
    *,
    strict_bridge_counts: bool = True,
) -> ForecastClimatology:
    """Batch convenience wrapper around the streaming climatology builder."""

    accumulator = EqualYearForecastClimatologyAccumulator(
        training_years=FORECAST_YEARS,
        half_width_days=15,
        expected_counts=FORECAST_COUNTS if strict_bridge_counts else None,
    )
    accumulator.update(initializations, ensemble_means)
    return accumulator.finalize()


def empirical_crps(
    members: np.ndarray,
    truth: np.ndarray,
    *,
    member_axis: int = 1,
    expected_member_count: int | None = 50,
) -> np.ndarray:
    """Finite-ensemble empirical CRPS without constructing an M-by-M array."""

    values = np.asarray(members, dtype=np.float64)
    target = np.asarray(truth, dtype=np.float64)
    if values.ndim < 2:
        raise ScoringContractError("members must contain a member dimension")
    axis = int(member_axis)
    if axis < 0:
        axis += values.ndim
    if axis < 0 or axis >= values.ndim:
        raise ScoringContractError("member_axis lies outside the ensemble dimensions")
    moved = np.moveaxis(values, axis, 0)
    if moved.shape[1:] != target.shape:
        raise ScoringContractError("ensemble and truth shapes are incompatible")
    count = moved.shape[0]
    if expected_member_count is not None and count != expected_member_count:
        raise ScoringContractError(
            f"expected {expected_member_count} ensemble members, found {count}"
        )
    if count < 1 or not np.isfinite(moved).all():
        raise ScoringContractError("ensemble members must be non-empty and finite")
    first = np.mean(np.abs(moved - target[None]), axis=0, dtype=np.float64)
    ordered = np.sort(moved, axis=0)
    coefficients = (2.0 * np.arange(count) - count + 1.0).reshape(
        (count,) + (1,) * target.ndim
    )
    dispersion = np.sum(ordered * coefficients, axis=0, dtype=np.float64) / count**2
    return first - dispersion


def _spatial_weights(weights: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(weights, dtype=np.float64)
    try:
        result = np.broadcast_to(array, shape)
    except ValueError as error:
        raise ScoringContractError("spatial weights do not broadcast to metric fields") from error
    if not np.isfinite(result).all() or np.any(result < 0.0):
        raise ScoringContractError("spatial weights must be finite and non-negative")
    if np.any(np.sum(result, axis=(-2, -1)) <= 0.0):
        raise ScoringContractError("every case/lead requires positive spatial support")
    return result


def weighted_field_mean(field: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Area-weighted spatial mean over the final two dimensions."""

    values = np.asarray(field, dtype=np.float64)
    if values.ndim < 2:
        raise ScoringContractError("metric field needs two spatial dimensions")
    spatial = _spatial_weights(weights, values.shape)
    supported = spatial > 0.0
    if not np.isfinite(values[supported]).all():
        raise ScoringContractError("metric field is non-finite on supported cells")
    safe = np.where(supported, values, 0.0)
    return np.sum(safe * spatial, axis=(-2, -1), dtype=np.float64) / np.sum(
        spatial, axis=(-2, -1), dtype=np.float64
    )


def weighted_raw_metrics(
    forecast: np.ndarray, truth: np.ndarray, weights: np.ndarray
) -> dict[str, np.ndarray]:
    """Raw ensemble-mean RMSE, MAE, and bias for each case/lead."""

    prediction = np.asarray(forecast, dtype=np.float64)
    observation = np.asarray(truth, dtype=np.float64)
    if prediction.shape != observation.shape or prediction.ndim < 2:
        raise ScoringContractError("forecast and truth fields are incompatible")
    spatial = _spatial_weights(weights, prediction.shape)
    supported = spatial > 0.0
    if not np.isfinite(prediction[supported]).all() or not np.isfinite(
        observation[supported]
    ).all():
        raise ScoringContractError("raw metric input is non-finite on supported cells")
    error = np.where(supported, prediction - observation, 0.0)
    return {
        "rmse": np.sqrt(weighted_field_mean(error**2, spatial)),
        "mae": weighted_field_mean(np.abs(error), spatial),
        "bias": weighted_field_mean(error, spatial),
        "valid_cell_count": np.sum(supported, axis=(-2, -1), dtype=np.int64),
        "effective_area_km2": np.sum(spatial, axis=(-2, -1), dtype=np.float64),
    }


def weighted_spatial_acc(
    forecast_anomaly: np.ndarray,
    observation_anomaly: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Area-weighted spatial anomaly correlation for each case/lead."""

    forecast = np.asarray(forecast_anomaly, dtype=np.float64)
    observation = np.asarray(observation_anomaly, dtype=np.float64)
    if forecast.shape != observation.shape or forecast.ndim < 2:
        raise ScoringContractError("forecast and observation anomalies are incompatible")
    spatial = _spatial_weights(weights, forecast.shape)
    supported = spatial > 0.0
    if np.any(np.sum(supported, axis=(-2, -1)) < 3):
        raise ScoringContractError("ACC requires at least three represented cells")
    if not np.isfinite(forecast[supported]).all() or not np.isfinite(
        observation[supported]
    ).all():
        raise ScoringContractError("ACC anomaly is non-finite on supported cells")
    total = np.sum(spatial, axis=(-2, -1), dtype=np.float64)
    safe_forecast = np.where(supported, forecast, 0.0)
    safe_observation = np.where(supported, observation, 0.0)
    forecast_mean = np.sum(safe_forecast * spatial, axis=(-2, -1)) / total
    observation_mean = np.sum(safe_observation * spatial, axis=(-2, -1)) / total
    forecast_centered = np.where(
        supported, forecast - forecast_mean[..., None, None], 0.0
    )
    observation_centered = np.where(
        supported, observation - observation_mean[..., None, None], 0.0
    )
    covariance = np.sum(
        spatial * forecast_centered * observation_centered,
        axis=(-2, -1),
        dtype=np.float64,
    )
    forecast_variance = np.sum(
        spatial * forecast_centered**2, axis=(-2, -1), dtype=np.float64
    )
    observation_variance = np.sum(
        spatial * observation_centered**2, axis=(-2, -1), dtype=np.float64
    )
    denominator = np.sqrt(forecast_variance * observation_variance)
    return np.divide(
        covariance,
        denominator,
        out=np.full(covariance.shape, np.nan, dtype=np.float64),
        where=denominator > 0.0,
    )


def build_region_base_weights(
    india_area_weight_km2: np.ndarray,
    cell_area_km2: np.ndarray,
    region_fractions: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Build All-India and four fractional homogeneous-region area weights."""

    india = np.asarray(india_area_weight_km2, dtype=np.float64)
    cell = np.asarray(cell_area_km2, dtype=np.float64)
    if india.ndim != 2 or cell.shape != india.shape:
        raise ScoringContractError("India and cell-area fields must share a 2-D grid")
    if not np.isfinite(india).all() or not np.isfinite(cell).all():
        raise ScoringContractError("area fields must be finite")
    if np.any(india < 0.0) or np.any(cell <= 0.0):
        raise ScoringContractError("area fields contain invalid values")
    expected = set(REGION_ORDER[1:])
    if set(region_fractions) != expected:
        raise ScoringContractError(
            f"region fractions must contain exactly {sorted(expected)}"
        )
    result = {"all_india": india.copy()}
    for name in REGION_ORDER[1:]:
        fraction = np.asarray(region_fractions[name], dtype=np.float64)
        if fraction.shape != india.shape or not np.isfinite(fraction).all():
            raise ScoringContractError(f"invalid fractional mask for {name}")
        if np.any((fraction < 0.0) | (fraction > 1.0)):
            raise ScoringContractError(f"fractional mask for {name} lies outside [0, 1]")
        result[name] = cell * fraction
    return result


def _season(month: int) -> str:
    if month in (1, 2):
        return "JF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8, 9):
        return "JJAS"
    return "OND"


def score_case_chunk(
    *,
    method: str,
    initializations: Iterable[Any],
    members: np.ndarray,
    truth: np.ndarray,
    forecast_climatology: ForecastClimatology | np.ndarray,
    observation_climatology: np.ndarray,
    weekly_observation_support: np.ndarray,
    region_base_weights: Mapping[str, np.ndarray],
    expected_member_count: int | None = 50,
) -> pd.DataFrame:
    """Score a truth-bearing chunk while retaining one row per case/lead/region."""

    inits = _dates(initializations, name="scoring chunk initializations")
    if imd_target_dates(inits)[:, -1, -1].max() > LAST_ALLOWED_TARGET_DATE:
        raise ScoringContractError("scoring chunk crosses the sealed 2025 target boundary")
    ensemble = np.asarray(members)
    observed = np.asarray(truth)
    observed_normal = np.asarray(observation_climatology)
    support = np.asarray(weekly_observation_support, dtype=np.float64)
    if ensemble.ndim != 5 or ensemble.shape[0] != len(inits) or ensemble.shape[2] != 6:
        raise ScoringContractError("members must be [case, member, 6, y, x]")
    expected_shape = (len(inits), 6) + ensemble.shape[-2:]
    if observed.shape != expected_shape or observed_normal.shape != expected_shape:
        raise ScoringContractError("truth and observation climatology shapes are invalid")
    if support.shape != expected_shape:
        raise ScoringContractError("weekly observation support shape is invalid")
    if not np.isfinite(support).all() or np.any((support < 0.0) | (support > 1.0)):
        raise ScoringContractError("weekly observation support must be finite in [0, 1]")
    if isinstance(forecast_climatology, ForecastClimatology):
        forecast_normal = forecast_climatology.select_loyo(inits)
    else:
        forecast_normal = np.asarray(forecast_climatology)
    if forecast_normal.shape != expected_shape:
        raise ScoringContractError("forecast climatology shape is invalid")
    if set(region_base_weights) != set(REGION_ORDER):
        raise ScoringContractError("region weights must contain the frozen five regions")

    point_crps = empirical_crps(
        ensemble,
        observed,
        member_axis=1,
        expected_member_count=expected_member_count,
    )
    ensemble_mean = ensemble.mean(axis=1, dtype=np.float64)
    forecast_anomaly = ensemble_mean - forecast_normal
    observation_anomaly = observed - observed_normal
    midpoints = valid_period_midpoints(inits)
    years = pd.DatetimeIndex(inits).year.to_numpy()
    rows: list[dict[str, Any]] = []
    for region in REGION_ORDER:
        base = np.asarray(region_base_weights[region], dtype=np.float64)
        if base.shape != ensemble.shape[-2:]:
            raise ScoringContractError(f"region weight shape is invalid for {region}")
        dynamic = support * base
        raw = weighted_raw_metrics(ensemble_mean, observed, dynamic)
        acc = weighted_spatial_acc(forecast_anomaly, observation_anomaly, dynamic)
        crps = weighted_field_mean(point_crps, dynamic)
        for case_index, init in enumerate(inits):
            for lead_index, lead_week in enumerate(LEAD_WEEKS):
                midpoint = pd.Timestamp(midpoints[case_index, lead_index])
                rows.append(
                    {
                        "method": method,
                        "variable": "tp",
                        "reference": "imd",
                        "reference_label": "IMD",
                        "units": "mm day-1",
                        "score_status": "available",
                        "verification_year": int(years[case_index]),
                        "init": np.datetime_as_string(init, unit="D"),
                        "valid_period_start": np.datetime_as_string(
                            init + np.timedelta64(7 * lead_index, "D"), unit="D"
                        ),
                        "valid_period_midpoint": midpoint.isoformat(),
                        "valid_period_end_exclusive": np.datetime_as_string(
                            init + np.timedelta64(7 * (lead_index + 1), "D"), unit="D"
                        ),
                        "season": _season(midpoint.month),
                        "region": region,
                        "region_label": REGION_LABELS[region],
                        "lead_week": lead_week,
                        "crps": float(crps[case_index, lead_index]),
                        "acc": float(acc[case_index, lead_index]),
                        "rmse": float(raw["rmse"][case_index, lead_index]),
                        "mae": float(raw["mae"][case_index, lead_index]),
                        "bias": float(raw["bias"][case_index, lead_index]),
                        "valid_cell_count": int(
                            raw["valid_cell_count"][case_index, lead_index]
                        ),
                        "effective_area_km2": float(
                            raw["effective_area_km2"][case_index, lead_index]
                        ),
                    }
                )
    return pd.DataFrame(rows)


def aggregate_case_scores(
    cases: pd.DataFrame,
    *,
    group_by: Sequence[str] = ("method", "region", "lead_week"),
) -> pd.DataFrame:
    """Arithmetic aggregation of per-initialization scores.

    In particular, reported RMSE is the mean of case-wise spatial RMSE values;
    it is never reconstructed as a square root of pooled squared errors.
    """

    required = {*group_by, "init", "crps", "acc", "rmse", "mae", "bias"}
    missing = sorted(required.difference(cases.columns))
    if missing:
        raise ScoringContractError(f"case table is missing columns: {missing}")
    if cases.duplicated([*group_by, "init"]).any():
        raise ScoringContractError("case table has duplicate initialization rows per group")
    rows: list[dict[str, Any]] = []
    for key, group in cases.groupby(list(group_by), sort=False, dropna=False):
        values = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_by, values, strict=True))
        row["case_count"] = int(len(group))
        for metric in ("crps", "acc", "rmse", "mae", "bias"):
            finite = group[metric].to_numpy(dtype=np.float64)
            finite = finite[np.isfinite(finite)]
            row[f"{metric}_valid_case_count"] = int(len(finite))
            row[metric] = float(finite.mean()) if len(finite) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def compare_raw_fuxi_identity(
    independent_cases: pd.DataFrame,
    benchmark_cases: pd.DataFrame,
    *,
    key_columns: Sequence[str] = ("init", "lead_week", "region"),
    metric_columns: Sequence[str] = (
        "acc",
        "rmse",
        "mae",
        "bias",
        "valid_cell_count",
        "effective_area_km2",
    ),
    atol: float = 1.0e-6,
    rtol: float = 1.0e-7,
    support_atol: float = 1.0e-6,
    support_rtol: float = 1.0e-12,
    require_complete_bridge: bool = True,
) -> dict[str, Any]:
    """Compare raw-FuXi case rows with the benchmark on identical IDs."""

    for name, frame in (
        ("independent", independent_cases),
        ("benchmark", benchmark_cases),
    ):
        missing = sorted(set((*key_columns, *metric_columns)).difference(frame.columns))
        if missing:
            raise ScoringContractError(f"{name} raw-FuXi table lacks columns: {missing}")
        if frame.empty:
            raise ScoringContractError(f"{name} raw-FuXi table is empty")
        if frame.duplicated(list(key_columns)).any():
            raise ScoringContractError(f"{name} raw-FuXi table has duplicate case IDs")
    left = independent_cases.set_index(list(key_columns)).sort_index()
    right = benchmark_cases.set_index(list(key_columns)).sort_index()
    if not left.index.equals(right.index):
        left_only = left.index.difference(right.index)
        right_only = right.index.difference(left.index)
        raise ScoringContractError(
            "raw-FuXi case IDs differ: "
            f"independent_only={len(left_only)}, benchmark_only={len(right_only)}"
        )

    case_id_hash = _raw_case_ids_sha256(left.reset_index())
    if require_complete_bridge:
        if tuple(key_columns) != ("init", "lead_week", "region"):
            raise ScoringContractError(
                "complete raw identity requires init/lead_week/region case keys"
            )
        if len(left) != RAW_IDENTITY_CASE_COUNT:
            raise ScoringContractError(
                "complete raw identity requires exactly "
                f"{RAW_IDENTITY_CASE_COUNT} case/lead/region rows"
            )
        init_values = np.asarray(
            sorted(pd.unique(left.reset_index()["init"])), dtype="datetime64[D]"
        )
        if (
            len(init_values) != 505
            or initialization_dates_sha256(init_values)
            != SCORING_INITIALIZATION_SHA256
        ):
            raise ScoringContractError(
                "complete raw identity does not contain the frozen 505 scoring dates"
            )
        if case_id_hash != RAW_IDENTITY_CASE_IDS_SHA256:
            raise ScoringContractError(
                "complete raw identity case IDs differ from the frozen Cartesian grid"
            )

    maximum: dict[str, float] = {}
    failures: dict[str, int] = {}
    for metric in metric_columns:
        first = left[metric].to_numpy(dtype=np.float64)
        second = right[metric].to_numpy(dtype=np.float64)
        both_nan = np.isnan(first) & np.isnan(second)
        comparable = np.isfinite(first) & np.isfinite(second)
        invalid = ~(both_nan | comparable)
        close = np.zeros(len(first), dtype=bool)
        close[both_nan] = True
        if metric == "valid_cell_count":
            close[comparable] = first[comparable] == second[comparable]
        elif metric == "effective_area_km2":
            close[comparable] = np.isclose(
                first[comparable],
                second[comparable],
                atol=support_atol,
                rtol=support_rtol,
            )
        else:
            close[comparable] = np.isclose(
                first[comparable], second[comparable], atol=atol, rtol=rtol
            )
        failures[metric] = int(np.count_nonzero(invalid | ~close))
        maximum[metric] = (
            float(np.max(np.abs(first[comparable] - second[comparable])))
            if np.any(comparable)
            else 0.0
        )
    return {
        "identity": not any(failures.values()),
        "matched_case_rows": int(len(left)),
        "case_ids_sha256": case_id_hash,
        "complete_bridge_contract": bool(require_complete_bridge),
        "key_columns": list(key_columns),
        "metric_columns": list(metric_columns),
        "atol": float(atol),
        "rtol": float(rtol),
        "support_atol": float(support_atol),
        "support_rtol": float(support_rtol),
        "failure_count_by_metric": failures,
        "max_absolute_difference_by_metric": maximum,
    }


def _raw_case_ids_sha256(frame: pd.DataFrame) -> str:
    """Hash normalized raw-FuXi init/lead/region IDs in sorted order."""

    required = {"init", "lead_week", "region"}
    if not required.issubset(frame.columns):
        raise ScoringContractError("raw case-ID hash requires init/lead_week/region")
    records = sorted(
        (
            np.datetime_as_string(np.datetime64(init, "D"), unit="D"),
            str(int(lead)),
            str(region),
        )
        for init, lead, region in frame[["init", "lead_week", "region"]].itertuples(
            index=False, name=None
        )
    )
    payload = "".join("|".join(record) + "\n" for record in records).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def assert_raw_fuxi_identity(
    independent_cases: pd.DataFrame,
    benchmark_cases: pd.DataFrame,
    **kwargs: Any,
) -> dict[str, Any]:
    """Return an identity receipt or fail closed on any metric mismatch."""

    receipt = compare_raw_fuxi_identity(independent_cases, benchmark_cases, **kwargs)
    if not receipt["identity"]:
        raise ScoringContractError(
            "raw-FuXi metrics differ from the benchmark: "
            f"{receipt['failure_count_by_metric']}"
        )
    return receipt


__all__ = [
    "EqualYearForecastClimatologyAccumulator",
    "FORECAST_COUNTS",
    "FORECAST_INITIALIZATION_SHA256",
    "FORECAST_YEARS",
    "ForecastClimatology",
    "LAST_ALLOWED_TARGET_DATE",
    "LEAD_WEEKS",
    "REGION_LABELS",
    "REGION_ORDER",
    "RAW_IDENTITY_CASE_COUNT",
    "RAW_IDENTITY_CASE_IDS_SHA256",
    "SCORING_COUNTS",
    "SCORING_INITIALIZATION_SHA256",
    "ScoringContractError",
    "TARGET_DAY_OFFSETS",
    "WeeklyIMDTargets",
    "aggregate_case_scores",
    "assert_raw_fuxi_identity",
    "build_equal_year_forecast_climatology",
    "build_region_base_weights",
    "circular_window_mean",
    "compare_raw_fuxi_identity",
    "empirical_crps",
    "fixed_climatology_day",
    "imd_target_dates",
    "initialization_dates_sha256",
    "score_case_chunk",
    "scoreable_initialization_mask",
    "valid_midpoint_jjas_mask",
    "valid_period_midpoints",
    "validate_bridge_cohorts",
    "weekly_imd_targets",
    "weighted_field_mean",
    "weighted_raw_metrics",
    "weighted_spatial_acc",
]
