"""Leakage-aware categorical probability calibration for FuXi-S2S.

This module implements a compact, independently written adaptation of the
probabilistic bias-correction (PBC) construction described by Guan et al.
(2026).  It deliberately contains no file-system or experiment-split logic;
the all-season driver owns those contracts.

CDF values use the strict convention ``P(Y < q)``.  That convention matters
for mixed-discrete precipitation, where several climatological quantiles can
be exactly zero.  Both nominal quantile probabilities and empirical,
tie-aware training probabilities are retained by :class:`CalendarQuantiles`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


class PBCContractError(ValueError):
    """Raised when an input violates a scientific or array-shape contract."""


def _calendar_positions(dates: np.ndarray) -> np.ndarray:
    """Return stable 0--365 month/day positions using a leap-year template."""

    values = np.asarray(dates, dtype="datetime64[D]")
    template = pd.date_range("2000-01-01", "2000-12-31", freq="D")
    lookup = {date.strftime("%m-%d"): index for index, date in enumerate(template)}
    strings = pd.DatetimeIndex(values.reshape(-1)).strftime("%m-%d")
    return np.asarray([lookup[value] for value in strings], dtype=np.int16).reshape(
        values.shape
    )


def verification_midpoints(
    initializations: np.ndarray, lead_count: int = 6
) -> np.ndarray:
    """Weekly verification midpoint dates for forecasts issued at ``init``.

    Lead week one verifies over ``[init, init + 6 days]`` and therefore has
    midpoint ``init + 3 days``.  Later leads advance in seven-day increments.
    """

    starts = np.asarray(initializations, dtype="datetime64[D]")
    if starts.ndim != 1 or starts.size == 0:
        raise PBCContractError("initializations must be a non-empty 1-D array")
    offsets = (3 + 7 * np.arange(lead_count)).astype("timedelta64[D]")
    return starts[:, None] + offsets[None]


def verification_end_dates(
    initializations: np.ndarray, lead_count: int = 6
) -> np.ndarray:
    """Inclusive end dates for all weekly forecast periods."""

    starts = np.asarray(initializations, dtype="datetime64[D]")
    offsets = (6 + 7 * np.arange(lead_count)).astype("timedelta64[D]")
    return starts[:, None] + offsets[None]


def _validate_levels(levels: Sequence[float]) -> np.ndarray:
    result = np.asarray(levels, dtype=np.float64)
    if (
        result.ndim != 1
        or result.size == 0
        or not np.isfinite(result).all()
        or np.any(result <= 0.0)
        or np.any(result >= 1.0)
        or np.any(np.diff(result) <= 0.0)
    ):
        raise PBCContractError(
            "quantile levels must be finite and strictly inside (0,1)"
        )
    return result


def _validate_indices(indices: np.ndarray, case_count: int, label: str) -> np.ndarray:
    values = np.asarray(indices, dtype=np.int64)
    if values.ndim != 1 or values.size == 0:
        raise PBCContractError(f"{label} must be a non-empty 1-D array")
    if np.any(values < 0) or np.any(values >= case_count):
        raise PBCContractError(f"{label} contains an out-of-range case index")
    if np.unique(values).size != values.size:
        raise PBCContractError(f"{label} contains duplicate case indices")
    return values


@dataclass(frozen=True)
class CalendarQuantiles:
    """Training-only calendar-conditioned precipitation quantiles.

    ``thresholds`` and ``empirical_cdf`` have shape
    ``[calendar_day=366, cut, latitude, longitude]``.  The latter records the
    strict empirical probability ``P_train(Y < threshold)`` and is therefore
    the defensible tie-aware climatological CDF for zero-inflated rainfall.
    """

    levels: np.ndarray
    thresholds: np.ndarray
    empirical_cdf: np.ndarray
    support: np.ndarray
    window_radius_days: int
    minimum_samples: int
    fit_indices: np.ndarray
    sample_count_by_day: np.ndarray
    unique_fit_window_count: int
    duplicate_fit_window_count: int


def fit_calendar_quantiles(
    truth: np.ndarray,
    initializations: np.ndarray,
    fit_indices: np.ndarray,
    levels: Sequence[float],
    support: np.ndarray,
    *,
    window_radius_days: int = 15,
    minimum_samples: int = 8,
) -> CalendarQuantiles:
    """Fit pooled-lead, calendar-local quantiles using only ``fit_indices``.

    Every weekly target is assigned by its midpoint month/day.  A centered
    calendar window is legitimate in the frozen-split setting because all
    samples come from earlier training years.  If a deliberately sparse smoke
    cache supplies fewer than ``minimum_samples`` in a window, the nearest
    calendar samples are used and their exact count is retained.
    """

    values = np.asarray(truth, dtype=np.float32)
    starts = np.asarray(initializations, dtype="datetime64[D]")
    mask = np.asarray(support, dtype=bool)
    cuts = _validate_levels(levels)
    if values.ndim != 4:
        raise PBCContractError(
            "truth must have shape [case, lead, latitude, longitude]"
        )
    if starts.shape != (values.shape[0],):
        raise PBCContractError("initializations do not align with truth")
    if mask.shape != values.shape[-2:] or not np.any(mask):
        raise PBCContractError("support must select at least one truth grid cell")
    if window_radius_days < 0 or window_radius_days > 183:
        raise PBCContractError("window_radius_days must be between 0 and 183")
    if minimum_samples < 1:
        raise PBCContractError("minimum_samples must be positive")
    selected = _validate_indices(fit_indices, len(values), "fit_indices")
    if not np.isfinite(values[selected][..., mask]).all():
        raise PBCContractError("training truth is non-finite on supported cells")

    midpoint = verification_midpoints(starts, values.shape[1])
    fit_midpoints = midpoint[selected].reshape(-1)
    positions = _calendar_positions(fit_midpoints).reshape(-1)
    samples = values[selected].reshape(-1, *mask.shape)
    # The twice-weekly archive reaches the same verification week from several
    # issue/lead pairs.  Treat that observed week as one climatology sample,
    # rather than silently multiplying its weight by the number of forecasts.
    date_numbers = fit_midpoints.astype("datetime64[D]").astype(np.int64)
    _, first, inverse, multiplicity = np.unique(
        date_numbers, return_index=True, return_inverse=True, return_counts=True
    )
    for group in np.flatnonzero(multiplicity > 1):
        rows = np.flatnonzero(inverse == group)
        if not np.allclose(
            samples[rows][:, mask],
            samples[rows[0]][mask][None],
            rtol=0.0,
            atol=1.0e-6,
        ):
            raise PBCContractError(
                "duplicate verification windows have inconsistent IMD truth"
            )
    samples = samples[first]
    positions = positions[first]
    thresholds = np.full((366, cuts.size, *mask.shape), np.nan, dtype=np.float32)
    empirical = np.full_like(thresholds, np.nan)
    counts = np.empty(366, dtype=np.int32)

    for day in range(366):
        distances = np.minimum((positions - day) % 366, (day - positions) % 366)
        chosen = np.flatnonzero(distances <= window_radius_days)
        if chosen.size < minimum_samples:
            take = min(minimum_samples, samples.shape[0])
            chosen = np.argsort(distances, kind="stable")[:take]
        counts[day] = chosen.size
        supported_values = samples[chosen][:, mask].astype(np.float64, copy=False)
        quantiles = np.quantile(supported_values, cuts, axis=0, method="linear")
        # Numerical safety; empirical quantiles should already be ordered.
        quantiles = np.maximum.accumulate(quantiles, axis=0)
        persisted_quantiles = quantiles.astype(np.float32)
        thresholds[day][:, mask] = persisted_quantiles
        empirical[day][:, mask] = np.mean(
            supported_values[:, None, :] < persisted_quantiles[None, :, :],
            axis=0,
            dtype=np.float64,
        ).astype(np.float32)

    return CalendarQuantiles(
        levels=cuts.astype(np.float32),
        thresholds=thresholds,
        empirical_cdf=empirical,
        support=mask.copy(),
        window_radius_days=int(window_radius_days),
        minimum_samples=int(minimum_samples),
        fit_indices=selected.copy(),
        sample_count_by_day=counts,
        unique_fit_window_count=int(first.size),
        duplicate_fit_window_count=int(date_numbers.size - first.size),
    )


def calendar_fields(
    model: CalendarQuantiles,
    initializations: np.ndarray,
    lead_count: int,
    *,
    empirical: bool = False,
) -> np.ndarray:
    """Materialize date-matched thresholds or tie-aware climatological CDFs."""

    source = model.empirical_cdf if empirical else model.thresholds
    positions = _calendar_positions(verification_midpoints(initializations, lead_count))
    result = source[positions]
    expected = (
        len(initializations),
        lead_count,
        len(model.levels),
        *model.support.shape,
    )
    if result.shape != expected:
        raise PBCContractError(
            f"calendar field shape {result.shape}, expected {expected}"
        )
    return np.asarray(result, dtype=np.float32)


def ensemble_cdf(
    members: np.ndarray,
    thresholds: np.ndarray,
    *,
    chunk_size: int = 16,
    case_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Convert members to strict empirical CDF values without materializing them."""

    if members.ndim != 5:
        raise PBCContractError(
            "members must have shape [case, member, lead, latitude, longitude]"
        )
    selected = (
        np.arange(members.shape[0], dtype=np.int64)
        if case_indices is None
        else _validate_indices(case_indices, members.shape[0], "case_indices")
    )
    cuts = np.asarray(thresholds, dtype=np.float32)
    if cuts.ndim != 5:
        raise PBCContractError("thresholds must have shape [case,lead,cut,y,x]")
    expected = (selected.size, members.shape[2], cuts.shape[2], *members.shape[-2:])
    if cuts.shape != expected:
        raise PBCContractError(f"threshold shape {cuts.shape}, expected {expected}")
    if chunk_size < 1:
        raise PBCContractError("chunk_size must be positive")
    output = np.empty(expected, dtype=np.float32)
    for start in range(0, selected.size, chunk_size):
        stop = min(start + chunk_size, selected.size)
        block = np.asarray(members[selected[start:stop]], dtype=np.float32)
        if not np.isfinite(block).all() or np.any(block < 0.0):
            raise PBCContractError("members contain invalid precipitation")
        output[start:stop] = np.mean(
            block[:, :, :, None] < cuts[start:stop, None],
            axis=1,
            dtype=np.float64,
        ).astype(np.float32)
    output[~np.isfinite(cuts)] = np.nan
    return output


def observation_cdf(truth: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Convert verifying observations to strict cumulative bin indicators."""

    values = np.asarray(truth, dtype=np.float32)
    cuts = np.asarray(thresholds, dtype=np.float32)
    expected = (values.shape[0], values.shape[1], cuts.shape[2], *values.shape[-2:])
    if values.ndim != 4 or cuts.ndim != 5 or cuts.shape != expected:
        raise PBCContractError("truth and threshold shapes are incompatible")
    result = (values[:, :, None] < cuts).astype(np.float32)
    result[~np.isfinite(cuts) | ~np.isfinite(values[:, :, None])] = np.nan
    return result


def _pava_row(values: np.ndarray) -> np.ndarray:
    """Equal-weight pool-adjacent-violators projection of one finite row."""

    size = len(values)
    levels = np.empty(size, dtype=np.float64)
    weights = np.empty(size, dtype=np.int32)
    block_count = 0
    for value in values:
        levels[block_count] = float(value)
        weights[block_count] = 1
        block_count += 1
        while block_count >= 2 and levels[block_count - 2] > levels[block_count - 1]:
            left = block_count - 2
            total = weights[left] + weights[left + 1]
            levels[left] = (
                levels[left] * weights[left] + levels[left + 1] * weights[left + 1]
            ) / total
            weights[left] = total
            block_count -= 1
    projected = np.empty(size, dtype=np.float64)
    cursor = 0
    for block in range(block_count):
        stop = cursor + int(weights[block])
        projected[cursor:stop] = levels[block]
        cursor = stop
    return projected


def project_cdf(values: np.ndarray, *, axis: int = -1) -> np.ndarray:
    """Project arbitrary bin predictions onto valid nondecreasing CDFs.

    This is the exact equal-weight least-squares isotonic projection, followed
    by the probability bounds.  Rows that are entirely NaN (unsupported grid
    cells) remain NaN; partially finite rows are rejected.
    """

    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        raise PBCContractError("CDF input must have at least one dimension")
    moved = np.moveaxis(array, axis, -1)
    flat = np.clip(moved.reshape(-1, moved.shape[-1]), 0.0, 1.0)
    finite = np.isfinite(flat)
    partial = np.any(finite, axis=1) & ~np.all(finite, axis=1)
    if np.any(partial):
        raise PBCContractError("CDF rows must be wholly finite or wholly NaN")
    valid_rows = np.flatnonzero(np.all(finite, axis=1))
    bad = valid_rows[np.any(np.diff(flat[valid_rows], axis=1) < 0.0, axis=1)]
    for row in bad:
        flat[row] = _pava_row(flat[row])
    projected = flat.reshape(moved.shape)
    return np.moveaxis(projected, -1, axis).astype(np.float32)


def is_valid_cdf(
    values: np.ndarray, *, axis: int = -1, tolerance: float = 1e-7
) -> bool:
    """Return whether every finite CDF row is bounded and nondecreasing."""

    array = np.asarray(values, dtype=np.float64)
    moved = np.moveaxis(array, axis, -1).reshape(-1, array.shape[axis])
    finite = np.isfinite(moved)
    if np.any(np.any(finite, axis=1) & ~np.all(finite, axis=1)):
        return False
    rows = moved[np.all(finite, axis=1)]
    return bool(
        np.all(rows >= -tolerance)
        and np.all(rows <= 1.0 + tolerance)
        and np.all(np.diff(rows, axis=1) >= -tolerance)
    )


def _threshold_row_structure(
    thresholds: np.ndarray,
    *,
    axis: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return flattened thresholds, supported rows, and unique-cut indicators."""

    cuts = np.asarray(thresholds, dtype=np.float64)
    if cuts.ndim == 0:
        raise PBCContractError("thresholds must have at least one dimension")
    moved = np.moveaxis(cuts, axis, -1)
    flat = moved.reshape(-1, moved.shape[-1])
    finite = np.isfinite(flat)
    partial = np.any(finite, axis=1) & ~np.all(finite, axis=1)
    if np.any(partial):
        raise PBCContractError("threshold rows must be wholly finite or wholly NaN")
    supported = np.all(finite, axis=1)
    if np.any(flat[supported] < 0.0):
        raise PBCContractError("physical precipitation thresholds cannot be negative")
    if np.any(np.diff(flat[supported], axis=1) < 0.0):
        raise PBCContractError(
            "physical thresholds must be nondecreasing along the cut axis"
        )
    unique = np.zeros_like(flat, dtype=bool)
    unique[supported, 0] = True
    unique[supported, 1:] = flat[supported, 1:] != flat[supported, :-1]
    return flat, supported, unique


def effective_cut_mask(thresholds: np.ndarray, *, axis: int = 2) -> np.ndarray:
    """Mark each distinct, informative positive precipitation threshold.

    Repeated climatological quantiles do not define distinct category
    boundaries. Under strict ``P(Y < q)`` with nonnegative precipitation,
    ``q == 0`` is the deterministic empty event and is not an informative
    boundary. Unsupported all-NaN rows contain no effective cuts.
    """

    cuts = np.asarray(thresholds)
    flat, _, unique = _threshold_row_structure(cuts, axis=axis)
    moved_shape = np.moveaxis(cuts, axis, -1).shape
    result = (unique & (flat > 0.0)).reshape(moved_shape)
    return np.moveaxis(result, -1, axis)


def project_cdf_for_thresholds(
    values: np.ndarray,
    thresholds: np.ndarray,
    *,
    axis: int = 2,
) -> np.ndarray:
    """Project CDFs in the geometry of unique physical thresholds.

    For each finite row, values within every exact-equality threshold run are
    first averaged symmetrically. The q=0 group is forced to zero and excluded;
    distinct positive cuts then receive equal weight in an exact
    pool-adjacent-violators projection. The projected group values are expanded
    back to the nominal slots, so identical thresholds receive bit-identical
    CDF values. Tie-run size is never used as an isotonic weight.
    """

    array = np.asarray(values, dtype=np.float64)
    cuts = np.asarray(thresholds, dtype=np.float64)
    if array.shape != cuts.shape:
        raise PBCContractError("CDF values and thresholds must have identical shape")
    moved = np.moveaxis(array, axis, -1)
    flat = moved.reshape(-1, moved.shape[-1])
    flat_cuts, supported, _ = _threshold_row_structure(cuts, axis=axis)
    finite = np.isfinite(flat)
    if not np.array_equal(np.all(finite, axis=1), supported) or np.any(
        np.any(finite, axis=1) & ~np.all(finite, axis=1)
    ):
        raise PBCContractError(
            "CDF finite support must exactly match threshold finite support"
        )

    output = np.clip(flat, 0.0, 1.0)
    output[~supported] = np.nan
    duplicate = flat_cuts[:, 1:] == flat_cuts[:, :-1]
    adjacent_difference = np.diff(output, axis=1)
    needs_projection = supported & (
        np.any(adjacent_difference < 0.0, axis=1)
        | np.any(duplicate & (adjacent_difference != 0.0), axis=1)
        | np.any((flat_cuts == 0.0) & (output != 0.0), axis=1)
    )
    for row in np.flatnonzero(needs_projection):
        starts = np.concatenate(
            (
                np.asarray([0], dtype=np.int64),
                np.flatnonzero(flat_cuts[row, 1:] != flat_cuts[row, :-1]) + 1,
            )
        )
        stops = np.concatenate((starts[1:], np.asarray([flat.shape[1]])))
        collapsed = np.asarray(
            [np.mean(flat[row, start:stop]) for start, stop in zip(starts, stops)],
            dtype=np.float64,
        )
        group_thresholds = flat_cuts[row, starts]
        positive = group_thresholds > 0.0
        projected = np.zeros_like(collapsed)
        if np.any(positive):
            projected[positive] = np.clip(_pava_row(collapsed[positive]), 0.0, 1.0)
        for value, start, stop in zip(projected, starts, stops):
            output[row, start:stop] = value

    projected = output.reshape(moved.shape)
    return np.moveaxis(projected, -1, axis).astype(np.float32)


def is_valid_cdf_for_thresholds(
    values: np.ndarray,
    thresholds: np.ndarray,
    *,
    axis: int = 2,
    tolerance: float = 1e-7,
) -> bool:
    """Check probability, monotonicity, support, and exact threshold equality."""

    array = np.asarray(values, dtype=np.float64)
    cuts = np.asarray(thresholds, dtype=np.float64)
    if array.shape != cuts.shape or not is_valid_cdf(
        array, axis=axis, tolerance=tolerance
    ):
        return False
    try:
        flat_cuts, supported, _ = _threshold_row_structure(cuts, axis=axis)
    except PBCContractError:
        return False
    moved = np.moveaxis(array, axis, -1).reshape(flat_cuts.shape)
    finite = np.isfinite(moved)
    if not np.array_equal(np.all(finite, axis=1), supported):
        return False
    duplicate = flat_cuts[supported, 1:] == flat_cuts[supported, :-1]
    differences = np.diff(moved[supported], axis=1)
    zero_anchor = flat_cuts[supported] == 0.0
    return bool(
        np.all(differences[duplicate] == 0.0)
        and np.all(moved[supported][zero_anchor] == 0.0)
    )


def threshold_equality_diagnostics(
    before: np.ndarray,
    after: np.ndarray,
    thresholds: np.ndarray,
    *,
    axis: int = 2,
) -> Mapping[str, np.ndarray]:
    """Per-row duplicate incidence and exact-equality projection evidence."""

    raw = np.asarray(before, dtype=np.float64)
    projected = np.asarray(after, dtype=np.float64)
    cuts = np.asarray(thresholds, dtype=np.float64)
    if raw.shape != cuts.shape or projected.shape != cuts.shape:
        raise PBCContractError("tie diagnostic arrays must have identical shapes")
    flat_cuts, supported, unique = _threshold_row_structure(cuts, axis=axis)
    informative = unique & (flat_cuts > 0.0)
    raw_flat = np.moveaxis(raw, axis, -1).reshape(flat_cuts.shape)
    projected_flat = np.moveaxis(projected, axis, -1).reshape(flat_cuts.shape)
    for label, values in (("before", raw_flat), ("after", projected_flat)):
        finite = np.isfinite(values)
        if not np.array_equal(np.all(finite, axis=1), supported) or np.any(
            np.any(finite, axis=1) & ~np.all(finite, axis=1)
        ):
            raise PBCContractError(
                f"{label} CDF support differs from threshold support"
            )
    duplicate = np.zeros_like(flat_cuts, dtype=bool)
    duplicate[supported, 1:] = flat_cuts[supported, 1:] == flat_cuts[supported, :-1]
    raw_difference = np.zeros_like(flat_cuts)
    projected_difference = np.zeros_like(flat_cuts)
    raw_difference[:, 1:] = np.abs(np.diff(raw_flat, axis=1))
    projected_difference[:, 1:] = np.abs(np.diff(projected_flat, axis=1))
    zero_threshold = supported[:, None] & (flat_cuts == 0.0)

    output_shape = np.moveaxis(cuts, axis, -1).shape[:-1]
    results: dict[str, np.ndarray] = {}
    for name, values in (
        (
            "unique_physical_cut_count",
            np.sum(unique, axis=1, dtype=np.int16),
        ),
        (
            "informative_positive_cut_count",
            np.sum(informative, axis=1, dtype=np.int16),
        ),
        ("duplicate_cut_count", np.sum(duplicate, axis=1, dtype=np.int16)),
        (
            "zero_threshold_cut_count",
            np.sum(zero_threshold, axis=1, dtype=np.int16),
        ),
        (
            "pre_equality_violation_count",
            np.sum(duplicate & (raw_difference > 0.0), axis=1, dtype=np.int16),
        ),
        (
            "post_equality_violation_count",
            np.sum(
                duplicate & (projected_difference > 0.0),
                axis=1,
                dtype=np.int16,
            ),
        ),
        (
            "pre_maximum_absolute_equality_difference",
            np.max(np.where(duplicate, raw_difference, 0.0), axis=1),
        ),
        (
            "post_maximum_absolute_equality_difference",
            np.max(np.where(duplicate, projected_difference, 0.0), axis=1),
        ),
        (
            "pre_zero_anchor_violation_count",
            np.sum(zero_threshold & (raw_flat != 0.0), axis=1, dtype=np.int16),
        ),
        (
            "post_zero_anchor_violation_count",
            np.sum(
                zero_threshold & (projected_flat != 0.0),
                axis=1,
                dtype=np.int16,
            ),
        ),
    ):
        results[name] = np.asarray(values).reshape(output_shape)
    return results


@dataclass(frozen=True)
class DebiasFit:
    """Calendar-local additive probability-bias corrections."""

    correction: np.ndarray
    half_window_days: int
    fit_indices: np.ndarray
    sample_count_by_day_lead: np.ndarray


def fit_debias(
    raw_cdf: np.ndarray,
    observed_cdf: np.ndarray,
    initializations: np.ndarray,
    fit_indices: np.ndarray,
    support: np.ndarray,
    *,
    half_window_days: int,
    minimum_samples: int = 8,
) -> DebiasFit:
    """Fit Debias++'s additive ``mean(observed - forecast)`` correction."""

    raw = np.asarray(raw_cdf, dtype=np.float32)
    observed = np.asarray(observed_cdf, dtype=np.float32)
    mask = np.asarray(support, dtype=bool)
    if raw.ndim != 5 or observed.shape != raw.shape:
        raise PBCContractError("raw and observed CDFs must share [case,lead,cut,y,x]")
    if mask.shape != raw.shape[-2:]:
        raise PBCContractError("support does not match CDF grid")
    if half_window_days < 0 or half_window_days > 183 or minimum_samples < 1:
        raise PBCContractError("invalid Debias++ calendar-window settings")
    selected = _validate_indices(fit_indices, raw.shape[0], "fit_indices")
    positions = _calendar_positions(
        verification_midpoints(initializations, raw.shape[1])
    )
    correction = np.full((366, *raw.shape[1:]), np.nan, dtype=np.float32)
    counts = np.empty((366, raw.shape[1]), dtype=np.int32)
    for lead in range(raw.shape[1]):
        selected_positions = positions[selected, lead]
        selected_delta = observed[selected, lead] - raw[selected, lead]
        if not np.isfinite(selected_delta[..., mask]).all():
            raise PBCContractError("non-finite supported Debias++ training values")

        # Aggregate once by exact calendar position, then obtain every centered
        # circular window with two cumulative-sum lookups.  This is exactly the
        # same arithmetic as repeatedly selecting cases, but avoids a costly
        # full training-array scan for all 366 target days.
        daily_sum = np.zeros((366, *raw.shape[2:]), dtype=np.float64)
        daily_count = np.bincount(selected_positions, minlength=366).astype(np.int32)
        for day_with_data in np.flatnonzero(daily_count):
            daily_sum[day_with_data] = np.sum(
                selected_delta[selected_positions == day_with_data],
                axis=0,
                dtype=np.float64,
            )
        repeated_sum = np.concatenate((daily_sum, daily_sum, daily_sum), axis=0)
        repeated_count = np.tile(daily_count, 3)
        prefix_sum = np.concatenate(
            (
                np.zeros((1, *daily_sum.shape[1:]), dtype=np.float64),
                np.cumsum(repeated_sum, axis=0),
            ),
            axis=0,
        )
        prefix_count = np.concatenate(
            (np.zeros(1, dtype=np.int64), np.cumsum(repeated_count, dtype=np.int64))
        )
        centres = np.arange(366, dtype=np.int64) + 366
        left = centres - half_window_days
        right = centres + half_window_days + 1
        window_sum = prefix_sum[right] - prefix_sum[left]
        window_count = prefix_count[right] - prefix_count[left]
        for day in range(366):
            if window_count[day] >= minimum_samples:
                counts[day, lead] = int(window_count[day])
                correction[day, lead][:, mask] = (
                    window_sum[day][:, mask] / float(window_count[day])
                ).astype(np.float32)
                continue
            distance = np.minimum(
                (selected_positions - day) % 366,
                (day - selected_positions) % 366,
            )
            take = min(minimum_samples, selected.size)
            chosen_local = np.argsort(distance, kind="stable")[:take]
            counts[day, lead] = chosen_local.size
            correction[day, lead][:, mask] = np.mean(
                selected_delta[chosen_local][..., mask], axis=0, dtype=np.float64
            ).astype(np.float32)
    return DebiasFit(
        correction=correction,
        half_window_days=int(half_window_days),
        fit_indices=selected.copy(),
        sample_count_by_day_lead=counts,
    )


def apply_debias(
    raw_cdf: np.ndarray,
    initializations: np.ndarray,
    fit: DebiasFit,
    thresholds: np.ndarray,
    *,
    project: bool = False,
) -> np.ndarray:
    """Apply Debias++; optionally project over unique physical thresholds."""

    raw = np.asarray(raw_cdf, dtype=np.float32)
    if raw.ndim != 5 or fit.correction.shape[1:] != raw.shape[1:]:
        raise PBCContractError("Debias++ fit does not match input CDF shape")
    positions = _calendar_positions(
        verification_midpoints(initializations, raw.shape[1])
    )
    lead = np.arange(raw.shape[1], dtype=np.int64)[None, :]
    adjusted = np.clip(raw + fit.correction[positions, lead], 0.0, 1.0)
    return (
        project_cdf_for_thresholds(adjusted, thresholds, axis=2)
        if project
        else adjusted.astype(np.float32)
    )


@dataclass(frozen=True)
class IssueTimeLags:
    """Two exact completed weekly observations available at each issuance."""

    values: np.ndarray
    source_indices: np.ndarray
    window_start: np.ndarray
    window_end: np.ndarray

    @property
    def available(self) -> np.ndarray:
        return np.all(self.source_indices >= 0, axis=1)


def build_issue_time_lags(
    initializations: np.ndarray,
    weekly_truth: np.ndarray,
    *,
    lag_weeks: Sequence[int] = (1, 2),
) -> IssueTimeLags:
    """Build W1 truth lags at ``init-7`` and ``init-14``.

    A source W1 period beginning at ``init-7*j`` ends one day before the next
    seven-day boundary.  The explicit end-date check below makes it impossible
    for a current or future observation to enter an issue-time feature.
    """

    starts = np.asarray(initializations, dtype="datetime64[D]")
    truth = np.asarray(weekly_truth, dtype=np.float32)
    lags = tuple(int(value) for value in lag_weeks)
    if truth.ndim != 4 or starts.shape != (truth.shape[0],):
        raise PBCContractError("initializations and weekly truth are misaligned")
    if not lags or any(value < 1 for value in lags) or len(set(lags)) != len(lags):
        raise PBCContractError("lag_weeks must contain unique positive integers")
    if np.unique(starts).size != starts.size:
        raise PBCContractError("initializations must be unique")
    integer_starts = starts.astype(np.int64)
    lookup = {int(value): index for index, value in enumerate(integer_starts)}
    shape = (len(starts), len(lags), *truth.shape[-2:])
    values = np.full(shape, np.nan, dtype=np.float32)
    sources = np.full((len(starts), len(lags)), -1, dtype=np.int64)
    window_start = np.full(
        (len(starts), len(lags)), np.datetime64("NaT"), dtype="datetime64[D]"
    )
    window_end = window_start.copy()
    for case, issue in enumerate(starts):
        for column, lag in enumerate(lags):
            source_start = issue - np.timedelta64(7 * lag, "D")
            source_index = lookup.get(int(source_start.astype(np.int64)))
            if source_index is None:
                continue
            source_end = source_start + np.timedelta64(6, "D")
            if not source_end < issue:
                raise PBCContractError(
                    "lag observation is not complete before issuance"
                )
            values[case, column] = truth[source_index, 0]
            sources[case, column] = source_index
            window_start[case, column] = source_start
            window_end[case, column] = source_end
    available = sources >= 0
    if np.any(
        window_end[available]
        >= np.broadcast_to(starts[:, None], sources.shape)[available]
    ):
        raise PBCContractError(
            "one or more lag windows reaches its forecast issue time"
        )
    return IssueTimeLags(values, sources, window_start, window_end)


def build_daily_issue_time_lags(
    initializations: np.ndarray,
    daily_dates: np.ndarray,
    daily_values: np.ndarray,
    *,
    lag_weeks: Sequence[int] = (1, 2),
) -> IssueTimeLags:
    """Aggregate exact issue-time lags from a complete daily observation archive.

    Forecast initialization dates need not form a continuous weekly sequence.  In
    particular, the FuXi archive restarts its date cadence at year and leap-day
    boundaries.  Operational rainfall observations are nevertheless available on
    every calendar day, so lag week ``j`` is the seven-day mean over
    ``[issue - 7*j, issue - 7*j + 6]``.  Each such window ends strictly before the
    forecast is issued.

    ``source_indices`` stores the index of the first daily observation in each
    complete lag window.  It is ``-1`` only when the daily archive does not contain
    all seven requested days (for example, the first few 2002 training issues when
    the supplied observation archive begins on 2002-01-01).
    """

    starts = np.asarray(initializations, dtype="datetime64[D]")
    dates = np.asarray(daily_dates, dtype="datetime64[D]")
    observations = np.asarray(daily_values, dtype=np.float32)
    lags = tuple(int(value) for value in lag_weeks)
    if starts.ndim != 1 or starts.size == 0 or np.unique(starts).size != starts.size:
        raise PBCContractError("initializations must be a non-empty unique 1-D array")
    if dates.ndim != 1 or dates.size == 0 or observations.ndim != 3:
        raise PBCContractError(
            "daily observations must have shape [day,latitude,longitude]"
        )
    if observations.shape[0] != dates.size:
        raise PBCContractError("daily dates and observations are misaligned")
    if np.unique(dates).size != dates.size or np.any(
        np.diff(dates) <= np.timedelta64(0, "D")
    ):
        raise PBCContractError("daily observation dates must be unique and increasing")
    if not lags or any(value < 1 for value in lags) or len(set(lags)) != len(lags):
        raise PBCContractError("lag_weeks must contain unique positive integers")

    shape = (len(starts), len(lags), *observations.shape[-2:])
    values = np.full(shape, np.nan, dtype=np.float32)
    sources = np.full((len(starts), len(lags)), -1, dtype=np.int64)
    window_start = np.full(
        (len(starts), len(lags)), np.datetime64("NaT"), dtype="datetime64[D]"
    )
    window_end = window_start.copy()
    for case, issue in enumerate(starts):
        for column, lag in enumerate(lags):
            requested_start = issue - np.timedelta64(7 * lag, "D")
            requested_dates = requested_start + np.arange(7).astype("timedelta64[D]")
            position = int(np.searchsorted(dates, requested_start))
            stop = position + 7
            if stop > dates.size or not np.array_equal(
                dates[position:stop], requested_dates
            ):
                continue
            requested_end = requested_dates[-1]
            if not requested_end < issue:
                raise PBCContractError(
                    "daily lag window is not complete before issuance"
                )
            values[case, column] = np.mean(
                observations[position:stop], axis=0, dtype=np.float64
            ).astype(np.float32)
            sources[case, column] = position
            window_start[case, column] = requested_start
            window_end[case, column] = requested_end
    available = sources >= 0
    issue_grid = np.broadcast_to(starts[:, None], sources.shape)
    if np.any(window_end[available] >= issue_grid[available]):
        raise PBCContractError(
            "one or more daily lag windows reaches forecast issuance"
        )
    return IssueTimeLags(values, sources, window_start, window_end)


def lag_observation_cdf(
    lags: IssueTimeLags,
    model: CalendarQuantiles,
    case_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Convert lagged rainfall to indicators under each lag period's threshold."""

    selected = (
        np.arange(lags.values.shape[0], dtype=np.int64)
        if case_indices is None
        else _validate_indices(case_indices, lags.values.shape[0], "case_indices")
    )
    starts = lags.window_start[selected]
    amounts = lags.values[selected]
    sources = lags.source_indices[selected]
    positions = np.zeros(starts.shape, dtype=np.int16)
    available = sources >= 0
    if np.any(available):
        midpoint = starts[available] + np.timedelta64(3, "D")
        positions[available] = _calendar_positions(midpoint)
    thresholds = model.thresholds[positions]
    result = (amounts[:, :, None] < thresholds).astype(np.float32)
    result = np.where(available[:, :, None, None, None], result, np.nan)
    result[..., ~model.support] = np.nan
    return result


@dataclass(frozen=True)
class PersistenceFit:
    """Per-lead, per-cut, per-grid Persistence++ ridge coefficients."""

    coefficients: np.ndarray
    ridge: float
    fit_indices: np.ndarray
    usable_fit_indices: np.ndarray
    feature_names: tuple[str, ...] = (
        "intercept",
        "training_empirical_climatology_cdf",
        "lag_1week_indicator",
        "lag_2week_indicator",
        "raw_fuxi_cdf",
    )


def fit_persistence(
    raw_cdf: np.ndarray,
    observed_cdf: np.ndarray,
    climatology_cdf: np.ndarray,
    lag_cdf: np.ndarray,
    lag_available: np.ndarray,
    fit_indices: np.ndarray,
    support: np.ndarray,
    *,
    ridge: float = 1.0e-3,
) -> PersistenceFit:
    """Fit the frozen-split Persistence++ regression independently per cell."""

    raw = np.asarray(raw_cdf, dtype=np.float32)
    observed = np.asarray(observed_cdf, dtype=np.float32)
    climate = np.asarray(climatology_cdf, dtype=np.float32)
    lagged = np.asarray(lag_cdf, dtype=np.float32)
    available = np.asarray(lag_available, dtype=bool)
    mask = np.asarray(support, dtype=bool)
    if raw.ndim != 5 or observed.shape != raw.shape or climate.shape != raw.shape:
        raise PBCContractError("Persistence++ CDF arrays must have identical shapes")
    expected_lags = (raw.shape[0], 2, raw.shape[2], *raw.shape[-2:])
    if lagged.shape != expected_lags or available.shape != (raw.shape[0],):
        raise PBCContractError("Persistence++ lag arrays have incompatible shapes")
    if mask.shape != raw.shape[-2:] or ridge < 0.0 or not np.isfinite(ridge):
        raise PBCContractError("invalid Persistence++ support or ridge")
    selected = _validate_indices(fit_indices, raw.shape[0], "fit_indices")
    usable = selected[available[selected]]
    if usable.size < 6:
        raise PBCContractError(
            "Persistence++ needs at least six issue-time-complete cases"
        )
    locations = np.flatnonzero(mask.reshape(-1))
    coefficients = np.full((*raw.shape[1:3], *mask.shape, 5), np.nan, dtype=np.float32)
    regularizer = np.diag([1.0e-10, ridge, ridge, ridge, ridge]).astype(np.float64)
    for lead in range(raw.shape[1]):
        for cut in range(raw.shape[2]):
            feature_fields = np.stack(
                (
                    np.ones_like(raw[usable, lead, cut]),
                    climate[usable, lead, cut],
                    lagged[usable, 0, cut],
                    lagged[usable, 1, cut],
                    raw[usable, lead, cut],
                ),
                axis=-1,
            )
            x = feature_fields.reshape(usable.size, -1, 5)[:, locations]
            y = observed[usable, lead, cut].reshape(usable.size, -1)[:, locations]
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                raise PBCContractError(
                    "non-finite supported Persistence++ training values"
                )
            cell_x = np.moveaxis(x.astype(np.float64), 1, 0)
            cell_y = np.moveaxis(y.astype(np.float64), 1, 0)
            xtx = np.einsum("cnf,cng->cfg", cell_x, cell_x, optimize=True)
            xty = np.einsum("cnf,cn->cf", cell_x, cell_y, optimize=True)
            solved = np.linalg.solve(xtx + regularizer[None], xty[..., None])[..., 0]
            field = coefficients[lead, cut].reshape(-1, 5)
            field[locations] = solved.astype(np.float32)
    return PersistenceFit(
        coefficients=coefficients,
        ridge=float(ridge),
        fit_indices=selected.copy(),
        usable_fit_indices=usable.copy(),
    )


def apply_persistence(
    raw_cdf: np.ndarray,
    climatology_cdf: np.ndarray,
    lag_cdf: np.ndarray,
    lag_available: np.ndarray,
    fit: PersistenceFit,
    thresholds: np.ndarray,
    *,
    project: bool = False,
) -> np.ndarray:
    """Apply Persistence++, falling back to climatology if exact lags are absent."""

    raw = np.asarray(raw_cdf, dtype=np.float32)
    climate = np.asarray(climatology_cdf, dtype=np.float32)
    lagged = np.asarray(lag_cdf, dtype=np.float32).copy()
    available = np.asarray(lag_available, dtype=bool)
    if raw.shape != climate.shape or raw.ndim != 5:
        raise PBCContractError("Persistence++ prediction arrays are incompatible")
    if lagged.shape != (raw.shape[0], 2, raw.shape[2], *raw.shape[-2:]):
        raise PBCContractError("Persistence++ prediction lags have wrong shape")
    if (
        fit.coefficients.shape[:2] != raw.shape[1:3]
        or fit.coefficients.shape[2:4] != raw.shape[-2:]
    ):
        raise PBCContractError("Persistence++ coefficients do not match predictions")
    if available.shape != (raw.shape[0],):
        raise PBCContractError("lag availability does not match predictions")
    # The canonical held-out cases have both lags.  This fallback only makes a
    # sparse smoke cache well-defined and mirrors climatological imputation.
    lagged[~available, 0] = climate[~available, 0]
    lagged[~available, 1] = climate[~available, 0]
    output = np.full_like(raw, np.nan)
    for lead in range(raw.shape[1]):
        for cut in range(raw.shape[2]):
            features = np.stack(
                (
                    np.ones_like(raw[:, lead, cut]),
                    climate[:, lead, cut],
                    lagged[:, 0, cut],
                    lagged[:, 1, cut],
                    raw[:, lead, cut],
                ),
                axis=-1,
            )
            output[:, lead, cut] = np.einsum(
                "nhwf,hwf->nhw",
                features,
                fit.coefficients[lead, cut],
                optimize=True,
            )
    output = np.clip(output, 0.0, 1.0)
    return (
        project_cdf_for_thresholds(output, thresholds, axis=2)
        if project
        else output.astype(np.float32)
    )


def ranked_probability_score(
    cdf: np.ndarray,
    observed_cdf: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    """Mean squared CDF error over distinct physical thresholds.

    Each case/lead/grid row receives total cut weight one over its distinct
    positive thresholds, irrespective of how many nominal quantiles collapse.
    The deterministic zero threshold is excluded. The score is bounded in
    [0,1]; a row with no positive threshold is NaN and therefore unscored. Both
    CDF inputs must already assign exactly equal values to identical thresholds.
    """

    forecast = np.asarray(cdf, dtype=np.float64)
    observed = np.asarray(observed_cdf, dtype=np.float64)
    cuts = np.asarray(thresholds, dtype=np.float64)
    if (
        forecast.shape != observed.shape
        or forecast.shape != cuts.shape
        or forecast.ndim != 5
    ):
        raise PBCContractError(
            "RPS inputs and thresholds must share [case,lead,cut,y,x]"
        )
    if not is_valid_cdf_for_thresholds(forecast, cuts, axis=2):
        raise PBCContractError("forecast CDF violates the physical-threshold contract")
    if not is_valid_cdf_for_thresholds(observed, cuts, axis=2):
        raise PBCContractError("observed CDF violates the physical-threshold contract")
    unique = effective_cut_mask(cuts, axis=2)
    count = np.sum(unique, axis=2, dtype=np.int16)
    squared = (forecast - observed) ** 2
    numerator = np.sum(np.where(unique, squared, 0.0), axis=2)
    result = np.divide(
        numerator,
        count,
        out=np.full_like(numerator, np.nan, dtype=np.float64),
        where=count > 0,
    )
    return result


def paper_nominal_cut_rps(
    cdf: np.ndarray,
    observed_cdf: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    """Guan-methods sensitivity: summed nominal-cut error where q80 is positive.

    This deliberately preserves all four nominal quintile slots and their
    summed-error scale after excluding rows whose persisted final (q80)
    threshold is nonpositive. Forecast and observed CDFs must still satisfy
    the equality-aware physical-threshold contract. This is a secondary
    formula sensitivity, not the primary normalized effective-cut score.
    """

    forecast = np.asarray(cdf, dtype=np.float64)
    observed = np.asarray(observed_cdf, dtype=np.float64)
    cuts = np.asarray(thresholds, dtype=np.float64)
    if (
        forecast.shape != observed.shape
        or forecast.shape != cuts.shape
        or forecast.ndim != 5
        or forecast.shape[2] != 4
    ):
        raise PBCContractError(
            "paper nominal-cut RPS requires matching four-cut quintile CDFs"
        )
    if not is_valid_cdf_for_thresholds(forecast, cuts, axis=2):
        raise PBCContractError(
            "paper nominal-cut forecast violates the physical-threshold contract"
        )
    if not is_valid_cdf_for_thresholds(observed, cuts, axis=2):
        raise PBCContractError(
            "paper nominal-cut observation violates the physical-threshold contract"
        )
    score = np.sum((forecast - observed) ** 2, axis=2, dtype=np.float64)
    return np.where(cuts[:, :, -1] > 0.0, score, np.nan)


def paper_fixed_nominal_reference_rps(
    observed_cdf: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    """Literal k/5 denominator for the secondary Guan-formula sensitivity.

    The fixed ``[0.2, 0.4, 0.6, 0.8]`` vector is intentionally *not*
    projected through tied physical thresholds. It is only a conventional
    formula reference and is never represented as a valid forecast CDF.
    """

    observed = np.asarray(observed_cdf, dtype=np.float64)
    cuts = np.asarray(thresholds, dtype=np.float64)
    if observed.shape != cuts.shape or observed.ndim != 5 or observed.shape[2] != 4:
        raise PBCContractError(
            "paper fixed-nominal reference requires matching four-cut quintiles"
        )
    if not is_valid_cdf_for_thresholds(observed, cuts, axis=2):
        raise PBCContractError(
            "paper fixed-nominal observation violates the physical-threshold contract"
        )
    nominal = np.asarray([0.2, 0.4, 0.6, 0.8], dtype=np.float64).reshape(1, 1, 4, 1, 1)
    score = np.sum((nominal - observed) ** 2, axis=2, dtype=np.float64)
    return np.where(cuts[:, :, -1] > 0.0, score, np.nan)


def weighted_spatial_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Area-weight the final two grid dimensions, respecting finite values."""

    array = np.asarray(values, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    if array.ndim < 2 or weight.shape != array.shape[-2:]:
        raise PBCContractError("weights do not match the scored grid")
    expanded = np.broadcast_to(weight, array.shape)
    valid = np.isfinite(array) & np.isfinite(expanded) & (expanded > 0.0)
    numerator = np.sum(np.where(valid, array * expanded, 0.0), axis=(-2, -1))
    denominator = np.sum(np.where(valid, expanded, 0.0), axis=(-2, -1))
    if np.any(denominator <= 0.0):
        raise PBCContractError("one or more fields have no positive scoring weight")
    return numerator / denominator


def mean_rps_by_lead(
    cdf: np.ndarray,
    observed_cdf: np.ndarray,
    thresholds: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Mean area-weighted normalized effective-cut RPS for each lead."""

    case_scores = weighted_spatial_mean(
        ranked_probability_score(cdf, observed_cdf, thresholds), weights
    )
    return np.mean(case_scores, axis=0, dtype=np.float64)


def select_debias_spans(
    fits: Sequence[DebiasFit],
    raw_validation_cdf: np.ndarray,
    observed_validation_cdf: np.ndarray,
    validation_thresholds: np.ndarray,
    validation_initializations: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Select one Debias++ calendar span per lead by validation RPS."""

    if not fits:
        raise PBCContractError("at least one Debias++ fit is required")
    scores: dict[int, np.ndarray] = {}
    for fit in fits:
        prediction = apply_debias(
            raw_validation_cdf,
            validation_initializations,
            fit,
            validation_thresholds,
            project=True,
        )
        scores[fit.half_window_days] = mean_rps_by_lead(
            prediction,
            observed_validation_cdf,
            validation_thresholds,
            weights,
        )
    spans = np.asarray(sorted(scores), dtype=np.int16)
    matrix = np.stack([scores[int(span)] for span in spans], axis=0)
    selected = spans[np.argmin(matrix, axis=0)]
    return selected.astype(np.int16), scores


def apply_selected_debias(
    raw_cdf: np.ndarray,
    initializations: np.ndarray,
    fits: Sequence[DebiasFit],
    selected_spans: np.ndarray,
    thresholds: np.ndarray,
    *,
    project: bool = True,
) -> np.ndarray:
    """Apply the validation-selected Debias++ span independently by lead."""

    lookup = {fit.half_window_days: fit for fit in fits}
    selected = np.asarray(selected_spans, dtype=np.int64)
    if selected.shape != (raw_cdf.shape[1],) or any(
        int(span) not in lookup for span in selected
    ):
        raise PBCContractError("selected Debias++ spans do not cover every lead")
    result = np.empty_like(np.asarray(raw_cdf, dtype=np.float32))
    for lead, span in enumerate(selected):
        fit = lookup[int(span)]
        positions = _calendar_positions(
            verification_midpoints(initializations, raw_cdf.shape[1])[:, lead]
        )
        result[:, lead] = np.clip(
            raw_cdf[:, lead] + fit.correction[positions, lead], 0.0, 1.0
        )
    return project_cdf_for_thresholds(result, thresholds, axis=2) if project else result


def combine_projected_components(
    debias_cdf: np.ndarray,
    persistence_cdf: np.ndarray,
    thresholds: np.ndarray,
    *,
    debias_weight: float = 0.5,
) -> np.ndarray:
    """Convex-average two already valid component CDFs."""

    left = np.asarray(debias_cdf, dtype=np.float32)
    right = np.asarray(persistence_cdf, dtype=np.float32)
    if left.shape != right.shape or not 0.0 <= debias_weight <= 1.0:
        raise PBCContractError("component shapes or blend weight are invalid")
    if not is_valid_cdf_for_thresholds(
        left, thresholds, axis=2
    ) or not is_valid_cdf_for_thresholds(right, thresholds, axis=2):
        raise PBCContractError("PBC components must be projected before averaging")
    result = debias_weight * left + (1.0 - debias_weight) * right
    result = project_cdf_for_thresholds(result, thresholds, axis=2)
    if not is_valid_cdf_for_thresholds(result, thresholds, axis=2):
        raise PBCContractError(
            "convex component average unexpectedly violated CDF validity"
        )
    return result.astype(np.float32)


def probability_bias(
    cdf: np.ndarray, observed_cdf: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Area-weighted signed CDF bias by case, lead, and probability cut."""

    forecast = np.asarray(cdf, dtype=np.float64)
    observed = np.asarray(observed_cdf, dtype=np.float64)
    if forecast.shape != observed.shape:
        raise PBCContractError("probability-bias inputs have different shapes")
    return weighted_spatial_mean(forecast - observed, weights)


def upper_tail_brier_score(
    cdf: np.ndarray,
    observed_cdf: np.ndarray,
    thresholds: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Brier score above the final informative positive threshold.

    Rows/cells with ``q95 <= 0`` are deterministic always-events for
    nonnegative rainfall and are excluded. Spatial denominators are therefore
    recomputed from the same threshold-only mask for every forecast/reference;
    :func:`weighted_spatial_mean` hard-fails if any case/lead has zero weight.
    """

    forecast_cdf = np.asarray(cdf, dtype=np.float64)
    observed_full = np.asarray(observed_cdf, dtype=np.float64)
    cuts = np.asarray(thresholds, dtype=np.float64)
    if forecast_cdf.shape != observed_full.shape or cuts.shape != forecast_cdf.shape:
        raise PBCContractError("upper-tail CDFs and thresholds have different shapes")
    if not is_valid_cdf_for_thresholds(forecast_cdf, cuts, axis=2):
        raise PBCContractError("upper-tail forecast violates threshold CDF contract")
    if not is_valid_cdf_for_thresholds(observed_full, cuts, axis=2):
        raise PBCContractError("upper-tail observation violates threshold CDF contract")
    forecast = 1.0 - forecast_cdf[:, :, -1]
    observed = 1.0 - observed_full[:, :, -1]
    informative = cuts[:, :, -1] > 0.0
    squared = np.where(informative, (forecast - observed) ** 2, np.nan)
    return weighted_spatial_mean(squared, weights)


def cdf_projection_diagnostics(
    before: np.ndarray, after: np.ndarray
) -> Mapping[str, float]:
    """Summarize how often and how strongly isotonic projection changed CDFs."""

    raw = np.asarray(before, dtype=np.float64)
    projected = np.asarray(after, dtype=np.float64)
    if raw.shape != projected.shape:
        raise PBCContractError("projection diagnostic arrays differ in shape")
    moved = np.moveaxis(raw, 2, -1).reshape(-1, raw.shape[2])
    finite = np.all(np.isfinite(moved), axis=1)
    violations = np.any(np.diff(moved[finite], axis=1) < 0.0, axis=1)
    delta = np.abs(projected - raw)
    return {
        "finite_cdf_rows": float(np.count_nonzero(finite)),
        "violating_cdf_rows": float(np.count_nonzero(violations)),
        "violating_fraction": float(np.mean(violations)) if violations.size else 0.0,
        "mean_absolute_projection_change": float(np.nanmean(delta)),
        "maximum_absolute_projection_change": float(np.nanmax(delta)),
    }
