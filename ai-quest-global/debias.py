"""Training-only Debias++ and validation-only uniform-blend calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _pava_1d(values: np.ndarray) -> np.ndarray:
    """Least-squares nondecreasing projection for one short vector."""

    values = np.asarray(values, dtype=np.float64)
    levels: list[float] = []
    weights: list[int] = []
    starts: list[int] = []
    for index, value in enumerate(values):
        levels.append(float(value))
        weights.append(1)
        starts.append(index)
        while len(levels) >= 2 and levels[-2] > levels[-1]:
            weight = weights[-2] + weights[-1]
            level = (
                levels[-2] * weights[-2] + levels[-1] * weights[-1]
            ) / weight
            levels[-2:] = [level]
            weights[-2:] = [weight]
            starts[-2:] = [starts[-2]]
    output = np.empty_like(values)
    for block, start in enumerate(starts):
        end = starts[block + 1] if block + 1 < len(starts) else len(values)
        output[start:end] = levels[block]
    return output


def isotonic_cdf(cdf: np.ndarray) -> np.ndarray:
    """Project the leading four-boundary axis onto valid nondecreasing CDFs."""

    values = np.asarray(cdf, dtype=np.float64)
    if values.shape[0] != 4:
        raise ValueError("CDF boundary axis must be the leading size-four axis")
    flat = values.reshape(4, -1)
    # Exact min-max characterization of one-dimensional L2 isotonic
    # regression. With four boundaries, enumerating the ten relevant interval
    # means is both clearer and orders of magnitude faster than a Python PAVA
    # loop over millions of global grid columns.
    prefix = np.concatenate((np.zeros((1, flat.shape[1])), np.cumsum(flat, axis=0)))
    projected = np.empty_like(flat)
    for index in range(4):
        lower_candidates = []
        for start in range(index + 1):
            upper_candidates = [
                (prefix[end + 1] - prefix[start]) / (end - start + 1)
                for end in range(index, 4)
            ]
            lower_candidates.append(np.minimum.reduce(upper_candidates))
        projected[index] = np.maximum.reduce(lower_candidates)
    return projected.reshape(values.shape).astype(np.float32)


def probabilities_to_cdf(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float32)
    if values.shape[-3] != 5:
        raise ValueError("probability category axis must be size five at position -3")
    return np.cumsum(values, axis=-3)[..., :4, :, :]


def cdf_to_probabilities(cdf: np.ndarray) -> np.ndarray:
    values = np.asarray(cdf, dtype=np.float32)
    if values.shape[-3] != 4:
        raise ValueError("CDF boundary axis must be size four at position -3")
    zeros = np.zeros((*values.shape[:-3], 1, *values.shape[-2:]), dtype=np.float32)
    ones = np.ones_like(zeros)
    result = np.diff(np.concatenate((zeros, values, ones), axis=-3), axis=-3)
    if np.any(result < -1.0e-6):
        raise RuntimeError("CDF is not nondecreasing")
    result = np.clip(result, 0.0, None)
    result /= result.sum(axis=-3, keepdims=True)
    return result.astype(np.float32)


@dataclass(frozen=True)
class DebiasPlusPlus:
    """Additive training-CDF bias followed by clipping and isotonic projection."""

    cdf_correction: np.ndarray  # [lead, boundary, lat, lon]
    clip: float = 1.0e-6

    @classmethod
    def fit(
        cls,
        train_probabilities: np.ndarray,
        train_targets: np.ndarray,
        train_valid: np.ndarray,
        *,
        clip: float = 1.0e-6,
    ) -> "DebiasPlusPlus":
        probability = np.asarray(train_probabilities, dtype=np.float32)
        target = np.asarray(train_targets)
        valid = np.asarray(train_valid, dtype=bool)
        if probability.ndim != 5 or probability.shape[1:3] != (2, 5):
            raise ValueError("train_probabilities must be [case,2,5,lat,lon]")
        if target.shape != (probability.shape[0], 2, *probability.shape[-2:]):
            raise ValueError("train_targets have incompatible shape")
        if valid.shape != target.shape:
            raise ValueError("train_valid must match train_targets")
        forecast_cdf = np.cumsum(probability, axis=2)[:, :, :4]
        observed_cdf = np.stack(
            [(target <= boundary) for boundary in range(4)], axis=2
        ).astype(np.float32)
        mask = valid[:, :, None]
        count = mask.sum(axis=0)
        if np.any(count == 0):
            raise ValueError("Debias++ has a lead/boundary/grid cell without training truth")
        correction = np.where(
            mask,
            observed_cdf - forecast_cdf,
            0.0,
        ).sum(axis=0) / count
        return cls(correction.astype(np.float32), float(clip))

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        values = np.asarray(probabilities, dtype=np.float32)
        if values.shape[-4:-2] != (2, 5):
            raise ValueError("probabilities must end with [lead=2,category=5,lat,lon]")
        raw_cdf = np.cumsum(values, axis=-3)[..., :4, :, :]
        corrected = np.clip(
            raw_cdf + self.cdf_correction,
            self.clip,
            1.0 - self.clip,
        )
        # Move the four-boundary axis first for the projection and restore it.
        boundary_axis = corrected.ndim - 3
        projected = np.moveaxis(
            isotonic_cdf(np.moveaxis(corrected, boundary_axis, 0)), 0, boundary_axis
        )
        return cdf_to_probabilities(projected)


@dataclass(frozen=True)
class SeasonalDebiasPlusPlus:
    """Paper-derived Debias++ with cyclic day-of-year windows.

    Sufficient statistics are fitted on training only. Validation chooses one
    of the paper's declared spans (14, 28, or 35 days) independently per lead.
    """

    error_sum: np.ndarray  # [366,2,4,lat,lon]
    valid_count: np.ndarray  # [366,2,1,lat,lon]
    clip: float = 1.0e-6

    @classmethod
    def empty(cls, height: int = 121, width: int = 240) -> "SeasonalDebiasPlusPlus":
        return cls(
            np.zeros((366, 2, 4, height, width), dtype=np.float32),
            np.zeros((366, 2, 1, height, width), dtype=np.uint16),
        )

    def add_case(
        self,
        probabilities: np.ndarray,
        target: np.ndarray,
        valid: np.ndarray,
        target_doy: np.ndarray,
    ) -> None:
        probability = np.asarray(probabilities, dtype=np.float32)
        target = np.asarray(target)
        valid = np.asarray(valid, dtype=bool)
        doys = np.asarray(target_doy, dtype=np.int16)
        if probability.shape[:2] != (2, 5) or target.shape != probability.shape[:1] + probability.shape[-2:]:
            raise ValueError("one case must contain [2,5,H,W] probabilities and [2,H,W] targets")
        if valid.shape != target.shape or doys.shape != (2,):
            raise ValueError("valid and target-day-of-year shapes are invalid")
        forecast_cdf = np.cumsum(probability, axis=1)[:, :4]
        observed_cdf = np.stack(
            [(target <= boundary) for boundary in range(4)], axis=1
        ).astype(np.float32)
        for lead in range(2):
            day = int(doys[lead])
            if not 0 <= day < 366:
                raise ValueError("target day of year must be zero-based in [0,365]")
            mask = valid[lead][None]
            self.error_sum[day, lead] += np.where(
                mask, observed_cdf[lead] - forecast_cdf[lead], 0.0
            )
            updated = self.valid_count[day, lead].astype(np.uint32) + mask
            if np.any(updated > np.iinfo(np.uint16).max):
                raise OverflowError("seasonal Debias++ valid count overflow")
            self.valid_count[day, lead] = updated.astype(np.uint16)

    @staticmethod
    def _window_days(day: int, span: int) -> np.ndarray:
        if span not in {14, 28, 35}:
            raise ValueError("Debias++ span must be 14, 28, or 35 days")
        return (day + np.arange(-span, span + 1)) % 366

    def correction(self, day: int, lead: int, span: int) -> np.ndarray:
        days = self._window_days(int(day), int(span))
        numerator = self.error_sum[days, lead].sum(axis=0, dtype=np.float64)
        denominator = self.valid_count[days, lead].sum(axis=0, dtype=np.uint32)
        return np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 0,
        ).astype(np.float32)

    def predict(
        self,
        probabilities: np.ndarray,
        target_doy: np.ndarray,
        span_by_lead: np.ndarray,
    ) -> np.ndarray:
        values = np.asarray(probabilities, dtype=np.float32)
        doys = np.asarray(target_doy, dtype=np.int16)
        spans = np.asarray(span_by_lead, dtype=np.int16)
        if values.ndim != 5 or values.shape[1:3] != (2, 5):
            raise ValueError("probabilities must be [case,2,5,lat,lon]")
        if doys.shape != (values.shape[0], 2) or spans.shape != (2,):
            raise ValueError("target_doy must be [case,2] and spans must be [2]")
        raw_cdf = np.cumsum(values, axis=2)[:, :, :4]
        corrected = np.empty_like(raw_cdf)
        correction_cache: dict[tuple[int, int, int], np.ndarray] = {}
        for case in range(values.shape[0]):
            for lead in range(2):
                key = (int(doys[case, lead]), lead, int(spans[lead]))
                if key not in correction_cache:
                    correction_cache[key] = self.correction(*key)
                corrected[case, lead] = np.clip(
                    raw_cdf[case, lead]
                    + correction_cache[key],
                    self.clip,
                    1.0 - self.clip,
                )
        boundary_axis = corrected.ndim - 3
        projected = np.moveaxis(
            isotonic_cdf(np.moveaxis(corrected, boundary_axis, 0)), 0, boundary_axis
        )
        return cdf_to_probabilities(projected)


def _rps_components(
    probabilities: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    spatial_weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probability = np.asarray(probabilities, dtype=np.float64)
    target = np.asarray(target)
    valid = np.asarray(valid, dtype=bool)
    weight = np.asarray(spatial_weight, dtype=np.float64)
    if probability.ndim != 5 or probability.shape[1:3] != (2, 5):
        raise ValueError("probabilities must be [case,2,5,lat,lon]")
    if weight.shape != probability.shape[-2:]:
        raise ValueError("spatial_weight must match the forecast grid")
    forecast_cdf = np.cumsum(probability, axis=2)[:, :, :4]
    observation_cdf = np.stack(
        [(target <= boundary) for boundary in range(4)], axis=2
    ).astype(np.float64)
    delta = forecast_cdf - observation_cdf
    full_weight = valid[:, :, None] * weight[None, None, None]
    return delta, full_weight, full_weight.sum(axis=(0, 2, 3, 4))


def fit_uniform_gamma(
    probabilities: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    spatial_weight: np.ndarray,
) -> np.ndarray:
    """Fit one closed-form validation RPS-optimal model weight per lead."""

    model_delta, weight, denominators = _rps_components(
        probabilities, target, valid, spatial_weight
    )
    uniform = np.full_like(probabilities, 0.2)
    uniform_delta, _, _ = _rps_components(uniform, target, valid, spatial_weight)
    direction = model_delta - uniform_delta
    numerator = -np.sum(weight * uniform_delta * direction, axis=(0, 2, 3, 4))
    denominator = np.sum(weight * direction * direction, axis=(0, 2, 3, 4))
    gamma = np.divide(
        numerator,
        denominator,
        out=np.zeros(2, dtype=np.float64),
        where=(denominator > 0) & (denominators > 0),
    )
    return np.clip(gamma, 0.0, 1.0).astype(np.float32)


def apply_uniform_gamma(probabilities: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float32)
    coefficients = np.asarray(gamma, dtype=np.float32)
    if values.shape[-4] != 2 or coefficients.shape != (2,):
        raise ValueError("probabilities need two leads and gamma must have shape [2]")
    shape = [1] * values.ndim
    shape[-4] = 2
    coefficient = coefficients.reshape(shape)
    result = coefficient * values + (1.0 - coefficient) * np.float32(0.2)
    return result.astype(np.float32)


__all__ = [
    "DebiasPlusPlus",
    "SeasonalDebiasPlusPlus",
    "apply_uniform_gamma",
    "cdf_to_probabilities",
    "fit_uniform_gamma",
    "isotonic_cdf",
    "probabilities_to_cdf",
]
