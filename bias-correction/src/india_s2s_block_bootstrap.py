#!/usr/bin/env python3
"""Paired, year-stratified circular block uncertainty for India S2S scores.

The benchmark resamples initialization starts, never individual grid cells or
lead/region rows.  Both forecasts in a comparison receive the same sampled
starts, preserving their paired design.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class BootstrapContractError(ValueError):
    """Raised when a requested resample would violate the paired contract."""


@dataclass(frozen=True)
class PairedBootstrapResult:
    estimate: float
    lower: float
    upper: float
    confidence: float
    block_length: int
    replicates: int
    seed: int
    paired_case_count: int
    year_counts: dict[int, int]

    def as_dict(self) -> dict[str, object]:
        return {
            "estimate": self.estimate,
            "lower": self.lower,
            "upper": self.upper,
            "confidence": self.confidence,
            "block_length": self.block_length,
            "replicates": self.replicates,
            "seed": self.seed,
            "paired_case_count": self.paired_case_count,
            "year_counts": self.year_counts,
        }


def _validated_inputs(
    initializations: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dates = np.asarray(initializations, dtype="datetime64[D]")
    left = np.asarray(candidate, dtype=np.float64)
    right = np.asarray(reference, dtype=np.float64)
    if dates.ndim != 1 or left.ndim != 1 or right.ndim != 1:
        raise BootstrapContractError("dates and paired values must be one-dimensional")
    if not (dates.size == left.size == right.size):
        raise BootstrapContractError("dates and paired values must have equal lengths")
    if dates.size == 0:
        raise BootstrapContractError("at least one paired case is required")
    if np.unique(dates).size != dates.size:
        raise BootstrapContractError("initialization dates must be unique")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise BootstrapContractError("paired values must be finite")
    order = np.argsort(dates)
    dates = dates[order]
    left = left[order]
    right = right[order]
    years = pd.DatetimeIndex(dates).year.to_numpy(dtype=np.int64)
    return dates, left, right, years


def circular_year_stratified_indices(
    initializations: np.ndarray,
    *,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw one bootstrap sample while retaining each year's case count.

    Dates are sorted within year.  Uniformly selected circular blocks are
    concatenated and truncated to that year's original number of starts.
    Returned indices refer to the original input order.
    """

    dates = np.asarray(initializations, dtype="datetime64[D]")
    if dates.ndim != 1 or dates.size == 0:
        raise BootstrapContractError("initializations must be a non-empty 1-D array")
    if np.unique(dates).size != dates.size:
        raise BootstrapContractError("initialization dates must be unique")
    if block_length < 1:
        raise BootstrapContractError("block_length must be positive")
    years = pd.DatetimeIndex(dates).year.to_numpy(dtype=np.int64)
    sampled: list[np.ndarray] = []
    for year in np.unique(years):
        original = np.flatnonzero(years == year)
        original = original[np.argsort(dates[original])]
        count = original.size
        blocks_needed = (count + block_length - 1) // block_length
        starts = rng.integers(0, count, size=blocks_needed)
        offsets = np.arange(block_length, dtype=np.int64)
        within_year = ((starts[:, None] + offsets[None, :]) % count).reshape(-1)
        sampled.append(original[within_year[:count]])
    return np.concatenate(sampled)


def circular_year_stratified_index_matrix(
    initializations: np.ndarray,
    *,
    block_length: int,
    replicates: int,
    seed: int,
) -> np.ndarray:
    """Vectorized bootstrap index matrix with shape ``[replicate, case]``."""

    dates = np.asarray(initializations, dtype="datetime64[D]")
    if dates.ndim != 1 or dates.size == 0:
        raise BootstrapContractError("initializations must be a non-empty 1-D array")
    if np.unique(dates).size != dates.size:
        raise BootstrapContractError("initialization dates must be unique")
    if block_length < 1:
        raise BootstrapContractError("block_length must be positive")
    if replicates < 1:
        raise BootstrapContractError("replicates must be positive")
    years = pd.DatetimeIndex(dates).year.to_numpy(dtype=np.int64)
    random = np.random.default_rng(seed)
    sampled: list[np.ndarray] = []
    offsets = np.arange(block_length, dtype=np.int64)
    for year in np.unique(years):
        original = np.flatnonzero(years == year)
        original = original[np.argsort(dates[original])]
        count = original.size
        blocks_needed = (count + block_length - 1) // block_length
        starts = random.integers(0, count, size=(replicates, blocks_needed))
        within_year = (
            (starts[:, :, None] + offsets[None, None, :]) % count
        ).reshape(replicates, -1)[:, :count]
        sampled.append(original[within_year])
    return np.concatenate(sampled, axis=1)


def paired_year_stratified_circular_block_bootstrap(
    initializations: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    block_length: int = 16,
    replicates: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[PairedBootstrapResult, np.ndarray]:
    """Bootstrap the arithmetic mean of ``candidate - reference``.

    The second return value contains all replicate differences, allowing the
    caller to archive the exact distribution or derive one-sided gates.
    """

    dates, left, right, years = _validated_inputs(
        initializations, candidate, reference
    )
    if block_length < 1:
        raise BootstrapContractError("block_length must be positive")
    if replicates < 1:
        raise BootstrapContractError("replicates must be positive")
    if not 0.0 < confidence < 1.0:
        raise BootstrapContractError("confidence must lie strictly between zero and one")
    differences = left - right
    selected = circular_year_stratified_index_matrix(
        dates,
        block_length=block_length,
        replicates=replicates,
        seed=seed,
    )
    samples = differences[selected].mean(axis=1, dtype=np.float64)
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(samples, (alpha, 1.0 - alpha))
    year_counts = {
        int(year): int(np.count_nonzero(years == year)) for year in np.unique(years)
    }
    result = PairedBootstrapResult(
        estimate=float(differences.mean(dtype=np.float64)),
        lower=float(lower),
        upper=float(upper),
        confidence=float(confidence),
        block_length=int(block_length),
        replicates=int(replicates),
        seed=int(seed),
        paired_case_count=int(dates.size),
        year_counts=year_counts,
    )
    return result, samples


__all__ = [
    "BootstrapContractError",
    "PairedBootstrapResult",
    "circular_year_stratified_index_matrix",
    "circular_year_stratified_indices",
    "paired_year_stratified_circular_block_bootstrap",
]
