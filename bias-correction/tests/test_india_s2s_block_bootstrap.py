from __future__ import annotations

import numpy as np
import pytest

from india_s2s_block_bootstrap import (
    BootstrapContractError,
    circular_year_stratified_index_matrix,
    circular_year_stratified_indices,
    paired_year_stratified_circular_block_bootstrap,
)


def test_circular_resample_preserves_every_year_count_and_pair_index() -> None:
    dates = np.asarray(
        [
            "2020-06-01",
            "2020-06-04",
            "2020-06-08",
            "2021-06-03",
            "2021-06-07",
        ],
        dtype="datetime64[D]",
    )
    selected = circular_year_stratified_indices(
        dates, block_length=2, rng=np.random.default_rng(7)
    )
    sampled_years = dates[selected].astype("datetime64[Y]")
    assert selected.shape == (5,)
    assert np.count_nonzero(sampled_years == np.datetime64("2020")) == 3
    assert np.count_nonzero(sampled_years == np.datetime64("2021")) == 2
    assert np.all((selected >= 0) & (selected < dates.size))

    matrix = circular_year_stratified_index_matrix(
        dates, block_length=2, replicates=17, seed=7
    )
    assert matrix.shape == (17, 5)
    for row in matrix:
        row_years = dates[row].astype("datetime64[Y]")
        assert np.count_nonzero(row_years == np.datetime64("2020")) == 3
        assert np.count_nonzero(row_years == np.datetime64("2021")) == 2


def test_paired_bootstrap_constant_difference_is_exact() -> None:
    dates = np.arange(
        np.datetime64("2020-01-01"), np.datetime64("2022-01-01"), np.timedelta64(7, "D")
    )
    reference = np.linspace(-1.0, 1.0, dates.size)
    candidate = reference + 0.25
    result, samples = paired_year_stratified_circular_block_bootstrap(
        dates,
        candidate,
        reference,
        block_length=16,
        replicates=200,
        seed=11,
    )
    assert result.estimate == pytest.approx(0.25)
    assert result.lower == pytest.approx(0.25)
    assert result.upper == pytest.approx(0.25)
    assert np.allclose(samples, 0.25)
    assert sum(result.year_counts.values()) == dates.size


def test_pairing_changes_the_variance_relative_to_unpaired_values() -> None:
    dates = np.arange(
        np.datetime64("2020-01-01"), np.datetime64("2023-01-01"), np.timedelta64(7, "D")
    )
    shared = np.sin(np.arange(dates.size, dtype=np.float64) / 3.0)
    candidate = shared + 0.1
    reference = shared
    result, samples = paired_year_stratified_circular_block_bootstrap(
        dates, candidate, reference, replicates=100, block_length=13
    )
    assert result.estimate == pytest.approx(0.1)
    assert samples.std() < 1.0e-12


@pytest.mark.parametrize(
    ("dates", "left", "right"),
    [
        (["2020-01-01"], [1.0, 2.0], [1.0]),
        (["2020-01-01", "2020-01-01"], [1.0, 2.0], [1.0, 2.0]),
        (["2020-01-01"], [np.nan], [1.0]),
    ],
)
def test_invalid_pair_contract_is_rejected(dates, left, right) -> None:
    with pytest.raises(BootstrapContractError):
        paired_year_stratified_circular_block_bootstrap(
            np.asarray(dates, dtype="datetime64[D]"),
            np.asarray(left),
            np.asarray(right),
            replicates=2,
        )


def test_invalid_bootstrap_controls_are_rejected() -> None:
    dates = np.asarray(["2020-01-01"], dtype="datetime64[D]")
    values = np.asarray([1.0])
    with pytest.raises(BootstrapContractError):
        paired_year_stratified_circular_block_bootstrap(
            dates, values, values, block_length=0
        )
    with pytest.raises(BootstrapContractError):
        paired_year_stratified_circular_block_bootstrap(
            dates, values, values, replicates=0
        )
