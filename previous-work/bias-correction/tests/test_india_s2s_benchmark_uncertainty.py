from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from india_s2s_benchmark_uncertainty import (
    BenchmarkUncertaintyError,
    compute_intervals,
)


def _synthetic_table() -> pd.DataFrame:
    dates = pd.date_range("2020-06-01", periods=169, freq="3D")
    rows = []
    for lead in range(1, 7):
        for date in dates:
            for model, offset in (("fuxi_s2s", 0.0), ("mme", 0.25)):
                rows.append(
                    {
                        "reference": "imd",
                        "season": "JJAS",
                        "region": "all_india",
                        "lead_week": lead,
                        "model": model,
                        "score_status": "available",
                        "init": date.strftime("%Y-%m-%d"),
                        "acc": 0.1 + offset,
                        "rmse": 4.0 + offset,
                        "mae": 3.0 + offset,
                        "bias": -0.2 + offset,
                    }
                )
    return pd.DataFrame(rows)


def test_compute_intervals_keeps_169_paired_rows_and_both_block_lengths() -> None:
    result, distributions = compute_intervals(
        _synthetic_table(), replicates=20, seed=3
    )
    assert len(result) == 6 * 4 * 2
    assert set(result["n_cases"]) == {169}
    assert set(result["block_length_starts"]) == {13, 16}
    assert np.allclose(result["estimate"], 0.25)
    assert np.allclose(result["ci_lower"], 0.25)
    assert np.allclose(result["ci_upper"], 0.25)
    assert len(distributions) == len(result)


def test_compute_intervals_rejects_unpaired_case() -> None:
    table = _synthetic_table().iloc[:-1]
    with pytest.raises(BenchmarkUncertaintyError, match="169 paired"):
        compute_intervals(table, replicates=2)
