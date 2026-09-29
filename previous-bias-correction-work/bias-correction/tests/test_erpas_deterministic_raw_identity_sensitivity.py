from __future__ import annotations

import numpy as np
import pandas as pd

from build_erpas_deterministic_raw_identity_sensitivity import (
    BOOTSTRAP_DRAWS,
    EXPECTED_SHARED_DATES,
    LEADS,
    METHODS,
    make_bootstrap_indices,
    strict_shared_dates,
    summarize_metric,
)


def test_strict_shared_dates_intersects_every_lead_and_source() -> None:
    common = ("2023-06-01", "2024-06-20")
    rows = []
    for lead in LEADS:
        for value in (*common, f"2023-05-{lead:02d}"):
            rows.append({"lead_week": lead, "fuxi_init": value})
    deterministic = np.asarray((*common, "2023-07-01"), dtype="datetime64[D]")
    result = strict_shared_dates(pd.DataFrame(rows), deterministic)
    assert result == tuple(np.asarray(common, dtype="datetime64[D]"))


def test_bootstrap_is_deterministic_paired_and_year_stratified() -> None:
    dates = tuple(np.asarray(EXPECTED_SHARED_DATES, dtype="datetime64[D]"))
    first = make_bootstrap_indices(dates)
    second = make_bootstrap_indices(dates)
    assert np.array_equal(first, second)
    assert first.shape == (len(LEADS), BOOTSTRAP_DRAWS, 26)
    assert np.all((first[:, :, :14] >= 0) & (first[:, :, :14] < 14))
    assert np.all((first[:, :, 14:] >= 14) & (first[:, :, 14:] < 26))


def test_summaries_keep_raw_identity_effect_direction() -> None:
    rows = []
    dates = tuple(np.asarray(EXPECTED_SHARED_DATES, dtype="datetime64[D]"))
    for lead in LEADS:
        for fuxi_init in dates:
            year = pd.Timestamp(fuxi_init).year
            erpas_init = fuxi_init - np.timedelta64(1, "D")
            for method, acc, rmse in zip(
                METHODS,
                (0.10, 0.20, 0.35),
                (7.0, 6.0, 5.5),
                strict=True,
            ):
                rows.append(
                    {
                        "year": year,
                        "erpas_init": np.datetime_as_string(erpas_init, unit="D"),
                        "fuxi_init": np.datetime_as_string(fuxi_init, unit="D"),
                        "lead_week": lead,
                        "method": method,
                        "acc": acc,
                        "rmse_mm_day": rmse,
                    }
                )
    metrics = pd.DataFrame(rows)
    indices = make_bootstrap_indices(dates)
    acc_absolute, acc_effects = summarize_metric(
        metrics, indices, value_column="acc"
    )
    rmse_absolute, rmse_effects = summarize_metric(
        metrics, indices, value_column="rmse_mm_day"
    )
    assert len(acc_absolute) == len(rmse_absolute) == 12
    assert len(acc_effects) == len(rmse_effects) == 8
    raw_acc = acc_effects.loc[acc_effects.second_method == "raw_fuxi"]
    raw_rmse = rmse_effects.loc[rmse_effects.second_method == "raw_fuxi"]
    assert np.allclose(raw_acc.estimate, 0.15)
    assert np.allclose(raw_rmse.estimate, -0.5)
    assert acc_effects.resolved_favorable.all()
    assert rmse_effects.resolved_favorable.all()
