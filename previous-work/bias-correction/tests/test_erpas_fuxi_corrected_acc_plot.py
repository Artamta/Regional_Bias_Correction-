from __future__ import annotations

import numpy as np
import pandas as pd

from plot_erpas_fuxi_corrected_acc import (
    METHODS,
    circular_block_indices,
    collect_pairs,
    paired_bootstrap,
    valid_midpoint_is_jjas,
    weighted_spatial_acc,
    weighted_spatial_rmse,
)


def test_collect_pairs_requires_following_thursday() -> None:
    erpas = np.asarray(["2024-06-05", "2024-06-12"], dtype="datetime64[D]")
    fuxi = np.asarray(["2024-06-06"], dtype="datetime64[D]")
    assert collect_pairs(erpas, fuxi) == [(erpas[0], fuxi[0])]


def test_valid_midpoint_membership_is_lead_specific() -> None:
    init = np.datetime64("2024-05-23", "D")
    assert not valid_midpoint_is_jjas(init, 1)
    assert valid_midpoint_is_jjas(init, 2)


def test_weighted_spatial_acc_uses_common_normal() -> None:
    truth = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    forecast = truth * 2.0
    normal = np.zeros_like(truth)
    weights = np.ones_like(truth)
    acc, cells, area = weighted_spatial_acc(forecast, truth, normal, weights)
    assert np.isclose(acc, 1.0)
    assert cells == 4
    assert area == 4.0


def test_weighted_spatial_rmse_uses_area_weights() -> None:
    truth = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    forecast = np.asarray([[2.0, 2.0], [1.0, 4.0]])
    weights = np.asarray([[1.0, 0.0], [3.0, 2.0]])
    rmse, cells, area = weighted_spatial_rmse(forecast, truth, weights)
    assert np.isclose(rmse, np.sqrt(13.0 / 6.0))
    assert cells == 3
    assert area == 6.0


def test_circular_blocks_stay_in_range() -> None:
    result = circular_block_indices(7, 4, np.random.default_rng(4))
    assert result.shape == (7,)
    assert np.all((result >= 0) & (result < 7))


def test_bootstrap_preserves_paired_method_effect() -> None:
    rows = []
    for lead in range(1, 5):
        for year in (2023, 2024):
            for case in range(5):
                for method, value in zip(METHODS, (0.1, 0.2, 0.35), strict=True):
                    rows.append(
                        {
                            "year": year,
                            "erpas_init": f"{year}-06-{case + 1:02d}",
                            "fuxi_init": f"{year}-06-{case + 2:02d}",
                            "lead_week": lead,
                            "method": method,
                            "acc": value,
                        }
                    )
    absolute, effects = paired_bootstrap(
        pd.DataFrame(rows), draws=100, block_length=2, seed=7
    )
    assert len(absolute) == 12
    delta = effects[
        effects.comparison == "location_spread_minus_raw_fuxi"
    ].estimate.to_numpy()
    assert np.allclose(delta, 0.15)
