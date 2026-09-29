from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "evaluate"
    / "plot_physical_validation_results.py"
)
SPEC = importlib.util.spec_from_file_location("plot_physical_validation_results", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
plots = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plots)


def test_india_boundary_loader_requires_checked_official_provenance(
    tmp_path: Path,
) -> None:
    ring = [[70.0, 10.0], [71.0, 10.0], [71.0, 11.0], [70.0, 10.0]]
    payload = {
        "type": "FeatureCollection",
        "source": {"name": "Survey of India synthetic ABDB fixture"},
        "features": [
            {
                "type": "Feature",
                "properties": {"name": f"region-{index}"},
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            }
            for index in range(30)
        ],
    }
    path = tmp_path / "india.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    segments, provenance = plots.load_india_boundary(path)

    assert len(segments) == 30
    assert provenance["feature_count"] == 30
    assert provenance["ring_count"] == 30
    assert provenance["boundary_path"] == str(path.resolve())
    payload["source"] = {"name": "unverified"}
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Survey of India"):
        plots.load_india_boundary(path)


def test_select_best_physical_candidate_uses_rank_and_completed_residuals() -> None:
    ranking = pd.DataFrame(
        {
            "configuration": [
                "physical_control",
                "physical_tcwv",
                "physical_moisture_circulation",
                "physical_full_compact",
            ],
            "rank": [3, 2, 1, 4],
        }
    )
    # The rank-1 candidate is ignored because its ensemble residual is absent.
    selected = plots.select_best_physical_candidate(
        ranking,
        ["physical_control", "physical_tcwv", "physical_full_compact"],
    )
    assert selected == "physical_tcwv"
    assert (
        plots.select_best_physical_candidate(
            ranking,
            ["physical_control", "physical_tcwv", "physical_full_compact"],
            explicit_candidate="physical_full_compact",
        )
        == "physical_full_compact"
    )
    with pytest.raises(ValueError, match="cannot be the control"):
        plots.select_best_physical_candidate(
            ranking,
            ["physical_control", "physical_tcwv"],
            explicit_candidate="physical_control",
        )


def test_spatial_bias_skill_and_case_rmse_math() -> None:
    truth = np.zeros((2, 2, 2, 2), dtype=np.float32)
    control = np.full_like(truth, 2.0)
    candidate = np.full_like(truth, 1.0)
    weights = np.asarray([[1.0, 3.0], [0.0, 2.0]])
    candidate[..., 1, 1] = 2.0

    bias = plots.spatial_mean_bias(candidate, truth, weights)
    assert bias.shape == (2, 2, 2)
    assert np.allclose(bias[:, 0, 0], 1.0)
    assert np.isnan(bias[:, 1, 0]).all()

    skill, candidate_rmse, control_rmse = plots.spatial_rmse_skill_vs_control(
        candidate, control, truth, weights
    )
    assert np.allclose(skill[:, 0, 0], 50.0)
    assert np.allclose(skill[:, 1, 1], 0.0)
    assert np.isnan(skill[:, 1, 0]).all()
    assert np.allclose(candidate_rmse[:, 0, 0], 1.0)
    assert np.allclose(control_rmse[:, 0, 0], 2.0)

    case_rmse = plots.area_weighted_case_rmse(candidate, truth, weights)
    expected = np.sqrt((1.0 + 3.0 + 2.0 * 4.0) / 6.0)
    assert case_rmse.shape == (2, 2)
    assert np.allclose(case_rmse, expected)


def _synthetic_case_metrics() -> pd.DataFrame:
    rows = []
    dates = list(pd.date_range("2018-06-01", periods=3, freq="7D")) + list(
        pd.date_range("2019-06-01", periods=3, freq="7D")
    )
    for method in ("physical_control", "physical_tcwv"):
        for case_index, date in enumerate(dates):
            for lead in range(1, 7):
                control_rmse = 4.0 + 0.1 * lead + 0.01 * case_index
                control_acc = 0.2 + 0.02 * lead + 0.001 * case_index
                control_bias = 1.0 + 0.05 * lead + 0.01 * case_index
                if method == "physical_control":
                    rmse, acc, bias = control_rmse, control_acc, control_bias
                else:
                    rmse = control_rmse - 0.2
                    acc = control_acc + 0.03
                    bias = control_bias - 0.1
                rows.append(
                    {
                        "method": method,
                        "case_id": date,
                        "lead": lead,
                        "rmse": rmse,
                        "acc": acc,
                        "bias": bias,
                    }
                )
    return pd.DataFrame(rows)


def test_all_lead_guard_table_has_directional_every_lead_guards() -> None:
    table, guards = plots.build_all_lead_guard_table(
        _synthetic_case_metrics(), "physical_tcwv"
    )
    assert table.lead.tolist() == [f"W{lead}" for lead in range(1, 7)]
    assert np.allclose(table.rmse_delta_candidate_minus_control, -0.2)
    assert np.allclose(table.acc_delta_candidate_minus_control, 0.03)
    assert np.allclose(table.abs_bias_delta_candidate_minus_control, -0.1)
    assert table.all_three_improved.all()
    assert all(guards.values())


def test_year_stratified_paired_moving_block_bootstrap_is_deterministic() -> None:
    frame = _synthetic_case_metrics()
    first = plots.stratified_paired_bootstrap(
        frame, "physical_tcwv", replicates=200, seed=17
    )
    second = plots.stratified_paired_bootstrap(
        frame, "physical_tcwv", replicates=200, seed=17
    )
    pd.testing.assert_frame_equal(first, second)
    assert set(first.block_length_initializations) == {13}
    assert set(first.scope) == {
        "W1",
        "W2",
        "W3",
        "W4",
        "W5",
        "W6",
        "ALL_WEEKS",
    }
    assert np.allclose(
        first.loc[
            first.metric.eq("rmse_delta_candidate_minus_control"), "point_delta"
        ],
        -0.2,
    )
    assert np.allclose(first.bootstrap_probability_improved, 1.0)
