from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import numpy as np
import pandas as pd
import pytest

import plot_locked_exploratory_acc_figures as acc_plot


def _synthetic_result(root: Path) -> Path:
    metrics = root / "metrics"
    metrics.mkdir(parents=True)
    dates = np.concatenate(
        [
            pd.date_range(f"{year}-06-01", periods=35, freq="3D").to_numpy()
            for year in acc_plot.EXPECTED_YEARS
        ]
    )
    raw_means = np.asarray([0.50, 0.36, 0.22, 0.17, 0.12, 0.15])
    log_gains = np.asarray([0.07, 0.06, 0.05, 0.04, 0.04, 0.03])
    corrected_gains = np.asarray([0.14, 0.12, 0.12, 0.09, 0.07, 0.09])
    rows = []
    for case_index, date in enumerate(dates):
        oscillation = 0.16 * np.sin(0.43 * case_index)
        for lead in acc_plot.EXPECTED_LEADS:
            raw = raw_means[lead - 1] + oscillation + 0.015 * np.cos(lead + case_index)
            values = {
                "raw_fuxi": raw,
                "log_bias": raw + log_gains[lead - 1],
                "corrected": raw + corrected_gains[lead - 1],
            }
            for method, value in values.items():
                rows.append(
                    {
                        "method": method,
                        "predictor": method,
                        "case_id": str(pd.Timestamp(date).date()),
                        "lead": lead,
                        "region": "india",
                        "season": "ALL",
                        "valid_cells": 171,
                        "weight_sum": 1.0,
                        "acc": value,
                        "rmse": 1.0,
                        "mae": 0.8,
                        "bias": 0.1,
                        "negative_fraction": 0.0,
                    }
                )
    case = pd.DataFrame(rows)
    case.to_csv(metrics / acc_plot.CASE_FILENAME, index=False)
    lead = (
        case.groupby(["method", "lead"], as_index=False)["acc"]
        .mean()
        .assign(rmse=1.0, mae=0.8, bias=0.1)
    )
    lead[["method", "lead", "rmse", "mae", "bias", "acc"]].to_csv(
        metrics / acc_plot.LEAD_FILENAME, index=False
    )

    bootstrap_rows = []
    for lead_week, effect in zip(
        acc_plot.EXPECTED_LEADS, corrected_gains, strict=True
    ):
        lower = effect - (0.03 if lead_week < 5 else 0.10)
        upper = effect + 0.04
        bootstrap_rows.append(
            {
                "candidate": "corrected",
                "baseline": "raw_fuxi",
                "scope": f"W{lead_week}",
                "metric": "acc",
                "effect_positive_is_better": effect,
                "effect_units": "ACC difference",
                "ci95_lower": lower,
                "ci95_upper": upper,
                # These legacy fields are intentionally nonsense: the new
                # plotting code must neither parse nor use them.
                "probability_improved": "unused",
                "centered_block_null_two_sided_p": "unused",
                "bh_q_across_six_leads": "unused",
                "bootstrap_supported_improvement": "unused",
            }
        )
    pooled = float(np.mean(corrected_gains))
    bootstrap_rows.append(
        {
            "candidate": "corrected",
            "baseline": "raw_fuxi",
            "scope": "ALL_WEEKS",
            "metric": "acc",
            "effect_positive_is_better": pooled,
            "effect_units": "ACC difference",
            "ci95_lower": pooled - 0.03,
            "ci95_upper": pooled + 0.03,
            "probability_improved": "unused",
            "centered_block_null_two_sided_p": "unused",
            "bh_q_across_six_leads": "unused",
            "bootstrap_supported_improvement": "unused",
        }
    )
    pd.DataFrame(bootstrap_rows).to_csv(
        metrics / acc_plot.BOOTSTRAP_FILENAME, index=False
    )
    return root


def test_loader_reads_only_three_csvs_and_builds_complete_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _synthetic_result(tmp_path / "result")
    original = acc_plot.pd.read_csv
    reads: list[Path] = []
    read_columns: list[set[str]] = []

    def recording_read_csv(path: Path, *args: object, **kwargs: object) -> pd.DataFrame:
        reads.append(Path(path))
        read_columns.append(set(kwargs["usecols"]))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(acc_plot.pd, "read_csv", recording_read_csv)
    data = acc_plot.load_acc_figure_data(result)
    assert [path.name for path in reads] == [
        acc_plot.CASE_FILENAME,
        acc_plot.LEAD_FILENAME,
        acc_plot.BOOTSTRAP_FILENAME,
    ]
    assert read_columns == [
        set(acc_plot.CASE_READ_COLUMNS),
        set(acc_plot.LEAD_READ_COLUMNS),
        set(acc_plot.BOOTSTRAP_READ_COLUMNS),
    ]
    assert not set(acc_plot.BOOTSTRAP_READ_COLUMNS) & {
        "probability_improved",
        "centered_block_null_two_sided_p",
        "bh_q_across_six_leads",
        "bootstrap_supported_improvement",
    }
    assert len(data.paired_cases) == 70 * 6
    assert set(data.paired_cases.columns) >= {
        "raw_fuxi",
        "log_bias",
        "corrected",
        "delta_corrected_raw",
    }
    assert np.allclose(
        data.corrected_vs_raw["effect_positive_is_better"],
        [0.14, 0.12, 0.12, 0.09, 0.07, 0.09],
    )


def test_figure_set_is_fresh_atomic_and_explicitly_not_independent(
    tmp_path: Path,
) -> None:
    result = _synthetic_result(tmp_path / "result")
    source_hashes = {
        path.name: acc_plot.sha256_file(path)
        for path in (result / "metrics").glob("*.csv")
    }
    data = acc_plot.load_acc_figure_data(result)
    output = acc_plot.generate_acc_figure_set(data, tmp_path / "figures")
    expected = [
        output / f"{acc_plot.FIGURE_A_STEM}.png",
        output / f"{acc_plot.FIGURE_A_STEM}.pdf",
        output / f"{acc_plot.FIGURE_B_STEM}.png",
        output / f"{acc_plot.FIGURE_B_STEM}.pdf",
    ]
    assert all(path.is_file() and path.stat().st_size > 10_000 for path in expected)
    assert not list(tmp_path.glob(f".{output.name}.partial-*"))
    assert source_hashes == {
        path.name: acc_plot.sha256_file(path)
        for path in (result / "metrics").glob("*.csv")
    }
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["genuine_independent_confirmation"] is False
    assert manifest["source_arrays_opened"] is False
    assert "exploratory/reused" in manifest["evaluation_scope"]
    assert "p-value" in manifest["unused_inference_fields"]
    assert set(manifest["figures"]) == {path.name for path in expected}


def test_incomplete_case_pair_is_refused(tmp_path: Path) -> None:
    result = _synthetic_result(tmp_path / "result")
    path = result / "metrics" / acc_plot.CASE_FILENAME
    frame = pd.read_csv(path)
    frame = frame.drop(
        frame.loc[
            frame["method"].eq("corrected")
            & frame["lead"].eq(1)
        ].index[0]
    )
    frame.to_csv(path, index=False)
    with pytest.raises(acc_plot.AccFigureContractError, match="70 cases|incomplete"):
        acc_plot.load_acc_figure_data(result)


def test_bootstrap_point_effect_must_match_locked_case_and_lead_tables(
    tmp_path: Path,
) -> None:
    result = _synthetic_result(tmp_path / "result")
    path = result / "metrics" / acc_plot.BOOTSTRAP_FILENAME
    frame = pd.read_csv(path)
    selected = frame["scope"].eq("W1")
    frame.loc[selected, "effect_positive_is_better"] += 0.01
    frame.to_csv(path, index=False)
    with pytest.raises(acc_plot.AccFigureContractError, match="point effect differs"):
        acc_plot.load_acc_figure_data(result)


def test_existing_destination_is_never_overwritten(tmp_path: Path) -> None:
    result = _synthetic_result(tmp_path / "result")
    data = acc_plot.load_acc_figure_data(result)
    output = tmp_path / "figures"
    output.mkdir()
    sentinel = output / "belongs_to_user.txt"
    sentinel.write_text("preserve\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="fresh ACC figure output"):
        acc_plot.generate_acc_figure_set(data, output)
    assert sentinel.read_text(encoding="utf-8") == "preserve\n"


def test_visible_scope_language_is_exploratory_reused_not_independent() -> None:
    assert "exploratory/reused" in acc_plot.SCOPE_LINE
    assert "NOT INDEPENDENT CONFIRMATION" in acc_plot.GUARD_LINE
