from __future__ import annotations

from pathlib import Path

import numpy as np

from quest_plots.group_report import plot_group_report
from quest_plots.figures import (
    plot_architecture,
    plot_category_comparison,
    plot_prediction_probabilities,
    plot_probabilistic_metrics,
    plot_scores,
    plot_spatial_scores,
    plot_training_history,
    plot_training_workflow,
)
from quest_plots.report import (
    _blend_uniform,
    _cell_rps,
    _format_init_date,
    _probabilistic_metric_rows,
    _spatial_summary,
    selected_score_rows,
)


def test_static_and_array_figures_write_png(tmp_path: Path) -> None:
    history = {
        "epoch": np.arange(3),
        "train_loss": np.array([1.1, 0.9, 0.8]),
        "validation_loss": np.array([1.2, 1.0, 0.95]),
        "train_rps": np.array([1.1, 0.9, 0.8]),
        "validation_rps": np.array([1.2, 1.0, 0.95]),
    }
    latitude = np.linspace(-90, 90, 5)
    longitude = np.linspace(0, 315, 8)
    probability = np.full((2, 5, 5, 8), 0.2)
    fields = np.full((2, 5, 8), 0.8)

    outputs = []
    outputs += plot_architecture(tmp_path / "architecture", ("png",))
    outputs += plot_training_workflow(tmp_path / "workflow", ("png",))
    outputs += plot_training_history({"seed 42": history}, tmp_path / "curves", ("png",))
    outputs += plot_prediction_probabilities(
        probability, latitude, longitude, "2020-06-16", tmp_path / "prediction", ("png",)
    )
    outputs += plot_category_comparison(
        probability,
        probability,
        np.zeros((2, 5, 8), dtype=np.int8),
        latitude,
        longitude,
        "2020-06-16",
        0,
        tmp_path / "categories",
        ("png",),
    )
    outputs += plot_spatial_scores(
        fields, fields - 0.1, latitude, longitude, tmp_path / "spatial", ("png",)
    )

    assert len(outputs) == 6
    assert all(path.is_file() and path.stat().st_size > 1_000 for path in outputs)


def test_score_selection_and_plot(tmp_path: Path) -> None:
    rows: list[dict[str, str]] = []
    systems = (
        ("uniform", "none", 0.83),
        ("p0", "none", 1.02),
        ("p0", "by_lead", 0.82),
        ("spatial_ensemble", "none", 0.86),
        ("spatial_ensemble", "by_lead", 0.80),
    )
    for split in ("fit", "evaluation"):
        for system, calibration, score in systems:
            rows.append(
                {
                    "split": split,
                    "system": system,
                    "calibration": calibration,
                    "lead": "pooled",
                    "rps": str(score),
                    "rpss_vs_uniform": str(1 - score / 0.83),
                    "relative_improvement_vs_p0_percent": str(100 * (1.02 - score) / 1.02),
                }
            )
    selected = selected_score_rows(rows)
    output = plot_scores(selected, tmp_path / "scores", ("png",))

    assert len(selected) == 10
    assert output[0].is_file()


def test_probabilistic_metrics_are_zero_for_perfect_forecast(tmp_path: Path) -> None:
    target = np.zeros((1, 2, 2, 3), dtype=np.int8)
    perfect = np.zeros((1, 2, 5, 2, 3), dtype=np.float32)
    perfect[:, :, 0] = 1.0
    uniform = np.full_like(perfect, 0.2)
    rows = _probabilistic_metric_rows(
        {"Perfect": perfect, "Uniform": uniform},
        target,
        np.array([-30.0, 30.0]),
    )
    perfect_rows = [row for row in rows if row["system"] == "Perfect"]
    assert all(row["rps"] == 0.0 for row in perfect_rows)
    assert all(row["brier"] == 0.0 for row in perfect_rows)
    assert all(row["log"] == 0.0 for row in perfect_rows)
    output = plot_probabilistic_metrics(rows, tmp_path / "metrics", ("png",))
    assert output[0].is_file()


def test_group_report_writes_shareable_pdf(tmp_path: Path) -> None:
    score_rows = [
        {"split": split, "label": label, "rps": rps, "rpss": 0.03, "improvement_vs_p0_percent": 20.0}
        for split in ("fit", "evaluation")
        for label, rps in (
            ("Selected calibrated ensemble", 0.80),
            ("Calibrated p₀", 0.82),
        )
    ]
    metric_rows = [
        {"lead": "Pooled", "system": system, "rps": rps, "brier": brier, "log": log}
        for system, rps, brier, log in (
            ("Uniform", 0.83, 0.80, 1.61),
            ("Raw FuXi p₀", 1.02, 0.94, 2.10),
            ("Selected calibrated ensemble", 0.80, 0.79, 1.59),
        )
    ]
    spatial = [
        {"area_fraction_improved": 0.65},
        {"area_fraction_improved": 0.66},
    ]
    paths = plot_group_report(
        score_rows,
        metric_rows,
        spatial,
        tmp_path / "group_report",
        ("pdf", "png"),
    )
    assert paths[0].read_bytes().startswith(b"%PDF")
    assert paths[1].is_file() and paths[1].stat().st_size > 10_000


def test_calibration_and_cell_rps_contract() -> None:
    raw = np.zeros((1, 2, 5, 2, 3), dtype=np.float32)
    raw[:, :, 0] = 1.0
    blended = _blend_uniform(raw, np.array([0.5, 0.25], dtype=np.float32))
    assert np.allclose(blended.sum(axis=2), 1.0)
    assert np.allclose(blended[:, 0, 0], 0.6)
    assert np.allclose(blended[:, 1, 0], 0.4)

    target = np.zeros((1, 2, 2, 3), dtype=np.int8)
    score = _cell_rps(raw, target)
    assert score.shape == (2, 2, 3)
    assert np.allclose(score, 0.0)
    assert _format_init_date("2020-06-16") == "2020-06-16"
    assert _format_init_date(20200616) == "2020-06-16"

    summary = _spatial_summary(
        np.ones((2, 2, 3)),
        np.stack((np.full((2, 3), 0.5), np.full((2, 3), 1.5))),
        np.array([-30.0, 30.0]),
    )
    assert np.isclose(summary[0]["area_fraction_improved"], 1.0)
    assert np.isclose(summary[0]["area_weighted_mean_rps_reduction"], 0.5)
    assert np.isclose(summary[1]["area_fraction_worse"], 1.0)
