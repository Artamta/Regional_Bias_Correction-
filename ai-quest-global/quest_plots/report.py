"""Build a complete, reproducible AI Quest plot package from run artifacts."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .group_report import plot_group_report
from .figures import (
    EXPLORATORY_NOTE,
    plot_architecture,
    plot_category_comparison,
    plot_prediction_probabilities,
    plot_probabilistic_metrics,
    plot_scores,
    plot_spatial_scores,
    plot_training_history,
    plot_training_workflow,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = PROJECT_ROOT / "runs/imerg_real_smoke_20260815_v2/cache/fuxi_imerg_exploratory_2017_2020.npz"
DEFAULT_COMPARISON = PROJECT_ROOT / "runs/imerg_real_smoke_20260816_fast_ablation_results_complete/comparison.csv"
DEFAULT_HISTORIES = (
    PROJECT_ROOT / "runs/imerg_real_smoke_20260815_v3/training_spatial_control_120ep/history.csv",
    PROJECT_ROOT / "runs/imerg_real_smoke_20260816_fast_ablation_spatial_seed43/history.csv",
    PROJECT_ROOT / "runs/imerg_real_smoke_20260816_fast_ablation_spatial_seed44/history.csv",
)
DEFAULT_CHECKPOINTS = (
    PROJECT_ROOT / "runs/imerg_real_smoke_20260815_v3/training_spatial_control_120ep/best.pt",
    PROJECT_ROOT / "runs/imerg_real_smoke_20260816_fast_ablation_spatial_seed43/best.pt",
    PROJECT_ROOT / "runs/imerg_real_smoke_20260816_fast_ablation_spatial_seed44/best.pt",
)
DEFAULT_OUTPUT = PROJECT_ROOT / "plot_packages/ai_quest_imerg_exploratory_20260816"


@dataclass(frozen=True)
class ReportInputs:
    """All source artifacts needed to reproduce the presentation package."""

    cache: Path = DEFAULT_CACHE
    comparison: Path = DEFAULT_COMPARISON
    histories: tuple[Path, ...] = DEFAULT_HISTORIES
    checkpoints: tuple[Path, ...] = DEFAULT_CHECKPOINTS
    evaluation_year: int = 2020
    case_index: int = 0
    formats: tuple[str, ...] = ("png", "pdf")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _history(path: Path) -> dict[str, np.ndarray]:
    rows = _read_csv(path)
    if not rows:
        raise ValueError(f"empty training history: {path}")
    return {
        name: np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        for name in rows[0]
    }


def _find_score(
    rows: list[dict[str, str]],
    *,
    split: str,
    system: str,
    calibration: str,
    lead: str = "pooled",
) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if row["split"] == split
        and row["system"] == system
        and row["calibration"] == calibration
        and row["lead"] == lead
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one score for {split}/{system}/{calibration}/{lead}; found {len(matches)}"
        )
    return matches[0]


def selected_score_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    """Return the five presentation systems for both artifact splits."""

    specifications = (
        ("Uniform", "uniform", "none"),
        ("Raw FuXi p₀", "p0", "none"),
        ("Calibrated p₀", "p0", "by_lead"),
        ("Raw spatial ensemble", "spatial_ensemble", "none"),
        ("Selected calibrated ensemble", "spatial_ensemble", "by_lead"),
    )
    split_labels = {
        "fit": "Validation / calibration 2019",
        "evaluation": "Retrospective evaluation 2020",
    }
    result: list[dict[str, object]] = []
    for split, split_label in split_labels.items():
        for label, system, calibration in specifications:
            row = _find_score(
                rows,
                split=split,
                system=system,
                calibration=calibration,
            )
            result.append(
                {
                    "split": split,
                    "split_label": split_label,
                    "label": label,
                    "system": system,
                    "calibration": calibration,
                    "rps": float(row["rps"]),
                    "rpss": float(row["rpss_vs_uniform"]),
                    "improvement_vs_p0_percent": float(
                        row["relative_improvement_vs_p0_percent"]
                    ),
                }
            )
    return result


def _write_score_summary(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _by_lead_alphas(rows: list[dict[str, str]]) -> np.ndarray:
    return np.asarray(
        [
            float(
                _find_score(
                    rows,
                    split="fit",
                    system="spatial_ensemble",
                    calibration="by_lead",
                    lead=lead,
                )["alpha"]
            )
            for lead in ("D19-25", "D26-32")
        ],
        dtype=np.float32,
    )


def _predict_ensemble(inputs: ReportInputs) -> tuple[Any, np.ndarray]:
    # Importing these here keeps diagram/CSV-only use of the package lightweight.
    from predict import load_checkpoint_model, load_prepared, run_model

    prepared = load_prepared(
        inputs.cache,
        require_target=True,
        years=[inputs.evaluation_year],
    )
    prediction_sum = np.zeros_like(prepared.p0, dtype=np.float32)
    for checkpoint in inputs.checkpoints:
        model, _ = load_checkpoint_model(
            checkpoint,
            device="cpu",
            in_channels=int(prepared.features.shape[2]),
        )
        prediction_sum += run_model(
            model,
            prepared.features,
            prepared.p0,
            device="cpu",
        )
    return prepared, prediction_sum / float(len(inputs.checkpoints))


def _blend_uniform(probability: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    if alphas.shape != (2,):
        raise ValueError("calibration alphas must contain one value per lead")
    alpha = alphas.reshape(1, 2, 1, 1, 1)
    return (alpha * probability + (1.0 - alpha) * 0.2).astype(np.float32)


def _cell_rps(probability: np.ndarray, target: np.ndarray) -> np.ndarray:
    categories = np.arange(5, dtype=np.int8).reshape(1, 1, 5, 1, 1)
    observed_cdf = target[:, :, None] <= categories
    forecast_cdf = np.cumsum(probability, axis=2)
    score = np.square(forecast_cdf - observed_cdf).sum(axis=2)
    score[(target < 0) | (target > 4)] = np.nan
    valid_count = np.isfinite(score).sum(axis=0)
    return np.divide(
        np.nansum(score, axis=0),
        valid_count,
        out=np.full(score.shape[1:], np.nan, dtype=np.float64),
        where=valid_count > 0,
    )


def _format_init_date(value: Any) -> str:
    """Format either the NPZ ISO-string or Zarr integer date convention."""

    text = str(np.asarray(value).item())
    compact = text.replace("-", "")[:8]
    if len(compact) != 8 or not compact.isdigit():
        raise ValueError(f"unsupported initialization date: {text!r}")
    return f"{compact[:4]}-{compact[4:6]}-{compact[6:]}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_label(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _report_markdown(score_rows: list[dict[str, object]], init_date: str) -> str:
    validation = next(
        row
        for row in score_rows
        if row["split"] == "fit" and row["label"] == "Selected calibrated ensemble"
    )
    evaluation = next(
        row
        for row in score_rows
        if row["split"] == "evaluation" and row["label"] == "Selected calibrated ensemble"
    )
    return f"""# AI Quest global precipitation plot package

This package describes the evaluated global precipitation adapter and its current
FuXi–IMERG smoke evidence. It is **not an official ERA5 or AI Weather Quest
validation result**.

## Figures

1. `01_architecture`: the simple residual spatial U-Net used by each ensemble member.
2. `02_training_workflow`: data, train-only normalization, RPS loss, splits, selection,
   and calibration.
3. `03_training_curves`: objective and RPS histories for the three spatial seeds.
4. `04_prediction_probabilities`: the actual 2 × 5 probabilistic output for the
   {init_date} example initialization.
5. `05_validation_test_scores`: uniform, raw FuXi, raw/calibrated baselines, and the
   selected ensemble on 2019 and retrospective 2020.
6. `06_spatial_improvement`: raw FuXi RPS, selected-model RPS, and their grid-cell
   difference over the four 2020 cases.
7. `07_category_comparison_D19-25`: all Q1–Q5 raw FuXi probabilities, selected
   probabilities, and one-hot ground truth for the first lead.
8. `08_category_comparison_D26-32`: the same complete comparison for the second lead.
9. `09_probabilistic_scorecard`: RPS, multicategory Brier score, and log score for
   uniform, raw FuXi, and the selected system. RPS is the categorical CRPS analogue;
   exact continuous CRPS is unavailable from this five-bin output.

Both PNG and vector PDF versions are generated. `score_summary.csv` is the compact
machine-readable table used by figure 05, and `manifest.json` records exact sources.
`AI_QUEST_GROUP_REPORT.pdf` is the concise one-page summary intended for sharing.

## Headline numbers

- Selected calibrated spatial ensemble 2019 RPS: **{float(validation['rps']):.6f}**
  (RPSS {100 * float(validation['rpss']):+.2f}% vs uniform).
- Retrospective 2020 RPS: **{float(evaluation['rps']):.6f}**
  (RPSS {100 * float(evaluation['rpss']):+.2f}% vs uniform).
- Reduction against raw FuXi p0 in 2020: **{float(evaluation['improvement_vs_p0_percent']):.2f}%**.

## Scientific boundary

The cache contains 16 initializations: 8 from 2017–2018 for training, 4 from 2019
for validation/calibration, and 4 already-opened 2020 cases for retrospective
evaluation. Truth is IMERG Final V07B, thresholds use the 2002–2016 IMERG calendar
climatology, and scoring is cosine-area weighted over supported global cells. The
official ERA5 cache and official land/aridity mask are absent, so these plots support
pipeline and ablation discussion only—not a competition-rank claim.

## Rebuild

From the `ai-quest-global` directory:

```bash
/home/raj.ayush/.conda/envs/fuxi/bin/python -m quest_plots
```
"""


def _spatial_summary(
    p0_rps: np.ndarray,
    selected_rps: np.ndarray,
    latitude: np.ndarray,
) -> list[dict[str, float]]:
    """Summarize the spatial comparison with cosine-area weighting."""

    improvement = np.asarray(p0_rps, dtype=np.float64) - np.asarray(
        selected_rps, dtype=np.float64
    )
    area = np.broadcast_to(
        np.clip(np.cos(np.deg2rad(np.asarray(latitude, dtype=np.float64))), 0, None)[
            :, None
        ],
        improvement.shape[1:],
    )
    result: list[dict[str, float]] = []
    for lead in range(2):
        valid = np.isfinite(improvement[lead])
        weights = area[valid]
        values = improvement[lead][valid]
        if not len(values) or float(weights.sum()) <= 0:
            raise ValueError(f"lead {lead + 1} has no valid spatial comparison cells")
        result.append(
            {
                "area_fraction_improved": float(weights[values > 0].sum() / weights.sum()),
                "area_fraction_worse": float(weights[values < 0].sum() / weights.sum()),
                "area_weighted_mean_rps_reduction": float(
                    np.average(values, weights=weights)
                ),
            }
        )
    return result


def _probabilistic_metric_rows(
    systems: dict[str, np.ndarray],
    target: np.ndarray,
    latitude: np.ndarray,
) -> list[dict[str, object]]:
    """Compute cosine-area-weighted proper scores for categorical probabilities."""

    target = np.asarray(target)
    if target.ndim != 4 or target.shape[1] != 2:
        raise ValueError("target must have shape [case, 2, latitude, longitude]")
    if target.shape[-2] != len(latitude):
        raise ValueError("latitude does not match target")
    expected = target.shape[:2] + (5,) + target.shape[-2:]
    area = np.broadcast_to(
        np.clip(np.cos(np.deg2rad(np.asarray(latitude, dtype=np.float64))), 0, None)[
            :, None
        ],
        target.shape[-2:],
    )
    categories = np.arange(5, dtype=np.int8).reshape(1, 1, 5, 1, 1)
    valid = (target >= 0) & (target <= 4)
    observed = target[:, :, None] == categories
    observed_cdf = target[:, :, None] <= categories
    rows: list[dict[str, object]] = []
    for system, raw_probability in systems.items():
        probability = np.asarray(raw_probability, dtype=np.float64)
        if probability.shape != expected:
            raise ValueError(f"{system} probability must have shape {expected}")
        if not np.isfinite(probability).all() or np.any(probability < 0):
            raise ValueError(f"{system} contains invalid probabilities")
        if not np.allclose(probability.sum(axis=2), 1.0, atol=2e-5):
            raise ValueError(f"{system} probabilities do not sum to one")
        forecast_cdf = probability.cumsum(axis=2)
        cell_rps = np.square(forecast_cdf - observed_cdf).sum(axis=2)
        cell_brier = np.square(probability - observed).sum(axis=2)
        observed_probability = np.take_along_axis(
            probability,
            np.clip(target, 0, 4)[:, :, None],
            axis=2,
        ).squeeze(2)
        cell_log = -np.log(np.clip(observed_probability, 1e-12, 1.0))
        for lead_name, lead_indices in (
            ("D19–25", (0,)),
            ("D26–32", (1,)),
            ("Pooled", (0, 1)),
        ):
            selected_valid = valid[:, lead_indices]
            weights = area[None, None] * selected_valid
            denominator = float(weights.sum())
            if denominator <= 0:
                raise ValueError(f"{lead_name} contains no valid metric cells")
            index = (slice(None), lead_indices)
            rows.append(
                {
                    "lead": lead_name,
                    "system": system,
                    "rps": float((cell_rps[index] * weights).sum() / denominator),
                    "brier": float(
                        (cell_brier[index] * weights).sum() / denominator
                    ),
                    "log": float((cell_log[index] * weights).sum() / denominator),
                }
            )
    return rows


def _short_report_markdown(
    score_rows: list[dict[str, object]],
    spatial: list[dict[str, float]],
) -> str:
    def score(split: str, label: str) -> dict[str, object]:
        return next(
            row
            for row in score_rows
            if row["split"] == split and row["label"] == label
        )

    raw_validation = score("fit", "Raw FuXi p₀")
    raw_evaluation = score("evaluation", "Raw FuXi p₀")
    raw_model_validation = score("fit", "Raw spatial ensemble")
    raw_model_evaluation = score("evaluation", "Raw spatial ensemble")
    calibrated_p0_evaluation = score("evaluation", "Calibrated p₀")
    selected_validation = score("fit", "Selected calibrated ensemble")
    selected_evaluation = score("evaluation", "Selected calibrated ensemble")
    added_value = 100 * (
        float(calibrated_p0_evaluation["rps"]) - float(selected_evaluation["rps"])
    ) / float(calibrated_p0_evaluation["rps"])
    return f"""# Short report: global AI Quest precipitation adapter

## What and why

The experiment adapts 51-member FuXi subseasonal forecasts into **five rainfall-category
probabilities** on the global 1.5° grid for **D19–25 and D26–32**. Raw FuXi probabilities
are the anchor (`p₀`). A small spatial U-Net learns residual corrections, three random
seeds are averaged, and a validation-fitted uniform blend reduces overconfidence. The
goal is better-calibrated probabilistic rainfall guidance, measured primarily by Ranked
Probability Score (RPS; lower is better).

## Evidence and result

This is a small **exploratory FuXi–IMERG smoke test**, not official ERA5 or competition
validation: 8 cases from 2017–2018 train the model, 4 cases from 2019 select/calibrate it,
and 4 already-opened 2020 cases provide retrospective evaluation.

| System | 2019 RPS | 2020 retrospective RPS |
|---|---:|---:|
| Raw FuXi `p₀` | {float(raw_validation['rps']):.3f} | {float(raw_evaluation['rps']):.3f} |
| Raw spatial ensemble | {float(raw_model_validation['rps']):.3f} | {float(raw_model_evaluation['rps']):.3f} |
| Selected calibrated ensemble | **{float(selected_validation['rps']):.3f}** | **{float(selected_evaluation['rps']):.3f}** |

The selected system reduces 2020 RPS by **{float(selected_evaluation['improvement_vs_p0_percent']):.2f}%**
relative to raw FuXi and has RPSS **{100 * float(selected_evaluation['rpss']):+.2f}%**
against uniform probabilities. Calibration supplies most of the gain: the neural ensemble
adds about **{added_value:.2f}%** beyond calibrated `p₀` on this retrospective slice.

## How to read the spatial plot

- Left: raw FuXi grid-cell RPS. Middle: selected-system RPS. They use the **same color
  scale**, so lower/lighter values in the middle mean less error.
- Right: `raw p₀ RPS − selected RPS`. **Green is improvement**, magenta is degradation,
  and white is little change.
- The model is not better everywhere. Cosine-area weighted, it improves
  **{100 * spatial[0]['area_fraction_improved']:.1f}%** of valid area for D19–25 and
  **{100 * spatial[1]['area_fraction_improved']:.1f}%** for D26–32; about 35% worsens.
  Mean grid-cell RPS reductions are **{spatial[0]['area_weighted_mean_rps_reduction']:.3f}**
  and **{spatial[1]['area_weighted_mean_rps_reduction']:.3f}**, respectively.

Thus “better” comes from the lower aggregate RPS and predominantly green difference map,
not from visual smoothness. With only four 2020 cases, the spatial pattern is illustrative,
not a stable regional-skill claim.

## CRPS and companion metrics

For this ordered five-category output, RPS is the defensible categorical analogue of
CRPS: it sums squared CDF errors across the category boundaries. The package also
reports multicategory Brier and log scores. Exact continuous CRPS in millimetres cannot
be calculated without a continuous predictive rainfall distribution inside and beyond
the five bins; assigning arbitrary bin midpoints would create a misleading score.

## Next steps

1. Finalize the completed ERA5 parts using the official land/aridity mask and issue
   schedule, then run the frozen evaluation.
2. Fit normalization, checkpoints, and calibration on training/validation only, then open
   a genuinely untouched test once.
3. Scale to more initialization dates; only then test the planned multivariable/context
   model and make regional-skill claims.
"""


def build_report(inputs: ReportInputs, output_dir: Path) -> list[Path]:
    """Build all figures and supporting files; return every written path."""

    source_paths = [inputs.cache, inputs.comparison, *inputs.histories, *inputs.checkpoints]
    missing = [path for path in source_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing report artifact(s): " + ", ".join(map(str, missing)))
    if len(inputs.histories) != len(inputs.checkpoints):
        raise ValueError("histories and checkpoints must have the same number of seeds")
    if not inputs.checkpoints:
        raise ValueError("at least one checkpoint is required")
    if inputs.case_index < 0:
        raise ValueError("case_index cannot be negative")

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_csv(inputs.comparison)
    scores = selected_score_rows(rows)
    written: list[Path] = []
    written.extend(plot_architecture(output_dir / "01_architecture", inputs.formats))
    written.extend(plot_training_workflow(output_dir / "02_training_workflow", inputs.formats))
    histories = {
        f"seed {42 + index}": _history(path)
        for index, path in enumerate(inputs.histories)
    }
    written.extend(
        plot_training_history(histories, output_dir / "03_training_curves", inputs.formats)
    )
    written.extend(plot_scores(scores, output_dir / "05_validation_test_scores", inputs.formats))

    prepared, raw_ensemble = _predict_ensemble(inputs)
    if inputs.case_index >= prepared.n_cases:
        raise IndexError(
            f"case_index {inputs.case_index} is outside the {prepared.n_cases} evaluation cases"
        )
    selected = _blend_uniform(raw_ensemble, _by_lead_alphas(rows))
    init_date = _format_init_date(np.asarray(prepared.init_dates)[inputs.case_index])
    written.extend(
        plot_prediction_probabilities(
            selected[inputs.case_index],
            np.asarray(prepared.latitude),
            np.asarray(prepared.longitude),
            init_date,
            output_dir / "04_prediction_probabilities",
            inputs.formats,
        )
    )
    assert prepared.target is not None
    for lead_index, lead_slug in enumerate(("D19-25", "D26-32")):
        written.extend(
            plot_category_comparison(
                prepared.p0[inputs.case_index],
                selected[inputs.case_index],
                prepared.target[inputs.case_index],
                np.asarray(prepared.latitude),
                np.asarray(prepared.longitude),
                init_date,
                lead_index,
                output_dir / f"{7 + lead_index:02d}_category_comparison_{lead_slug}",
                inputs.formats,
            )
        )
    metric_rows = _probabilistic_metric_rows(
        {
            "Uniform": np.full(prepared.p0.shape, 0.2, dtype=np.float32),
            "Raw FuXi p₀": prepared.p0,
            "Selected calibrated ensemble": selected,
        },
        prepared.target,
        np.asarray(prepared.latitude),
    )
    written.extend(
        plot_probabilistic_metrics(
            metric_rows,
            output_dir / "09_probabilistic_scorecard",
            inputs.formats,
        )
    )
    metric_path = output_dir / "probabilistic_metrics_2020.csv"
    _write_score_summary(metric_path, metric_rows)
    written.append(metric_path)
    p0_spatial_rps = _cell_rps(prepared.p0, prepared.target)
    selected_spatial_rps = _cell_rps(selected, prepared.target)
    written.extend(
        plot_spatial_scores(
            p0_spatial_rps,
            selected_spatial_rps,
            np.asarray(prepared.latitude),
            np.asarray(prepared.longitude),
            output_dir / "06_spatial_improvement",
            inputs.formats,
        )
    )

    score_path = output_dir / "score_summary.csv"
    _write_score_summary(score_path, scores)
    written.append(score_path)
    readme_path = output_dir / "README.md"
    readme_path.write_text(_report_markdown(scores, init_date), encoding="utf-8")
    written.append(readme_path)
    short_report_path = output_dir / "SHORT_REPORT.md"
    spatial_summary = _spatial_summary(
        p0_spatial_rps,
        selected_spatial_rps,
        np.asarray(prepared.latitude),
    )
    short_report_path.write_text(
        _short_report_markdown(
            scores,
            spatial_summary,
        ),
        encoding="utf-8",
    )
    written.append(short_report_path)
    written.extend(
        plot_group_report(
            scores,
            metric_rows,
            spatial_summary,
            output_dir / "AI_QUEST_GROUP_REPORT",
            ("pdf", "png"),
        )
    )
    manifest = {
        "status": "exploratory_not_official_era5_or_competition_validation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "note": EXPLORATORY_NOTE,
        "evaluation_year": inputs.evaluation_year,
        "example_case_index": inputs.case_index,
        "formats": list(inputs.formats),
        "sources": {
            _source_label(path): _sha256(path) for path in source_paths
        },
        "outputs": [path.name for path in written] + ["manifest.json"],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    written.append(manifest_path)
    return written


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    result.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON)
    result.add_argument("--history", type=Path, action="append", dest="histories")
    result.add_argument("--checkpoint", type=Path, action="append", dest="checkpoints")
    result.add_argument("--evaluation-year", type=int, default=2020)
    result.add_argument("--case-index", type=int, default=0)
    result.add_argument("--format", action="append", dest="formats", choices=("png", "pdf", "svg"))
    result.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return result


def main() -> None:
    args = parser().parse_args()
    inputs = ReportInputs(
        cache=args.cache,
        comparison=args.comparison,
        histories=tuple(args.histories) if args.histories else DEFAULT_HISTORIES,
        checkpoints=tuple(args.checkpoints) if args.checkpoints else DEFAULT_CHECKPOINTS,
        evaluation_year=args.evaluation_year,
        case_index=args.case_index,
        formats=tuple(args.formats) if args.formats else ("png", "pdf"),
    )
    paths = build_report(inputs, args.output_dir)
    print(f"wrote {len(paths)} files to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
