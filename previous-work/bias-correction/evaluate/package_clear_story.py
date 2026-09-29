#!/usr/bin/env python3
"""Package the locked FuXi–IMD clear-story figures in presentation order.

This script only validates and copies already-rendered presentation artifacts.
It does not open source forecast, observation, target, checkpoint, or prediction
arrays, and it does not recompute a metric.  The output is a fresh immutable
directory containing numbered PNG/PDF files, concise slide notes, and a
checksum manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


EXPECTED_EVALUATION_ROLE = "exploratory_reused_hindcast_evaluation"
EXPECTED_CONFIGURATION = "physical_full_compact"
EXPECTED_YEARS = (2020, 2021)

SPATIAL_FILES = (
    "01_six_week_spatial_atlas_native_grid",
    "02_six_week_spatial_atlas_visual_interpolation",
)
ACC_FILES = (
    "01_paired_case_acc_raw_vs_corrected_exploratory_2020_2021",
    "02_acc_by_lead_and_paired_gain_exploratory_2020_2021",
)
JJAS_FILES = (
    "jjas_month_lead_improvement_tradeoffs",
    "jjas_month_lead_paired_uncertainty",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def validate_sources(spatial: Path, acc: Path, jjas: Path) -> dict[str, Any]:
    spatial_manifest_path = spatial / "spatial_atlas_manifest.json"
    acc_manifest_path = acc / "manifest.json"
    jjas_manifest_path = jjas / "diagnostic_manifest.json"
    spatial_manifest = read_json(spatial_manifest_path)
    acc_manifest = read_json(acc_manifest_path)
    jjas_manifest = read_json(jjas_manifest_path)

    spatial_scope = str(spatial_manifest.get("evaluation_scope", "")).lower()
    acc_scope = str(acc_manifest.get("evaluation_scope", "")).lower()
    jjas_scope = str(jjas_manifest.get("evaluation_scope", "")).lower()
    if not all(
        "exploratory/reused" in scope and "not independent" in scope
        for scope in (spatial_scope, acc_scope, jjas_scope)
    ):
        raise ValueError("every source must explicitly state exploratory/reused scope")

    if spatial_manifest.get("status") != "complete":
        raise ValueError("spatial atlas is not complete")
    if spatial_manifest.get("selected_configuration") != EXPECTED_CONFIGURATION:
        raise ValueError("unexpected spatial-atlas configuration")
    if spatial_manifest.get("visual_interpolation_used_for_metrics") is not False:
        raise ValueError("spatial interpolation must not enter metrics")
    if spatial_manifest.get("cases") != 70:
        raise ValueError("spatial atlas must contain 70 JJAS starts")
    if spatial_manifest.get("lead_weeks") != list(range(1, 7)):
        raise ValueError("spatial atlas must contain all six lead weeks")

    if acc_manifest.get("genuine_independent_confirmation") is not False:
        raise ValueError("ACC scope must be exploratory/reused")
    if acc_manifest.get("source_arrays_opened") is not False:
        raise ValueError("ACC figure generation opened forbidden arrays")
    if acc_manifest.get("cases") != 70 or acc_manifest.get("paired_points") != 420:
        raise ValueError("ACC figure case coverage is incomplete")

    if jjas_manifest.get("status") != "complete":
        raise ValueError("JJAS diagnostic is not complete")
    if jjas_manifest.get("evaluation_role") != EXPECTED_EVALUATION_ROLE:
        raise ValueError("unexpected JJAS evaluation role")
    if jjas_manifest.get("selected_configuration") != EXPECTED_CONFIGURATION:
        raise ValueError("unexpected JJAS configuration")
    if tuple(jjas_manifest.get("test_years", ())) != EXPECTED_YEARS:
        raise ValueError("JJAS diagnostic must use 2020–2021")
    if jjas_manifest.get("genuine_independent_test") is not False:
        raise ValueError("JJAS scope must be exploratory/reused")
    if jjas_manifest.get("parameter_updates") != 0:
        raise ValueError("JJAS diagnostic unexpectedly refit the model")
    uncertainty = jjas_manifest.get("uncertainty", {})
    if uncertainty.get("p_values_computed") is not False:
        raise ValueError("p-values are outside the presentation contract")
    if uncertainty.get("significance_claimed") is not False:
        raise ValueError("significance claims are outside the presentation contract")

    for source, stems in (
        (spatial, SPATIAL_FILES),
        (acc, ACC_FILES),
        (jjas, JJAS_FILES),
    ):
        for stem in stems:
            for suffix in (".png", ".pdf"):
                path = source / f"{stem}{suffix}"
                if not path.is_file() or path.stat().st_size < 10_000:
                    raise ValueError(f"missing or truncated figure: {path}")

    for name, expected in spatial_manifest.get("artifacts", {}).items():
        if sha256_file(spatial / name) != expected:
            raise ValueError(f"spatial artifact checksum mismatch: {name}")
    for name, expected in acc_manifest.get("figures", {}).items():
        if sha256_file(acc / name) != expected:
            raise ValueError(f"ACC artifact checksum mismatch: {name}")
    for name, expected in jjas_manifest.get("artifacts", {}).items():
        if sha256_file(jjas / name) != expected:
            raise ValueError(f"JJAS artifact checksum mismatch: {name}")

    return {
        "spatial": {
            "path": str(spatial_manifest_path.resolve()),
            "sha256": sha256_file(spatial_manifest_path),
        },
        "acc": {
            "path": str(acc_manifest_path.resolve()),
            "sha256": sha256_file(acc_manifest_path),
        },
        "jjas": {
            "path": str(jjas_manifest_path.resolve()),
            "sha256": sha256_file(jjas_manifest_path),
        },
    }


def _story_markdown() -> str:
    return """# FuXi–IMD bias-correction: clear presentation story

## Scope to state on every result slide

- Frozen `physical_full_compact` correction; trained on 2002–2017 and selected on blocked 2018–2019 validation.
- Evaluation shown here: 70 JJAS initializations from 2020–2021, six lead weeks, and 171 IMD-supported India cells.
- 2020–2021 has already been inspected during development. Treat it as locked exploratory/reused hindcasts, **not independent confirmation**.
- ACC is India-area-weighted centred spatial anomaly correlation against IMD, using training-only climatology.
- Interpolation in Figure 02 is for display only. Every metric uses the native 1.5° grid.

## Recommended slide order and narration

### 01 — Native-grid six-week spatial composites

**Claim:** the frozen correction moves the FuXi rainfall pattern toward the IMD composite across all six leads.

Read the first three rows as IMD, raw FuXi, and corrected rainfall. Read the last two as `IMD − forecast`: positive means the forecast is too dry, negative means it is too wet. This is a composite over the same 70 starts, not one forecast case and not a daily map.

### 02 — Interpolated visualization of the same composites

Use this version for a large projected slide. Say explicitly that smoothing is visual only; do not use it as quantitative evidence or imply extra spatial resolution.

### 03 — Paired ACC behavior, case by case

**Claim:** improvement is not caused only by a few cases. Of 420 paired initialization × lead cases, 329 (78.3%) lie above the 1:1 line. The pooled mean corrected-minus-raw ACC is +0.106.

IMD is the verification reference on both axes. Do not add a separate “IMD ACC” curve: IMD verified against itself would be 1 by definition and would not be a model comparison.

### 04 — Mean ACC by lead and paired gain

**Claim:** corrected ACC exceeds raw FuXi and the training-only log-bias baseline at every lead week. Corrected-minus-raw ΔACC is +0.139, +0.118, +0.121, +0.094, +0.073, and +0.092 for W1–W6. The pooled effect is +0.106 with a paired 95% percentile interval [+0.044, +0.149].

The paired interval is wholly above zero for W1–W4; W5–W6 include zero. Call these “bootstrap-supported exploratory improvements,” not statistically significant independent results.

### 05 — Detailed JJAS month × lead result

**Claim:** the benefit is seasonally structured. ACC and RMSE point estimates improve in 22/24 month–lead cells. Gains are strongest and most consistent for June–July initializations, weaken through August, and become slightly negative for September W5–W6.

**Important tradeoff:** national absolute mean bias improves in only 6/24 cells and generally worsens in August–September. The current model is a spatial-pattern and RMSE correction, not a universally calibrated mean-rainfall correction.

### 06 — JJAS uncertainty and limitations

Whiskers are paired, month-stratified, two-stage circular moving-block 95% percentile intervals with all six leads kept together. They are approximate because only two year clusters are available. No p-values, FDR, or independent-significance claim is made.

## One-sentence result

> A compact frozen FuXi–IMD correction improves India-wide rainfall spatial pattern skill and RMSE across most JJAS month–lead combinations, with the largest gains at early-to-middle leads, while exposing a remaining national-mean bias and late-September calibration problem.

## Exact headline numbers

- Mean ACC: raw 0.253; training-only log-bias 0.305; corrected 0.359.
- Pooled corrected-minus-raw ΔACC: +0.106; paired 95% percentile interval [+0.044, +0.149].
- Pooled RMSE reduction versus raw: 11.55%; paired 95% percentile interval [8.34%, 14.69%].
- Absolute mean-bias change versus raw: −0.614 mm day⁻¹ under the positive-is-better convention; the correction worsens absolute bias overall.
- Corrected ACC is higher in 329/420 paired initialization × lead cases (78.3%; descriptive).
- Month × lead: ACC and RMSE improve in 22/24 point-estimate cells; absolute mean bias improves in 6/24.

## What not to claim

- Do not call 2020–2021 an independent test or final confirmation.
- Do not say all metrics improve: absolute mean bias worsens overall.
- Do not call the display interpolation a higher-resolution prediction.
- Do not claim significance from the exploratory percentile intervals.
- Do not show IMD as a third ACC curve; IMD is the verifying reference.
"""


def _captions_markdown() -> str:
    return """# Figure captions

## Figure 01

Six-week rainfall composites over India for IMD observations, raw FuXi-S2S, and the frozen neural correction, followed by IMD-minus-forecast error fields. Each column is a lead week and each field is averaged over 70 JJAS initializations in the 2020–2021 locked exploratory/reused hindcasts. Maps use the native 1.5° evaluation grid and 171 IMD-supported cells.

## Figure 02

The same six-week composites rendered with cubic visual interpolation inside the nearest-cell IMD support footprint. Interpolation is display-only and is not used for any metric, uncertainty calculation, observation, or prediction.

## Figure 03

Paired India-area-weighted spatial ACC of the frozen correction versus raw FuXi-S2S against IMD for 420 initialization × lead cases. Points above the 1:1 line favor the correction; large diamonds show lead-wise means. The case fraction and scatter are descriptive because starts and lead weeks are dependent.

## Figure 04

Mean spatial ACC against IMD for raw FuXi-S2S, training-only log-bias correction, and the frozen neural correction, with paired corrected-minus-raw effects. Error bars are saved two-stage moving-block 95% percentile intervals that resample years and chronological initialization blocks while retaining all six leads. Filled markers indicate descriptive intervals wholly above zero; they are not independent significance tests.

## Figure 05

Initialization-month × lead-week changes in spatial ACC, RMSE, and absolute national-mean bias. Positive values favor correction. Bold cells have approximate paired 95% percentile intervals wholly above zero. The panel reveals broad ACC/RMSE gains but a mean-bias tradeoff and weak September weeks 5–6.

## Figure 06

Paired month × lead effects with approximate 95% percentile intervals from 5,000 month-stratified, two-stage circular moving-block resamples. Only two year clusters are available, so the intervals quantify exploratory stability and do not support p-value or independent-confirmation claims.
"""


def _copy_artifact(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copy2(source, destination)


def _atomic_write_text(path: Path, content: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=path.suffix, dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def package(
    spatial: Path,
    acc: Path,
    jjas: Path,
    output: Path,
) -> Path:
    spatial = Path(spatial).expanduser().resolve()
    acc = Path(acc).expanduser().resolve()
    jjas = Path(jjas).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"fresh output directory required: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    sources = validate_sources(spatial, acc, jjas)

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=output.parent)
    )
    try:
        mapping = (
            (spatial, SPATIAL_FILES[0], "01_spatial_native_grid"),
            (spatial, SPATIAL_FILES[1], "02_spatial_interpolated_display_only"),
            (acc, ACC_FILES[0], "03_paired_case_acc"),
            (acc, ACC_FILES[1], "04_acc_by_lead_and_gain"),
            (jjas, JJAS_FILES[0], "05_jjas_month_lead_tradeoffs"),
            (jjas, JJAS_FILES[1], "06_jjas_month_lead_uncertainty"),
        )
        artifacts: dict[str, str] = {}
        for source, stem, target_stem in mapping:
            for suffix in (".png", ".pdf"):
                destination = temporary / f"{target_stem}{suffix}"
                _copy_artifact(source / f"{stem}{suffix}", destination)
                artifacts[destination.name] = sha256_file(destination)

        summary_source = jjas / "jjas_initialization_month_by_lead_summary.csv"
        summary_target = temporary / "jjas_initialization_month_by_lead_summary.csv"
        _copy_artifact(summary_source, summary_target)
        artifacts[summary_target.name] = sha256_file(summary_target)

        story_path = temporary / "STORY_AND_SLIDE_NOTES.md"
        caption_path = temporary / "CAPTIONS.md"
        _atomic_write_text(story_path, _story_markdown())
        _atomic_write_text(caption_path, _captions_markdown())
        artifacts[story_path.name] = sha256_file(story_path)
        artifacts[caption_path.name] = sha256_file(caption_path)

        manifest = {
            "schema_name": "fuxi_imd_clear_story_presentation_package",
            "schema_version": 1,
            "status": "complete",
            "evaluation_role": EXPECTED_EVALUATION_ROLE,
            "evaluation_scope": (
                "2020-2021 exploratory/reused locked hindcasts; "
                "not independent confirmation"
            ),
            "selected_configuration": EXPECTED_CONFIGURATION,
            "training_years_inclusive": [2002, 2017],
            "blocked_validation_years_inclusive": [2018, 2019],
            "evaluation_years": list(EXPECTED_YEARS),
            "cases": 70,
            "lead_weeks": list(range(1, 7)),
            "imd_supported_cells": 171,
            "source_or_prediction_arrays_opened_by_packager": False,
            "metrics_recomputed_by_packager": False,
            "independent_confirmation_claimed": False,
            "statistical_significance_claimed": False,
            "source_manifests": sources,
            "artifacts": dict(sorted(artifacts.items())),
        }
        manifest_path = temporary / "MANIFEST.json"
        _atomic_write_text(
            manifest_path,
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        )
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spatial_figure_directory", type=Path)
    parser.add_argument("acc_figure_directory", type=Path)
    parser.add_argument("jjas_figure_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = package(
        args.spatial_figure_directory,
        args.acc_figure_directory,
        args.jjas_figure_directory,
        args.output_directory,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
