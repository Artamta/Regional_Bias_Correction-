#!/usr/bin/env python3
"""Build a non-destructive, evidence-labeled presentation asset package.

The packager copies only already-rendered PNG, PDF, and SVG files.  It never
loads forecast, prediction, residual, observation, or target arrays and never
computes a new evaluation metric.  A 2018--2019 validation-publication
directory is required.  A locked 2020--2021 exploratory/reused-test directory
can be supplied later with ``--exploratory-test-directory``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METHOD_ASSETS = PROJECT_ROOT / "presentation" / "figures"
FIGURE_SUFFIXES = (".png", ".pdf", ".svg")
VALIDATION_YEARS = (2018, 2019)
EXPLORATORY_TEST_YEARS = (2020, 2021)
ARCHITECTURE_STEM = "physical_temporal_unet_architecture"
ARCHITECTURE_STEM_V2 = "physical_temporal_unet_architecture_v2"
TRAINING_STEM = "physical_full_compact_training_curves"
TRAINING_STEM_V2 = "physical_full_compact_training_curves_v2"

KNOWN_TITLES = {
    ARCHITECTURE_STEM: "Physics-guided temporal residual architecture",
    ARCHITECTURE_STEM_V2: "Physics-guided temporal residual architecture",
    TRAINING_STEM: "Training and blocked-validation learning curves",
    TRAINING_STEM_V2: "Training and blocked-validation learning curves",
    "01_spatial_mean_bias_by_lead": "Spatial mean bias by lead week",
    "02_spatial_rmse_skill_vs_control": "Spatial RMSE skill versus compact control",
    "03_india_weighted_metrics_by_lead": "India-area-weighted metrics by lead week",
    "04_validation_time_lead_comparison": "Temporal and lead-wise validation behavior",
    "01_acc_rmse_bias_by_lead_exploratory_test": (
        "ACC, RMSE, and bias by lead week"
    ),
}


@dataclass
class AssetFamily:
    """One logical figure and its available render formats."""

    source_stem: str
    title: str
    section: str
    claim_scope: str
    evidence_years: str
    recommended_use: str
    caveat: str
    files: dict[str, Path]


@dataclass
class SlideEntry:
    """One row in the generated slide index."""

    slide: int
    section: str
    title: str
    claim_scope: str
    evidence_years: str
    status: str
    recommended_use: str
    caveat: str
    png: str = ""
    pdf: str = ""
    svg: str = ""
    source_stem: str = ""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _humanize(stem: str) -> str:
    value = Path(stem).name
    value = re.sub(r"^\d+[_-]*", "", value)
    return value.replace("_", " ").replace("-", " ").strip().title()


def _slug(stem: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", stem).strip("_").lower()
    if not value:
        raise ValueError(f"cannot form an asset name from {stem!r}")
    return value


def _discover_families(figures_directory: Path) -> dict[str, dict[str, Path]]:
    """Group rendered files by relative path without loading their contents."""

    figures_directory = figures_directory.resolve()
    if not figures_directory.is_dir():
        raise FileNotFoundError(figures_directory)
    grouped: dict[str, dict[str, Path]] = {}
    for path in sorted(figures_directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in FIGURE_SUFFIXES:
            continue
        resolved = path.resolve()
        if figures_directory not in resolved.parents:
            raise ValueError(f"figure escapes its declared source directory: {path}")
        relative = resolved.relative_to(figures_directory)
        stem = str(relative.with_suffix(""))
        suffix = resolved.suffix.lower()
        if suffix in grouped.setdefault(stem, {}):
            raise ValueError(f"duplicate {suffix} rendering for {stem}")
        grouped[stem][suffix] = resolved
    return grouped


def _verify_declared_artifacts(
    source_directory: Path,
    manifest: Mapping[str, Any],
    grouped: Mapping[str, Mapping[str, Path]],
) -> None:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("source manifest must contain an artifacts checksum mapping")
    for renderings in grouped.values():
        for path in renderings.values():
            relative = str(path.relative_to(source_directory))
            expected = artifacts.get(relative)
            if not isinstance(expected, str):
                raise ValueError(f"source manifest does not declare figure {relative}")
            actual = _sha256(path)
            if actual != expected:
                raise ValueError(f"source figure checksum differs: {relative}")


def _validate_validation_publication(
    directory: Path,
) -> tuple[Mapping[str, Any], dict[str, dict[str, Path]], str]:
    directory = directory.resolve()
    manifest = _load_json(directory / "postprocessing_manifest.json")
    if manifest.get("status") != "complete":
        raise ValueError("validation publication is not complete")
    if tuple(manifest.get("validation_years", ())) != VALIDATION_YEARS:
        raise ValueError("validation publication must contain only 2018--2019")
    scope = str(manifest.get("evaluation_scope", "")).lower()
    if "blocked validation" not in scope or "no 2020+" not in scope:
        raise ValueError(
            "validation publication must explicitly declare blocked validation "
            "and no 2020+ metric"
        )
    if manifest.get("source_training_mode") != "full":
        raise ValueError("validation publication must derive from a full run")
    selected = str(manifest.get("selected_best_physical_candidate", ""))
    if not selected:
        raise ValueError("validation publication lacks a selected physical candidate")
    grouped = _discover_families(directory / "figures")
    if not grouped or any(".png" not in files for files in grouped.values()):
        raise ValueError("every validation figure must have a PNG preview")
    _verify_declared_artifacts(directory, manifest, grouped)
    return manifest, grouped, selected


def _first_present(
    manifest: Mapping[str, Any], names: Sequence[str], default: Any = None
) -> Any:
    for name in names:
        if name in manifest:
            return manifest[name]
    return default


def _validate_exploratory_test(
    directory: Path,
    selected_validation_configuration: str,
) -> tuple[Mapping[str, Any], dict[str, dict[str, Path]]]:
    """Enforce an explicit locked, reused-test manifest contract."""

    directory = directory.resolve()
    manifest_path = directory / "postprocessing_manifest.json"
    if not manifest_path.is_file():
        manifest_path = directory / "manifest.json"
    manifest = _load_json(manifest_path)
    if manifest.get("status") != "complete":
        raise ValueError("exploratory-test output is not complete")
    years = _first_present(
        manifest,
        ("test_years", "evaluation_years", "exploratory_test_years"),
    )
    if tuple(years or ()) != EXPLORATORY_TEST_YEARS:
        raise ValueError("exploratory-test years must be exactly 2020--2021")
    if manifest.get("evaluation_role") != "exploratory_reused_hindcast_evaluation":
        raise ValueError(
            "evaluation_role must be exploratory_reused_hindcast_evaluation"
        )
    scope = str(manifest.get("evaluation_scope", "")).lower()
    for required in ("exploratory", "reused", "not independent"):
        if required not in scope:
            raise ValueError(
                f"exploratory-test evaluation_scope must contain {required!r}"
            )
    locked = _first_present(
        manifest,
        (
            "selection_locked_before_test",
            "selection_locked_before_evaluation",
            "model_selection_locked",
        ),
    )
    if locked is not True:
        raise ValueError("model selection must be explicitly locked before test")
    if manifest.get("selection_locked_before_target_access") is not True:
        raise ValueError("selection must be explicitly locked before target access")
    if manifest.get("test_used_for_selection") is not False:
        raise ValueError("test_used_for_selection must be explicitly false")
    if manifest.get("parameter_updates") != 0:
        raise ValueError("locked exploratory evaluation must have parameter_updates=0")
    if manifest.get("reused_test_period") is not True:
        raise ValueError("reused_test_period must be explicitly true")
    if manifest.get("genuine_independent_test") is not False:
        raise ValueError("genuine_independent_test must be explicitly false")
    selected = str(manifest.get("selected_configuration", ""))
    if selected != selected_validation_configuration:
        raise ValueError(
            "exploratory-test selected_configuration differs from validation selection"
        )
    grouped = _discover_families(directory / "figures")
    if not grouped or any(".png" not in files for files in grouped.values()):
        raise ValueError("every exploratory-test figure must have a PNG preview")
    _verify_declared_artifacts(directory, manifest, grouped)
    return manifest, grouped


def _method_families(directory: Path) -> list[AssetFamily]:
    grouped = _discover_families(directory)

    def preferred_stem(stems: Sequence[str], required_suffixes: set[str]) -> str:
        for stem in stems:
            if stem in grouped and required_suffixes.issubset(grouped[stem]):
                return stem
        raise ValueError(
            f"one of {list(stems)!r} must provide {sorted(required_suffixes)}"
        )

    architecture_stem = preferred_stem(
        (ARCHITECTURE_STEM_V2, ARCHITECTURE_STEM),
        {".png", ".pdf", ".svg"},
    )
    training_stem = preferred_stem(
        (TRAINING_STEM_V2, TRAINING_STEM),
        {".png", ".pdf"},
    )
    architecture_is_v2 = architecture_stem == ARCHITECTURE_STEM_V2
    training_is_v2 = training_stem == TRAINING_STEM_V2
    return [
        AssetFamily(
            source_stem=architecture_stem,
            title=KNOWN_TITLES[architecture_stem],
            section="01_method",
            claim_scope="Method diagram; no evaluation result",
            evidence_years="Train 2002–2017; selection 2018–2019",
            recommended_use="Main method slide; split or crop for readability",
            caveat=(
                "The v2 split panel explicitly labels 2020–2021 as a reused "
                "check and not independent confirmation."
                if architecture_is_v2
                else "Its TEST 2020–2021 box must be described as "
                "exploratory/reused, not independent confirmation."
            ),
            files=grouped[architecture_stem],
        ),
        AssetFamily(
            source_stem=training_stem,
            title=KNOWN_TITLES[training_stem],
            section="02_training",
            claim_scope="Optimization and blocked-validation selection",
            evidence_years="Train 2002–2017; validation 2018–2019",
            recommended_use="Training/overfitting diagnostic slide",
            caveat=(
                "The v2 mean and shading stop at the last epoch shared by all "
                "three seeds; individual seed traces retain their true stop lengths."
                if training_is_v2
                else "Late-epoch mean is not three-seed: two seeds remain from "
                "epoch 21 and one from epoch 27 (display numbering)."
            ),
            files=grouped[training_stem],
        ),
    ]


def _validation_families(
    grouped: Mapping[str, Mapping[str, Path]],
) -> list[AssetFamily]:
    families = []
    for stem, files in sorted(grouped.items()):
        basename = Path(stem).name
        use = "Main validation result slide"
        caveat = "Blocked validation used for model selection; not an independent test."
        if basename == "01_spatial_mean_bias_by_lead":
            use = "Paper/supplement; too dense for a single projected slide"
            caveat += " Descriptive spatial means; boundary is presentation reference."
        elif basename == "02_spatial_rmse_skill_vs_control":
            caveat += (
                " Descriptive cell estimates only; mean effects are below one percent "
                "and no cell-wise significance is claimed."
            )
        elif basename == "03_india_weighted_metrics_by_lead":
            caveat += (
                " Curves have no uncertainty intervals and several methods overlap."
            )
        elif basename == "04_validation_time_lead_comparison":
            caveat += (
                " Per-case heatmap is descriptive and can show large local swings."
            )
        families.append(
            AssetFamily(
                source_stem=stem,
                title=KNOWN_TITLES.get(basename, _humanize(basename)),
                section="03_blocked_validation",
                claim_scope="Blocked validation",
                evidence_years="JJAS 2018–2019",
                recommended_use=use,
                caveat=caveat,
                files=dict(files),
            )
        )
    return families


def _exploratory_families(
    grouped: Mapping[str, Mapping[str, Path]],
) -> list[AssetFamily]:
    families = []
    for stem, files in sorted(grouped.items()):
        basename = Path(stem).name
        week_match = re.fullmatch(r"02_spatial_week_(\d+)_exploratory_test", basename)
        title = KNOWN_TITLES.get(basename, _humanize(basename))
        if week_match:
            title = f"India spatial comparison · lead week {week_match.group(1)}"
        families.append(
            AssetFamily(
                source_stem=stem,
                title=title,
                section="04_exploratory_reused_test",
                claim_scope=(
                    "Exploratory/reused hindcast test; not independent confirmation"
                ),
                evidence_years="JJAS 2020–2021",
                recommended_use=(
                    "Exploratory result only; label reused test on the slide"
                ),
                caveat=(
                    "This period was previously examined during project development. "
                    "Do not describe it as untouched, held-out, or independent."
                ),
                files=dict(files),
            )
        )
    return families


def _copy_families(output: Path, families: Sequence[AssetFamily]) -> list[SlideEntry]:
    entries: list[SlideEntry] = []
    for slide, family in enumerate(families, start=1):
        destination_directory = output / "assets" / family.section
        destination_directory.mkdir(parents=True, exist_ok=True)
        destination_stem = destination_directory / (
            f"{slide:02d}_{_slug(family.source_stem)}"
        )
        copied: dict[str, str] = {}
        for suffix, source in sorted(family.files.items()):
            destination = destination_stem.with_suffix(suffix)
            shutil.copy2(source, destination)
            copied[suffix] = str(destination.relative_to(output))
        entries.append(
            SlideEntry(
                slide=slide,
                section=family.section,
                title=family.title,
                claim_scope=family.claim_scope,
                evidence_years=family.evidence_years,
                status="available",
                recommended_use=family.recommended_use,
                caveat=family.caveat,
                png=copied.get(".png", ""),
                pdf=copied.get(".pdf", ""),
                svg=copied.get(".svg", ""),
                source_stem=family.source_stem,
            )
        )
    return entries


def _placeholder_entries(
    start: int, *, exploratory_available: bool
) -> list[SlideEntry]:
    entries = []
    if not exploratory_available:
        entries.append(
            SlideEntry(
                slide=start,
                section="04_exploratory_reused_test",
                title="2020–2021 exploratory/reused test results",
                claim_scope="Exploratory/reused hindcast test; not independent confirmation",
                evidence_years="JJAS 2020–2021",
                status="pending — no locked output supplied",
                recommended_use="Add only after the locked evaluator completes",
                caveat="Never relabel this reused period as independent confirmation.",
            )
        )
        start += 1
    entries.append(
        SlideEntry(
            slide=start,
            section="05_unavailable_independent_test",
            title="Genuine independent physical-model evaluation",
            claim_scope="No result available",
            evidence_years="2025",
            status="unavailable",
            recommended_use="Limitations/future-evaluation statement only",
            caveat=(
                "No genuine independent 2025 FuXi physical-model evaluation is "
                "available; do not show or claim a 2025 score."
            ),
        )
    )
    return entries


def _write_slide_index(output: Path, entries: Sequence[SlideEntry]) -> None:
    columns = tuple(SlideEntry.__dataclass_fields__)
    with (output / "SLIDE_INDEX.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for entry in entries:
            writer.writerow(vars(entry))
    lines = [
        "# Slide asset index",
        "",
        "Every result asset is labeled by evidence scope. Blank asset links are "
        "intentional status placeholders, not missing computed results.",
        "",
        "| # | Asset | Scope | Years | Status | Caveat |",
        "|---:|---|---|---|---|---|",
    ]
    for entry in entries:
        label = entry.title
        if entry.png:
            label = f"[{label}]({entry.png})"
        escaped = [
            str(entry.slide),
            label,
            entry.claim_scope,
            entry.evidence_years,
            entry.status,
            entry.caveat,
        ]
        escaped = [value.replace("|", "\\|").replace("\n", " ") for value in escaped]
        lines.append("| " + " | ".join(escaped) + " |")
    (output / "SLIDE_INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _thumbnail(path: Path, maximum: tuple[int, int] = (1200, 700)) -> np.ndarray:
    with Image.open(path) as image:
        converted = image.convert("RGB")
        converted.thumbnail(maximum, Image.Resampling.LANCZOS)
        return np.asarray(converted).copy()


def _write_contact_sheet(output: Path, entries: Sequence[SlideEntry]) -> None:
    columns = 3
    rows = int(math.ceil(len(entries) / columns))
    figure = plt.figure(figsize=(16.0, 1.45 + 3.7 * rows), facecolor="#F4F7F9")
    grid = figure.add_gridspec(
        rows,
        columns,
        left=0.035,
        right=0.965,
        top=0.90,
        bottom=0.04,
        hspace=0.34,
        wspace=0.16,
    )
    for index, entry in enumerate(entries):
        axis = figure.add_subplot(grid[index // columns, index % columns])
        axis.set_facecolor("white")
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color("#CBD6DE")
            spine.set_linewidth(0.9)
        if entry.png:
            axis.imshow(_thumbnail(output / entry.png))
        else:
            axis.text(
                0.5,
                0.55,
                "RESULT NOT AVAILABLE",
                ha="center",
                va="center",
                transform=axis.transAxes,
                fontsize=15,
                fontweight="bold",
                color="#A33D35",
            )
            axis.text(
                0.5,
                0.42,
                entry.status,
                ha="center",
                va="center",
                transform=axis.transAxes,
                fontsize=9,
                color="#5D7180",
            )
        axis.set_title(
            f"{entry.slide:02d}  {entry.title}",
            loc="left",
            fontsize=10.5,
            fontweight="semibold",
            color="#172B3A",
            pad=8,
        )
        axis.text(
            0.0,
            -0.08,
            f"{entry.claim_scope} · {entry.evidence_years}",
            transform=axis.transAxes,
            fontsize=7.8,
            color="#526777",
            va="top",
        )
    for index in range(len(entries), rows * columns):
        axis = figure.add_subplot(grid[index // columns, index % columns])
        axis.axis("off")
    figure.suptitle(
        "FuXi–IMD physical bias correction · presentation asset index",
        x=0.035,
        ha="left",
        y=0.965,
        fontsize=20,
        fontweight="bold",
        color="#172B3A",
    )
    figure.text(
        0.035,
        0.928,
        "2018–2019 = blocked validation  ·  2020–2021 = exploratory/reused test  ·  2025 independent evaluation = unavailable",
        ha="left",
        fontsize=10.5,
        color="#526777",
    )
    figure.savefig(
        output / "CONTACT_SHEET.png", dpi=180, facecolor=figure.get_facecolor()
    )
    figure.savefig(output / "CONTACT_SHEET.pdf", facecolor=figure.get_facecolor())
    plt.close(figure)


def _write_readme(
    output: Path,
    validation_directory: Path,
    validation_manifest: Mapping[str, Any],
    exploratory_directory: Path | None,
    exploratory_manifest: Mapping[str, Any] | None,
    entries: Sequence[SlideEntry],
) -> None:
    source_stems = {entry.source_stem for entry in entries}
    architecture_note = (
        "- The packaged v2 architecture explicitly labels 2020–2021 as a "
        "reused check and not independent confirmation. Preserve that wording."
        if ARCHITECTURE_STEM_V2 in source_stems
        else "- The architecture figure's green `TEST · 2020–2021` box is a "
        "split-design label, not proof of an untouched test. Verbally relabel "
        "it as exploratory/reused."
    )
    training_note = (
        "- The packaged v2 training mean/shading uses only epochs shared by all "
        "three seeds; individual seed traces retain their actual stop lengths."
        if TRAINING_STEM_V2 in source_stems
        else "- The training figure's late-epoch mean is not based on all three "
        "seeds: two seeds remain from display epoch 21 and one from display "
        "epoch 27. Use it as an overfitting/early-stopping diagnostic, not a "
        "three-seed late-epoch comparison."
    )
    exploratory_status = (
        "Included from a completed locked evaluator output."
        if exploratory_manifest is not None
        else "Not included: no completed locked evaluator output was supplied."
    )
    lines = [
        "# FuXi–IMD presentation asset package",
        "",
        "This is a non-destructive copy/index of existing rendered figures. The "
        "packager did **not** read target, observation, prediction, residual, or "
        "forecast arrays and did not calculate any new result.",
        "",
        "## Evidence status",
        "",
        "| Evidence | Years | Honest interpretation |",
        "|---|---|---|",
        "| Training | 2002–2017 | Model fitting and train-only normalization. |",
        "| Blocked validation | 2018–2019 | Used for architecture, checkpoint, and variable selection; **not an independent test**. |",
        f"| Exploratory/reused hindcast test | 2020–2021 | {exploratory_status} Even when present, it is **not independent confirmation** because this period was examined earlier in the project. |",
        "| Genuine independent physical-model evaluation | 2025 | **Unavailable. No 2025 metric or result may be claimed.** |",
        "",
        "## Required language",
        "",
        "Use “blocked validation (2018–2019)” for validation figures and "
        "“exploratory/reused test (2020–2021), not independent confirmation” for "
        "any locked exploratory-test figures. Do not use “held-out”, “untouched”, "
        "or “independent test” for 2020–2021.",
        "",
        "## Known visual caveats",
        "",
        architecture_note,
        training_note,
        "- The 24-panel spatial-bias figure is too dense for a normal projected "
        "slide; use it in the paper/supplement or split it across slides.",
        "- In the 2018–2019 physical-vs-compact validation ablation, spatial skill "
        "maps are descriptive point estimates: area-improved fractions do not "
        "establish cell-wise significance and mean RMSE effects are below one percent.",
        "- The 2018–2019 physical/control metric curves have no uncertainty and often "
        "overlap. Use the separate bootstrap table for inference.",
        "- In the 2020–2021 exploratory reused check, corrected-vs-raw RMSE/ACC "
        "improve, while absolute national mean bias worsens; preserve all three panels.",
        "- Administrative boundaries are presentation references, not legal or "
        "cadastral determinations.",
        "",
        "## Contents",
        "",
        "- `assets/`: copied PNG/PDF/SVG families, separated by evidence scope.",
        "- `SLIDE_INDEX.md` and `SLIDE_INDEX.csv`: ordered asset inventory and claim guards.",
        "- `CONTACT_SHEET.png` and `CONTACT_SHEET.pdf`: visual index with unavailable-result placeholders.",
        "- `PACKAGE_MANIFEST.json`: checksums and source provenance for copied assets.",
        "",
        "## Sources",
        "",
        f"- Validation publication: `{validation_directory}`",
        f"- Validation publication created: `{validation_manifest.get('created_utc', 'unknown')}`",
        f"- Selected physical configuration: `{validation_manifest.get('selected_best_physical_candidate')}`",
        f"- Exploratory/reused test output: `{exploratory_directory if exploratory_directory else 'not supplied'}`",
        "",
        f"Indexed slide entries: {len(entries)}.",
    ]
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def package_assets(
    validation_publication_directory: Path,
    output_directory: Path,
    *,
    exploratory_test_directory: Path | None = None,
    method_assets_directory: Path = DEFAULT_METHOD_ASSETS,
) -> Path:
    """Validate, copy, index, and label existing figure assets."""

    validation_directory = Path(validation_publication_directory).expanduser().resolve()
    output = Path(output_directory).expanduser().resolve()
    method_directory = Path(method_assets_directory).expanduser().resolve()
    exploratory_directory = (
        None
        if exploratory_test_directory is None
        else Path(exploratory_test_directory).expanduser().resolve()
    )
    source_directories = [validation_directory, method_directory]
    if exploratory_directory is not None:
        source_directories.append(exploratory_directory)
    for source in source_directories:
        if output == source or source in output.parents:
            raise ValueError("output must not be inside a source-results directory")
    if output.exists():
        raise FileExistsError(f"refusing to replace an existing package: {output}")

    validation_manifest, validation_grouped, selected = (
        _validate_validation_publication(validation_directory)
    )
    exploratory_manifest = None
    exploratory_grouped: dict[str, dict[str, Path]] = {}
    if exploratory_directory is not None:
        exploratory_manifest, exploratory_grouped = _validate_exploratory_test(
            exploratory_directory, selected
        )
    families = [
        *_method_families(method_directory),
        *_validation_families(validation_grouped),
        *_exploratory_families(exploratory_grouped),
    ]

    output.mkdir(parents=True, exist_ok=False)
    entries = _copy_families(output, families)
    entries.extend(
        _placeholder_entries(
            len(entries) + 1,
            exploratory_available=exploratory_manifest is not None,
        )
    )
    _write_slide_index(output, entries)
    _write_contact_sheet(output, entries)
    _write_readme(
        output,
        validation_directory,
        validation_manifest,
        exploratory_directory,
        exploratory_manifest,
        entries,
    )

    manifest = {
        "status": "complete",
        "created_utc": _utc_now(),
        "operation": "copy/index rendered PNG, PDF, and SVG assets only",
        "arrays_read": False,
        "metrics_computed": False,
        "source_results_modified": False,
        "validation": {
            "directory": str(validation_directory),
            "years": list(VALIDATION_YEARS),
            "role": "blocked validation used for selection; not independent test",
            "manifest_sha256": _sha256(
                validation_directory / "postprocessing_manifest.json"
            ),
        },
        "exploratory_reused_test": {
            "included": exploratory_manifest is not None,
            "directory": (
                None if exploratory_directory is None else str(exploratory_directory)
            ),
            "years": list(EXPLORATORY_TEST_YEARS),
            "role": "exploratory/reused hindcast test; not independent confirmation",
        },
        "genuine_independent_2025_evaluation": {
            "available": False,
            "result_claimed": False,
        },
        "selected_configuration": selected,
        "slide_entries": [vars(entry) for entry in entries],
    }
    manifest["artifacts"] = {
        str(path.relative_to(output)): {
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "PACKAGE_MANIFEST.json"
    }
    (output / "PACKAGE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("validation_publication_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--exploratory-test-directory", type=Path)
    parser.add_argument(
        "--method-assets-directory", type=Path, default=DEFAULT_METHOD_ASSETS
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = package_assets(
        args.validation_publication_directory,
        args.output_directory,
        exploratory_test_directory=args.exploratory_test_directory,
        method_assets_directory=args.method_assets_directory,
    )
    print(f"PASS: presentation asset package complete: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
