#!/usr/bin/env python3
"""Render provenance-locked figures for the India S2S rainfall paper.

Only frozen tabular evidence, JSON receipts, and the compact spatial-support
inputs needed for Figure 1 are read.  The generator never opens a forecast,
checkpoint, or observation array.  A fresh staging directory is atomically
published after every input hash and scientific gate has been checked a
second time.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import platform
import shutil
import stat
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from project_paths import PROJECT_ROOT


EXPERIMENT = "india_s2s_paper_figures_v2"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "resultsv3/india_s2s_paper_figures"
STUDY_ROOT = PROJECT_ROOT.parent / "studies/india_s2s_verification_v2"

BENCHMARK_EXPERIMENT = "india_s2s_benchmark_uncertainty_v2"
SCORING_EXPERIMENT = "india_s2s_probabilistic_bridge_scoring_v1"
AUDIT_EXPERIMENT = "india_s2s_probabilistic_bridge_score_audit_v1"
COMPARISON_EXPERIMENT = "india_s2s_paper_comparisons_v1"
CASE_COUNT_PER_LEAD = 169
RAW_IDENTITY_ROWS = 15_150
CASE_ROWS = 45_450
SELECTED_SEED = 43
SELECTED_CHECKPOINT_SHA256 = (
    "a9a71465e2399773d5968bc94e4a5d535826b9ec5d1c4d3ec555fd8449586fee"
)
STORY_COHORT = "2022_2024_valid_midpoint_jjas_synchronized_pooled_w1_w6"
PRIMARY_BLOCK_LENGTH = 16
LEADS = tuple(range(1, 7))
MODELS = (
    "cma",
    "dlesym_v0",
    "ecmwf",
    "fuxi_s2s",
    "ncep",
    "neuralgcm",
    "ukmo",
    "mme",
)
MODEL_LABELS = {
    "cma": "CMA",
    "dlesym_v0": "DLESyM-v0",
    "ecmwf": "ECMWF",
    "fuxi_s2s": "FuXi-S2S",
    "ncep": "NCEP",
    "neuralgcm": "NeuralGCM",
    "ukmo": "UKMO",
    "mme": "Equal-system MME",
}
T2M_MODELS = (
    "cma",
    "dlesym_v0",
    "dlesym_v1",
    "ecmwf",
    "fcn3",
    "fuxi_s2s",
    "ukmo",
    "mme",
)
T2M_MODEL_LABELS = {
    "cma": "CMA",
    "dlesym_v0": "DLESyM-v0",
    "dlesym_v1": "DLESyM-v1",
    "ecmwf": "ECMWF",
    "fcn3": "FourCastNet 3",
    "fuxi_s2s": "FuXi-S2S",
    "ukmo": "UKMO",
    "mme": "Equal-system MME",
}
T2M_MME_COMPONENTS = (
    "cma",
    "dlesym_v1",
    "ecmwf",
    "fcn3",
    "fuxi_s2s",
    "ukmo",
)
METHODS = ("raw_fuxi", "moment_calibration", "location_spread")
METHOD_LABELS = {
    "raw_fuxi": "Raw FuXi-S2S",
    "moment_calibration": "Train-only moment calibration",
    "location_spread": "Neural location-spread adapter",
}
REGIONS = (
    "all_india",
    "northwest_india",
    "central_india",
    "south_peninsula",
    "east_northeast_india",
)
REGION_LABELS = {
    "all_india": "All India",
    "northwest_india": "Northwest India",
    "central_india": "Central India",
    "south_peninsula": "South Peninsula",
    "east_northeast_india": "East & Northeast India",
}
FIXED_MME_COMPONENTS = (
    "cma",
    "dlesym_v0",
    "fuxi_s2s",
    "ncep",
    "neuralgcm",
    "ukmo",
)
REGION_COLORS = {
    "northwest_india": "#0072B2",
    "central_india": "#D55E00",
    "south_peninsula": "#009E73",
    "east_northeast_india": "#CC79A7",
}
SPATIAL_ARRAY_DTYPES = {
    "latitude": "<f8",
    "longitude": "<f8",
    "india_area_weight_km2": "<f8",
    "northwest_india_fraction": "<f4",
    "central_india_fraction": "<f4",
    "south_peninsula_fraction": "<f4",
    "east_northeast_india_fraction": "<f4",
}
_RENAME_NOREPLACE = 1


class PaperFigureError(RuntimeError):
    """Raised when frozen inputs or scientific display contracts differ."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PaperFigureError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(values: np.ndarray, dtype: str) -> str:
    """Hash an array after canonical little-endian dtype conversion."""

    canonical = np.ascontiguousarray(np.asarray(values).astype(dtype, copy=False))
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PaperFigureError(f"cannot read JSON: {path}") from error
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class FigureInputs:
    """All tabular and receipt inputs used by the renderer."""

    benchmark_manifest: Path
    scoring_manifest: Path
    audit_receipt: Path
    comparison_manifest: Path
    imd_methods_manifest: Path
    imd_seasonal_summary: Path
    imd_weekly_summary: Path
    imd_case_metrics: Path
    imerg_methods_manifest: Path
    imerg_seasonal_summary: Path
    imerg_weekly_summary: Path
    imerg_case_metrics: Path
    calibration_manifest: Path
    adapter_support: Path
    spatial_support_metadata: Path


@dataclass(frozen=True)
class InputHashes:
    """Expected bytes for every top-level frozen input."""

    benchmark_manifest: str
    scoring_manifest: str
    audit_receipt: str
    comparison_manifest: str
    imd_methods_manifest: str
    imd_seasonal_summary: str
    imd_weekly_summary: str
    imd_case_metrics: str
    imerg_methods_manifest: str
    imerg_seasonal_summary: str
    imerg_weekly_summary: str
    imerg_case_metrics: str
    calibration_manifest: str
    adapter_support: str
    spatial_support_metadata: str


@dataclass(frozen=True)
class SpatialArrayHashes:
    """Canonical logical-array hashes for the Figure 1 geometry source."""

    latitude: str
    longitude: str
    india_area_weight_km2: str
    northwest_india_fraction: str
    central_india_fraction: str
    south_peninsula_fraction: str
    east_northeast_india_fraction: str


@dataclass(frozen=True)
class Figure1Geometry:
    """Small immutable geometry used to render the Figure 1 domain panel."""

    latitude: np.ndarray
    longitude: np.ndarray
    india_geometry: np.ndarray
    adapter_support: np.ndarray
    region_fractions: Mapping[str, np.ndarray]
    logical_sha256: Mapping[str, str]


CANONICAL_INPUTS = FigureInputs(
    benchmark_manifest=(
        PROJECT_ROOT
        / "resultsv3/india_s2s_benchmark_uncertainty/full_v2_20260825T232000Z/manifest.json"
    ),
    scoring_manifest=(
        PROJECT_ROOT
        / "resultsv3/india_s2s_probabilistic_bridge/scoring_full_20260825T234808Z/manifest.json"
    ),
    audit_receipt=(
        PROJECT_ROOT
        / "resultsv3/india_s2s_probabilistic_bridge/audit_full_20260826T000133Z/audit_receipt.json"
    ),
    comparison_manifest=(
        PROJECT_ROOT
        / "resultsv3/india_s2s_paper_comparisons/full_v2_20260826T001721Z/manifest.json"
    ),
    imd_methods_manifest=(
        STUDY_ROOT / "results/imd_tp_sensitivity/methods_manifest.json"
    ),
    imd_seasonal_summary=(
        STUDY_ROOT / "results/imd_tp_sensitivity/tables/seasonal_summary.csv"
    ),
    imd_weekly_summary=(
        STUDY_ROOT / "results/imd_tp_sensitivity/tables/weekly_summary.csv"
    ),
    imd_case_metrics=(
        STUDY_ROOT / "results/imd_tp_sensitivity/tables/case_metrics.csv"
    ),
    imerg_methods_manifest=(
        STUDY_ROOT / "results/imerg_era5_primary/methods_manifest.json"
    ),
    imerg_seasonal_summary=(
        STUDY_ROOT / "results/imerg_era5_primary/tables/seasonal_summary.csv"
    ),
    imerg_weekly_summary=(
        STUDY_ROOT / "results/imerg_era5_primary/tables/weekly_summary.csv"
    ),
    imerg_case_metrics=(
        STUDY_ROOT / "results/imerg_era5_primary/tables/case_metrics.csv"
    ),
    calibration_manifest=(
        PROJECT_ROOT
        / "resultsv3/fuxi_allseason_ensemble_calibration/full_20260825T224711Z/manifest.json"
    ),
    adapter_support=(
        PROJECT_ROOT
        / "resultsv3/fuxi_allseason_ensemble_calibration/full_20260825T224711Z/evaluation/scoring_support.npz"
    ),
    spatial_support_metadata=Path(
        "/storage/raj.ayush/s2s_final_data/final_iteration/standardized/"
        "india_s2s_benchmark_v1/spatial/spatial_support.zarr/.zmetadata"
    ),
)
CANONICAL_HASHES = InputHashes(
    benchmark_manifest="96c209146e334c7d8223b350d287b186897add882aa292069eb0de414c352d90",
    scoring_manifest="4ab303a3fa0025bfb2b8654bf34770091f117df18f236c0c724d8e1078450943",
    audit_receipt="d7c822e0c09528b443e1dddd83872aab2894d82da5dedce6b4d0f1fbc3807bcc",
    comparison_manifest="3b754c7024537e56691f56fa8f6932dd8ae40f6b358716cb3fffa8e9051c85da",
    imd_methods_manifest="9b515c0230fc345323565a429f185ea86cf8e7c5e47a208b6117a7b649c26e6d",
    imd_seasonal_summary="fe13e44db6a2c7e78bd45fd09767a54498379d9da704b21268bf86d664502770",
    imd_weekly_summary="33e2513a38ac1e86e7250d787e13fe285d8cb6478ae2804a4dc45172cdba69cc",
    imd_case_metrics="000544615d004669b41dd87fc647bd979cd92c3638e7deee5ea77992046e1a14",
    imerg_methods_manifest="a9bb596b3222c87647ccd14f9e4e9fbb6f72a6bfb16f67dbc784bd9b6eeedf4e",
    imerg_seasonal_summary="9a17028aa581fc1619126b63777f79bf31979488a6be08a5155f97d47d5238ea",
    imerg_weekly_summary="da3013b08afb1f0e947dff067ff44eb97d72490a06296401275a50871e055932",
    imerg_case_metrics="6f3c6433de533a319f38523a88c9f19f6b376a03f1b3bde67ec36a59fe8dd26a",
    calibration_manifest="a98bf491a41c7245ef2ee0904e4882eb0e3ca3081db2b0291b1f0fa1669fd5df",
    adapter_support="492d7c54167163f8a77cb7ea3c3f16a142be5427e88b477f367bc6278f2659c4",
    spatial_support_metadata="07bb0e60a396a6056df0cea9c3b96861aeb5fe0f1db9640173b9e166306cfbe4",
)

CANONICAL_SPATIAL_ARRAY_HASHES = SpatialArrayHashes(
    latitude="a1ca9eb14bbab26c6ac5f911e32bb5ebe22458a49820e7a12daa17a6dc800ce7",
    longitude="f18a6f780547a636f81300d5f360c7691b375705d2ce046d743b858ad8262e0c",
    india_area_weight_km2="147ccd9bbff94449ee62da84813f2834d04abe9e507e53b5a34ef1b6b10d9e3b",
    northwest_india_fraction="606d3058d7164ac02d0cb15252f10c209e50bd36b81c7316949ae964bb5ca92a",
    central_india_fraction="40e5ca9e3905d22d4e42bad470cffdc2409a3a2fda5be2fd2cd98f3bbbdfe2b1",
    south_peninsula_fraction="a6aecd239b7066530f4674345863bf788dbbbbd43c64bb244fb6602e5f33be25",
    east_northeast_india_fraction="3fe8d6ded4146c05a5220ec4a31ecbf94af4fb43ee61b87400ec240e2356a785",
)

CANONICAL_VISUAL_INSPECTION: dict[str, Any] = {
    "status": "passed",
    "inspected_utc": "2026-08-26T01:08:11Z",
    "method": "manual visual inspection of every rendered PNG at high/original detail",
    "inspection_source_package": str(
        PROJECT_ROOT
        / "resultsv3/india_s2s_paper_figures/full_v3_20260826T010700Z"
    ),
    "inspection_source_manifest_sha256": (
        "e42b97a1e88c69cb2f0cbf84dad7eedde7a92a04b7c7333f03d03f5736bb0da4"
    ),
    "figure_png_sha256": {
        "figures/fig01_domain_windows_splits.png": (
            "8c1312459e21953ba7619ba23b2bc27a211dfd35b2d7e2bfa06b972fd2f0e59e"
        ),
        "figures/fig02_all_model_imd_jjas_acc_rmse.png": (
            "ee43bfc0fe6ddc76c544e0ba36eebaee3d0b4351cbc058bc6057c32194423b68"
        ),
        "figures/fig03_regional_reference_sensitivity.png": (
            "ee521c7564d5e1f1ffe6c36df47c798bc4583fff24b099134182e93ae5d04279"
        ),
        "figures/fig04_retrospective_calibrated_fuxi.png": (
            "72e7f59250ce2a7f66fa1d98951fb4e3102a2484995cc7568e437dd6ee05de2a"
        ),
        "figures/figS01_allseason_imd_rainfall.png": (
            "be9ff95334e9160b55f0234154e2ccadd2421c805e6d332076b2f3e0bb977da6"
        ),
        "figures/figS02_allseason_era5_temperature.png": (
            "a256858cd5ca88ef3cda2a488c4f9b6a0831c89fd09cb8a08824535cd54f60c8"
        ),
    },
    "checks": [
        "all six PNGs opened successfully",
        "titles, axes, legends, annotations, and panel labels are legible",
        "no clipped scientific content or unintended overlaps were observed",
        "Figure 1 support cells, fractional-region mixtures, hatching, weekly windows, and temporal firewall are distinguishable",
        "Figures 2--4 and S1--S2 preserve their intended line, marker, heatmap, interval, and missing-data encodings",
    ],
    "limitations": (
        "Visual inspection checks rendering integrity and readability; it does not replace "
        "the numerical, provenance, or scientific-contract tests."
    ),
    "sealed_2025_target_opened": False,
}


@dataclass
class HashLedger:
    """Input hashes that are rechecked immediately before publication."""

    expected: dict[Path, str]

    def __init__(self) -> None:
        self.expected = {}

    def verify(self, path: Path, expected: str, label: str) -> None:
        resolved = Path(path).resolve()
        _require(
            len(str(expected)) == 64
            and all(character in "0123456789abcdef" for character in str(expected)),
            f"{label} has an invalid SHA-256",
        )
        _require(resolved.is_file(), f"{label} is missing: {resolved}")
        previous = self.expected.get(resolved)
        _require(previous in (None, expected), f"conflicting hash bindings for {label}")
        observed = sha256_file(resolved)
        _require(observed == expected, f"{label} SHA-256 differs: {observed} != {expected}")
        self.expected[resolved] = expected

    def reverify(self) -> None:
        for path, expected in self.expected.items():
            _require(path.is_file(), f"pinned input disappeared: {path}")
            _require(sha256_file(path) == expected, f"pinned input changed: {path}")


def _resolve_child(root: Path, relative: str) -> Path:
    child = Path(str(relative))
    _require(not child.is_absolute(), f"artifact path must be relative: {relative}")
    resolved_root = Path(root).resolve()
    resolved = (resolved_root / child).resolve()
    _require(resolved_root in resolved.parents, f"artifact escapes run directory: {relative}")
    return resolved


def _verify_artifact_inventory(
    receipt_path: Path,
    receipt: Mapping[str, Any],
    ledger: HashLedger,
    label: str,
) -> int:
    root = Path(receipt_path).resolve().parent
    hashes = receipt.get("artifact_sha256")
    _require(isinstance(hashes, dict) and hashes, f"{label} has no artifact inventory")
    expected_paths = {str(relative) for relative in hashes}
    actual_paths = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
        and path.name not in {Path(receipt_path).name, "failure.json"}
        and ".tmp" not in path.name
    }
    _require(
        actual_paths == expected_paths,
        f"{label} inventory differs: extra={sorted(actual_paths - expected_paths)}, "
        f"missing={sorted(expected_paths - actual_paths)}",
    )
    for relative, digest in hashes.items():
        ledger.verify(_resolve_child(root, str(relative)), str(digest), f"{label} {relative}")
    return len(hashes)


def _load_receipts(
    inputs: FigureInputs, hashes: InputHashes, ledger: HashLedger
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, int],
]:
    for field in inputs.__dataclass_fields__:
        ledger.verify(
            Path(getattr(inputs, field)),
            str(getattr(hashes, field)),
            field.replace("_", " "),
        )
    benchmark = _read_json(inputs.benchmark_manifest)
    scoring = _read_json(inputs.scoring_manifest)
    audit = _read_json(inputs.audit_receipt)
    comparison = _read_json(inputs.comparison_manifest)
    imd_methods = _read_json(inputs.imd_methods_manifest)
    imerg_methods = _read_json(inputs.imerg_methods_manifest)
    calibration = _read_json(inputs.calibration_manifest)
    _require(
        benchmark.get("experiment") == BENCHMARK_EXPERIMENT
        and benchmark.get("status") == "complete",
        "benchmark uncertainty receipt is not complete v2",
    )
    _require(
        scoring.get("experiment") == SCORING_EXPERIMENT
        and scoring.get("status") == "complete",
        "probabilistic scoring receipt is not complete v1",
    )
    _require(
        audit.get("experiment") == AUDIT_EXPERIMENT and audit.get("status") == "passed",
        "independent scoring audit did not pass",
    )
    _require(
        comparison.get("experiment") == COMPARISON_EXPERIMENT
        and comparison.get("status") == "complete",
        "paper-comparison addendum is not complete v1",
    )
    _require(
        calibration.get("experiment") == "fuxi_allseason_ensemble_calibration_v2_aligned"
        and calibration.get("status") == "complete",
        "aligned calibration receipt is not complete v2",
    )
    artifact_counts = {
        "benchmark": _verify_artifact_inventory(
            inputs.benchmark_manifest, benchmark, ledger, "benchmark"
        ),
        "scoring": _verify_artifact_inventory(
            inputs.scoring_manifest, scoring, ledger, "scoring"
        ),
        "audit": _verify_artifact_inventory(inputs.audit_receipt, audit, ledger, "audit"),
        "comparison": _verify_artifact_inventory(
            inputs.comparison_manifest, comparison, ledger, "comparison"
        ),
    }
    return (
        benchmark,
        scoring,
        audit,
        comparison,
        imd_methods,
        imerg_methods,
        calibration,
        artifact_counts,
    )


def _validate_methods_manifest(
    manifest: Mapping[str, Any], *, track: str, reference: str, output_group: str
) -> None:
    _require(manifest.get("output_group") == output_group, f"{reference} output group differs")
    _require(manifest.get("included_tracks") and track in manifest["included_tracks"], f"{track} missing")
    methods = manifest.get("methods", {})
    _require(
        methods.get("season_assignment")
        == "season of the exact seven-day valid-window midpoint (init + 7*lead_week - 3.5 days)",
        f"{reference} season assignment differs",
    )
    contract = manifest.get("tracks", {}).get(track, {})
    _require(contract.get("reference") == reference, f"{track} reference differs")
    _require(tuple(contract.get("models", ())) == MODELS[:-1], f"{track} model inventory differs")
    _require(
        tuple(contract.get("mme_components", ())) == FIXED_MME_COMPONENTS,
        f"{track} fixed MME membership differs",
    )
    _require(
        contract.get("unavailable_lead_weeks") == {"ecmwf": [3]},
        f"{track} unavailable-lead contract differs",
    )
    _require(contract.get("total_paired_initializations") == 517, f"{track} paired starts differ")


def _validate_temperature_methods_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate the existing 516-start ERA5 temperature supplement contract."""

    _require(
        "t2m_era5" in manifest.get("included_tracks", ()),
        "ERA5 temperature track is missing",
    )
    contract = manifest.get("tracks", {}).get("t2m_era5", {})
    _require(
        contract.get("variable") == "t2m"
        and contract.get("reference") == "era5"
        and tuple(contract.get("models", ())) == T2M_MODELS[:-1]
        and tuple(contract.get("mme_components", ())) == T2M_MME_COMPONENTS
        and contract.get("unavailable_lead_weeks") == {}
        and contract.get("total_paired_initializations") == 516,
        "ERA5 temperature method contract differs",
    )


def _validate_global_gates(
    inputs: FigureInputs,
    hashes: InputHashes,
    benchmark: Mapping[str, Any],
    scoring: Mapping[str, Any],
    audit: Mapping[str, Any],
    comparison: Mapping[str, Any],
    imd_methods: Mapping[str, Any],
    imerg_methods: Mapping[str, Any],
    calibration: Mapping[str, Any],
) -> None:
    _require(
        benchmark.get("cohort") == "IMD valid-midpoint JJAS"
        and benchmark.get("case_count_per_lead_region") == CASE_COUNT_PER_LEAD,
        "benchmark cohort is not 169-case valid-midpoint JJAS",
    )
    _require(
        Path(str(benchmark.get("source_case_metrics", ""))).resolve()
        == Path(inputs.imd_case_metrics).resolve()
        and benchmark.get("source_case_metrics_sha256") == hashes.imd_case_metrics,
        "benchmark is not bound to the requested IMD case table",
    )
    cohort = scoring.get("cohort", {})
    _require(
        cohort.get("valid_midpoint_jjas_count_per_lead") == CASE_COUNT_PER_LEAD
        and cohort.get("maximum_target_label_opened") == "2024-12-30",
        "scoring cohort or maximum target label differs",
    )
    _require(
        scoring.get("opened_observation_years") == [2020, 2021, 2022, 2023, 2024]
        and scoring.get("opened_2025_observation") is False
        and scoring.get("sealed_2025_target_opened") is False,
        "scoring violates the sealed-2025 contract",
    )
    _require(
        scoring.get("selected_seed") == SELECTED_SEED
        and scoring.get("selected_checkpoint_sha256") == SELECTED_CHECKPOINT_SHA256,
        "scoring did not use frozen seed 43 checkpoint",
    )
    parent = scoring.get("preflight_parent_receipt", {})
    _require(
        parent.get("selected_seed") == SELECTED_SEED
        and parent.get("selected_checkpoint_sha256") == SELECTED_CHECKPOINT_SHA256
        and parent.get("sealed_2025_target_opened") is False,
        "preflight parent selection or seal differs",
    )
    _require(
        Path(str(scoring.get("parent_manifest", ""))).resolve()
        == Path(inputs.calibration_manifest).resolve()
        and scoring.get("parent_manifest_sha256") == hashes.calibration_manifest
        and Path(str(parent.get("parent_manifest", ""))).resolve()
        == Path(inputs.calibration_manifest).resolve()
        and parent.get("parent_manifest_sha256") == hashes.calibration_manifest,
        "scoring is not bound to the aligned calibration manifest",
    )
    calibration_evaluation = calibration.get("evaluation", {})
    support_relative = str(calibration_evaluation.get("scoring_support_artifact", ""))
    support_path = (Path(inputs.calibration_manifest).resolve().parent / support_relative).resolve()
    _require(
        support_path == Path(inputs.adapter_support).resolve()
        and calibration_evaluation.get("scoring_support_sha256") == hashes.adapter_support
        and calibration.get("artifact_sha256", {}).get(support_relative)
        == hashes.adapter_support
        and calibration_evaluation.get("support_cells") == 171,
        "calibration does not bind the frozen 171-cell support artifact",
    )
    raw = scoring.get("raw_fuxi_identity", {})
    _require(
        raw.get("identity") is True
        and raw.get("complete_bridge_contract") is True
        and raw.get("matched_case_rows") == RAW_IDENTITY_ROWS,
        "raw FuXi identity gate failed",
    )
    _require(scoring.get("table_rows", {}).get("case_metrics") == CASE_ROWS, "case rows differ")
    truth = scoring.get("truth_access", {})
    _require(
        truth.get("opened_2025") is False
        and truth.get("sealed_2025_target_opened") is False
        and truth.get("maximum_target_label") == "2024-12-30",
        "truth-access receipt violates the 2025 seal",
    )
    _require(
        Path(str(truth.get("spatial_support_store", ""))).resolve()
        == Path(inputs.spatial_support_metadata).resolve().parent
        and truth.get("spatial_support_zmetadata_sha256")
        == hashes.spatial_support_metadata
        and Path(str(calibration_evaluation.get("spatial_area_source", ""))).resolve()
        == Path(inputs.spatial_support_metadata).resolve().parent,
        "scoring and calibration do not share the pinned spatial-support store",
    )
    _require(
        audit.get("scoring_manifest_sha256") == hashes.scoring_manifest
        and Path(str(audit.get("scoring_manifest", ""))).resolve()
        == Path(inputs.scoring_manifest).resolve()
        and audit.get("input_hashes_reverified_after_semantic_audit") is True,
        "audit is not bound to the requested scoring manifest",
    )
    semantic = audit.get("semantic_checks", {})
    _require(
        semantic.get("selected_seed") == SELECTED_SEED
        and semantic.get("raw_identity_rows") == RAW_IDENTITY_ROWS
        and semantic.get("full_case_cartesian_rows") == CASE_ROWS
        and semantic.get("sealed_2025_target_opened") is False
        and semantic.get("forecast_or_truth_arrays_opened_by_auditor") is False
        and semantic.get("valid_midpoint_jjas_summary_reconstructed") is True
        and semantic.get("synchronized_pooled_story_gate_reconstructed") is True,
        "independent semantic-audit gates differ",
    )
    story = audit.get("story_gate", {})
    _require(
        story.get("analysis_cohort") == STORY_COHORT
        and story.get("cases_per_lead") == 100
        and story.get("case_lead_rows") == 600
        and story.get("date_intersection_count") == 70
        and story.get("date_union_count") == 130
        and story.get("primary_block_length_starts") == PRIMARY_BLOCK_LENGTH
        and story.get("retrospective_2022_2024_component_passes") is True
        and story.get("final_headline_gate")
        == "pending_sealed_2025_point-estimate_direction_check"
        and story.get("sealed_2025_target_opened") is False,
        "retrospective story gate or sealed prospective gate differs",
    )
    comparison_inputs = comparison.get("inputs", {})
    comparison_safety = comparison.get("scoring_safety_contract", {})
    comparison_mme = comparison.get("calibrated_fuxi_vs_mme", {})
    comparison_common = comparison.get("exact_common_midpoint", {})
    _require(
        comparison_inputs.get("scoring_manifest_sha256") == hashes.scoring_manifest
        and comparison_inputs.get("audit_receipt_sha256") == hashes.audit_receipt
        and comparison_inputs.get("benchmark_cases_sha256") == hashes.imd_case_metrics,
        "comparison addendum is not bound to the frozen score/audit/benchmark inputs",
    )
    _require(
        comparison_inputs.get("methods_manifest_sha256") == hashes.imd_methods_manifest
        and Path(str(comparison_inputs.get("methods_manifest", ""))).resolve()
        == Path(inputs.imd_methods_manifest).resolve(),
        "comparison addendum is not bound to the fixed-MME methods manifest",
    )
    _require(
        comparison.get("selected_seed") == SELECTED_SEED
        and comparison.get("retrospective_only") is True
        and comparison.get("sealed_2025_target_opened") is False
        and comparison.get("latest_target_year") == 2024
        and comparison_safety.get("valid_midpoint_jjas_count_per_lead")
        == CASE_COUNT_PER_LEAD
        and comparison_safety.get("maximum_target_label_opened") == "2024-12-30",
        "comparison addendum selection or 2025 safety contract differs",
    )
    _require(
        comparison_mme.get("cases_per_lead") == CASE_COUNT_PER_LEAD
        and comparison_mme.get("ecmwf_in_mme") is False
        and tuple(comparison_mme.get("mme_members", ())) == FIXED_MME_COMPONENTS,
        "comparison addendum does not use the fixed six-system MME",
    )
    _require(
        comparison_common.get("common_midpoints_per_lead") == 85
        and comparison_common.get("case_lead_rows") == 510
        and comparison_common.get("sampling_unit")
        == "valid-period midpoint synchronized exactly across six leads",
        "exact-common midpoint sensitivity contract differs",
    )
    _validate_methods_manifest(
        imd_methods, track="tp_imd", reference="imd", output_group="imd_tp_sensitivity"
    )
    _validate_methods_manifest(
        imerg_methods,
        track="tp_imerg",
        reference="imerg",
        output_group="imerg_era5_primary",
    )
    _validate_temperature_methods_manifest(imerg_methods)


def _load_figure1_geometry(
    inputs: FigureInputs,
    expected_hashes: SpatialArrayHashes,
) -> Figure1Geometry:
    """Load only the pinned static geometry and frozen calibration support."""

    with np.load(inputs.adapter_support, allow_pickle=False) as archive:
        required = {
            "latitude",
            "longitude",
            "observation_fraction",
            "support_mask",
            "scoring_weight_km2_fraction",
        }
        _require(required.issubset(archive.files), "adapter support keys differ")
        adapter_latitude = np.asarray(archive["latitude"], dtype=np.float64)
        adapter_longitude = np.asarray(archive["longitude"], dtype=np.float64)
        adapter_support = np.asarray(archive["support_mask"], dtype=bool)
        observation_fraction = np.asarray(
            archive["observation_fraction"], dtype=np.float32
        )
        scoring_weight = np.asarray(
            archive["scoring_weight_km2_fraction"], dtype=np.float64
        )
    _require(
        adapter_latitude.shape == (27,)
        and adapter_longitude.shape == (27,)
        and adapter_support.shape == (27, 27),
        "adapter support grid is not 27 x 27",
    )
    _require(
        observation_fraction.shape == (27, 27)
        and scoring_weight.shape == (27, 27)
        and np.isfinite(observation_fraction).all()
        and np.isfinite(scoring_weight).all()
        and np.all((observation_fraction >= 0.0) & (observation_fraction <= 1.0))
        and np.all(scoring_weight >= 0.0),
        "adapter support weights are invalid",
    )
    _require(
        int(np.count_nonzero(adapter_support)) == 171
        and np.array_equal(adapter_support, scoring_weight > 0.0),
        "adapter support is not the frozen 171-cell mask",
    )

    try:
        import xarray as xr
    except ImportError as error:  # pragma: no cover - dependency is part of the project env
        raise PaperFigureError("xarray is required to load Figure 1 geometry") from error
    store = Path(inputs.spatial_support_metadata).resolve().parent
    with xr.open_zarr(store, consolidated=True) as dataset:
        latitude = np.asarray(dataset.latitude.values, dtype=np.float64)
        longitude = np.asarray(dataset.longitude.values, dtype=np.float64)
        india_area = np.asarray(
            dataset.india_area_weight_km2.load().values, dtype=np.float64
        )
        region_fractions = {
            region: np.asarray(
                dataset[f"{region}_fraction"].load().values, dtype=np.float32
            )
            for region in REGIONS[1:]
        }
    arrays = {
        "latitude": latitude,
        "longitude": longitude,
        "india_area_weight_km2": india_area,
        **{f"{region}_fraction": values for region, values in region_fractions.items()},
    }
    logical_hashes: dict[str, str] = {}
    for name, dtype in SPATIAL_ARRAY_DTYPES.items():
        observed = array_sha256(arrays[name], dtype)
        expected = str(getattr(expected_hashes, name))
        _require(observed == expected, f"spatial logical array differs: {name}")
        logical_hashes[name] = observed
    _require(
        latitude.shape == (27,)
        and longitude.shape == (27,)
        and india_area.shape == (27, 27)
        and np.array_equal(latitude, adapter_latitude)
        and np.array_equal(longitude, adapter_longitude),
        "spatial and adapter grids differ",
    )
    _require(
        np.isfinite(india_area).all() and np.all(india_area >= 0.0),
        "India geometry weights are invalid",
    )
    india_geometry = india_area > 0.0
    _require(
        int(np.count_nonzero(india_geometry)) == 174
        and np.all(~adapter_support | india_geometry),
        "India geometry or its relation to adapter support differs",
    )
    for region, fraction in region_fractions.items():
        _require(
            fraction.shape == (27, 27)
            and np.isfinite(fraction).all()
            and np.all((fraction >= 0.0) & (fraction <= 1.0))
            and np.any(fraction > 0.0),
            f"invalid fractional geometry for {region}",
        )
    fraction_sum = np.sum(np.stack(list(region_fractions.values())), axis=0)
    _require(
        int(np.count_nonzero(fraction_sum > 0.0)) == 174
        and np.array_equal(fraction_sum > 0.0, india_geometry)
        and np.all(fraction_sum <= 1.0 + 1.0e-6),
        "fractional regions do not partition the India geometry",
    )
    return Figure1Geometry(
        latitude=latitude,
        longitude=longitude,
        india_geometry=india_geometry,
        adapter_support=adapter_support,
        region_fractions=region_fractions,
        logical_sha256=logical_hashes,
    )


def _figure1_geometry_frame(geometry: Figure1Geometry) -> pd.DataFrame:
    """Return a deterministic, paper-facing long table for the domain panel."""

    rows: list[dict[str, Any]] = []
    fractions = np.stack([geometry.region_fractions[name] for name in REGIONS[1:]])
    for row_index, latitude in enumerate(geometry.latitude):
        for column_index, longitude in enumerate(geometry.longitude):
            values = fractions[:, row_index, column_index]
            fraction_sum = float(values.sum(dtype=np.float64))
            dominant = REGIONS[1:][int(np.argmax(values))] if fraction_sum > 0.0 else ""
            row: dict[str, Any] = {
                "row_index": row_index,
                "column_index": column_index,
                "latitude": float(latitude),
                "longitude": float(longitude),
                "india_geometry": bool(geometry.india_geometry[row_index, column_index]),
                "adapter_support": bool(geometry.adapter_support[row_index, column_index]),
                "fraction_sum": fraction_sum,
                "dominant_region": dominant,
            }
            for region_index, region in enumerate(REGIONS[1:]):
                row[f"{region}_fraction"] = float(values[region_index])
            rows.append(row)
    return pd.DataFrame(rows)


def _read_csv(path: Path, label: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as error:
        raise PaperFigureError(f"cannot read {label}: {path}") from error


def _load_seasonal(path: Path, *, track: str, reference: str) -> pd.DataFrame:
    frame = _read_csv(path, f"{reference} seasonal summary")
    required = {
        "track",
        "variable",
        "reference",
        "season",
        "region",
        "region_label",
        "model",
        "lead_week",
        "case_count",
        "acc_valid_case_count",
        "acc",
        "rmse_valid_case_count",
        "rmse",
    }
    _require(required.issubset(frame.columns), f"{reference} seasonal table lacks columns")
    selected = frame[
        frame["track"].eq(track)
        & frame["variable"].eq("tp")
        & frame["reference"].eq(reference)
        & frame["season"].eq("JJAS")
        & frame["region"].isin(REGIONS)
        & frame["model"].isin(MODELS)
    ].copy()
    _require(
        not selected.duplicated(["region", "model", "lead_week"]).any(),
        f"{reference} seasonal table has duplicate rows",
    )
    _require(
        set(selected["region"]) == set(REGIONS)
        and set(selected["model"]) == set(MODELS)
        and set(selected["lead_week"].astype(int)) == set(LEADS),
        f"{reference} seasonal inventory differs",
    )
    expected_rows = len(REGIONS) * len(MODELS) * len(LEADS)
    _require(len(selected) == expected_rows, f"{reference} seasonal Cartesian rows differ")
    _require((selected["case_count"] == CASE_COUNT_PER_LEAD).all(), f"{reference} case count differs")
    missing = selected[
        selected["model"].eq("ecmwf") & selected["lead_week"].astype(int).eq(3)
    ]
    _require(
        len(missing) == len(REGIONS)
        and (missing["acc_valid_case_count"] == 0).all()
        and (missing["rmse_valid_case_count"] == 0).all()
        and missing[["acc", "rmse"]].isna().all().all(),
        f"{reference} does not encode ECMWF W3 as unavailable",
    )
    available = selected.drop(missing.index)
    _require(
        (available["acc_valid_case_count"] == CASE_COUNT_PER_LEAD).all()
        and (available["rmse_valid_case_count"] == CASE_COUNT_PER_LEAD).all()
        and np.isfinite(available[["acc", "rmse"]].to_numpy(dtype=float)).all(),
        f"{reference} has incomplete available rows",
    )
    return selected


def _load_allseason_summary(
    path: Path,
    *,
    track: str,
    variable: str,
    reference: str,
    models: Sequence[str],
    case_count: int,
    unavailable: set[tuple[str, int]],
) -> pd.DataFrame:
    """Load a validated all-season weekly summary for a supplement."""

    frame = _read_csv(path, f"{reference} all-season summary")
    required = {
        "track",
        "variable",
        "reference",
        "region",
        "region_label",
        "model",
        "lead_week",
        "case_count",
        "acc_valid_case_count",
        "acc",
        "rmse_valid_case_count",
        "rmse",
    }
    _require(required.issubset(frame.columns), f"{reference} all-season table lacks columns")
    selected = frame[
        frame["track"].eq(track)
        & frame["variable"].eq(variable)
        & frame["reference"].eq(reference)
        & frame["region"].isin(REGIONS)
        & frame["model"].isin(models)
    ].copy()
    _require(
        len(selected) == len(REGIONS) * len(models) * len(LEADS)
        and not selected.duplicated(["region", "model", "lead_week"]).any()
        and set(selected["region"]) == set(REGIONS)
        and set(selected["model"]) == set(models)
        and set(selected["lead_week"].astype(int)) == set(LEADS)
        and (selected["case_count"] == case_count).all(),
        f"{reference} all-season inventory differs",
    )
    unavailable_mask = pd.Series(False, index=selected.index)
    for model, lead in unavailable:
        unavailable_mask |= selected["model"].eq(model) & selected["lead_week"].astype(int).eq(lead)
    missing = selected[unavailable_mask]
    available = selected[~unavailable_mask]
    _require(
        len(missing) == len(unavailable) * len(REGIONS)
        and (
            missing.empty
            or (
                (missing["acc_valid_case_count"] == 0).all()
                and (missing["rmse_valid_case_count"] == 0).all()
                and missing[["acc", "rmse"]].isna().all().all()
            )
        ),
        f"{reference} unavailable all-season rows differ",
    )
    _require(
        (available["acc_valid_case_count"] == case_count).all()
        and (available["rmse_valid_case_count"] == case_count).all()
        and np.isfinite(available[["acc", "rmse"]].to_numpy(dtype=float)).all(),
        f"{reference} available all-season rows are incomplete",
    )
    return selected


def _case_id_set(path: Path, *, track: str, reference: str) -> set[tuple[str, str, int, str]]:
    frame = _read_csv(path, f"{reference} case metrics")
    required = {
        "track",
        "variable",
        "reference",
        "season",
        "region",
        "model",
        "init",
        "valid_period_midpoint",
        "lead_week",
        "score_status",
    }
    _require(required.issubset(frame.columns), f"{reference} case table lacks columns")
    midpoint = pd.to_datetime(frame["valid_period_midpoint"], errors="raise")
    selected = frame[
        frame["track"].eq(track)
        & frame["variable"].eq("tp")
        & frame["reference"].eq(reference)
        & frame["season"].eq("JJAS")
        & midpoint.dt.month.isin((6, 7, 8, 9))
        & frame["region"].isin(REGIONS)
        & frame["model"].isin(("mme", "fuxi_s2s"))
        & frame["score_status"].eq("available")
    ].copy()
    _require(
        not selected.duplicated(["model", "init", "lead_week", "region"]).any(),
        f"{reference} matched case IDs contain duplicates",
    )
    counts = selected.groupby(["model", "region", "lead_week"], observed=True).size()
    _require(
        len(counts) == 2 * len(REGIONS) * len(LEADS)
        and (counts == CASE_COUNT_PER_LEAD).all(),
        f"{reference} matched case cohort is not 169 per model/region/lead",
    )
    return {
        (str(row.model), str(row.init), int(row.lead_week), str(row.region))
        for row in selected.itertuples(index=False)
    }


def _assert_frames_equal(left: pd.DataFrame, right: pd.DataFrame, label: str) -> None:
    _require(set(left.columns) == set(right.columns), f"{label} columns differ")
    columns = sorted(left.columns)
    left_normalized = left[columns].copy()
    right_normalized = right[columns].copy()
    for column in columns:
        left_numeric = pd.to_numeric(left_normalized[column], errors="coerce")
        right_numeric = pd.to_numeric(right_normalized[column], errors="coerce")
        if (
            left_numeric.notna().sum() == left_normalized[column].notna().sum()
            and right_numeric.notna().sum() == right_normalized[column].notna().sum()
        ):
            left_normalized[column] = left_numeric.astype(float)
            right_normalized[column] = right_numeric.astype(float)
        else:
            left_normalized[column] = left_normalized[column].astype("string")
            right_normalized[column] = right_normalized[column].astype("string")
    left_sorted = left_normalized.sort_values(columns, kind="stable").reset_index(drop=True)
    right_sorted = right_normalized.sort_values(columns, kind="stable").reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(
            left_sorted,
            right_sorted,
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
    except AssertionError as error:
        raise PaperFigureError(f"{label} values differ") from error


def _load_calibration_tables(
    inputs: FigureInputs,
    scoring: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scoring_root = Path(inputs.scoring_manifest).resolve().parent
    audit_root = Path(inputs.audit_receipt).resolve().parent
    summary = _read_csv(scoring_root / "tables/jjas_valid_midpoint_summary.csv", "scoring JJAS summary")
    audit_summary = _read_csv(
        audit_root / "reconstructed/jjas_valid_midpoint_summary.csv", "audited JJAS summary"
    )
    _assert_frames_equal(summary, audit_summary, "scoring/audit JJAS summary")
    required = {
        "method",
        "region",
        "lead_week",
        "case_count",
        "crps",
        "acc",
        "coverage90",
        "pooled_spread_skill_ratio",
    }
    _require(required.issubset(summary.columns), "JJAS summary lacks figure columns")
    _require(
        len(summary) == len(METHODS) * len(REGIONS) * len(LEADS)
        and set(summary["method"]) == set(METHODS)
        and set(summary["region"]) == set(REGIONS)
        and set(summary["lead_week"].astype(int)) == set(LEADS)
        and (summary["case_count"] == CASE_COUNT_PER_LEAD).all(),
        "JJAS summary does not satisfy its 3 x 5 x 6, N=169 contract",
    )
    all_india = summary[summary["region"].eq("all_india")].copy()
    _require(
        np.isfinite(
            all_india[["crps", "acc", "coverage90", "pooled_spread_skill_ratio"]].to_numpy(
                dtype=float
            )
        ).all(),
        "all-India calibration curves contain non-finite values",
    )

    intervals = _read_csv(scoring_root / "tables/paired_block_intervals.csv", "paired intervals")
    audit_intervals = _read_csv(
        audit_root / "reconstructed/paired_block_intervals.csv", "audited paired intervals"
    )
    scoring_nonstory = intervals[
        ~intervals["analysis_cohort"].eq(STORY_COHORT)
    ][audit_intervals.columns].copy()
    _assert_frames_equal(
        scoring_nonstory, audit_intervals, "scoring/audit non-pooled paired intervals"
    )
    effects = intervals[
        intervals["analysis_cohort"].eq("2020_2024_valid_midpoint_jjas")
        & intervals["region"].eq("all_india")
        & intervals["candidate"].eq("location_spread")
        & intervals["baseline"].eq("raw_fuxi")
        & intervals["block_length_starts"].eq(PRIMARY_BLOCK_LENGTH)
        & (
            (
                intervals["metric"].eq("crps")
                & intervals["effect"].eq("skill_pct_vs_baseline")
            )
            | (
                intervals["metric"].eq("acc")
                & intervals["effect"].eq("candidate_minus_baseline")
            )
        )
    ].copy()
    effects["lead_week"] = pd.to_numeric(effects["lead_week"], errors="coerce")
    _require(
        len(effects) == 12
        and set(effects["lead_week"].astype(int)) == set(LEADS)
        and (effects["n_cases"] == CASE_COUNT_PER_LEAD).all()
        and (effects["replicates"] == 10_000).all(),
        "primary per-lead effect interval contract differs",
    )

    story = audit.get("story_gate", {})
    audit_story = _read_csv(
        audit_root / "reconstructed/story_gate_intervals.csv", "audited story intervals"
    )
    primary_story = audit_story[audit_story["block_length_starts"].eq(PRIMARY_BLOCK_LENGTH)]
    _require(
        len(primary_story) == 8
        and (primary_story["analysis_cohort"] == STORY_COHORT).all()
        and (primary_story["n_cases_per_lead"] == 100).all()
        and (primary_story["n_case_lead_rows"] == 600).all(),
        "primary retrospective story interval contract differs",
    )
    specifications = {
        "crps_skill_pct_vs_raw": ("crps", "raw_fuxi", "skill_pct_vs_baseline"),
        "crps_skill_pct_vs_moment": (
            "crps",
            "moment_calibration",
            "skill_pct_vs_baseline",
        ),
        "acc_delta_vs_raw": ("acc", "raw_fuxi", "candidate_minus_baseline"),
        "bias_delta_vs_raw": ("bias", "raw_fuxi", "candidate_minus_baseline"),
    }
    for key, (metric, baseline, effect) in specifications.items():
        match = primary_story[
            primary_story["metric"].eq(metric)
            & primary_story["baseline"].eq(baseline)
            & primary_story["effect"].eq(effect)
        ]
        _require(len(match) == 1, f"missing retrospective story interval: {key}")
        values = story.get(key, {})
        for column in ("estimate", "ci_lower", "ci_upper"):
            _require(
                np.isclose(float(match.iloc[0][column]), float(values.get(column, np.nan)), atol=1e-12),
                f"audit story JSON disagrees for {key}.{column}",
            )
    return all_india, effects, primary_story


def _load_comparison_tables(
    inputs: FigureInputs,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate the exact-common and calibrated-vs-MME addendum."""

    root = Path(inputs.comparison_manifest).resolve().parent
    exact = _read_csv(
        root / "tables/exact_common_midpoint_sensitivity.csv",
        "exact-common midpoint sensitivity",
    )
    mme = _read_csv(
        root / "tables/calibrated_fuxi_vs_mme.csv",
        "calibrated FuXi versus MME",
    )
    exact_primary = exact[
        exact["block_length_valid_midpoints"].eq(PRIMARY_BLOCK_LENGTH)
    ].copy()
    _require(
        len(exact_primary) == 8
        and (exact_primary["analysis_cohort"]
             == "2022_2024_exact_common_valid_midpoints_pooled_w1_w6").all()
        and (exact_primary["candidate"] == "location_spread").all()
        and (exact_primary["n_common_midpoints_per_lead"] == 85).all()
        and (exact_primary["n_case_lead_rows"] == 510).all()
        and (exact_primary["replicates"] == 10_000).all(),
        "exact-common primary interval table differs",
    )
    exact_crps = exact_primary[
        exact_primary["metric"].eq("crps")
        & exact_primary["baseline"].eq("raw_fuxi")
        & exact_primary["effect"].eq("skill_pct_vs_baseline")
    ]
    exact_acc = exact_primary[
        exact_primary["metric"].eq("acc")
        & exact_primary["baseline"].eq("raw_fuxi")
        & exact_primary["effect"].eq("candidate_minus_baseline")
    ]
    _require(
        len(exact_crps) == 1
        and len(exact_acc) == 1
        and float(exact_crps.iloc[0]["ci_lower"]) > 0.0
        and float(exact_acc.iloc[0]["ci_lower"]) > 0.0,
        "exact-common CRPS/ACC sensitivity is not supported",
    )

    mme_primary = mme[mme["block_length_starts"].eq(PRIMARY_BLOCK_LENGTH)].copy()
    _require(
        len(mme_primary) == 36
        and set(mme_primary["lead_week"].astype(int)) == set(LEADS)
        and (mme_primary["analysis_cohort"] == "2020_2024_valid_midpoint_jjas").all()
        and (mme_primary["candidate"] == "location_spread").all()
        and (mme_primary["baseline"] == "equal_system_mme_no_ecmwf").all()
        and (mme_primary["n_cases"] == CASE_COUNT_PER_LEAD).all()
        and (mme_primary["replicates"] == 10_000).all(),
        "calibrated-FuXi versus fixed-MME primary table differs",
    )
    rmse = mme_primary[
        mme_primary["metric"].eq("rmse")
        & mme_primary["effect"].eq("candidate_minus_baseline")
    ].sort_values("lead_week")
    acc = mme_primary[
        mme_primary["metric"].eq("acc")
        & mme_primary["effect"].eq("candidate_minus_baseline")
    ].sort_values("lead_week")
    _require(
        len(rmse) == len(LEADS)
        and (rmse["estimate"] < 0.0).all()
        and (rmse["ci_upper"] < 0.0).all(),
        "calibrated neural RMSE is not lower than the fixed MME at W1-W6",
    )
    significant_acc_leads = set(
        acc.loc[(acc["estimate"] > 0.0) & (acc["ci_lower"] > 0.0), "lead_week"].astype(int)
    )
    week2 = acc[acc["lead_week"].astype(int).eq(2)]
    _require(
        significant_acc_leads == {1, 3, 4, 5, 6}
        and len(week2) == 1
        and float(week2.iloc[0]["ci_lower"]) <= 0.0 <= float(week2.iloc[0]["ci_upper"]),
        "calibrated neural ACC versus fixed-MME lead pattern differs",
    )
    return exact_primary, mme_primary


def _style() -> dict[str, Any]:
    return {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.transparent": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }


MODEL_STYLES = {
    "cma": ("#0072B2", "o", "-"),
    "dlesym_v0": ("#56B4E9", "s", "--"),
    "ecmwf": ("#CC79A7", "^", "-"),
    "fuxi_s2s": ("#D55E00", "D", "-"),
    "ncep": ("#009E73", "v", "-."),
    "neuralgcm": ("#E69F00", "P", ":"),
    "ukmo": ("#882255", "X", "--"),
    "mme": ("#000000", "*", "-"),
}
T2M_MODEL_STYLES = {
    "cma": ("#0072B2", "o", "-"),
    "dlesym_v0": ("#56B4E9", "s", "--"),
    "dlesym_v1": ("#009E73", "P", "-"),
    "ecmwf": ("#CC79A7", "^", "-"),
    "fcn3": ("#E69F00", "v", "-."),
    "fuxi_s2s": ("#D55E00", "D", "-"),
    "ukmo": ("#882255", "X", "--"),
    "mme": ("#000000", "*", "-"),
}
METHOD_STYLES = {
    "raw_fuxi": ("#4D4D4D", "o", "--"),
    "moment_calibration": ("#E69F00", "s", "-."),
    "location_spread": ("#0072B2", "D", "-"),
}


def _panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.13,
        1.07,
        label,
        transform=axis.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
    )


def _save_figure(figure: plt.Figure, base: Path, title: str) -> tuple[Path, Path]:
    png = base.with_suffix(".png")
    pdf = base.with_suffix(".pdf")
    figure.savefig(
        png,
        dpi=220,
        bbox_inches="tight",
        metadata={"Software": EXPERIMENT, "Title": title},
    )
    figure.savefig(
        pdf,
        bbox_inches="tight",
        metadata={
            "Title": title,
            "Author": "",
            "Subject": "India S2S retrospective paper evidence",
            "Keywords": "subseasonal; India; monsoon; retrospective",
            "Creator": EXPERIMENT,
            "Producer": EXPERIMENT,
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(figure)
    return png, pdf


def _figure2(imd: pd.DataFrame, output: Path) -> tuple[Path, Path]:
    all_india = imd[imd["region"].eq("all_india")]
    with plt.rc_context(_style()):
        figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.25), sharex=True)
        for metric, axis, ylabel in (
            ("acc", axes[0], "Spatial anomaly correlation (ACC)"),
            ("rmse", axes[1], "RMSE (mm day$^{-1}$)"),
        ):
            for model in MODELS:
                rows = all_india[all_india["model"].eq(model)].sort_values("lead_week")
                color, marker, linestyle = MODEL_STYLES[model]
                emphasized = model in {"mme", "fuxi_s2s"}
                axis.plot(
                    rows["lead_week"],
                    rows[metric],
                    color=color,
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=2.25 if emphasized else 1.15,
                    markersize=6 if emphasized else 4.2,
                    alpha=1.0 if emphasized else 0.78,
                    label=MODEL_LABELS[model],
                    zorder=4 if emphasized else 2,
                )
            axis.set_xticks(LEADS, [f"W{lead}" for lead in LEADS])
            axis.set_xlabel("Forecast lead")
            axis.set_ylabel(ylabel)
            axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.8)
            floor, ceiling = axis.get_ylim()
            marker_y = floor + 0.035 * (ceiling - floor)
            axis.scatter([3], [marker_y], marker="x", color="#CC79A7", s=42, linewidth=1.5)
            axis.annotate(
                "ECMWF W3 unavailable",
                xy=(3, marker_y),
                xytext=(3.35, floor + 0.13 * (ceiling - floor)),
                arrowprops={"arrowstyle": "-", "color": "#777777", "lw": 0.7},
                color="#555555",
                fontsize=7.5,
            )
        axes[0].set_title("JJAS spatial anomaly skill")
        axes[1].set_title("JJAS rainfall-magnitude error")
        _panel_label(axes[0], "a")
        _panel_label(axes[1], "b")
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.035),
            ncol=4,
            frameon=False,
        )
        figure.suptitle(
            "Five-year India JJAS rainfall benchmark (IMD; 169 cases per lead)",
            fontsize=12,
            y=1.01,
        )
        figure.subplots_adjust(bottom=0.22, wspace=0.27)
        return _save_figure(
            figure,
            output / "fig02_all_model_imd_jjas_acc_rmse",
            "Five-year India JJAS rainfall benchmark",
        )


def _supplement_allseason_two_panel(
    frame: pd.DataFrame,
    *,
    models: Sequence[str],
    labels: Mapping[str, str],
    styles: Mapping[str, tuple[str, str, str]],
    case_count: int,
    reference_label: str,
    variable_label: str,
    rmse_label: str,
    output: Path,
    stem: str,
    unavailable: set[tuple[str, int]],
) -> tuple[Path, Path]:
    """Render a compact descriptive all-season ACC/RMSE supplement."""

    all_india = frame[frame["region"].eq("all_india")]
    with plt.rc_context(_style()):
        figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.25), sharex=True)
        for metric, axis, ylabel, title in (
            ("acc", axes[0], "Spatial anomaly correlation (ACC)", "Spatial anomaly skill"),
            ("rmse", axes[1], rmse_label, "Raw-field error"),
        ):
            for model in models:
                rows = all_india[all_india["model"].eq(model)].sort_values("lead_week")
                color, marker, linestyle = styles[model]
                emphasized = model in {"mme", "fuxi_s2s"}
                axis.plot(
                    rows["lead_week"],
                    rows[metric],
                    color=color,
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=2.25 if emphasized else 1.15,
                    markersize=6 if emphasized else 4.2,
                    alpha=1.0 if emphasized else 0.78,
                    label=labels[model],
                    zorder=4 if emphasized else 2,
                )
            axis.set_xticks(LEADS, [f"W{lead}" for lead in LEADS])
            axis.set_xlabel("Forecast lead")
            axis.set_ylabel(ylabel)
            axis.set_title(title)
            axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.8)
            if unavailable:
                floor, ceiling = axis.get_ylim()
                for model, lead in sorted(unavailable):
                    marker_y = floor + 0.035 * (ceiling - floor)
                    axis.scatter(
                        [lead],
                        [marker_y],
                        marker="x",
                        color=styles[model][0],
                        s=38,
                        linewidth=1.4,
                    )
                    axis.annotate(
                        f"{labels[model]} W{lead} unavailable",
                        xy=(lead, marker_y),
                        xytext=(lead + 0.35, floor + 0.13 * (ceiling - floor)),
                        arrowprops={"arrowstyle": "-", "color": "#777777", "lw": 0.7},
                        color="#555555",
                        fontsize=7.5,
                    )
        _panel_label(axes[0], "a")
        _panel_label(axes[1], "b")
        handles, legend_labels = axes[0].get_legend_handles_labels()
        figure.legend(
            handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.035),
            ncol=4,
            frameon=False,
        )
        figure.suptitle(
            f"All-season India {variable_label} benchmark ({reference_label}; {case_count} starts)",
            fontsize=12,
            y=1.01,
        )
        figure.subplots_adjust(bottom=0.22, wspace=0.27)
        return _save_figure(
            figure,
            output / stem,
            f"All-season India {variable_label} benchmark",
        )


def _matrix(frame: pd.DataFrame, model: str, metric: str = "acc") -> np.ndarray:
    matrix = np.full((len(REGIONS), len(LEADS)), np.nan, dtype=float)
    for row_index, region in enumerate(REGIONS):
        for column_index, lead in enumerate(LEADS):
            match = frame[
                frame["region"].eq(region)
                & frame["model"].eq(model)
                & frame["lead_week"].astype(int).eq(lead)
            ]
            _require(len(match) == 1, f"missing {model}/{region}/W{lead}")
            matrix[row_index, column_index] = float(match.iloc[0][metric])
    return matrix


def _relative_luminance(rgba: Sequence[float]) -> float:
    """Return WCAG relative luminance for one rendered RGBA color."""

    rgb = np.asarray(rgba[:3], dtype=float)
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return float(np.dot(linear, np.array([0.2126, 0.7152, 0.0722])))


def _annotation_text_color(image: Any, value: float) -> str:
    """Choose the higher-contrast light or dark text for a heatmap cell."""

    luminance = _relative_luminance(image.cmap(image.norm(value)))
    dark_luminance = _relative_luminance((0.08, 0.08, 0.08, 1.0))
    white_contrast = 1.05 / (luminance + 0.05)
    dark_contrast = (luminance + 0.05) / (dark_luminance + 0.05)
    return "white" if white_contrast >= dark_contrast else "#141414"


def _annotate_heatmap(
    axis: plt.Axes,
    image: Any,
    values: np.ndarray,
    *,
    difference: bool,
) -> None:
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if difference:
                label = f"{value:+.02f}"
            else:
                label = f"{value:.02f}"
            axis.text(
                column,
                row,
                label,
                ha="center",
                va="center",
                fontsize=7,
                color=_annotation_text_color(image, float(value)),
            )


def _format_heatmap(axis: plt.Axes, title: str, show_y: bool) -> None:
    axis.set_xticks(range(len(LEADS)), [f"W{lead}" for lead in LEADS])
    axis.set_yticks(
        range(len(REGIONS)),
        [REGION_LABELS[region] for region in REGIONS] if show_y else [""] * len(REGIONS),
    )
    axis.set_xlabel("Forecast lead")
    axis.set_title(title)
    axis.tick_params(length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)


def _figure3(imd: pd.DataFrame, imerg: pd.DataFrame, output: Path) -> tuple[Path, Path]:
    mme_imd = _matrix(imd, "mme")
    fuxi_imd = _matrix(imd, "fuxi_s2s")
    mme_reference_delta = _matrix(imerg, "mme") - mme_imd
    fuxi_reference_delta = _matrix(imerg, "fuxi_s2s") - fuxi_imd
    acc_min = min(float(np.nanmin(mme_imd)), float(np.nanmin(fuxi_imd)))
    acc_max = max(float(np.nanmax(mme_imd)), float(np.nanmax(fuxi_imd)))
    delta_limit = max(
        abs(float(np.nanmin(mme_reference_delta))),
        abs(float(np.nanmax(mme_reference_delta))),
        abs(float(np.nanmin(fuxi_reference_delta))),
        abs(float(np.nanmax(fuxi_reference_delta))),
        0.01,
    )
    with plt.rc_context(_style()):
        figure, axes = plt.subplots(2, 2, figsize=(11.2, 7.1), constrained_layout=True)
        image0 = axes[0, 0].imshow(mme_imd, cmap="viridis", vmin=acc_min, vmax=acc_max, aspect="auto")
        image1 = axes[0, 1].imshow(fuxi_imd, cmap="viridis", vmin=acc_min, vmax=acc_max, aspect="auto")
        image2 = axes[1, 0].imshow(
            mme_reference_delta,
            cmap="PuOr_r",
            vmin=-delta_limit,
            vmax=delta_limit,
            aspect="auto",
        )
        image3 = axes[1, 1].imshow(
            fuxi_reference_delta,
            cmap="PuOr_r",
            vmin=-delta_limit,
            vmax=delta_limit,
            aspect="auto",
        )
        titles = (
            "Equal-system MME: ACC against IMD",
            "FuXi-S2S: ACC against IMD",
            "MME reference sensitivity: IMERG minus IMD ACC",
            "FuXi reference sensitivity: IMERG minus IMD ACC",
        )
        images = (image0, image1, image2, image3)
        for index, (axis, title) in enumerate(zip(axes.flat, titles)):
            _format_heatmap(axis, title, show_y=index % 2 == 0)
            _annotate_heatmap(
                axis,
                images[index],
                (mme_imd, fuxi_imd, mme_reference_delta, fuxi_reference_delta)[index],
                difference=index >= 2,
            )
            _panel_label(axis, "abcd"[index])
        colorbar0 = figure.colorbar(image0, ax=axes[0, :], shrink=0.8, pad=0.02)
        colorbar0.set_label("ACC")
        colorbar2 = figure.colorbar(image2, ax=axes[1, :], shrink=0.8, pad=0.02)
        colorbar2.set_label("ACC difference")
        figure.suptitle(
            "Regional JJAS skill and observation-reference sensitivity\n"
            "169 matched valid-midpoint cases per lead; reference differences are descriptive",
            fontsize=12,
        )
        return _save_figure(
            figure,
            output / "fig03_regional_reference_sensitivity",
            "Regional JJAS skill and observation-reference sensitivity",
        )


def _plot_method_curves(
    axis: plt.Axes,
    summary: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    target: float | None = None,
) -> None:
    for method in METHODS:
        rows = summary[summary["method"].eq(method)].sort_values("lead_week")
        color, marker, linestyle = METHOD_STYLES[method]
        axis.plot(
            rows["lead_week"],
            rows[metric],
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=2.0 if method == "location_spread" else 1.4,
            markersize=5,
            label=METHOD_LABELS[method],
        )
    if target is not None:
        axis.axhline(target, color="#777777", linestyle=":", linewidth=1.0, label=f"Target = {target:g}")
    axis.set_xticks(LEADS, [f"W{lead}" for lead in LEADS])
    axis.set_xlabel("Forecast lead")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.8)


def _effect_panel(
    axis: plt.Axes,
    effects: pd.DataFrame,
    *,
    metric: str,
    ylabel: str,
    title: str,
    color: str,
) -> None:
    rows = effects[effects["metric"].eq(metric)].sort_values("lead_week")
    estimate = rows["estimate"].to_numpy(dtype=float)
    lower = rows["ci_lower"].to_numpy(dtype=float)
    upper = rows["ci_upper"].to_numpy(dtype=float)
    axis.errorbar(
        rows["lead_week"].to_numpy(dtype=int),
        estimate,
        yerr=np.vstack((estimate - lower, upper - estimate)),
        fmt="D",
        color=color,
        ecolor=color,
        markersize=5,
        capsize=3,
        linewidth=1.4,
    )
    axis.axhline(0.0, color="#555555", linestyle="--", linewidth=0.9)
    axis.set_xticks(LEADS, [f"W{lead}" for lead in LEADS])
    axis.set_xlabel("Forecast lead")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.8)


def _gate_text(
    story: Mapping[str, Any],
    exact_common: pd.DataFrame,
) -> str:
    crps_raw = story["crps_skill_pct_vs_raw"]
    crps_moment = story["crps_skill_pct_vs_moment"]
    acc = story["acc_delta_vs_raw"]
    bias = story["bias_delta_vs_raw"]
    exact_crps = exact_common[
        exact_common["metric"].eq("crps")
        & exact_common["baseline"].eq("raw_fuxi")
        & exact_common["effect"].eq("skill_pct_vs_baseline")
    ].iloc[0]
    exact_acc = exact_common[
        exact_common["metric"].eq("acc")
        & exact_common["baseline"].eq("raw_fuxi")
        & exact_common["effect"].eq("candidate_minus_baseline")
    ].iloc[0]
    return (
        "RETROSPECTIVE GATE ONLY — 2022–2024\n"
        "100 cases/lead; 600 case-lead rows (intersection 70; union 130)\n"
        f"CRPS skill vs raw: {crps_raw['estimate']:.2f}% "
        f"[{crps_raw['ci_lower']:.2f}, {crps_raw['ci_upper']:.2f}]\n"
        f"CRPS skill vs moment: {crps_moment['estimate']:.2f}% "
        f"[{crps_moment['ci_lower']:.2f}, {crps_moment['ci_upper']:.2f}]\n"
        f"ACC delta vs raw: {acc['estimate']:+.3f} "
        f"[{acc['ci_lower']:+.3f}, {acc['ci_upper']:+.3f}]\n"
        f"Bias delta vs raw: {bias['estimate']:+.3f} "
        f"[{bias['ci_lower']:+.3f}, {bias['ci_upper']:+.3f}] mm day⁻¹ (unresolved)\n"
        f"Exact-common sensitivity (85 midpoints/lead): CRPSS {exact_crps['estimate']:.2f}% "
        f"[{exact_crps['ci_lower']:.2f}, {exact_crps['ci_upper']:.2f}]; "
        f"ACC {exact_acc['estimate']:+.3f} [{exact_acc['ci_lower']:+.3f}, "
        f"{exact_acc['ci_upper']:+.3f}]\n"
        "Vs fixed six-system MME (169/lead): lower RMSE W1–W6; ACC higher at "
        "W1,W3–W6 (paired intervals); W2 unresolved\n"
        "Retrospective component: PASS\n"
        "Final headline gate: PENDING sealed 2025 direction check"
    )


def _figure4(
    summary: pd.DataFrame,
    effects: pd.DataFrame,
    story: Mapping[str, Any],
    exact_common: pd.DataFrame,
    output: Path,
) -> tuple[Path, Path]:
    with plt.rc_context(_style()):
        figure = plt.figure(figsize=(11.3, 11.2), constrained_layout=True)
        grid = figure.add_gridspec(4, 2, height_ratios=(1.0, 1.0, 1.0, 0.72))
        axes = np.array(
            [
                [figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1])],
                [figure.add_subplot(grid[1, 0]), figure.add_subplot(grid[1, 1])],
                [figure.add_subplot(grid[2, 0]), figure.add_subplot(grid[2, 1])],
            ]
        )
        gate_axis = figure.add_subplot(grid[3, :])
        gate_axis.set_axis_off()
        _plot_method_curves(
            axes[0, 0], summary, "crps", "CRPS (mm day$^{-1}$)", "Empirical ensemble CRPS"
        )
        _plot_method_curves(axes[0, 1], summary, "acc", "ACC", "Ensemble-mean spatial ACC")
        _plot_method_curves(
            axes[1, 0],
            summary,
            "coverage90",
            "Observed coverage",
            "Nominal 90% interval coverage",
            target=0.9,
        )
        _plot_method_curves(
            axes[1, 1],
            summary,
            "pooled_spread_skill_ratio",
            "Pooled spread / RMSE",
            "Pooled spread–error ratio",
            target=1.0,
        )
        _effect_panel(
            axes[2, 0],
            effects,
            metric="crps",
            ylabel="CRPS skill vs raw (%)",
            title="Neural vs raw: paired CRPS effect",
            color="#0072B2",
        )
        _effect_panel(
            axes[2, 1],
            effects,
            metric="acc",
            ylabel="ACC delta vs raw",
            title="Neural vs raw: paired ACC effect",
            color="#D55E00",
        )
        for index, axis in enumerate(axes.flat):
            _panel_label(axis, "abcdef"[index])
        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.958),
            ncol=3,
            frameon=False,
        )
        figure.suptitle(
            "Retrospective calibrated FuXi evaluation (IMD JJAS, 2020–2024)",
            fontsize=12,
            y=0.995,
        )
        gate_axis.text(
            0.5,
            0.5,
            _gate_text(story, exact_common),
            transform=gate_axis.transAxes,
            ha="center",
            va="center",
            fontsize=8.1,
            linespacing=1.25,
            bbox={
                "boxstyle": "round,pad=0.55",
                "facecolor": "#F4F8FB",
                "edgecolor": "#4C78A8",
                "linewidth": 0.9,
            },
        )
        figure.get_layout_engine().set(
            h_pad=0.06,
            w_pad=0.08,
            hspace=0.10,
            wspace=0.10,
            rect=(0.0, 0.0, 1.0, 0.91),
        )
        return _save_figure(
            figure,
            output / "fig04_retrospective_calibrated_fuxi",
            "Retrospective calibrated FuXi evaluation",
        )


def _figure1(
    geometry: Figure1Geometry,
    output: Path,
) -> tuple[Path, Path]:
    """Render domain, lead-window, and temporal-firewall panels."""

    with plt.rc_context(_style()):
        figure = plt.figure(figsize=(12.0, 7.35), constrained_layout=True)
        grid = figure.add_gridspec(
            2,
            2,
            width_ratios=(0.88, 1.32),
            height_ratios=(0.82, 1.18),
        )
        map_axis = figure.add_subplot(grid[:, 0])
        window_axis = figure.add_subplot(grid[0, 1])
        split_axis = figure.add_subplot(grid[1, 1])

        color_matrix = np.asarray(
            [matplotlib.colors.to_rgb(REGION_COLORS[name]) for name in REGIONS[1:]],
            dtype=np.float64,
        )
        fractions = np.stack([geometry.region_fractions[name] for name in REGIONS[1:]])
        for row_index, latitude in enumerate(geometry.latitude):
            for column_index, longitude in enumerate(geometry.longitude):
                if not geometry.india_geometry[row_index, column_index]:
                    continue
                cell_fractions = fractions[:, row_index, column_index].astype(np.float64)
                total = float(cell_fractions.sum())
                if geometry.adapter_support[row_index, column_index]:
                    mixture = np.sum(cell_fractions[:, None] * color_matrix, axis=0) / total
                    strength = 0.42 + 0.58 * min(total, 1.0)
                    face = 1.0 - strength * (1.0 - mixture)
                    patch = matplotlib.patches.Rectangle(
                        (float(longitude) - 0.75, float(latitude) - 0.75),
                        1.5,
                        1.5,
                        facecolor=face,
                        edgecolor="white",
                        linewidth=0.28,
                    )
                else:
                    patch = matplotlib.patches.Rectangle(
                        (float(longitude) - 0.75, float(latitude) - 0.75),
                        1.5,
                        1.5,
                        facecolor="white",
                        edgecolor="#555555",
                        linewidth=0.9,
                        hatch="////",
                    )
                map_axis.add_patch(patch)
        map_axis.add_patch(
            matplotlib.patches.Rectangle(
                (59.25, -0.75),
                40.5,
                40.5,
                fill=False,
                edgecolor="#A6A6A6",
                linewidth=0.8,
            )
        )
        map_axis.set_xlim(59.25, 99.75)
        map_axis.set_ylim(-0.75, 39.75)
        map_axis.set_aspect(1.0 / np.cos(np.deg2rad(20.0)))
        map_axis.set_xticks(np.arange(60, 100, 6))
        map_axis.set_yticks(np.arange(0, 40, 6))
        map_axis.set_xlabel("Longitude (°E)")
        map_axis.set_ylabel("Latitude (°N)")
        map_axis.grid(color="#E6E6E6", linewidth=0.45, zorder=-5)
        map_axis.set_title("27 × 27 verification grid and IMD regions")
        map_axis.text(
            0.03,
            0.025,
            "171 fixed support cells\n174 India-overlap cells",
            transform=map_axis.transAxes,
            fontsize=7.5,
            va="bottom",
            color="#303030",
            bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.92, "pad": 3},
        )
        legend_handles = [
            matplotlib.patches.Patch(
                facecolor=REGION_COLORS[name],
                edgecolor="none",
                label=REGION_LABELS[name],
            )
            for name in REGIONS[1:]
        ]
        legend_handles.append(
            matplotlib.patches.Patch(
                facecolor="white",
                edgecolor="#555555",
                hatch="////",
                label="India overlap outside fixed support",
            )
        )
        map_axis.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(0.0, -0.095),
            ncol=2,
            frameon=False,
            fontsize=7.2,
            handlelength=1.5,
            columnspacing=1.0,
        )

        week_colors = ("#D9EAF7", "#C6DDF1", "#B3D0EB", "#A0C3E5", "#8DB6DF", "#7AA9D9")
        for lead, color in zip(LEADS, week_colors):
            start = 7 * (lead - 1)
            window_axis.add_patch(
                matplotlib.patches.Rectangle(
                    (start, 0.42),
                    7,
                    0.34,
                    facecolor=color,
                    edgecolor="#3C6686",
                    linewidth=0.8,
                )
            )
            window_axis.text(
                start + 3.5,
                0.59,
                f"W{lead}",
                ha="center",
                va="center",
                fontsize=8.5,
                fontweight="bold",
            )
            window_axis.plot(start + 3.5, 0.87, marker="v", color="#D55E00", markersize=4.2)
        window_axis.text(
            21,
            0.98,
            "valid-period midpoints determine season",
            ha="center",
            fontsize=7.4,
            color="#7A3518",
        )
        window_axis.text(
            0,
            0.20,
            "IMD end labels: W1 = init+1…+7",
            ha="left",
            fontsize=7.6,
            color="#303030",
        )
        window_axis.text(
            42,
            0.20,
            "W6 = init+36…+42",
            ha="right",
            fontsize=7.6,
            color="#303030",
        )
        window_axis.set_xlim(-0.6, 42.6)
        window_axis.set_ylim(0.08, 1.10)
        window_axis.set_xticks(np.arange(0, 43, 7))
        window_axis.set_xlabel("Days after initialization; forecast windows are left-closed, right-open")
        window_axis.set_yticks([])
        window_axis.spines[["left", "right", "top"]].set_visible(False)
        window_axis.set_title("Six weekly valid windows and corrected IMD alignment")

        split_rows = {
            "Adapter fit": (
                (2002, 2018, "Train\n2002–17", "#56B4E9"),
                (2018, 2020, "Select\n2018–19", "#E69F00"),
            ),
            "Evidence": (
                (2020, 2022, "Dev\n2020–21", "#B8B8B8"),
                (2022, 2025, "Retro\n2022–24", "#009E73"),
                (2025, 2026, "Sealed\n2025", "#FFFFFF"),
            ),
        }
        row_y = {"Adapter fit": 1.30, "Evidence": 0.37}
        for label, segments in split_rows.items():
            y = row_y[label]
            for start, stop, text_value, color in segments:
                split_axis.add_patch(
                    matplotlib.patches.Rectangle(
                        (start, y),
                        stop - start,
                        0.50,
                        facecolor=color,
                        edgecolor="#4D4D4D",
                        linewidth=0.8,
                        hatch="////" if start == 2025 else None,
                    )
                )
                split_axis.text(
                    (start + stop) / 2,
                    y + 0.25,
                    "2025" if start == 2025 else text_value,
                    ha="center",
                    va="center",
                    fontsize=7.0 if stop - start <= 2 else 7.6,
                    color="#202020",
                    rotation=90 if start == 2025 else 0,
                )
            split_axis.text(2001.25, y + 0.25, label, ha="right", va="center", fontsize=8)
        split_axis.plot([2020, 2025], [2.16, 2.16], color="#4D4D4D", linewidth=0.9)
        split_axis.plot([2020, 2020], [2.10, 2.22], color="#4D4D4D", linewidth=0.9)
        split_axis.plot([2025, 2025], [2.10, 2.22], color="#4D4D4D", linewidth=0.9)
        split_axis.text(
            2022.5,
            2.25,
            "Operational bridge 2020–2024 (all retrospective)",
            ha="center",
            va="bottom",
            fontsize=7.6,
        )
        split_axis.text(
            2025.5,
            0.20,
            "sealed · no result",
            ha="center",
            va="top",
            fontsize=6.8,
            color="#555555",
        )
        split_axis.set_xlim(2001.3, 2026.2)
        split_axis.set_ylim(0.05, 2.55)
        split_axis.set_xticks((2002, 2006, 2010, 2014, 2018, 2020, 2022, 2024, 2025))
        split_axis.tick_params(axis="x", labelrotation=35)
        split_axis.set_yticks([])
        split_axis.spines[["left", "right", "top"]].set_visible(False)
        split_axis.set_title("Temporal split and sealed-2025 firewall")

        _panel_label(map_axis, "a")
        _panel_label(window_axis, "b")
        _panel_label(split_axis, "c")
        figure.suptitle(
            "India S2S benchmark domain and frozen evaluation protocol",
            fontsize=12,
            y=1.01,
        )
        figure.get_layout_engine().set(h_pad=0.08, w_pad=0.11, hspace=0.13, wspace=0.12)
        return _save_figure(
            figure,
            output / "fig01_domain_windows_splits",
            "India S2S benchmark domain and frozen evaluation protocol",
        )


def _figure1_contract(geometry: Figure1Geometry) -> dict[str, Any]:
    return {
        "figure": 1,
        "status": "rendered_from_hash_bound_static_geometry",
        "panels": {
            "domain": {
                "grid_shape": [27, 27],
                "frozen_adapter_supported_cells": int(
                    np.count_nonzero(geometry.adapter_support)
                ),
                "fractional_india_geometry_cells": int(
                    np.count_nonzero(geometry.india_geometry)
                ),
                "regions": list(REGIONS[1:]),
                "boundary_encoding": "RGB mixtures of the four fractional-region masks",
                "scoring_distinction": (
                    "weekly observation coverage remains a separate dynamic scoring factor"
                ),
            },
            "forecast_windows": {
                "W1": "[init, init+7 days); IMD end labels init+1...+7",
                "W6": "[init+35, init+42 days); IMD end labels init+36...+42",
            },
            "splits": {
                "train": "2002-2017",
                "validation": "2018-2019",
                "development": "2020-2021",
                "retrospective": "2020-2024 operational bridge",
                "prospective": "2025 sealed; no result",
            },
        },
        "spatial_logical_array_sha256": dict(geometry.logical_sha256),
        "sealed_2025_target_opened": False,
        "forecast_or_truth_arrays_opened": False,
        "static_spatial_support_arrays_opened": True,
    }


def _figure4_data_contract(
    inputs: FigureInputs,
    scoring: Mapping[str, Any],
) -> dict[str, Any]:
    """Describe packaged aggregates separately from upstream case rows."""

    relative = "tables/case_metrics.csv"
    source = _resolve_child(Path(inputs.scoring_manifest).resolve().parent, relative)
    source_hash = scoring.get("artifact_sha256", {}).get(relative)
    _require(
        isinstance(source_hash, str)
        and sha256_file(source) == source_hash
        and scoring.get("table_rows", {}).get("case_metrics") == CASE_ROWS,
        "Figure 4 case-level source is not bound to the scoring receipt",
    )
    columns = list(pd.read_csv(source, nrows=0).columns)
    _require(columns, "Figure 4 case-level source has no schema")
    return {
        "figure": 4,
        "packaged_data_granularity": "aggregate summaries and paired interval estimates",
        "packaged_case_level_rows": False,
        "packaged_tables": {
            "data/figure4_values.csv": "method x region x lead aggregate curves",
            "data/figure4_primary_block16_effects.csv": (
                "lead-wise paired block-bootstrap interval summaries"
            ),
            "data/figure4_retrospective_gate.csv": (
                "pooled retrospective gate interval summaries"
            ),
            "data/figure4_exact_common_sensitivity.csv": (
                "exact-common midpoint sensitivity interval summaries"
            ),
            "data/figure4_calibrated_fuxi_vs_mme.csv": (
                "paired calibrated-FuXi versus fixed-MME interval summaries"
            ),
        },
        "upstream_case_level_source": {
            "path": str(source),
            "sha256": source_hash,
            "row_count": CASE_ROWS,
            "columns": columns,
            "copied_into_figure_package": False,
            "reason_not_copied": (
                "the immutable scoring artifact is already fully hash-verified; the figure "
                "package avoids duplicating its 45,450 rows"
            ),
        },
        "sealed_2025_target_opened": False,
    }


def _record_visual_inspection(
    staging: Path,
    inspection: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Bind an explicit prior manual inspection to byte-identical rendered PNGs."""

    if inspection is None:
        return {"status": "not_recorded"}
    _require(inspection.get("status") == "passed", "visual inspection did not pass")
    expected = inspection.get("figure_png_sha256")
    _require(isinstance(expected, dict) and len(expected) == 6, "visual inspection inventory differs")
    actual_pngs = {
        str(path.relative_to(staging))
        for path in (staging / "figures").glob("*.png")
    }
    _require(actual_pngs == set(expected), "rendered PNG inventory differs from visual inspection")
    for relative, digest in expected.items():
        _require(
            isinstance(digest, str)
            and len(digest) == 64
            and sha256_file(staging / relative) == digest,
            f"rendered PNG differs from inspected bytes: {relative}",
        )
    source_root = Path(str(inspection.get("inspection_source_package", ""))).resolve()
    source_manifest = source_root / "manifest.json"
    _require(
        source_manifest.is_file()
        and sha256_file(source_manifest)
        == inspection.get("inspection_source_manifest_sha256"),
        "visual-inspection source package manifest differs",
    )
    for relative, digest in expected.items():
        _require(
            sha256_file(source_root / relative) == digest,
            f"visual-inspection source image differs: {relative}",
        )
    receipt = dict(inspection)
    receipt["current_render_png_sha256_equal_inspected_bytes"] = True
    receipt["current_render_png_count"] = len(expected)
    _write_json(staging / "visual_inspection_receipt.json", receipt)
    return {
        "status": "passed",
        "receipt": "visual_inspection_receipt.json",
        "inspected_png_count": len(expected),
        "current_render_png_sha256_equal_inspected_bytes": True,
    }


def rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish a directory without replacing an existing run."""

    source = Path(source).resolve()
    destination = Path(destination).resolve()
    _require(source.parent == destination.parent, "publication requires one parent directory")
    flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(source.parent, flags)
    try:
        metadata = os.stat(source.name, dir_fd=descriptor, follow_symlinks=False)
        _require(stat.S_ISDIR(metadata.st_mode), "publication source is not a real directory")
        library = ctypes.CDLL(None, use_errno=True)
        operation = getattr(library, "renameat2", None)
        if operation is None:
            raise RuntimeError("Linux renameat2 is required for no-clobber publication")
        operation.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        operation.restype = ctypes.c_int
        result = operation(
            descriptor,
            os.fsencode(source.name),
            descriptor,
            os.fsencode(destination.name),
            _RENAME_NOREPLACE,
        )
        if result != 0:
            error = ctypes.get_errno()
            if error == errno.EEXIST:
                raise FileExistsError(destination)
            raise OSError(error, os.strerror(error), destination)
    finally:
        os.close(descriptor)


def _assert_output_path(destination: Path, output_root: Path) -> None:
    resolved = Path(destination).resolve()
    root = Path(output_root).resolve()
    _require(root in resolved.parents, "output must be a child of resultsv3/india_s2s_paper_figures")
    _require(resolved != root, "output must be a fresh child directory")


def _input_manifest(inputs: FigureInputs, hashes: InputHashes) -> dict[str, dict[str, str]]:
    return {
        field: {
            "path": str(Path(getattr(inputs, field)).resolve()),
            "sha256": str(getattr(hashes, field)),
        }
        for field in inputs.__dataclass_fields__
    }


def generate_figures(
    inputs: FigureInputs,
    hashes: InputHashes,
    output: Path,
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    spatial_array_hashes: SpatialArrayHashes = CANONICAL_SPATIAL_ARRAY_HASHES,
    visual_inspection: Mapping[str, Any] | None = None,
) -> Path:
    """Validate frozen inputs and atomically publish Figures 1--4."""

    destination = Path(output).resolve()
    _assert_output_path(destination, output_root)
    if destination.exists():
        raise FileExistsError(f"fresh paper-figure output required: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.incomplete-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir()
    ledger = HashLedger()
    try:
        (
            benchmark,
            scoring,
            audit,
            comparison,
            imd_methods,
            imerg_methods,
            calibration,
            artifact_counts,
        ) = _load_receipts(inputs, hashes, ledger)
        _validate_global_gates(
            inputs,
            hashes,
            benchmark,
            scoring,
            audit,
            comparison,
            imd_methods,
            imerg_methods,
            calibration,
        )
        geometry = _load_figure1_geometry(inputs, spatial_array_hashes)
        imd = _load_seasonal(inputs.imd_seasonal_summary, track="tp_imd", reference="imd")
        imerg = _load_seasonal(inputs.imerg_seasonal_summary, track="tp_imerg", reference="imerg")
        imd_allseason = _load_allseason_summary(
            inputs.imd_weekly_summary,
            track="tp_imd",
            variable="tp",
            reference="imd",
            models=MODELS,
            case_count=517,
            unavailable={("ecmwf", 3)},
        )
        temperature_allseason = _load_allseason_summary(
            inputs.imerg_weekly_summary,
            track="t2m_era5",
            variable="t2m",
            reference="era5",
            models=T2M_MODELS,
            case_count=516,
            unavailable=set(),
        )
        _require(
            _case_id_set(inputs.imd_case_metrics, track="tp_imd", reference="imd")
            == _case_id_set(inputs.imerg_case_metrics, track="tp_imerg", reference="imerg"),
            "IMD and IMERG MME/FuXi valid-midpoint case IDs differ",
        )
        calibration, effects, primary_story = _load_calibration_tables(inputs, scoring, audit)
        exact_common, calibrated_vs_mme = _load_comparison_tables(inputs)

        figures_directory = staging / "figures"
        data_directory = staging / "data"
        figures_directory.mkdir()
        data_directory.mkdir()
        _figure1(geometry, figures_directory)
        _figure2(imd, figures_directory)
        _figure3(imd, imerg, figures_directory)
        _figure4(
            calibration,
            effects,
            audit["story_gate"],
            exact_common,
            figures_directory,
        )
        _supplement_allseason_two_panel(
            imd_allseason,
            models=MODELS,
            labels=MODEL_LABELS,
            styles=MODEL_STYLES,
            case_count=517,
            reference_label="IMD",
            variable_label="rainfall",
            rmse_label="RMSE (mm day$^{-1}$)",
            output=figures_directory,
            stem="figS01_allseason_imd_rainfall",
            unavailable={("ecmwf", 3)},
        )
        _supplement_allseason_two_panel(
            temperature_allseason,
            models=T2M_MODELS,
            labels=T2M_MODEL_LABELS,
            styles=T2M_MODEL_STYLES,
            case_count=516,
            reference_label="ERA5",
            variable_label="2 m temperature",
            rmse_label="RMSE (K)",
            output=figures_directory,
            stem="figS02_allseason_era5_temperature",
            unavailable=set(),
        )

        figure2_values = imd[imd["region"].eq("all_india")][
            [
                "reference",
                "season",
                "region",
                "model",
                "lead_week",
                "case_count",
                "acc_valid_case_count",
                "acc",
                "rmse_valid_case_count",
                "rmse",
            ]
        ].sort_values(["model", "lead_week"])
        figure3_values = pd.concat(
            (
                imd[imd["model"].isin(("mme", "fuxi_s2s"))],
                imerg[imerg["model"].isin(("mme", "fuxi_s2s"))],
            ),
            ignore_index=True,
        )[
            [
                "reference",
                "season",
                "region",
                "region_label",
                "model",
                "lead_week",
                "case_count",
                "acc_valid_case_count",
                "acc",
            ]
        ].sort_values(["reference", "model", "region", "lead_week"])
        figure4_values = calibration[
            [
                "method",
                "region",
                "lead_week",
                "case_count",
                "crps",
                "acc",
                "coverage90",
                "pooled_spread_skill_ratio",
            ]
        ].sort_values(["method", "lead_week"])
        figure2_values.to_csv(data_directory / "figure2_values.csv", index=False, lineterminator="\n")
        figure3_values.to_csv(data_directory / "figure3_values.csv", index=False, lineterminator="\n")
        figure4_values.to_csv(data_directory / "figure4_values.csv", index=False, lineterminator="\n")
        effects.sort_values(["metric", "lead_week"]).to_csv(
            data_directory / "figure4_primary_block16_effects.csv", index=False, lineterminator="\n"
        )
        primary_story.sort_values(["baseline", "metric", "effect"]).to_csv(
            data_directory / "figure4_retrospective_gate.csv", index=False, lineterminator="\n"
        )
        exact_common.sort_values(["baseline", "metric", "effect"]).to_csv(
            data_directory / "figure4_exact_common_sensitivity.csv",
            index=False,
            lineterminator="\n",
        )
        calibrated_vs_mme.sort_values(["lead_week", "metric", "effect"]).to_csv(
            data_directory / "figure4_calibrated_fuxi_vs_mme.csv",
            index=False,
            lineterminator="\n",
        )
        _figure1_geometry_frame(geometry).to_csv(
            data_directory / "figure1_grid_geometry.csv",
            index=False,
            lineterminator="\n",
            float_format="%.10g",
        )
        imd_allseason[imd_allseason["region"].eq("all_india")].sort_values(
            ["model", "lead_week"]
        ).to_csv(
            data_directory / "figureS1_allseason_imd_rainfall.csv",
            index=False,
            lineterminator="\n",
        )
        temperature_allseason[
            temperature_allseason["region"].eq("all_india")
        ].sort_values(["model", "lead_week"]).to_csv(
            data_directory / "figureS2_allseason_era5_temperature.csv",
            index=False,
            lineterminator="\n",
        )
        _write_json(staging / "figure1_data_contract.json", _figure1_contract(geometry))
        _write_json(
            staging / "figure4_data_contract.json",
            _figure4_data_contract(inputs, scoring),
        )

        captions = {
            "figure1": (
                "Frozen India S2S domain and evaluation protocol. (a) The 27 x 27, 1.5-degree "
                "grid contains 171 fixed calibration-support cells. Cell colors are mixtures "
                "of the four fractional IMD homogeneous-region masks; hatching marks three "
                "additional cells with fractional India overlap but no fixed training support. "
                "Dynamic weekly observation coverage is applied separately during scoring. "
                "The cell schematic is not a national- or state-boundary map. (b) Forecast "
                "weeks span [init, init+42 days), while corrected end-labelled IMD targets use "
                "init+1 through init+42; valid-period midpoints assign JJAS membership. (c) "
                "The adapter is trained on 2002-2017 and selected on 2018-2019. All operational "
                "2020-2024 evidence is retrospective; the 2025 target remains sealed and has "
                "no result."
            ),
            "figure2": (
                "All-system 2020-2024 IMD JJAS benchmark, assigned by seven-day valid-period "
                "midpoint (169 cases per lead). ACC is the arithmetic mean of per-case "
                "area-weighted spatial anomaly correlations; RMSE is the arithmetic mean of "
                "per-case area x observation-coverage-weighted raw-field errors. ECMWF W3 is "
                "unavailable; the fixed equal-system MME excludes ECMWF at every lead. Lines "
                "are descriptive; paired MME-FuXi uncertainty is not encoded as uncertainty "
                "for the other systems."
            ),
            "figure3": (
                "Regional MME and FuXi-S2S ACC under IMD and descriptive IMERG-minus-IMD "
                "reference sensitivity on identical 2020-2024 valid-midpoint JJAS case IDs "
                "(169 cases per lead). All India uses area x coverage weights; regions also "
                "use fractional IMD homogeneous-region masks. Reference differences have no "
                "paired interval here and are not significance claims."
            ),
            "figure4": (
                "Retrospective calibrated-FuXi evaluation against IMD for 2020-2024 "
                "valid-midpoint JJAS (169 identical 50-member cases per method, region, and "
                "lead). Curves are arithmetic means of per-case area-weighted scores; pooled "
                "spread/error is sqrt(pooled ensemble variance / pooled ensemble-mean MSE). "
                "Effect bars are paired 95% intervals from 10,000 year-stratified circular "
                "block resamples with 16-start blocks. The inset is a separate synchronized "
                "2022-2024 retrospective cohort (100 lead-specific cases per lead; 600 "
                "case-lead rows; intersection 70, union 130). The final headline gate remains "
                "pending the sealed-2025 direction check. A separately frozen exact-common "
                "85-midpoint-per-lead sensitivity and paired calibrated-neural versus fixed "
                "six-system MME comparison are shown as retrospective annotations."
                " The Figure 4 CSVs packaged here contain aggregate curves and interval "
                "summaries, not case-level rows; the 45,450 case rows remain in the immutable, "
                "hash-bound upstream scoring artifact identified by figure4_data_contract.json."
            ),
            "figureS1": (
                "Descriptive all-season rainfall benchmark against IMD for the 517 paired "
                "2020-2024 initialization archive. ACC and RMSE are arithmetic means of "
                "per-case spatial scores. ECMWF W3 is unavailable, and the fixed rainfall "
                "MME excludes ECMWF at every lead. This supplement is rendered from the "
                "previously validated weekly-summary table; no observation array is opened."
            ),
            "figureS2": (
                "Descriptive all-season 2 m temperature benchmark against ERA5 for 516 paired "
                "2020-2024 initializations. ACC and RMSE are arithmetic means of per-case "
                "spatial scores. The temperature MME contains CMA, DLESyM-v1, ECMWF, "
                "FourCastNet 3, FuXi-S2S, and UKMO; DLESyM-v0 is displayed but excluded. "
                "This supplement is rendered from the previously validated weekly-summary "
                "table; no observation array is opened."
            ),
        }
        _write_json(staging / "captions.json", captions)
        visual_inspection_status = _record_visual_inspection(staging, visual_inspection)

        source = Path(__file__).resolve()
        snapshot = staging / "code/src" / source.name
        snapshot.parent.mkdir(parents=True)
        shutil.copy2(source, snapshot)

        ledger.reverify()
        artifacts = {
            str(path.relative_to(staging)): sha256_file(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file() and path.name not in {"manifest.json", "failure.json"}
        }
        canonical = hashes == CANONICAL_HASHES
        manifest = {
            "experiment": EXPERIMENT,
            "status": "complete",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "output_path": str(destination),
            "scientific_status": (
                "retrospective 2020-2024 evidence; final headline gate pending sealed 2025 check"
            ),
            "software": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "matplotlib": matplotlib.__version__,
                "matplotlib_backend": matplotlib.get_backend(),
            },
            "rendering": {
                "png_dpi": 220,
                "formats": ["png", "pdf"],
                "accessible_encoding": "color plus marker plus line style",
                "pdf_creation_and_modification_dates_omitted": True,
            },
            "inputs": _input_manifest(inputs, hashes),
            "hard_gates": {
                "canonical_known_input_hashes": canonical,
                "all_input_sha256_verified": True,
                "all_input_hashes_reverified_before_publication": True,
                "validated_manifest_artifact_counts": artifact_counts,
                "valid_midpoint_jjas_cases_per_lead": CASE_COUNT_PER_LEAD,
                "matched_imd_imerg_case_ids": True,
                "independent_scoring_audit": "passed",
                "raw_fuxi_identity": True,
                "raw_fuxi_identity_rows": RAW_IDENTITY_ROWS,
                "selected_seed": SELECTED_SEED,
                "selected_checkpoint_sha256": SELECTED_CHECKPOINT_SHA256,
                "retrospective_story_gate": "passed",
                "exact_common_midpoint_sensitivity_cases_per_lead": 85,
                "calibrated_neural_vs_fixed_mme": {
                    "fixed_mme_excludes_ecmwf": True,
                    "rmse_lower_primary_interval_leads": [1, 2, 3, 4, 5, 6],
                    "acc_higher_primary_interval_leads": [1, 3, 4, 5, 6],
                    "acc_unresolved_leads": [2],
                },
                "final_headline_gate": "pending_sealed_2025_point-estimate_direction_check",
                "sealed_2025_target_opened": False,
                "forecast_or_truth_arrays_opened_by_generator": False,
                "static_spatial_support_arrays_opened_by_generator": True,
                "frozen_adapter_support_cells": 171,
                "fractional_india_geometry_cells": 174,
                "spatial_logical_array_sha256": dict(geometry.logical_sha256),
                "allseason_rainfall_paired_initializations": 517,
                "allseason_temperature_paired_initializations": 516,
                "rendered_images_visually_inspected": (
                    visual_inspection_status.get("status") == "passed"
                ),
            },
            "figures": {
                "figure1": {
                    "status": "rendered",
                    "png": "figures/fig01_domain_windows_splits.png",
                    "pdf": "figures/fig01_domain_windows_splits.pdf",
                    "data": "data/figure1_grid_geometry.csv",
                    "contract": "figure1_data_contract.json",
                },
                "figure2": {
                    "png": "figures/fig02_all_model_imd_jjas_acc_rmse.png",
                    "pdf": "figures/fig02_all_model_imd_jjas_acc_rmse.pdf",
                    "data": "data/figure2_values.csv",
                    "ecmwf_w3": "explicitly marked unavailable",
                },
                "figure3": {
                    "png": "figures/fig03_regional_reference_sensitivity.png",
                    "pdf": "figures/fig03_regional_reference_sensitivity.pdf",
                    "data": "data/figure3_values.csv",
                    "uncertainty_claim": "none; descriptive reference sensitivity",
                },
                "figure4": {
                    "png": "figures/fig04_retrospective_calibrated_fuxi.png",
                    "pdf": "figures/fig04_retrospective_calibrated_fuxi.pdf",
                    "curve_data": "data/figure4_values.csv",
                    "effect_data": "data/figure4_primary_block16_effects.csv",
                    "gate_data": "data/figure4_retrospective_gate.csv",
                    "exact_common_sensitivity_data": (
                        "data/figure4_exact_common_sensitivity.csv"
                    ),
                    "calibrated_fuxi_vs_mme_data": (
                        "data/figure4_calibrated_fuxi_vs_mme.csv"
                    ),
                    "contract": "figure4_data_contract.json",
                    "packaged_data_granularity": (
                        "aggregate summaries and paired interval estimates"
                    ),
                    "packaged_case_level_rows": False,
                    "gate_scope": "retrospective only",
                },
            },
            "supplementary_figures": {
                "figureS1": {
                    "png": "figures/figS01_allseason_imd_rainfall.png",
                    "pdf": "figures/figS01_allseason_imd_rainfall.pdf",
                    "data": "data/figureS1_allseason_imd_rainfall.csv",
                    "scope": "all season; 517 paired initializations",
                },
                "figureS2": {
                    "png": "figures/figS02_allseason_era5_temperature.png",
                    "pdf": "figures/figS02_allseason_era5_temperature.pdf",
                    "data": "data/figureS2_allseason_era5_temperature.csv",
                    "scope": "all season; 516 paired initializations",
                },
            },
            "captions": captions,
            "visual_inspection": visual_inspection_status,
            "source_snapshot_sha256": {
                str(snapshot.relative_to(staging)): artifacts[str(snapshot.relative_to(staging))]
            },
            "artifact_sha256": artifacts,
        }
        _write_json(staging / "manifest.json", manifest)
        ledger.reverify()
        rename_noreplace(staging, destination)
    except BaseException as error:
        if staging.exists():
            _write_json(
                staging / "failure.json",
                {
                    "experiment": EXPERIMENT,
                    "status": "failed",
                    "failed_utc": datetime.now(timezone.utc).isoformat(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "sealed_2025_target_opened": False,
                    "forecast_or_truth_arrays_opened_by_generator": False,
                },
            )
        raise
    return destination / "manifest.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--benchmark-manifest", type=Path, default=CANONICAL_INPUTS.benchmark_manifest
    )
    parser.add_argument("--scoring-manifest", type=Path, default=CANONICAL_INPUTS.scoring_manifest)
    parser.add_argument("--audit-receipt", type=Path, default=CANONICAL_INPUTS.audit_receipt)
    parser.add_argument(
        "--comparison-manifest",
        type=Path,
        default=CANONICAL_INPUTS.comparison_manifest,
    )
    parser.add_argument(
        "--imd-methods-manifest", type=Path, default=CANONICAL_INPUTS.imd_methods_manifest
    )
    parser.add_argument(
        "--imd-seasonal-summary", type=Path, default=CANONICAL_INPUTS.imd_seasonal_summary
    )
    parser.add_argument(
        "--imd-weekly-summary", type=Path, default=CANONICAL_INPUTS.imd_weekly_summary
    )
    parser.add_argument("--imd-case-metrics", type=Path, default=CANONICAL_INPUTS.imd_case_metrics)
    parser.add_argument(
        "--imerg-methods-manifest", type=Path, default=CANONICAL_INPUTS.imerg_methods_manifest
    )
    parser.add_argument(
        "--imerg-seasonal-summary", type=Path, default=CANONICAL_INPUTS.imerg_seasonal_summary
    )
    parser.add_argument(
        "--imerg-weekly-summary", type=Path, default=CANONICAL_INPUTS.imerg_weekly_summary
    )
    parser.add_argument(
        "--imerg-case-metrics", type=Path, default=CANONICAL_INPUTS.imerg_case_metrics
    )
    parser.add_argument(
        "--calibration-manifest",
        type=Path,
        default=CANONICAL_INPUTS.calibration_manifest,
    )
    parser.add_argument(
        "--adapter-support", type=Path, default=CANONICAL_INPUTS.adapter_support
    )
    parser.add_argument(
        "--spatial-support-metadata",
        type=Path,
        default=CANONICAL_INPUTS.spatial_support_metadata,
    )
    parser.add_argument(
        "--record-canonical-visual-inspection",
        action="store_true",
        help=(
            "bind the manual inspection of the canonical v3 PNG bytes; custom renders "
            "fail if any image differs"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    inputs = FigureInputs(
        benchmark_manifest=arguments.benchmark_manifest,
        scoring_manifest=arguments.scoring_manifest,
        audit_receipt=arguments.audit_receipt,
        comparison_manifest=arguments.comparison_manifest,
        imd_methods_manifest=arguments.imd_methods_manifest,
        imd_seasonal_summary=arguments.imd_seasonal_summary,
        imd_weekly_summary=arguments.imd_weekly_summary,
        imd_case_metrics=arguments.imd_case_metrics,
        imerg_methods_manifest=arguments.imerg_methods_manifest,
        imerg_seasonal_summary=arguments.imerg_seasonal_summary,
        imerg_weekly_summary=arguments.imerg_weekly_summary,
        imerg_case_metrics=arguments.imerg_case_metrics,
        calibration_manifest=arguments.calibration_manifest,
        adapter_support=arguments.adapter_support,
        spatial_support_metadata=arguments.spatial_support_metadata,
    )
    manifest = generate_figures(
        inputs,
        CANONICAL_HASHES,
        arguments.output,
        visual_inspection=(
            CANONICAL_VISUAL_INSPECTION
            if arguments.record_canonical_visual_inspection
            else None
        ),
    )
    print(f"PASS: paper figures published at {manifest}")


if __name__ == "__main__":
    main()
