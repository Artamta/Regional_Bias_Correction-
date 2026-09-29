#!/usr/bin/env python3
"""Build two claim-closing India S2S paper comparisons.

The module reads only immutable, already-scored tables.  It adds (1) an
exact-common-valid-midpoint sensitivity for the retrospective 2022--2024
calibration gate and (2) a paired deterministic comparison between the
selected calibrated FuXi ensemble mean and the frozen equal-system MME.
No forecast, observation, or 2025 array is opened.
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
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from india_s2s_block_bootstrap import circular_year_stratified_index_matrix
from project_paths import PROJECT_ROOT


EXPERIMENT = "india_s2s_paper_comparisons_v1"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "resultsv3/india_s2s_paper_comparisons"
DEFAULT_SCORING_MANIFEST = (
    PROJECT_ROOT
    / "resultsv3/india_s2s_probabilistic_bridge"
    / "scoring_full_20260825T234808Z/manifest.json"
)
DEFAULT_AUDIT_RECEIPT = (
    PROJECT_ROOT
    / "resultsv3/india_s2s_probabilistic_bridge"
    / "audit_full_20260826T000133Z/audit_receipt.json"
)
DEFAULT_BENCHMARK_CASES = (
    PROJECT_ROOT.parent
    / "studies/india_s2s_verification_v2/results/imd_tp_sensitivity/tables/case_metrics.csv"
)
DEFAULT_METHODS_MANIFEST = (
    PROJECT_ROOT.parent
    / "studies/india_s2s_verification_v2/results/imd_tp_sensitivity/methods_manifest.json"
)

SCORING_MANIFEST_SHA256 = (
    "4ab303a3fa0025bfb2b8654bf34770091f117df18f236c0c724d8e1078450943"
)
SCORING_CASES_SHA256 = (
    "36e2e1ccadf43e29144c4267e290ca001fc3d1f41658b630f665fd158b1d4f8c"
)
AUDIT_RECEIPT_SHA256 = (
    "d7c822e0c09528b443e1dddd83872aab2894d82da5dedce6b4d0f1fbc3807bcc"
)
BENCHMARK_CASES_SHA256 = (
    "000544615d004669b41dd87fc647bd979cd92c3638e7deee5ea77992046e1a14"
)
METHODS_MANIFEST_SHA256 = (
    "9b515c0230fc345323565a429f185ea86cf8e7c5e47a208b6117a7b649c26e6d"
)
MME_COMPONENTS = ("cma", "dlesym_v0", "fuxi_s2s", "ncep", "neuralgcm", "ukmo")
METHODS = ("raw_fuxi", "moment_calibration", "location_spread")
LEADS = (1, 2, 3, 4, 5, 6)
BLOCK_LENGTHS = (16, 13)
_RENAME_NOREPLACE = 1


class PaperComparisonError(RuntimeError):
    """Raised when a paper comparison violates its frozen contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PaperComparisonError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_hash(path: Path, expected: str, label: str) -> None:
    _require(Path(path).is_file(), f"{label} is missing: {path}")
    observed = sha256_file(Path(path))
    _require(observed == expected, f"{label} SHA-256 differs: {observed}")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PaperComparisonError(f"cannot read JSON: {path}") from error
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _resolve_child(root: Path, relative: str) -> Path:
    child = Path(relative)
    _require(not child.is_absolute(), f"artifact path must be relative: {relative}")
    resolved_root = Path(root).resolve()
    resolved = (resolved_root / child).resolve()
    _require(resolved_root in resolved.parents, f"artifact escapes run: {relative}")
    return resolved


def _dates_sha256(values: Iterable[Any]) -> str:
    dates = np.asarray(list(values), dtype="datetime64[D]")
    _require(dates.ndim == 1 and dates.size > 0, "date receipt is empty")
    _require(np.unique(dates).size == dates.size, "date receipt has duplicates")
    payload = "".join(
        f"{np.datetime_as_string(value, unit='D')}\n" for value in np.sort(dates)
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _rename_noreplace(source: Path, destination: Path) -> None:
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    _require(source.parent == destination.parent, "publication parents differ")
    flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(source.parent, flags)
    try:
        metadata = os.stat(source.name, dir_fd=descriptor, follow_symlinks=False)
        _require(stat.S_ISDIR(metadata.st_mode), "publication source is not a directory")
        library = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(library, "renameat2", None)
        _require(renameat2 is not None, "Linux renameat2 is required")
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
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


def validate_inputs(
    scoring_manifest_path: Path,
    audit_receipt_path: Path,
    benchmark_cases_path: Path,
    methods_manifest_path: Path = DEFAULT_METHODS_MANIFEST,
) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate the complete immutable evidence line and return case-table path."""

    scoring_manifest_path = Path(scoring_manifest_path).resolve()
    audit_receipt_path = Path(audit_receipt_path).resolve()
    benchmark_cases_path = Path(benchmark_cases_path).resolve()
    methods_manifest_path = Path(methods_manifest_path).resolve()
    _require_hash(scoring_manifest_path, SCORING_MANIFEST_SHA256, "scoring manifest")
    _require_hash(audit_receipt_path, AUDIT_RECEIPT_SHA256, "audit receipt")
    _require_hash(benchmark_cases_path, BENCHMARK_CASES_SHA256, "benchmark cases")
    _require_hash(methods_manifest_path, METHODS_MANIFEST_SHA256, "methods manifest")
    scoring = _read_json(scoring_manifest_path)
    audit = _read_json(audit_receipt_path)
    methods = _read_json(methods_manifest_path)
    required_scoring = {
        "experiment": "india_s2s_probabilistic_bridge_scoring_v1",
        "status": "complete",
        "selected_seed": 43,
        "member_count_each_method": 50,
        "opened_2025_observation": False,
        "sealed_2025_target_opened": False,
    }
    for key, expected in required_scoring.items():
        _require(scoring.get(key) == expected, f"scoring {key} differs")
    _require(
        scoring.get("opened_observation_years") == [2020, 2021, 2022, 2023, 2024],
        "scoring observation years differ",
    )
    raw_identity = scoring.get("raw_fuxi_identity", {})
    _require(
        raw_identity.get("identity") is True
        and raw_identity.get("matched_case_rows") == 15150,
        "raw FuXi benchmark identity is not complete",
    )
    relative = "tables/case_metrics.csv"
    _require(
        scoring.get("artifact_sha256", {}).get(relative) == SCORING_CASES_SHA256,
        "scoring manifest does not bind the expected case table",
    )
    scoring_cases_path = _resolve_child(scoring_manifest_path.parent, relative)
    _require_hash(scoring_cases_path, SCORING_CASES_SHA256, "scoring cases")
    _require(audit.get("status") == "passed", "independent table audit did not pass")
    _require(
        audit.get("scoring_manifest_sha256") == SCORING_MANIFEST_SHA256,
        "audit does not bind this scoring run",
    )
    semantics = audit.get("semantic_checks", {})
    _require(
        semantics.get("sealed_2025_target_opened") is False
        and semantics.get("full_case_cartesian_rows") == 45450,
        "audit safety or row contract differs",
    )
    track = methods.get("tracks", {}).get("tp_imd", {})
    _require(track.get("track") == "tp_imd", "methods manifest lacks TP/IMD track")
    _require(
        tuple(track.get("mme_components", ())) == MME_COMPONENTS,
        "methods manifest MME components differ",
    )
    _require(
        track.get("unavailable_lead_weeks", {}).get("ecmwf") == [3]
        and "ecmwf" not in track.get("mme_components", ()),
        "methods manifest ECMWF/MME contract differs",
    )
    return scoring_cases_path, scoring, audit, methods


def _bootstrap_pooled(
    candidate: np.ndarray,
    baseline: np.ndarray,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    candidate_samples = np.take(candidate, indices, axis=1).mean(
        axis=(0, 2), dtype=np.float64
    )
    baseline_samples = np.take(baseline, indices, axis=1).mean(
        axis=(0, 2), dtype=np.float64
    )
    return candidate_samples, baseline_samples, candidate_samples - baseline_samples


def exact_common_midpoint_sensitivity(
    cases: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
    block_lengths: Sequence[int] = BLOCK_LENGTHS,
    strict: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Pool six leads on the dates whose valid midpoints are exactly common."""

    required = {
        "method",
        "region",
        "lead_week",
        "verification_year",
        "valid_period_midpoint",
        "season",
        "crps",
        "acc",
        "bias",
    }
    _require(required.issubset(cases.columns), "scoring cases lack sensitivity columns")
    frame = cases[
        cases.method.isin(METHODS)
        & cases.region.eq("all_india")
        & cases.lead_week.isin(LEADS)
        & cases.verification_year.between(2022, 2024)
        & cases.season.eq("JJAS")
    ].copy()
    frame["midpoint_date"] = pd.to_datetime(frame.valid_period_midpoint).values.astype(
        "datetime64[D]"
    )
    _require(
        not frame.duplicated(["method", "lead_week", "midpoint_date"]).any(),
        "valid-midpoint cases are duplicated",
    )
    counts = frame.groupby(["method", "lead_week"]).size()
    if strict:
        _require(len(counts) == 18 and set(counts) == {100}, "expected 100 cases/lead")
    raw = frame[frame.method.eq("raw_fuxi")]
    lead_sets = {
        lead: set(raw[raw.lead_week.eq(lead)].midpoint_date.tolist()) for lead in LEADS
    }
    common = np.asarray(sorted(set.intersection(*lead_sets.values())), dtype="datetime64[D]")
    year_counts = {
        int(year): int(count)
        for year, count in pd.Series(pd.DatetimeIndex(common).year).value_counts().sort_index().items()
    }
    if strict:
        _require(len(common) == 85, f"expected 85 common midpoints, found {len(common)}")
        _require(year_counts == {2022: 35, 2023: 35, 2024: 15}, "common-year counts differ")
    _require(len(common) > 0, "no midpoint is common to all leads")

    matrices: dict[tuple[str, str], np.ndarray] = {}
    for method in METHODS:
        for metric in ("crps", "acc", "bias"):
            values: list[np.ndarray] = []
            for lead in LEADS:
                selected = frame[
                    frame.method.eq(method)
                    & frame.lead_week.eq(lead)
                    & frame.midpoint_date.isin(common)
                ].sort_values("midpoint_date")
                dates = selected.midpoint_date.to_numpy(dtype="datetime64[D]")
                _require(np.array_equal(dates, common), f"{method}/W{lead} midpoint mismatch")
                values.append(selected[metric].to_numpy(dtype=np.float64))
            matrices[(method, metric)] = np.stack(values)

    rows: list[dict[str, Any]] = []
    sampling: dict[str, Any] = {}
    for block in block_lengths:
        draw_seed = int(seed + 17000 + int(block))
        indices = circular_year_stratified_index_matrix(
            common,
            block_length=int(block),
            replicates=int(replicates),
            seed=draw_seed,
        )
        sampling[f"block{block}"] = {
            "seed": draw_seed,
            "shape": list(indices.shape),
            "index_matrix_sha256": hashlib.sha256(
                np.ascontiguousarray(indices).tobytes()
            ).hexdigest(),
        }
        for baseline_name in ("raw_fuxi", "moment_calibration"):
            for metric in ("crps", "acc", "bias"):
                candidate = matrices[("location_spread", metric)]
                baseline = matrices[(baseline_name, metric)]
                candidate_samples, baseline_samples, effects = _bootstrap_pooled(
                    candidate, baseline, indices
                )
                lower, upper = np.quantile(effects, (0.025, 0.975))
                common_fields = {
                    "reference": "imd",
                    "season": "JJAS_valid_midpoint",
                    "analysis_cohort": "2022_2024_exact_common_valid_midpoints_pooled_w1_w6",
                    "region": "all_india",
                    "lead_week": "pooled_w1_w6",
                    "metric": metric,
                    "candidate": "location_spread",
                    "baseline": baseline_name,
                    "candidate_mean": float(candidate.mean()),
                    "baseline_mean": float(baseline.mean()),
                    "n_common_midpoints_per_lead": int(len(common)),
                    "n_case_lead_rows": int(6 * len(common)),
                    "confidence": 0.95,
                    "block_length_valid_midpoints": int(block),
                    "replicates": int(replicates),
                    "seed": draw_seed,
                }
                rows.append(
                    {
                        **common_fields,
                        "effect": "candidate_minus_baseline",
                        "estimate": float(candidate.mean() - baseline.mean()),
                        "ci_lower": float(lower),
                        "ci_upper": float(upper),
                    }
                )
                if metric == "crps":
                    _require(np.all(baseline_samples > 0.0), "baseline CRPS is non-positive")
                    skill = 100.0 * (1.0 - candidate_samples / baseline_samples)
                    skill_lower, skill_upper = np.quantile(skill, (0.025, 0.975))
                    rows.append(
                        {
                            **common_fields,
                            "effect": "skill_pct_vs_baseline",
                            "estimate": float(100.0 * (1.0 - candidate.mean() / baseline.mean())),
                            "ci_lower": float(skill_lower),
                            "ci_upper": float(skill_upper),
                        }
                    )
    receipt = {
        "cohort": "2022_2024_exact_common_valid_midpoints_pooled_w1_w6",
        "common_midpoints_per_lead": int(len(common)),
        "case_lead_rows": int(6 * len(common)),
        "year_counts": year_counts,
        "common_midpoints_sha256": _dates_sha256(common),
        "sampling": sampling,
        "sampling_unit": "valid-period midpoint synchronized exactly across six leads",
    }
    return pd.DataFrame(rows), receipt


def calibrated_fuxi_vs_mme(
    scoring_cases: pd.DataFrame,
    benchmark_cases: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
    block_lengths: Sequence[int] = BLOCK_LENGTHS,
    strict: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Pair calibrated FuXi and the fixed six-system MME by initialization."""

    neural = scoring_cases[
        scoring_cases.method.eq("location_spread")
        & scoring_cases.region.eq("all_india")
        & scoring_cases.season.eq("JJAS")
        & scoring_cases.lead_week.isin(LEADS)
    ].copy()
    mme = benchmark_cases[
        benchmark_cases.model.eq("mme")
        & benchmark_cases.region.eq("all_india")
        & benchmark_cases.season.eq("JJAS")
        & benchmark_cases.lead_week.isin(LEADS)
        & benchmark_cases.score_status.eq("available")
    ].copy()
    _require(set(mme.model_label) == {"MME (no ECMWF)"}, "MME definition differs")
    if strict:
        _require(set(neural.groupby("lead_week").size()) == {169}, "neural count differs")
        _require(set(mme.groupby("lead_week").size()) == {169}, "MME count differs")

    rows: list[dict[str, Any]] = []
    sampling: dict[str, Any] = {}
    for lead in LEADS:
        left = neural[neural.lead_week.eq(lead)].sort_values("init")
        right = mme[mme.lead_week.eq(lead)].sort_values("init")
        left_dates = left.init.to_numpy(dtype="datetime64[D]")
        right_dates = right.init.to_numpy(dtype="datetime64[D]")
        _require(np.array_equal(left_dates, right_dates), f"W{lead} case IDs differ")
        _require(
            np.array_equal(
                pd.to_datetime(left.valid_period_midpoint).values.astype("datetime64[h]"),
                pd.to_datetime(right.valid_period_midpoint).values.astype("datetime64[h]"),
            ),
            f"W{lead} valid midpoints differ",
        )
        for block in block_lengths:
            draw_seed = int(seed + 23000 + 100 * lead + int(block))
            indices = circular_year_stratified_index_matrix(
                left_dates,
                block_length=int(block),
                replicates=int(replicates),
                seed=draw_seed,
            )
            sampling[f"w{lead}__block{block}"] = {
                "seed": draw_seed,
                "shape": list(indices.shape),
                "dates_sha256": _dates_sha256(left_dates),
                "index_matrix_sha256": hashlib.sha256(
                    np.ascontiguousarray(indices).tobytes()
                ).hexdigest(),
            }
            for metric in ("acc", "rmse", "mae", "bias"):
                candidate = left[metric].to_numpy(dtype=np.float64)
                baseline = right[metric].to_numpy(dtype=np.float64)
                candidate_samples = candidate[indices].mean(axis=1, dtype=np.float64)
                baseline_samples = baseline[indices].mean(axis=1, dtype=np.float64)
                effects = candidate_samples - baseline_samples
                lower, upper = np.quantile(effects, (0.025, 0.975))
                common_fields = {
                    "reference": "imd",
                    "season": "JJAS_valid_midpoint",
                    "analysis_cohort": "2020_2024_valid_midpoint_jjas",
                    "region": "all_india",
                    "lead_week": int(lead),
                    "metric": metric,
                    "candidate": "location_spread",
                    "baseline": "equal_system_mme_no_ecmwf",
                    "candidate_mean": float(candidate.mean()),
                    "baseline_mean": float(baseline.mean()),
                    "n_cases": int(len(left_dates)),
                    "confidence": 0.95,
                    "block_length_starts": int(block),
                    "replicates": int(replicates),
                    "seed": draw_seed,
                }
                rows.append(
                    {
                        **common_fields,
                        "effect": "candidate_minus_baseline",
                        "estimate": float(candidate.mean() - baseline.mean()),
                        "ci_lower": float(lower),
                        "ci_upper": float(upper),
                    }
                )
                if metric in {"rmse", "mae"}:
                    _require(np.all(baseline_samples > 0.0), f"MME {metric} is non-positive")
                    skill = 100.0 * (1.0 - candidate_samples / baseline_samples)
                    skill_lower, skill_upper = np.quantile(skill, (0.025, 0.975))
                    rows.append(
                        {
                            **common_fields,
                            "effect": "skill_pct_vs_baseline",
                            "estimate": float(100.0 * (1.0 - candidate.mean() / baseline.mean())),
                            "ci_lower": float(skill_lower),
                            "ci_upper": float(skill_upper),
                        }
                    )
    receipt = {
        "cohort": "2020_2024_valid_midpoint_jjas",
        "cases_per_lead": int(len(neural[neural.lead_week.eq(1)])),
        "candidate": "selected seed-43 location-spread ensemble mean",
        "baseline": "fixed six-system equal mean",
        "mme_members": list(MME_COMPONENTS),
        "ecmwf_in_mme": False,
        "sampling": sampling,
    }
    return pd.DataFrame(rows), receipt


def _artifact_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in {"manifest.json", "failure.json"}
    }


def publish(
    *,
    scoring_manifest: Path,
    audit_receipt: Path,
    benchmark_cases: Path,
    methods_manifest: Path = DEFAULT_METHODS_MANIFEST,
    output: Path,
    replicates: int = 10_000,
    bootstrap_seed: int = 42,
) -> Path:
    """Validate inputs, compute both comparisons, and publish atomically."""

    destination = Path(output).resolve()
    root = DEFAULT_OUTPUT_ROOT.resolve()
    _require(root in destination.parents, "output must be under resultsv3 comparison root")
    _require(not destination.exists(), f"fresh output path required: {destination}")
    _require(replicates > 0, "replicates must be positive")
    staging = destination.with_name(f".{destination.name}.incomplete-{os.getpid()}")
    _require(not staging.exists(), f"staging path exists: {staging}")
    staging.mkdir(parents=True)
    frozen_inputs = {
        Path(scoring_manifest).resolve(): SCORING_MANIFEST_SHA256,
        Path(audit_receipt).resolve(): AUDIT_RECEIPT_SHA256,
        Path(benchmark_cases).resolve(): BENCHMARK_CASES_SHA256,
        Path(methods_manifest).resolve(): METHODS_MANIFEST_SHA256,
    }
    try:
        scoring_cases_path, scoring, _, _ = validate_inputs(
            scoring_manifest, audit_receipt, benchmark_cases, methods_manifest
        )
        frozen_inputs[scoring_cases_path] = SCORING_CASES_SHA256
        scoring_cases = pd.read_csv(scoring_cases_path)
        benchmark = pd.read_csv(benchmark_cases)
        _require(len(scoring_cases) == 45450, "scoring case-row count differs")
        _require(
            pd.to_datetime(scoring_cases.valid_period_midpoint).dt.year.max() <= 2024,
            "scoring table crosses the 2025 firewall",
        )
        exact, exact_receipt = exact_common_midpoint_sensitivity(
            scoring_cases,
            replicates=replicates,
            seed=bootstrap_seed,
        )
        versus_mme, mme_receipt = calibrated_fuxi_vs_mme(
            scoring_cases,
            benchmark,
            replicates=replicates,
            seed=bootstrap_seed,
        )
        tables = staging / "tables"
        receipts = staging / "receipts"
        source_dir = staging / "code/src"
        tables.mkdir(parents=True)
        receipts.mkdir(parents=True)
        source_dir.mkdir(parents=True)
        exact.to_csv(tables / "exact_common_midpoint_sensitivity.csv", index=False)
        versus_mme.to_csv(tables / "calibrated_fuxi_vs_mme.csv", index=False)
        _write_json(receipts / "exact_common_midpoint.json", exact_receipt)
        _write_json(receipts / "calibrated_fuxi_vs_mme.json", mme_receipt)
        sources = (
            Path(__file__).resolve(),
            Path(circular_year_stratified_index_matrix.__code__.co_filename).resolve(),
        )
        _require(len({source.name for source in sources}) == len(sources), "source names collide")
        for source in sources:
            shutil.copy2(source, source_dir / source.name)

        for path, expected in frozen_inputs.items():
            _require_hash(path, expected, "comparison input changed during run")
        source_hashes = {
            f"code/src/{source.name}": sha256_file(source) for source in sources
        }
        for source in sources:
            _require_hash(
                source_dir / source.name,
                source_hashes[f"code/src/{source.name}"],
                "source snapshot",
            )

        exact_primary = exact[exact.block_length_valid_midpoints.eq(16)]
        mme_primary = versus_mme[versus_mme.block_length_starts.eq(16)]
        manifest: dict[str, Any] = {
            "experiment": EXPERIMENT,
            "status": "complete",
            "scientific_status": "retrospective sensitivity and paired deterministic comparison",
            "output_path": str(destination),
            "inputs": {
                "scoring_manifest": str(Path(scoring_manifest).resolve()),
                "scoring_manifest_sha256": SCORING_MANIFEST_SHA256,
                "scoring_cases": str(scoring_cases_path),
                "scoring_cases_sha256": SCORING_CASES_SHA256,
                "audit_receipt": str(Path(audit_receipt).resolve()),
                "audit_receipt_sha256": AUDIT_RECEIPT_SHA256,
                "benchmark_cases": str(Path(benchmark_cases).resolve()),
                "benchmark_cases_sha256": BENCHMARK_CASES_SHA256,
                "methods_manifest": str(Path(methods_manifest).resolve()),
                "methods_manifest_sha256": METHODS_MANIFEST_SHA256,
            },
            "selected_seed": 43,
            "parameter_averaging": False,
            "prediction_averaging": False,
            "retrospective_only": True,
            "sealed_2025_target_opened": False,
            "latest_target_year": 2024,
            "exact_common_midpoint": exact_receipt,
            "calibrated_fuxi_vs_mme": mme_receipt,
            "bootstrap": {
                "replicates": int(replicates),
                "base_seed": int(bootstrap_seed),
                "block_lengths": list(BLOCK_LENGTHS),
                "confidence": 0.95,
            },
            "software_versions": {
                "python": platform.python_version(),
                "python_implementation": platform.python_implementation(),
                "python_executable": sys.executable,
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
            "table_rows": {
                "exact_common_midpoint_sensitivity": int(len(exact)),
                "calibrated_fuxi_vs_mme": int(len(versus_mme)),
            },
            "primary_results": {
                "exact_common_midpoint_rows": int(len(exact_primary)),
                "calibrated_fuxi_vs_mme_rows": int(len(mme_primary)),
            },
            "source_snapshot_sha256": source_hashes,
            "scoring_safety_contract": scoring["cohort"],
        }
        manifest["artifact_sha256"] = _artifact_hashes(staging)
        _write_json(staging / "manifest.json", manifest)
        for path, expected in frozen_inputs.items():
            _require_hash(path, expected, "comparison input changed before publication")
        for source in sources:
            _require_hash(
                source,
                source_hashes[f"code/src/{source.name}"],
                "comparison source changed during run",
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        _rename_noreplace(staging, destination)
    except BaseException as error:
        _write_json(
            staging / "failure.json",
            {
                "experiment": EXPERIMENT,
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "forecast_or_observation_arrays_opened": False,
                "sealed_2025_target_opened": False,
            },
        )
        raise
    return destination / "manifest.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scoring-manifest", type=Path, default=DEFAULT_SCORING_MANIFEST)
    parser.add_argument("--audit-receipt", type=Path, default=DEFAULT_AUDIT_RECEIPT)
    parser.add_argument("--benchmark-cases", type=Path, default=DEFAULT_BENCHMARK_CASES)
    parser.add_argument("--methods-manifest", type=Path, default=DEFAULT_METHODS_MANIFEST)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = publish(
        scoring_manifest=args.scoring_manifest,
        audit_receipt=args.audit_receipt,
        benchmark_cases=args.benchmark_cases,
        methods_manifest=args.methods_manifest,
        output=args.output,
        replicates=args.replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(f"PASS: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
