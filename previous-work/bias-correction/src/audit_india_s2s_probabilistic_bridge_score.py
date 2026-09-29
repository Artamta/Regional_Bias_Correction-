#!/usr/bin/env python3
"""Independent post-run audit for the India S2S probabilistic bridge score.

The auditor never imports the scoring implementation and never opens a
forecast or observation array.  It verifies the immutable file graph, checks
the complete case-key and temporal contracts, rebuilds every published
summary and bootstrap interval from ``case_metrics.csv``, and writes a new
no-clobber audit receipt.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import shutil
import stat
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from project_paths import PROJECT_ROOT


AUDIT_EXPERIMENT = "india_s2s_probabilistic_bridge_score_audit_v1"
SCORING_EXPERIMENT = "india_s2s_probabilistic_bridge_scoring_v1"
INFERENCE_EXPERIMENT = "india_s2s_probabilistic_bridge_inference_v1"
PREFLIGHT_EXPERIMENT = "india_s2s_probabilistic_bridge_v1"
PARENT_EXPERIMENT = "fuxi_allseason_ensemble_calibration_v2_aligned"
CONTRACT_REVISION = "imd_end_labelled_init_plus_1_through_42_v2"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "resultsv3/india_s2s_probabilistic_bridge"

METHOD_ORDER = ("raw_fuxi", "moment_calibration", "location_spread")
METHOD_LABELS = {
    "raw_fuxi": "Raw FuXi-S2S",
    "moment_calibration": "Train-only moment calibration",
    "location_spread": "Selected neural location-spread adapter",
}
REGION_ORDER = (
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
LEAD_WEEKS = (1, 2, 3, 4, 5, 6)
TARGET_DAY_OFFSETS = tuple(range(1, 43))
FORECAST_YEARS = (2020, 2021, 2022, 2023, 2024)
SCORING_YEAR_COUNTS = {2020: 105, 2021: 104, 2022: 104, 2023: 104, 2024: 88}
SCORING_DATES_SHA256 = (
    "41b67539c1d9196dfe5b598e0613a80a794dfea9da6ecbaae40cc102ebd66062"
)
FORECAST_DATES_SHA256 = (
    "e95a7f88fdc0ca13b775e1c8708b510310e4a647ff2040e099f65db71baeef56"
)
RAW_CASE_IDS_SHA256 = (
    "623713c5d2b93dd0a8501086e1c9dd2d2596fa6017c2b1572f94ba1e0a709983"
)
BENCHMARK_CASES_SHA256 = (
    "000544615d004669b41dd87fc647bd979cd92c3638e7deee5ea77992046e1a14"
)
CASE_ROW_COUNT = 45_450
RAW_CASE_ROW_COUNT = 15_150
JJAS_CASE_ROW_COUNT = 15_210
CASE_METRICS = (
    "crps",
    "acc",
    "rmse",
    "mae",
    "bias",
    "coverage90",
    "spread_skill_ratio",
)
LOSS_METRICS = ("crps", "rmse", "mae")
COMPARISON_PAIRS = (
    ("moment_calibration", "raw_fuxi"),
    ("location_spread", "raw_fuxi"),
    ("location_spread", "moment_calibration"),
)
BLOCK_LENGTHS = (16, 13)
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 42
SELECTED_SEED = 43
_RENAME_NOREPLACE = 1


class ScoreAuditError(RuntimeError):
    """Raised when a saved score run violates an independent audit contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ScoreAuditError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ScoreAuditError(f"cannot read JSON: {path}") from error
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _normalized(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))


def _resolve_child(root: Path, relative: str) -> Path:
    child = Path(str(relative))
    _require(not child.is_absolute(), f"artifact path must be relative: {relative}")
    resolved_root = Path(root).resolve()
    resolved = (resolved_root / child).resolve()
    _require(resolved_root in resolved.parents, f"artifact escapes run: {relative}")
    return resolved


@dataclass
class HashLedger:
    """Files whose bytes must remain stable throughout the audit."""

    expected: dict[Path, str] = field(default_factory=dict)
    labels: dict[Path, str] = field(default_factory=dict)

    def verify(self, path: Path, expected: Any, label: str) -> None:
        resolved = Path(path).resolve()
        _require(_valid_sha256(expected), f"{label} has an invalid SHA-256")
        _require(resolved.is_file(), f"{label} is missing: {resolved}")
        expected_text = str(expected)
        previous = self.expected.get(resolved)
        _require(
            previous in (None, expected_text),
            f"{label} conflicts with another hash binding for {resolved}",
        )
        observed = sha256_file(resolved)
        _require(
            observed == expected_text,
            f"{label} SHA-256 differs: {observed} != {expected_text}",
        )
        self.expected[resolved] = expected_text
        self.labels[resolved] = label

    def reverify(self) -> None:
        for path, expected in self.expected.items():
            _require(path.is_file(), f"audited input disappeared: {path}")
            _require(
                sha256_file(path) == expected,
                f"audited input changed during audit: {self.labels[path]}",
            )


def _artifact_inventory(root: Path) -> set[str]:
    return {
        str(path.relative_to(root))
        for path in Path(root).rglob("*")
        if path.is_file()
        and path.name not in {"manifest.json", "failure.json"}
        and ".tmp" not in path.name
    }


def _verify_artifacts(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    ledger: HashLedger,
    *,
    label: str,
) -> int:
    root = Path(manifest_path).resolve().parent
    expected = manifest.get("artifact_sha256")
    _require(isinstance(expected, dict) and expected, f"{label} has no artifact hashes")
    expected_keys = {str(key) for key in expected}
    actual_keys = _artifact_inventory(root)
    _require(
        actual_keys == expected_keys,
        f"{label} artifact inventory differs: "
        f"extra={sorted(actual_keys - expected_keys)}, "
        f"missing={sorted(expected_keys - actual_keys)}",
    )
    for relative, digest in expected.items():
        ledger.verify(_resolve_child(root, str(relative)), digest, f"{label} artifact {relative}")
    source_hashes = manifest.get("source_snapshot_sha256", {})
    _require(isinstance(source_hashes, dict), f"{label} source hashes are invalid")
    for relative, digest in source_hashes.items():
        _require(
            expected.get(relative) == digest,
            f"{label} source snapshot is not identically artifact-bound: {relative}",
        )
    return len(expected)


def initialization_dates_sha256(values: Iterable[Any]) -> str:
    dates = np.asarray(list(values), dtype="datetime64[D]")
    _require(dates.ndim == 1 and dates.size > 0, "date hash requires a non-empty vector")
    _require(not np.isnat(dates).any(), "date hash contains NaT")
    _require(np.unique(dates).size == dates.size, "date hash contains duplicates")
    dates = np.sort(dates)
    payload = "".join(
        f"{np.datetime_as_string(value, unit='D')}\n" for value in dates
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def raw_case_ids_sha256(frame: pd.DataFrame) -> str:
    required = {"init", "lead_week", "region"}
    _require(required.issubset(frame.columns), "raw case-ID columns are absent")
    records = sorted(
        (
            np.datetime_as_string(np.datetime64(init, "D"), unit="D"),
            str(int(lead)),
            str(region),
        )
        for init, lead, region in frame[["init", "lead_week", "region"]].itertuples(
            index=False, name=None
        )
    )
    payload = "".join("|".join(record) + "\n" for record in records).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _season(month: int) -> str:
    if month in (1, 2):
        return "JF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8, 9):
        return "JJAS"
    return "OND"


def _year_counts(dates: np.ndarray) -> dict[int, int]:
    years = pd.DatetimeIndex(dates).year.to_numpy(dtype=np.int64)
    return {year: int(np.count_nonzero(years == year)) for year in FORECAST_YEARS}


def _read_csv(path: Path, label: str) -> pd.DataFrame:
    _require(path.is_file(), f"{label} is missing: {path}")
    frame = pd.read_csv(path)
    _require(not frame.empty, f"{label} is empty")
    _require(
        not any(str(column).startswith("Unnamed:") for column in frame.columns),
        f"{label} has an unnamed column",
    )
    return frame


def _sort_frame(frame: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    _require(set(keys).issubset(frame.columns), f"comparison keys are absent: {keys}")
    order = sorted(
        range(len(frame)),
        key=lambda index: tuple(str(frame.iloc[index][key]) for key in keys),
    )
    return frame.iloc[order].reset_index(drop=True)


def _require_frames_equivalent(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    keys: Sequence[str],
    label: str,
    atol: float = 2.0e-10,
) -> None:
    _require(set(actual.columns) == set(expected.columns), f"{label} columns differ")
    _require(len(actual) == len(expected), f"{label} row count differs")
    _require(not actual.duplicated(list(keys)).any(), f"{label} has duplicate identities")
    _require(
        not expected.duplicated(list(keys)).any(),
        f"reconstructed {label} has duplicate identities",
    )
    left = _sort_frame(actual, keys)
    right = _sort_frame(expected, keys)
    for column in sorted(left.columns):
        first = left[column]
        second = right[column]
        if pd.api.types.is_numeric_dtype(first) and pd.api.types.is_numeric_dtype(second):
            _require(
                np.allclose(
                    first.to_numpy(dtype=np.float64),
                    second.to_numpy(dtype=np.float64),
                    rtol=2.0e-10,
                    atol=atol,
                    equal_nan=True,
                ),
                f"{label} numeric column differs: {column}",
            )
        else:
            left_values = first.fillna("<NA>").astype(str).to_numpy()
            right_values = second.fillna("<NA>").astype(str).to_numpy()
            _require(
                np.array_equal(left_values, right_values),
                f"{label} categorical column differs: {column}",
            )


def validate_case_metrics(cases: pd.DataFrame) -> np.ndarray:
    """Validate all 45,450 semantic rows and return the frozen 505 starts."""

    required = {
        "track",
        "method",
        "variable",
        "reference",
        "reference_label",
        "units",
        "score_status",
        "verification_year",
        "init",
        "valid_period_start",
        "valid_period_midpoint",
        "valid_period_end_exclusive",
        "season",
        "region",
        "region_label",
        "lead_week",
        "valid_cell_count",
        "effective_area_km2",
        "ensemble_variance",
        "mean_squared_error",
        "ensemble_spread",
        "model",
        "model_label",
        "member_count",
        "selected_seed",
        *CASE_METRICS,
    }
    missing = sorted(required.difference(cases.columns))
    _require(not missing, f"case metrics lacks columns: {missing}")
    _require(len(cases) == CASE_ROW_COUNT, f"case row count differs: {len(cases)}")
    keys = ["method", "init", "region", "lead_week"]
    _require(not cases.duplicated(keys).any(), "case metrics has duplicate Cartesian keys")

    dates = np.asarray(sorted(cases["init"].astype(str).unique()), dtype="datetime64[D]")
    _require(len(dates) == 505, "case metrics does not contain exactly 505 starts")
    _require(_year_counts(dates) == SCORING_YEAR_COUNTS, "scoring year counts differ")
    _require(
        initialization_dates_sha256(dates) == SCORING_DATES_SHA256,
        "scoring initialization dates differ from the frozen hash",
    )
    expected_keys = {
        (method, np.datetime_as_string(init, unit="D"), region, lead)
        for method in METHOD_ORDER
        for init in dates
        for region in REGION_ORDER
        for lead in LEAD_WEEKS
    }
    observed_keys = set(cases[keys].itertuples(index=False, name=None))
    _require(observed_keys == expected_keys, "case metrics is not the full Cartesian grid")

    fixed_columns = {
        "track": "tp_imd",
        "variable": "tp",
        "reference": "imd",
        "reference_label": "IMD",
        "units": "mm day-1",
        "score_status": "available",
        "member_count": 50,
    }
    for column, expected in fixed_columns.items():
        _require(
            set(cases[column].dropna().unique()) == {expected},
            f"case {column} contract differs",
        )
    _require(set(cases.method) == set(METHOD_ORDER), "case methods differ")
    _require(set(cases.region) == set(REGION_ORDER), "case regions differ")
    _require(set(pd.to_numeric(cases.lead_week)) == set(LEAD_WEEKS), "case leads differ")
    _require((cases.model.astype(str) == cases.method.astype(str)).all(), "model/method differs")
    for method, label in METHOD_LABELS.items():
        selected = cases.method.eq(method)
        _require(set(cases.loc[selected, "model_label"]) == {label}, f"{method} label differs")
    for region, label in REGION_LABELS.items():
        selected = cases.region.eq(region)
        _require(set(cases.loc[selected, "region_label"]) == {label}, f"{region} label differs")

    seed_text = cases.selected_seed.astype(str)
    neural = cases.method.eq("location_spread")
    _require(set(seed_text[neural]) <= {"43", "43.0"}, "neural rows are not seed 43")
    _require(len(set(seed_text[neural])) == 1, "neural rows mix seed labels")
    _require(
        set(seed_text[~neural]) == {"not_applicable"},
        "non-neural rows improperly carry optimization seeds",
    )
    forbidden_seed_tokens = ("mean", "average", "ensemble_of_seed", "seed_42", "seed_44")
    serialized_methods = " ".join(cases.method.astype(str).unique()).lower()
    _require(
        not any(token in serialized_methods for token in forbidden_seed_tokens),
        "case table contains a seed-averaged method",
    )

    init = cases["init"].to_numpy(dtype="datetime64[D]")
    lead = pd.to_numeric(cases["lead_week"], errors="raise").to_numpy(dtype=np.int64)
    expected_start = init + (7 * (lead - 1)).astype("timedelta64[D]")
    expected_end = init + (7 * lead).astype("timedelta64[D]")
    expected_midpoint = init.astype("datetime64[h]") + (
        84 + 168 * (lead - 1)
    ).astype("timedelta64[h]")
    _require(
        np.array_equal(
            cases.valid_period_start.to_numpy(dtype="datetime64[D]"), expected_start
        ),
        "valid-period start labels differ",
    )
    _require(
        np.array_equal(
            cases.valid_period_end_exclusive.to_numpy(dtype="datetime64[D]"), expected_end
        ),
        "valid-period end labels differ",
    )
    _require(
        np.array_equal(
            cases.valid_period_midpoint.to_numpy(dtype="datetime64[h]"), expected_midpoint
        ),
        "valid-period midpoint labels differ",
    )
    expected_season = np.asarray(
        [_season(month) for month in pd.DatetimeIndex(expected_midpoint).month]
    )
    _require(np.array_equal(cases.season.astype(str), expected_season), "season labels differ")
    _require(
        np.array_equal(
            pd.to_numeric(cases.verification_year).to_numpy(dtype=np.int64),
            pd.DatetimeIndex(init).year.to_numpy(dtype=np.int64),
        ),
        "verification years differ from initialization years",
    )
    _require(
        (dates + np.timedelta64(42, "D")).max() == np.datetime64("2024-12-30"),
        "maximum opened +42 target label differs",
    )
    _require(
        (dates + np.timedelta64(42, "D")).max() < np.datetime64("2025-01-01"),
        "case table crosses the sealed 2025 target firewall",
    )

    finite_columns = [
        "crps",
        "rmse",
        "mae",
        "bias",
        "coverage90",
        "ensemble_variance",
        "mean_squared_error",
        "ensemble_spread",
        "effective_area_km2",
        "valid_cell_count",
    ]
    values = cases[finite_columns].to_numpy(dtype=np.float64)
    _require(np.isfinite(values).all(), "case metrics contains non-finite required values")
    _require(np.isfinite(cases.acc.to_numpy(dtype=np.float64)).all(), "case ACC is non-finite")
    _require(
        (cases[["crps", "rmse", "mae", "ensemble_variance", "mean_squared_error"]] >= 0.0)
        .all()
        .all(),
        "case loss/variance metrics contain negative values",
    )
    _require(cases.acc.between(-1.0, 1.0).all(), "case ACC lies outside [-1,1]")
    _require(cases.coverage90.between(0.0, 1.0).all(), "case coverage lies outside [0,1]")
    _require((cases.valid_cell_count > 0).all(), "case valid-cell count is non-positive")
    _require((cases.effective_area_km2 > 0.0).all(), "case effective area is non-positive")
    _require(
        np.allclose(
            cases.mean_squared_error.to_numpy(dtype=np.float64),
            np.square(cases.rmse.to_numpy(dtype=np.float64)),
            rtol=3.0e-6,
            atol=3.0e-8,
        ),
        "case MSE is inconsistent with RMSE",
    )
    _require(
        np.allclose(
            cases.ensemble_spread.to_numpy(dtype=np.float64),
            np.sqrt(cases.ensemble_variance.to_numpy(dtype=np.float64)),
            rtol=3.0e-6,
            atol=3.0e-8,
        ),
        "case ensemble spread is inconsistent with variance",
    )
    ratio = cases.spread_skill_ratio.to_numpy(dtype=np.float64)
    variance = cases.ensemble_variance.to_numpy(dtype=np.float64)
    mse = cases.mean_squared_error.to_numpy(dtype=np.float64)
    positive_mse = mse > 0.0
    _require(
        np.isfinite(ratio[positive_mse]).all()
        and np.allclose(
            ratio[positive_mse],
            np.sqrt(variance[positive_mse] / mse[positive_mse]),
            rtol=3.0e-6,
            atol=3.0e-8,
        ),
        "case spread-skill ratio is inconsistent with positive MSE",
    )
    _require(
        np.isnan(ratio[~positive_mse]).all(),
        "zero-MSE cases must store undefined spread-skill ratio as NaN",
    )
    support = cases.pivot(
        index=["init", "region", "lead_week"],
        columns="method",
        values=["valid_cell_count", "effective_area_km2"],
    )
    for metric in ("valid_cell_count", "effective_area_km2"):
        matrix = support[metric].to_numpy(dtype=np.float64)
        _require(
            np.allclose(matrix, matrix[:, :1], rtol=1.0e-12, atol=1.0e-6),
            f"{metric} changes between methods on identical cases",
        )

    jjas_counts = cases[cases.season.eq("JJAS")].groupby(
        ["method", "region", "lead_week"]
    ).size()
    _require(
        len(jjas_counts) == len(METHOD_ORDER) * len(REGION_ORDER) * len(LEAD_WEEKS)
        and set(jjas_counts) == {169},
        "valid-midpoint JJAS does not contain 169 cases per method/region/lead",
    )
    development = cases[
        cases.season.eq("JJAS") & cases.verification_year.between(2022, 2024)
    ]
    development_counts = development.groupby(["method", "region", "lead_week"]).size()
    _require(set(development_counts) == {100}, "2022-2024 JJAS does not contain 100/lead")
    return dates


def aggregate_case_metrics(
    cases: pd.DataFrame,
    *,
    group_by: Sequence[str] = ("method", "region", "lead_week"),
) -> pd.DataFrame:
    """Independently reproduce the scorer's arithmetic case aggregation."""

    rows: list[dict[str, Any]] = []
    for key, group in cases.groupby(list(group_by), sort=False, dropna=False):
        values = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_by, values, strict=True))
        row["case_count"] = int(len(group))
        for metric in CASE_METRICS:
            data = group[metric].to_numpy(dtype=np.float64)
            finite = data[np.isfinite(data)]
            row[f"{metric}_valid_case_count"] = int(len(finite))
            if metric == "spread_skill_ratio":
                row["mean_case_spread_skill_ratio"] = (
                    float(finite.mean()) if finite.size else np.nan
                )
            else:
                row[metric] = float(finite.mean()) if finite.size else np.nan
        variance = float(group.ensemble_variance.mean())
        mse = float(group.mean_squared_error.mean())
        row["ensemble_variance"] = variance
        row["mean_squared_error"] = mse
        row["ensemble_spread"] = float(np.sqrt(max(variance, 0.0)))
        row["spread_skill_ratio"] = (
            float(np.sqrt(max(variance, 0.0) / mse)) if mse > 0.0 else np.nan
        )
        row["pooled_spread_skill_ratio"] = row["spread_skill_ratio"]
        rows.append(row)
    return pd.DataFrame(rows)


def add_raw_comparisons(summary: pd.DataFrame) -> pd.DataFrame:
    result = summary.copy()
    keys = [column for column in ("region", "lead_week") if column in result.columns]
    raw = result[result.method.eq("raw_fuxi")].set_index(keys)
    for metric in LOSS_METRICS:
        result[f"{metric}_skill_pct_vs_raw"] = np.nan
    for metric in ("acc", "bias", "coverage90", "spread_skill_ratio"):
        result[f"delta_{metric}_vs_raw"] = np.nan
    for index, row in result.iterrows():
        key: Any = tuple(row[column] for column in keys)
        if len(key) == 1:
            key = key[0]
        reference = raw.loc[key]
        for metric in LOSS_METRICS:
            baseline = float(reference[metric])
            result.loc[index, f"{metric}_skill_pct_vs_raw"] = (
                100.0 * (1.0 - float(row[metric]) / baseline)
                if baseline > 0.0
                else np.nan
            )
        for metric in ("acc", "bias", "coverage90", "spread_skill_ratio"):
            result.loc[index, f"delta_{metric}_vs_raw"] = float(row[metric]) - float(
                reference[metric]
            )
    return result


def independent_index_matrix(
    initializations: np.ndarray,
    *,
    block_length: int,
    replicates: int,
    seed: int,
) -> np.ndarray:
    """Reproduce year-stratified circular indices without scorer imports."""

    dates = np.asarray(initializations, dtype="datetime64[D]")
    _require(dates.ndim == 1 and dates.size > 0, "bootstrap dates are empty")
    _require(np.unique(dates).size == len(dates), "bootstrap dates are duplicated")
    _require(block_length > 0 and replicates > 0, "bootstrap dimensions are invalid")
    years = pd.DatetimeIndex(dates).year.to_numpy(dtype=np.int64)
    random = np.random.default_rng(seed)
    sampled: list[np.ndarray] = []
    offsets = np.arange(block_length, dtype=np.int64)
    for year in np.unique(years):
        original = np.flatnonzero(years == year)
        original = original[np.argsort(dates[original])]
        count = original.size
        blocks_needed = (count + block_length - 1) // block_length
        starts = random.integers(0, count, size=(replicates, blocks_needed))
        within = (
            (starts[:, :, None] + offsets[None, None, :]) % count
        ).reshape(replicates, -1)[:, :count]
        sampled.append(original[within])
    return np.concatenate(sampled, axis=1)


def _paired_arrays(
    selected: pd.DataFrame, candidate: str, baseline: str, metric: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    subset = selected[selected.method.isin((candidate, baseline))]
    _require(not subset.duplicated(["method", "init"]).any(), "paired rows are duplicated")
    wide = subset.pivot(index="init", columns="method", values=metric).sort_index()
    _require(
        set(wide.columns) == {candidate, baseline} and not wide.isna().any().any(),
        f"incomplete pairing for {candidate}/{baseline}/{metric}",
    )
    return (
        wide.index.to_numpy(dtype="datetime64[D]"),
        wide[candidate].to_numpy(dtype=np.float64),
        wide[baseline].to_numpy(dtype=np.float64),
    )


def _sampling_receipt(
    dates: np.ndarray, indices: np.ndarray, *, seed: int
) -> dict[str, Any]:
    return {
        "dates_sha256": initialization_dates_sha256(dates),
        "index_matrix_sha256": hashlib.sha256(
            np.ascontiguousarray(indices).tobytes()
        ).hexdigest(),
        "shape": list(indices.shape),
        "seed": int(seed),
    }


def reconstruct_lead_intervals(
    cases: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
    block_lengths: Sequence[int],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Rebuild both lead-wise JJAS cohorts and their sampling receipts."""

    all_rows: list[pd.DataFrame] = []
    all_receipts: dict[str, Any] = {}
    cohorts = (
        ("2020_2024_valid_midpoint_jjas", cases[cases.season.eq("JJAS")], 169),
        (
            "2022_2024_valid_midpoint_jjas",
            cases[cases.season.eq("JJAS") & cases.verification_year.between(2022, 2024)],
            100,
        ),
    )
    for cohort_name, cohort, expected_count in cohorts:
        rows: list[dict[str, Any]] = []
        cohort_receipt: dict[str, Any] = {}
        for lead in LEAD_WEEKS:
            lead_cases = cohort[cohort.lead_week.eq(lead)]
            reference_dates: np.ndarray | None = None
            matrices: dict[int, np.ndarray] = {}
            for region in REGION_ORDER:
                selected = lead_cases[lead_cases.region.eq(region)]
                raw_dates = np.asarray(
                    sorted(selected.loc[selected.method.eq("raw_fuxi"), "init"].unique()),
                    dtype="datetime64[D]",
                )
                _require(
                    len(raw_dates) == expected_count,
                    f"{cohort_name}/{region}/W{lead} date count differs",
                )
                if reference_dates is None:
                    reference_dates = raw_dates
                    for block in block_lengths:
                        draw_seed = int(seed + 1000 * lead + int(block))
                        matrix = independent_index_matrix(
                            raw_dates,
                            block_length=int(block),
                            replicates=replicates,
                            seed=draw_seed,
                        )
                        matrices[int(block)] = matrix
                        cohort_receipt[f"w{lead}__block{block}"] = _sampling_receipt(
                            raw_dates, matrix, seed=draw_seed
                        )
                else:
                    _require(
                        np.array_equal(raw_dates, reference_dates),
                        f"{cohort_name} regional dates differ at W{lead}",
                    )
                for candidate, baseline in COMPARISON_PAIRS:
                    for metric in CASE_METRICS:
                        if metric == "spread_skill_ratio":
                            dates, candidate_variance, baseline_variance = _paired_arrays(
                                selected, candidate, baseline, "ensemble_variance"
                            )
                            mse_dates, candidate_mse, baseline_mse = _paired_arrays(
                                selected, candidate, baseline, "mean_squared_error"
                            )
                            _require(np.array_equal(dates, mse_dates), "spread/MSE dates differ")
                            candidate_estimate = float(
                                np.sqrt(candidate_variance.mean() / candidate_mse.mean())
                            )
                            baseline_estimate = float(
                                np.sqrt(baseline_variance.mean() / baseline_mse.mean())
                            )
                            candidate_values = baseline_values = None
                        else:
                            dates, candidate_values, baseline_values = _paired_arrays(
                                selected, candidate, baseline, metric
                            )
                            candidate_estimate = float(candidate_values.mean())
                            baseline_estimate = float(baseline_values.mean())
                        _require(
                            reference_dates is not None
                            and np.array_equal(dates, reference_dates),
                            f"{cohort_name} paired dates differ",
                        )
                        for block in block_lengths:
                            indices = matrices[int(block)]
                            if metric == "spread_skill_ratio":
                                candidate_samples = np.sqrt(
                                    candidate_variance[indices].mean(axis=1)
                                    / candidate_mse[indices].mean(axis=1)
                                )
                                baseline_samples = np.sqrt(
                                    baseline_variance[indices].mean(axis=1)
                                    / baseline_mse[indices].mean(axis=1)
                                )
                                samples = candidate_samples - baseline_samples
                                estimate = candidate_estimate - baseline_estimate
                            else:
                                assert candidate_values is not None
                                assert baseline_values is not None
                                difference = candidate_values - baseline_values
                                samples = difference[indices].mean(axis=1, dtype=np.float64)
                                estimate = float(difference.mean())
                            lower, upper = np.quantile(samples, (0.025, 0.975))
                            common = {
                                "reference": "imd",
                                "season": "JJAS_valid_midpoint",
                                "analysis_cohort": cohort_name,
                                "region": region,
                                "lead_week": int(lead),
                                "metric": metric,
                                "candidate": candidate,
                                "baseline": baseline,
                                "candidate_mean": candidate_estimate,
                                "baseline_mean": baseline_estimate,
                                "n_cases": int(expected_count),
                                "confidence": 0.95,
                                "block_length_starts": int(block),
                                "replicates": int(replicates),
                                "seed": int(seed + 1000 * lead + int(block)),
                            }
                            rows.append(
                                {
                                    **common,
                                    "effect": "candidate_minus_baseline",
                                    "estimate": estimate,
                                    "ci_lower": float(lower),
                                    "ci_upper": float(upper),
                                }
                            )
                            if metric in LOSS_METRICS:
                                assert candidate_values is not None
                                assert baseline_values is not None
                                candidate_mean = candidate_values[indices].mean(axis=1)
                                baseline_mean = baseline_values[indices].mean(axis=1)
                                _require(
                                    np.all(baseline_mean > 0.0),
                                    f"non-positive bootstrap baseline {metric}",
                                )
                                skill = 100.0 * (1.0 - candidate_mean / baseline_mean)
                                skill_lower, skill_upper = np.quantile(skill, (0.025, 0.975))
                                rows.append(
                                    {
                                        **common,
                                        "effect": "skill_pct_vs_baseline",
                                        "estimate": float(
                                            100.0
                                            * (
                                                1.0
                                                - candidate_values.mean()
                                                / baseline_values.mean()
                                            )
                                        ),
                                        "ci_lower": float(skill_lower),
                                        "ci_upper": float(skill_upper),
                                    }
                                )
        all_rows.append(pd.DataFrame(rows))
        all_receipts[cohort_name] = cohort_receipt
    return pd.concat(all_rows, ignore_index=True), all_receipts


def reconstruct_synchronized_story_gate(
    cases: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
    block_lengths: Sequence[int],
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Rebuild the synchronized six-lead 2022--2024 valid-midpoint gate."""

    selected = cases[
        cases.season.eq("JJAS")
        & cases.verification_year.between(2022, 2024)
        & cases.region.eq("all_india")
    ].copy()
    arrays: dict[tuple[str, str, int], np.ndarray] = {}
    lead_dates: dict[int, np.ndarray] = {}
    for lead in LEAD_WEEKS:
        subset = selected[selected.lead_week.eq(lead)]
        dates = np.asarray(
            sorted(subset.loc[subset.method.eq("raw_fuxi"), "init"].unique()),
            dtype="datetime64[D]",
        )
        _require(len(dates) == 100, f"pooled W{lead} does not contain 100 starts")
        _require(
            _year_counts(dates) == {2020: 0, 2021: 0, 2022: 35, 2023: 35, 2024: 30},
            f"pooled W{lead} year counts differ",
        )
        lead_dates[lead] = dates
        for method in METHOD_ORDER:
            method_rows = subset[subset.method.eq(method)].sort_values("init")
            _require(
                np.array_equal(method_rows.init.to_numpy(dtype="datetime64[D]"), dates),
                f"pooled {method}/W{lead} dates differ",
            )
            for metric in ("crps", "acc", "bias"):
                arrays[(method, metric, lead)] = method_rows[metric].to_numpy(
                    dtype=np.float64
                )
    year_sequences = [
        tuple(pd.DatetimeIndex(lead_dates[lead]).year) for lead in LEAD_WEEKS
    ]
    _require(
        all(sequence == year_sequences[0] for sequence in year_sequences[1:]),
        "pooled lead cohorts do not share the same sorted year-count sequence",
    )
    date_sets = [set(values.tolist()) for values in lead_dates.values()]
    intersection_count = len(set.intersection(*date_sets))
    union_count = len(set.union(*date_sets))
    _require(
        (intersection_count, union_count) == (70, 130),
        "pooled lead-specific date intersection/union differs",
    )

    reference_dates = lead_dates[1]
    rows: list[dict[str, Any]] = []
    receipts: dict[str, Any] = {}
    for block in block_lengths:
        draw_seed = int(seed + 9000 + int(block))
        indices = independent_index_matrix(
            reference_dates,
            block_length=int(block),
            replicates=replicates,
            seed=draw_seed,
        )
        receipt = {
            "ordinal_reference_lead": 1,
            "lead_dates_sha256": {
                f"w{lead}": initialization_dates_sha256(lead_dates[lead])
                for lead in LEAD_WEEKS
            },
            "index_matrix_sha256": hashlib.sha256(
                np.ascontiguousarray(indices).tobytes()
            ).hexdigest(),
            "shape": list(indices.shape),
            "seed": draw_seed,
            "year_counts_per_lead": {2022: 35, 2023: 35, 2024: 30},
            "n_cases_per_lead": 100,
            "n_case_lead_rows": 600,
            "lead_date_intersection_count": intersection_count,
            "lead_date_union_count": union_count,
            "sampling_unit": (
                "within-verification-year ordinal synchronized across "
                "six lead-specific valid-midpoint cohorts"
            ),
        }
        receipts[f"pooled_w1_w6__block{block}"] = receipt
        for candidate, baseline in (
            ("location_spread", "raw_fuxi"),
            ("location_spread", "moment_calibration"),
        ):
            for metric in ("crps", "acc", "bias"):
                candidate_values = np.stack(
                    [arrays[(candidate, metric, lead)] for lead in LEAD_WEEKS]
                )
                baseline_values = np.stack(
                    [arrays[(baseline, metric, lead)] for lead in LEAD_WEEKS]
                )
                differences = candidate_values - baseline_values
                samples = differences[:, indices].mean(axis=(0, 2), dtype=np.float64)
                lower, upper = np.quantile(samples, (0.025, 0.975))
                common = {
                    "reference": "imd",
                    "season": "JJAS_valid_midpoint",
                    "analysis_cohort": (
                        "2022_2024_valid_midpoint_jjas_synchronized_pooled_w1_w6"
                    ),
                    "region": "all_india",
                    "lead_week": "pooled_w1_w6",
                    "lead_aggregation": (
                        "arithmetic mean over six lead-specific 100-case cohorts; "
                        "bootstrap draws synchronized by within-year ordinal"
                    ),
                    "metric": metric,
                    "candidate": candidate,
                    "baseline": baseline,
                    "candidate_mean": float(candidate_values.mean()),
                    "baseline_mean": float(baseline_values.mean()),
                    "n_cases_per_lead": 100,
                    "n_case_lead_rows": 600,
                    "lead_date_intersection_count": intersection_count,
                    "lead_date_union_count": union_count,
                    "confidence": 0.95,
                    "block_length_starts": int(block),
                    "replicates": int(replicates),
                    "seed": draw_seed,
                }
                rows.append(
                    {
                        **common,
                        "effect": "candidate_minus_baseline",
                        "estimate": float(differences.mean()),
                        "ci_lower": float(lower),
                        "ci_upper": float(upper),
                    }
                )
                if metric == "crps":
                    candidate_samples = candidate_values[:, indices].mean(
                        axis=(0, 2), dtype=np.float64
                    )
                    baseline_samples = baseline_values[:, indices].mean(
                        axis=(0, 2), dtype=np.float64
                    )
                    _require(np.all(baseline_samples > 0.0), "pooled CRPS baseline is non-positive")
                    skill = 100.0 * (1.0 - candidate_samples / baseline_samples)
                    skill_lower, skill_upper = np.quantile(skill, (0.025, 0.975))
                    rows.append(
                        {
                            **common,
                            "effect": "skill_pct_vs_baseline",
                            "estimate": float(
                                100.0
                                * (1.0 - candidate_values.mean() / baseline_values.mean())
                            ),
                            "ci_lower": float(skill_lower),
                            "ci_upper": float(skill_upper),
                        }
                    )
    frame = pd.DataFrame(rows)
    primary_block = int(block_lengths[0])
    crps = frame[
        frame.candidate.eq("location_spread")
        & frame.baseline.eq("raw_fuxi")
        & frame.metric.eq("crps")
        & frame.effect.eq("skill_pct_vs_baseline")
        & frame.block_length_starts.eq(primary_block)
    ].iloc[0]
    acc = frame[
        frame.candidate.eq("location_spread")
        & frame.baseline.eq("raw_fuxi")
        & frame.metric.eq("acc")
        & frame.effect.eq("candidate_minus_baseline")
        & frame.block_length_starts.eq(primary_block)
    ].iloc[0]
    bias = frame[
        frame.candidate.eq("location_spread")
        & frame.baseline.eq("raw_fuxi")
        & frame.metric.eq("bias")
        & frame.effect.eq("candidate_minus_baseline")
        & frame.block_length_starts.eq(primary_block)
    ].iloc[0]
    neural_vs_moment = frame[
        frame.candidate.eq("location_spread")
        & frame.baseline.eq("moment_calibration")
        & frame.metric.eq("crps")
        & frame.effect.eq("skill_pct_vs_baseline")
        & frame.block_length_starts.eq(primary_block)
    ].iloc[0]
    retrospective_pass = bool(
        crps["estimate"] > 0.0
        and crps["ci_lower"] > 0.0
        and acc["estimate"] > 0.0
        and acc["ci_lower"] > 0.0
    )
    gate = {
        "analysis_cohort": (
            "2022_2024_valid_midpoint_jjas_synchronized_pooled_w1_w6"
        ),
        "region": "all_india",
        "primary_block_length_starts": primary_block,
        "cases_per_lead": 100,
        "case_lead_rows": 600,
        "lead_specific_dates": True,
        "date_intersection_count": intersection_count,
        "date_union_count": union_count,
        "crps_skill_pct_vs_raw": {
            "estimate": float(crps.estimate),
            "ci_lower": float(crps.ci_lower),
            "ci_upper": float(crps.ci_upper),
        },
        "acc_delta_vs_raw": {
            "estimate": float(acc.estimate),
            "ci_lower": float(acc.ci_lower),
            "ci_upper": float(acc.ci_upper),
        },
        "bias_delta_vs_raw": {
            "estimate": float(bias.estimate),
            "ci_lower": float(bias.ci_lower),
            "ci_upper": float(bias.ci_upper),
        },
        "crps_skill_pct_vs_moment": {
            "estimate": float(neural_vs_moment.estimate),
            "ci_lower": float(neural_vs_moment.ci_lower),
            "ci_upper": float(neural_vs_moment.ci_upper),
        },
        "retrospective_2022_2024_component_passes": retrospective_pass,
        "sealed_2025_target_opened": False,
        "final_headline_gate": "pending_sealed_2025_point-estimate_direction_check",
    }
    return frame, receipts, gate


def validate_raw_identity(
    cases: pd.DataFrame,
    benchmark_path: Path,
    stored_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Independently check the 15,150 raw rows against the frozen benchmark."""

    benchmark = _read_csv(benchmark_path, "frozen benchmark case metrics")
    allowed = set(cases.loc[cases.method.eq("raw_fuxi"), "init"].astype(str))
    reference = benchmark[
        benchmark.model.eq("fuxi_s2s")
        & benchmark.reference.eq("imd")
        & benchmark.score_status.eq("available")
        & benchmark.init.astype(str).isin(allowed)
    ].copy()
    raw = cases[cases.method.eq("raw_fuxi")].copy()
    key_columns = ("init", "lead_week", "region")
    metric_columns = (
        "acc",
        "rmse",
        "mae",
        "bias",
        "valid_cell_count",
        "effective_area_km2",
    )
    _require(len(raw) == RAW_CASE_ROW_COUNT, "raw case table is not 15,150 rows")
    _require(len(reference) == RAW_CASE_ROW_COUNT, "benchmark raw identity is not 15,150 rows")
    _require(not raw.duplicated(list(key_columns)).any(), "raw case IDs are duplicated")
    _require(not reference.duplicated(list(key_columns)).any(), "benchmark case IDs are duplicated")
    left = raw.set_index(list(key_columns)).sort_index()
    right = reference.set_index(list(key_columns)).sort_index()
    _require(left.index.equals(right.index), "raw and benchmark case IDs differ")
    case_hash = raw_case_ids_sha256(left.reset_index())
    _require(case_hash == RAW_CASE_IDS_SHA256, "raw Cartesian case-ID hash differs")
    failures: dict[str, int] = {}
    maximum: dict[str, float] = {}
    for metric in metric_columns:
        first = left[metric].to_numpy(dtype=np.float64)
        second = right[metric].to_numpy(dtype=np.float64)
        both_nan = np.isnan(first) & np.isnan(second)
        comparable = np.isfinite(first) & np.isfinite(second)
        invalid = ~(both_nan | comparable)
        close = np.zeros(len(first), dtype=bool)
        close[both_nan] = True
        if metric == "valid_cell_count":
            close[comparable] = first[comparable] == second[comparable]
        elif metric == "effective_area_km2":
            close[comparable] = np.isclose(
                first[comparable], second[comparable], atol=1.0e-6, rtol=1.0e-12
            )
        else:
            close[comparable] = np.isclose(
                first[comparable], second[comparable], atol=1.0e-6, rtol=1.0e-7
            )
        failures[metric] = int(np.count_nonzero(invalid | ~close))
        maximum[metric] = (
            float(np.max(np.abs(first[comparable] - second[comparable])))
            if np.any(comparable)
            else 0.0
        )
    result = {
        "identity": not any(failures.values()),
        "matched_case_rows": RAW_CASE_ROW_COUNT,
        "case_ids_sha256": case_hash,
        "complete_bridge_contract": True,
        "key_columns": list(key_columns),
        "metric_columns": list(metric_columns),
        "atol": 1.0e-6,
        "rtol": 1.0e-7,
        "support_atol": 1.0e-6,
        "support_rtol": 1.0e-12,
        "failure_count_by_metric": failures,
        "max_absolute_difference_by_metric": maximum,
        "benchmark_case_metrics": str(Path(benchmark_path).resolve()),
        "benchmark_case_metrics_sha256": BENCHMARK_CASES_SHA256,
    }
    _require(result["identity"], f"raw benchmark identity failed: {failures}")
    stored_core = dict(stored_receipt)
    stored_maximum = stored_core.pop("max_absolute_difference_by_metric", None)
    result_core = dict(result)
    result_maximum = result_core.pop("max_absolute_difference_by_metric")
    _require(
        _normalized(result_core) == _normalized(stored_core),
        "stored raw-FuXi identity receipt differs from independent reconstruction",
    )
    _require(
        isinstance(stored_maximum, dict)
        and set(stored_maximum) == set(result_maximum),
        "stored raw-FuXi maximum-difference fields differ",
    )
    for metric, value in result_maximum.items():
        _require(
            np.isclose(float(stored_maximum[metric]), value, rtol=1.0e-9, atol=2.0e-15),
            f"stored raw-FuXi maximum difference differs for {metric}",
        )
    return result


def _verify_external_record(
    record: Mapping[str, Any],
    ledger: HashLedger,
    *,
    label: str,
    year: int | None = None,
    require_manifest: bool = False,
) -> None:
    store = Path(str(record.get("store", ""))).resolve()
    _require(store.name.endswith(".zarr"), f"{label} store is not Zarr metadata")
    if year is not None:
        _require(record.get("year") == year, f"{label} year differs")
        _require(store.name == f"{year}.zarr", f"{label} store basename differs")
        _require(year in FORECAST_YEARS, f"{label} escapes 2020-2024")
    ledger.verify(store / ".zmetadata", record.get("zmetadata_sha256"), f"{label} .zmetadata")
    if require_manifest:
        source_manifest = Path(str(record.get("manifest", ""))).resolve()
        if year is not None:
            _require(source_manifest.name == f"{year}.json", f"{label} manifest basename differs")
        ledger.verify(
            source_manifest, record.get("manifest_sha256"), f"{label} source manifest"
        )


def _verify_scoring_header(manifest: Mapping[str, Any], root: Path) -> None:
    expected = {
        "experiment": SCORING_EXPERIMENT,
        "status": "complete",
        "scientific_status": "retrospective 2020-2024 benchmark evidence",
        "output_path": str(root),
        "selected_seed": SELECTED_SEED,
        "member_count_each_method": 50,
        "methods": list(METHOD_ORDER),
        "metrics": list(CASE_METRICS),
        "opened_observation_years": list(FORECAST_YEARS),
        "opened_2025_observation": False,
        "sealed_2025_target_opened": False,
    }
    for key, value in expected.items():
        _require(manifest.get(key) == value, f"scoring manifest {key} differs")
    cohort = manifest.get("cohort", {})
    expected_cohort = {
        "forecast_only_climatology_count": 517,
        "scoring_count": 505,
        "valid_midpoint_jjas_count_per_lead": 169,
        "target_day_offsets": list(TARGET_DAY_OFFSETS),
        "maximum_target_label_opened": "2024-12-30",
    }
    _require(cohort == expected_cohort, "scoring cohort manifest differs")
    _require(
        manifest.get("forecast_climatology")
        == "method-specific all-517 fixed-366-day centered-31-day equal-year four-year LOYO",
        "forecast climatology contract differs",
    )
    uncertainty = manifest.get("uncertainty", {})
    expected_pooled = {
        "n_cases_per_lead": 100,
        "n_case_lead_rows": 600,
        "lead_date_intersection_count": 70,
        "lead_date_union_count": 130,
    }
    _require(
        uncertainty.get("cohorts", {}).get("2020_2024_valid_midpoint_jjas") == 169
        and uncertainty.get("cohorts", {}).get("2022_2024_valid_midpoint_jjas") == 100
        and uncertainty.get("cohorts", {}).get(
            "2022_2024_valid_midpoint_jjas_synchronized_pooled_w1_w6"
        )
        == expected_pooled,
        "uncertainty cohorts differ",
    )
    _require(
        uncertainty.get("block_lengths_starts") == list(BLOCK_LENGTHS),
        "bootstrap block lengths differ",
    )
    _require(
        uncertainty.get("replicates") == BOOTSTRAP_REPLICATES,
        "bootstrap replicate count differs",
    )
    _require(uncertainty.get("seed") == BOOTSTRAP_SEED, "bootstrap seed differs")
    _require(
        uncertainty.get("comparisons") == [list(pair) for pair in COMPARISON_PAIRS],
        "bootstrap comparisons differ",
    )


def _verify_parent_selection(
    parent_manifest: Mapping[str, Any],
    scoring_manifest: Mapping[str, Any],
    parent_path: Path,
    ledger: HashLedger,
) -> Mapping[str, Any]:
    _require(parent_manifest.get("experiment") == PARENT_EXPERIMENT, "parent experiment differs")
    _require(parent_manifest.get("status") == "complete", "parent status differs")
    _require(parent_manifest.get("mode") == "full", "parent is not a full run")
    contract = parent_manifest.get("contract", {})
    _require(contract.get("revision") == CONTRACT_REVISION, "parent alignment differs")
    _require(contract.get("target_day_offsets") == list(TARGET_DAY_OFFSETS), "parent offsets differ")
    _require(contract.get("initialization_day_included_in_target") is False, "parent includes day 0")
    _require(contract.get("legacy_v1_zero_offset_checkpoint_loaded") is False, "parent loaded v1")
    _require(contract.get("sealed_2025_target_opened") is False, "parent opened 2025")
    _require(parent_manifest.get("configurations") == ["location_spread"], "parent searched architecture")
    _require(parent_manifest.get("seeds") == [42, 43, 44], "parent seed set differs")

    selection = parent_manifest.get("deployment_selection", {})
    candidates = selection.get("candidates", [])
    _require(
        isinstance(candidates, list)
        and sorted(int(item.get("seed", -1)) for item in candidates) == [42, 43, 44],
        "parent validation candidates differ",
    )
    for item in candidates:
        _require(np.isfinite(float(item.get("best_validation_crps", np.nan))), "candidate CRPS invalid")
        _require(_valid_sha256(item.get("checkpoint_sha256")), "candidate checkpoint hash invalid")
    chosen = min(candidates, key=lambda item: (float(item["best_validation_crps"]), int(item["seed"])))
    _require(int(chosen["seed"]) == SELECTED_SEED, "validation CRPS does not select seed 43")
    expected_selection = {
        "selected_seed": SELECTED_SEED,
        "configuration": "location_spread",
        "checkpoint": chosen["checkpoint"],
        "checkpoint_sha256": chosen["checkpoint_sha256"],
        "best_validation_crps": chosen["best_validation_crps"],
        "criterion": "minimum 2018-2019 validation CRPS; exact ties use lowest seed",
        "test_metrics_used_for_selection": False,
        "parameter_averaging": False,
        "prediction_averaging": False,
        "candidates": candidates,
    }
    _require(selection == expected_selection, "parent deployable selection semantics differ")
    seed_aggregation = parent_manifest.get("seed_aggregation", {})
    _require(seed_aggregation.get("parameter_averaging") is False, "parameters were averaged")
    _require(seed_aggregation.get("prediction_averaging") is False, "predictions were averaged")
    _require(
        seed_aggregation.get("headline_case_metrics")
        == "single validation-selected deployable checkpoint",
        "parent headline is not the selected checkpoint",
    )
    _require(
        scoring_manifest.get("selected_checkpoint_sha256") == chosen["checkpoint_sha256"],
        "scoring checkpoint differs from validation selection",
    )
    checkpoint_path = _resolve_child(parent_path.parent, str(chosen["checkpoint"]))
    _require(
        parent_manifest.get("artifact_sha256", {}).get(chosen["checkpoint"])
        == chosen["checkpoint_sha256"],
        "selected checkpoint is not parent artifact-bound",
    )
    ledger.verify(checkpoint_path, chosen["checkpoint_sha256"], "selected seed-43 checkpoint")
    return selection


def _verify_inputs_and_sources(
    scoring_path: Path,
    scoring_manifest: Mapping[str, Any],
    ledger: HashLedger,
) -> dict[str, Any]:
    """Verify all referenced input manifests, artifacts, and metadata hashes."""

    verified_artifacts = {
        "scoring": _verify_artifacts(scoring_path, scoring_manifest, ledger, label="scoring"),
    }
    input_specs = (
        ("input_inference_manifest", "input_inference_manifest_sha256", "inference"),
        ("input_preflight_manifest", "input_preflight_manifest_sha256", "preflight"),
        ("parent_manifest", "parent_manifest_sha256", "parent"),
    )
    inputs: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path_key, hash_key, label in input_specs:
        path = Path(str(scoring_manifest.get(path_key, ""))).resolve()
        ledger.verify(path, scoring_manifest.get(hash_key), f"{label} manifest")
        payload = _read_json(path)
        inputs[label] = (path, payload)
        verified_artifacts[label] = _verify_artifacts(path, payload, ledger, label=label)

    inference_path, inference_manifest = inputs["inference"]
    preflight_path, preflight_manifest = inputs["preflight"]
    parent_path, parent_manifest = inputs["parent"]
    _require(
        inference_manifest.get("experiment") == INFERENCE_EXPERIMENT
        and inference_manifest.get("status") == "complete"
        and inference_manifest.get("mode") == "full",
        "inference header differs",
    )
    _require(inference_manifest.get("output_path") == str(inference_path.parent), "inference path differs")
    for key in ("verification_truth_opened", "sealed_2025_target_opened"):
        _require(inference_manifest.get(key) is False, f"inference {key} differs")
    inference_selection = inference_manifest.get("inference_selection", {})
    expected_inference_selection = {
        "archive_count": 517,
        "selected_count": 517,
        "scoreable_count": 505,
        "forecast_only_count": 12,
        "selected_dates_sha256": FORECAST_DATES_SHA256,
        "scoreable_dates_sha256": SCORING_DATES_SHA256,
        "verification_truth_opened": False,
        "sealed_2025_target_opened": False,
    }
    for key, value in expected_inference_selection.items():
        _require(inference_selection.get(key) == value, f"inference selection {key} differs")
    storage = inference_manifest.get("storage_contract", {})
    _require(storage.get("operational_member_count") == 50, "inference member count differs")
    _require(storage.get("corrected_members_stored") is False, "corrected members were stored")
    _require(
        storage.get("corrected_members_exactly_reconstructible") is True,
        "corrected members are not reconstructible",
    )

    parent_selection = _verify_parent_selection(
        parent_manifest, scoring_manifest, parent_path, ledger
    )
    parent_receipt = inference_manifest.get("parent_receipt", {})
    expected_parent_receipt = {
        "experiment": PARENT_EXPERIMENT,
        "contract_revision": CONTRACT_REVISION,
        "parent_manifest": str(parent_path),
        "parent_manifest_sha256": scoring_manifest["parent_manifest_sha256"],
        "target_day_offsets": list(TARGET_DAY_OFFSETS),
        "selected_seed": SELECTED_SEED,
        "selected_checkpoint": parent_selection["checkpoint"],
        "selected_checkpoint_sha256": parent_selection["checkpoint_sha256"],
        "candidate_validation_crps": [
            {
                "seed": int(item["seed"]),
                "best_validation_crps": float(item["best_validation_crps"]),
            }
            for item in parent_selection["candidates"]
        ],
        "sealed_2025_target_opened": False,
    }
    _require(
        _normalized(parent_receipt) == _normalized(expected_parent_receipt),
        "inference parent receipt differs",
    )
    _require(
        _normalized(scoring_manifest.get("preflight_parent_receipt"))
        == _normalized(parent_receipt),
        "scoring/preflight parent receipt differs",
    )
    _require(
        preflight_manifest.get("experiment") == PREFLIGHT_EXPERIMENT
        and preflight_manifest.get("status") == "preflight_complete"
        and preflight_manifest.get("sealed_2025_target_opened") is False,
        "preflight header differs",
    )
    _require(preflight_manifest.get("output_path") == str(preflight_path.parent), "preflight path differs")
    _require(
        _normalized(preflight_manifest.get("parent_receipt")) == _normalized(parent_receipt),
        "preflight parent receipt differs",
    )
    _require(
        preflight_manifest.get("cohort_contract")
        == {
            "forecast_only_count": 517,
            "scoring_count": 505,
            "jjas_initialization_count": 170,
            "jjas_valid_midpoint_count_by_lead": {
                str(lead): 169 for lead in LEAD_WEEKS
            },
        },
        "preflight cohort contract differs",
    )

    moment_path = Path(str(scoring_manifest.get("moment_fit", ""))).resolve()
    ledger.verify(moment_path, scoring_manifest.get("moment_fit_sha256"), "moment fit")
    _require(parent_path.parent in moment_path.parents, "moment fit escapes parent run")
    moment_relative = str(moment_path.relative_to(parent_path.parent))
    _require(
        parent_manifest.get("artifact_sha256", {}).get(moment_relative)
        == scoring_manifest.get("moment_fit_sha256"),
        "moment fit is not parent artifact-bound",
    )
    benchmark_path = Path(str(scoring_manifest.get("benchmark_case_metrics", ""))).resolve()
    _require(
        scoring_manifest.get("benchmark_case_metrics_sha256") == BENCHMARK_CASES_SHA256,
        "benchmark case hash contract differs",
    )
    ledger.verify(benchmark_path, BENCHMARK_CASES_SHA256, "benchmark case metrics")

    shards = inference_manifest.get("inference_shards", [])
    _require(isinstance(shards, list) and len(shards) == 5, "inference shards differ")
    shards_by_year = {int(item.get("year", -1)): item for item in shards}
    _require(tuple(sorted(shards_by_year)) == FORECAST_YEARS, "inference shard years differ")
    forecast_sources = scoring_manifest.get("forecast_sources", [])
    _require(
        isinstance(forecast_sources, list) and len(forecast_sources) == 5,
        "scoring forecast-source receipts differ",
    )
    sources_by_year = {int(item.get("year", -1)): item for item in forecast_sources}
    _require(tuple(sorted(sources_by_year)) == FORECAST_YEARS, "forecast source years differ")
    expected_counts = {2020: 105, 2021: 104, 2022: 104, 2023: 104, 2024: 100}
    scoreable_counts = SCORING_YEAR_COUNTS
    for year in FORECAST_YEARS:
        shard = shards_by_year[year]
        _require(shard.get("case_count") == expected_counts[year], f"{year} shard count differs")
        _require(
            shard.get("scoreable_case_count") == scoreable_counts[year],
            f"{year} scoreable shard count differs",
        )
        _require(shard.get("member_count") == 50, f"{year} member count differs")
        _require(shard.get("selected_seed") == SELECTED_SEED, f"{year} shard seed differs")
        _require(
            shard.get("checkpoint_sha256") == parent_selection["checkpoint_sha256"],
            f"{year} shard checkpoint differs",
        )
        _require(shard.get("verification_truth_opened") is False, f"{year} inference opened truth")
        _require(shard.get("sealed_2025_target_opened") is False, f"{year} inference opened 2025")
        raw_source = shard.get("raw_source", {})
        score_source = sources_by_year[year]
        for key in (
            "year",
            "store",
            "manifest",
            "manifest_sha256",
            "zmetadata_sha256",
            "raw_weekly_members_sha256",
        ):
            _require(raw_source.get(key) == score_source.get(key), f"{year} source {key} differs")
        _require(_valid_sha256(raw_source.get("raw_weekly_members_sha256")), f"{year} content hash invalid")
        _verify_external_record(
            raw_source, ledger, label=f"FuXi {year}", year=year, require_manifest=True
        )

    truth_receipt_path = scoring_path.parent / "receipts/truth_access.json"
    truth_receipt = _read_json(truth_receipt_path)
    _require(
        _normalized(truth_receipt) == _normalized(scoring_manifest.get("truth_access")),
        "truth-access artifact and manifest differ",
    )
    expected_truth = {
        "source": "imd",
        "variable": "tp",
        "opened_years": list(FORECAST_YEARS),
        "opened_2025": False,
        "target_day_offsets": list(TARGET_DAY_OFFSETS),
        "scoreable_case_count": 505,
        "maximum_target_label": "2024-12-30",
        "sealed_2025_target_opened": False,
        "climatology_baseline": [1991, 2019],
    }
    for key, value in expected_truth.items():
        _require(truth_receipt.get(key) == value, f"truth receipt {key} differs")
    daily = truth_receipt.get("daily_sources", [])
    _require(isinstance(daily, list) and len(daily) == 5, "IMD daily receipts differ")
    daily_by_year = {int(item.get("year", -1)): item for item in daily}
    _require(tuple(sorted(daily_by_year)) == FORECAST_YEARS, "IMD opened years differ")
    for year in FORECAST_YEARS:
        _verify_external_record(daily_by_year[year], ledger, label=f"IMD {year}", year=year)
    climatology_source = truth_receipt.get("climatology_source", {})
    _verify_external_record(climatology_source, ledger, label="IMD 1991-2019 climatology")
    support_store = Path(str(truth_receipt.get("spatial_support_store", ""))).resolve()
    ledger.verify(
        support_store / ".zmetadata",
        truth_receipt.get("spatial_support_zmetadata_sha256"),
        "spatial-support metadata",
    )
    return {
        "verified_artifacts_by_manifest": verified_artifacts,
        "inference_manifest": inference_manifest,
        "preflight_manifest": preflight_manifest,
        "parent_manifest": parent_manifest,
        "benchmark_path": benchmark_path,
    }


def _story_receipt_from_saved(value: Mapping[str, Any]) -> Mapping[str, Any] | None:
    names = (
        "2022_2024_valid_midpoint_jjas_synchronized_pooled_w1_w6",
        "2022_2024_valid_midpoint_jjas_pooled_w1_w6",
        "pooled_w1_w6_2022_2024_valid_midpoint_jjas",
    )
    for name in names:
        candidate = value.get(name)
        if isinstance(candidate, dict):
            return candidate
    development = value.get("2022_2024_valid_midpoint_jjas")
    if isinstance(development, dict):
        selected = {
            key: item
            for key, item in development.items()
            if str(key).startswith("pooled_w1_w6__")
        }
        if selected:
            return selected
    return None


def validate_output_semantics(
    scoring_path: Path,
    manifest: Mapping[str, Any],
    ledger: HashLedger,
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Reconstruct every result table from the saved 45,450 case rows."""

    root = Path(scoring_path).resolve().parent
    cases = _read_csv(root / "tables/case_metrics.csv", "case metrics")
    scoring_dates = validate_case_metrics(cases)
    table_rows = manifest.get("table_rows", {})
    _require(table_rows.get("case_metrics") == len(cases), "manifest case row count differs")

    reconstructed_allseason = add_raw_comparisons(aggregate_case_metrics(cases))
    actual_allseason = _read_csv(
        root / "tables/allseason_summary.csv", "all-season summary"
    )
    _require_frames_equivalent(
        actual_allseason,
        reconstructed_allseason,
        keys=("method", "region", "lead_week"),
        label="all-season summary",
    )
    _require(
        table_rows.get("allseason_summary") == len(actual_allseason) == 90,
        "all-season summary row count differs",
    )

    jjas_cases = cases[cases.season.eq("JJAS")].copy()
    _require(len(jjas_cases) == JJAS_CASE_ROW_COUNT, "JJAS case-row count differs")
    reconstructed_jjas = add_raw_comparisons(aggregate_case_metrics(jjas_cases))
    actual_jjas = _read_csv(
        root / "tables/jjas_valid_midpoint_summary.csv", "JJAS summary"
    )
    _require_frames_equivalent(
        actual_jjas,
        reconstructed_jjas,
        keys=("method", "region", "lead_week"),
        label="valid-midpoint JJAS summary",
    )
    _require(
        table_rows.get("jjas_valid_midpoint_summary") == len(actual_jjas) == 90,
        "JJAS summary row count differs",
    )

    uncertainty = manifest["uncertainty"]
    replicates = int(uncertainty["replicates"])
    seed = int(uncertainty["seed"])
    block_lengths = tuple(int(value) for value in uncertainty["block_lengths_starts"])
    reconstructed_lead, lead_receipts = reconstruct_lead_intervals(
        cases,
        replicates=replicates,
        seed=seed,
        block_lengths=block_lengths,
    )
    reconstructed_story, story_receipts, story_gate = reconstruct_synchronized_story_gate(
        cases,
        replicates=replicates,
        seed=seed,
        block_lengths=block_lengths,
    )
    actual_intervals = _read_csv(
        root / "tables/paired_block_intervals.csv", "paired intervals"
    )
    pooled_name = "2022_2024_valid_midpoint_jjas_synchronized_pooled_w1_w6"
    if pooled_name in set(actual_intervals.analysis_cohort.astype(str)):
        reconstructed_intervals = pd.concat(
            [reconstructed_lead, reconstructed_story], ignore_index=True, sort=False
        )
        interval_keys = (
            "analysis_cohort",
            "region",
            "lead_week",
            "metric",
            "candidate",
            "baseline",
            "effect",
            "block_length_starts",
        )
        _require_frames_equivalent(
            actual_intervals,
            reconstructed_intervals,
            keys=interval_keys,
            label="paired intervals including synchronized story gate",
            atol=2.0e-9,
        )
    else:
        _require_frames_equivalent(
            actual_intervals,
            reconstructed_lead,
            keys=(
                "analysis_cohort",
                "region",
                "lead_week",
                "metric",
                "candidate",
                "baseline",
                "effect",
                "block_length_starts",
            ),
            label="lead-wise paired intervals",
            atol=2.0e-9,
        )
        story_path = root / "tables/story_gate_intervals.csv"
        _require(story_path.is_file(), "synchronized pooled story-gate intervals are absent")
        actual_story = _read_csv(story_path, "story-gate intervals")
        _require_frames_equivalent(
            actual_story,
            reconstructed_story,
            keys=(
                "analysis_cohort",
                "region",
                "lead_week",
                "metric",
                "candidate",
                "baseline",
                "effect",
                "block_length_starts",
            ),
            label="synchronized pooled story-gate intervals",
            atol=2.0e-9,
        )
    _require(
        table_rows.get("paired_block_intervals") == len(actual_intervals),
        "manifest paired-interval row count differs",
    )

    stored_sampling = _read_json(root / "receipts/bootstrap_sampling.json")
    for cohort_name, expected in lead_receipts.items():
        _require(
            _normalized(stored_sampling.get(cohort_name)) == _normalized(expected),
            f"{cohort_name} bootstrap sampling receipt differs",
        )
    stored_story = _story_receipt_from_saved(stored_sampling)
    _require(stored_story is not None, "synchronized story bootstrap receipt is absent")
    _require(
        _normalized(stored_story) == _normalized(story_receipts),
        "synchronized story bootstrap sampling receipt differs",
    )

    raw_receipt_path = root / "receipts/raw_fuxi_identity.json"
    raw_receipt = _read_json(raw_receipt_path)
    _require(
        _normalized(raw_receipt) == _normalized(manifest.get("raw_fuxi_identity")),
        "raw identity artifact and manifest differ",
    )
    raw_identity = validate_raw_identity(
        cases, Path(inputs["benchmark_path"]), raw_receipt
    )
    _require(
        raw_identity["matched_case_rows"] == RAW_CASE_ROW_COUNT,
        "raw identity row receipt differs",
    )

    truth_receipt = _read_json(root / "receipts/truth_access.json")
    _require(truth_receipt.get("opened_2025") is False, "truth receipt opened 2025")
    _require(
        truth_receipt.get("sealed_2025_target_opened") is False,
        "truth receipt opened sealed 2025 target",
    )
    cohort_receipt = _read_json(root / "receipts/cohort.json")
    _require(cohort_receipt.get("sealed_2025_target_opened") is False, "cohort opened 2025")
    scoring_contract = cohort_receipt.get("scoring", {})
    _require(
        scoring_contract.get("count") == 505
        and scoring_contract.get("dates_sha256") == SCORING_DATES_SHA256
        and scoring_contract.get("maximum_target_label") == "2024-12-30",
        "saved scoring cohort receipt differs",
    )

    story_file = root / "receipts/story_gate.json"
    if story_file.is_file():
        _require(
            _normalized(_read_json(story_file)) == _normalized(story_gate),
            "published story-gate quantities differ from independent reconstruction",
        )
    _require(
        initialization_dates_sha256(scoring_dates) == SCORING_DATES_SHA256,
        "post-semantic scoring date hash differs",
    )
    return {
        "cases": cases,
        "allseason": reconstructed_allseason,
        "jjas": reconstructed_jjas,
        "lead_intervals": reconstructed_lead,
        "story_intervals": reconstructed_story,
        "story_gate": story_gate,
        "raw_identity": raw_identity,
        "bootstrap_sampling_receipts_checked": len(lead_receipts) + 1,
    }


def rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish a directory without replacing an existing audit."""

    source = Path(source).resolve()
    destination = Path(destination).resolve()
    _require(source.parent == destination.parent, "audit publication must share a parent")
    flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(source.parent, flags)
    try:
        metadata = os.stat(source.name, dir_fd=descriptor, follow_symlinks=False)
        _require(stat.S_ISDIR(metadata.st_mode), "audit staging path is not a directory")
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


def _audit_artifact_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in {"audit_receipt.json", "failure.json"}
    }


def audit_scoring_run(scoring_manifest: Path, output: Path) -> Path:
    """Audit one completed score run and publish a separate immutable receipt."""

    scoring_path = Path(scoring_manifest).resolve()
    destination = Path(output).resolve()
    output_root = DEFAULT_OUTPUT_ROOT.resolve()
    _require(output_root in destination.parents, "audit output must remain under resultsv3 bridge")
    _require(destination != scoring_path.parent, "audit cannot overwrite the scoring run")
    _require(scoring_path.parent not in destination.parents, "audit cannot be nested in scoring run")
    if destination.exists():
        raise FileExistsError(f"fresh audit output required: {destination}")
    staging = destination.with_name(f".{destination.name}.incomplete-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    ledger = HashLedger()
    try:
        auditor_source = Path(__file__).resolve()
        auditor_source_digest = sha256_file(auditor_source)
        ledger.verify(auditor_source, auditor_source_digest, "independent auditor source")
        scoring_digest = sha256_file(scoring_path)
        ledger.verify(scoring_path, scoring_digest, "scoring manifest")
        manifest = _read_json(scoring_path)
        _verify_scoring_header(manifest, scoring_path.parent)
        inputs = _verify_inputs_and_sources(scoring_path, manifest, ledger)
        semantic = validate_output_semantics(scoring_path, manifest, ledger, inputs)

        reconstructed = staging / "reconstructed"
        reconstructed.mkdir(parents=True)
        semantic["allseason"].to_csv(reconstructed / "allseason_summary.csv", index=False)
        semantic["jjas"].to_csv(
            reconstructed / "jjas_valid_midpoint_summary.csv", index=False
        )
        semantic["lead_intervals"].to_csv(
            reconstructed / "paired_block_intervals.csv", index=False
        )
        semantic["story_intervals"].to_csv(
            reconstructed / "story_gate_intervals.csv", index=False
        )
        _write_json(staging / "story_gate.json", semantic["story_gate"])
        source_destination = staging / "code/src" / auditor_source.name
        source_destination.parent.mkdir(parents=True)
        shutil.copy2(auditor_source, source_destination)

        ledger.reverify()
        receipt: dict[str, Any] = {
            "experiment": AUDIT_EXPERIMENT,
            "status": "passed",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "output_path": str(destination),
            "scoring_manifest": str(scoring_path),
            "scoring_manifest_sha256": scoring_digest,
            "scoring_experiment": SCORING_EXPERIMENT,
            "verified_file_hash_count": len(ledger.expected),
            "verified_artifacts_by_manifest": inputs["verified_artifacts_by_manifest"],
            "semantic_checks": {
                "full_case_cartesian_rows": CASE_ROW_COUNT,
                "raw_identity_rows": RAW_CASE_ROW_COUNT,
                "raw_case_ids_sha256": semantic["raw_identity"]["case_ids_sha256"],
                "allseason_summary_reconstructed": True,
                "valid_midpoint_jjas_summary_reconstructed": True,
                "paired_intervals_reconstructed_for_both_cohorts": True,
                "synchronized_pooled_story_gate_reconstructed": True,
                "bootstrap_sampling_receipts_checked": semantic[
                    "bootstrap_sampling_receipts_checked"
                ],
                "selected_seed": SELECTED_SEED,
                "parameter_averaging": False,
                "prediction_averaging": False,
                "forecast_or_truth_arrays_opened_by_auditor": False,
                "forecast_source_manifests_and_zmetadata_hash_verified": True,
                "raw_member_content_hashes_cross_bound_without_reopening_arrays": 5,
                "observation_zmetadata_hash_verified_without_opening_arrays": True,
                "sealed_2025_target_opened": False,
            },
            "story_gate": semantic["story_gate"],
            "input_hashes_reverified_after_semantic_audit": True,
        }
        receipt["artifact_sha256"] = _audit_artifact_hashes(staging)
        _write_json(staging / "audit_receipt.json", receipt)
        ledger.reverify()
        destination.parent.mkdir(parents=True, exist_ok=True)
        rename_noreplace(staging, destination)
    except BaseException as error:
        _write_json(
            staging / "failure.json",
            {
                "experiment": AUDIT_EXPERIMENT,
                "status": "failed",
                "failed_utc": datetime.now(timezone.utc).isoformat(),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "forecast_or_truth_arrays_opened_by_auditor": False,
                "sealed_2025_target_opened": False,
            },
        )
        raise
    return destination / "audit_receipt.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scoring-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    receipt = audit_scoring_run(args.scoring_manifest, args.output)
    print(f"PASS: {receipt}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
