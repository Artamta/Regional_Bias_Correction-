#!/usr/bin/env python3
"""Preflight and safety contracts for the corrected India S2S bridge."""

from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import json
import os
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from project_paths import PROJECT_ROOT


EXPERIMENT = "india_s2s_probabilistic_bridge_v1"
PARENT_EXPERIMENT = "fuxi_allseason_ensemble_calibration_v2_aligned"
PARENT_ROOT = (
    PROJECT_ROOT / "resultsv3" / "fuxi_allseason_ensemble_calibration"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "resultsv3" / "india_s2s_probabilistic_bridge"
PLAN_PATH = PROJECT_ROOT / "plan" / "INDIA_S2S_PROBABILISTIC_BRIDGE_20260826.md"
BENCHMARK_CODE_ROOT = PROJECT_ROOT.parent / "studies" / "india_s2s_verification_v2"
BENCHMARK_DATA_CONFIG = BENCHMARK_CODE_ROOT / "config" / "data_sources.json"
BENCHMARK_EVALUATION_CONFIG = BENCHMARK_CODE_ROOT / "config" / "evaluation.json"
FUXI_EXPERIMENT_ID = (
    "model-run/fuxi/fuxi_s2s_strict00z_twice_weekly_2020_2025_ens50"
)

YEARS = (2020, 2021, 2022, 2023, 2024)
TARGET_DAY_OFFSETS = tuple(range(1, 43))
LAST_ALLOWED_TARGET_DATE = np.datetime64("2024-12-31", "D")
EXPECTED_FORECAST_COUNTS = {2020: 105, 2021: 104, 2022: 104, 2023: 104, 2024: 100}
EXPECTED_SCORING_COUNTS = {2020: 105, 2021: 104, 2022: 104, 2023: 104, 2024: 88}
EXPECTED_INITIALIZATION_JJAS_COUNTS = {
    2020: 35,
    2021: 35,
    2022: 35,
    2023: 35,
    2024: 30,
}
EXPECTED_DATE_HASHES = {
    "forecast_only_517": "e95a7f88fdc0ca13b775e1c8708b510310e4a647ff2040e099f65db71baeef56",
    "scoring_505": "41b67539c1d9196dfe5b598e0613a80a794dfea9da6ecbaae40cc102ebd66062",
    "jjas_initialization_170": "506dd4806ae49ddd19ef31edba2255d1ab9445e2558a61a4f6166d2950729ae0",
}


class BridgeContractError(RuntimeError):
    """Raised when a bridge input crosses a frozen scientific boundary."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _date_strings(initializations: np.ndarray) -> list[str]:
    dates = np.asarray(initializations, dtype="datetime64[D]")
    if dates.ndim != 1 or dates.size == 0:
        raise BridgeContractError("initializations must be a non-empty 1-D array")
    if np.unique(dates).size != dates.size or np.any(np.diff(dates) <= np.timedelta64(0, "D")):
        raise BridgeContractError("initializations must be unique and strictly increasing")
    return [np.datetime_as_string(value, unit="D") for value in dates]


def initialization_dates_sha256(initializations: np.ndarray) -> str:
    payload = "".join(f"{value}\n" for value in _date_strings(initializations))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def target_valid_dates(initializations: np.ndarray) -> np.ndarray:
    starts = np.asarray(initializations, dtype="datetime64[D]")
    _date_strings(starts)
    offsets = np.asarray(TARGET_DAY_OFFSETS, dtype="timedelta64[D]").reshape(1, 6, 7)
    return starts[:, None, None] + offsets


def weekly_means_from_daily_offsets() -> np.ndarray:
    """Return the executable target-window oracle [4, 11, ..., 39]."""

    values = np.asarray(TARGET_DAY_OFFSETS, dtype=np.float64).reshape(6, 7)
    return values.mean(axis=1)


def scoring_mask(initializations: np.ndarray) -> np.ndarray:
    dates = target_valid_dates(initializations)
    return dates[:, -1, -1] <= LAST_ALLOWED_TARGET_DATE


def _year_counts(initializations: np.ndarray) -> dict[int, int]:
    years = pd.DatetimeIndex(np.asarray(initializations, dtype="datetime64[D]")).year
    return {year: int(np.count_nonzero(years == year)) for year in YEARS}


def initialization_jjas_mask(initializations: np.ndarray) -> np.ndarray:
    months = pd.DatetimeIndex(np.asarray(initializations, dtype="datetime64[D]")).month
    return np.isin(months, (6, 7, 8, 9))


def valid_midpoint_jjas_mask(
    initializations: np.ndarray, lead_week: int
) -> np.ndarray:
    if lead_week not in range(1, 7):
        raise ValueError("lead_week must be in 1..6")
    starts = pd.DatetimeIndex(np.asarray(initializations, dtype="datetime64[D]"))
    midpoints = starts + pd.to_timedelta((lead_week - 1) * 7 + 3.5, unit="D")
    return np.isin(midpoints.month, (6, 7, 8, 9))


def build_cohort_receipt(
    initializations: np.ndarray, *, strict: bool = True
) -> dict[str, Any]:
    forecast = np.asarray(initializations, dtype="datetime64[D]")
    _date_strings(forecast)
    years = set(pd.DatetimeIndex(forecast).year)
    if not years.issubset(set(YEARS)):
        raise BridgeContractError(f"forecast years escape 2020-2024: {sorted(years)}")
    score = forecast[scoring_mask(forecast)]
    init_jjas = forecast[initialization_jjas_mask(forecast)]
    midpoint = {
        str(lead): forecast[valid_midpoint_jjas_mask(forecast, lead)]
        for lead in range(1, 7)
    }
    receipt = {
        "forecast_only": {
            "name": "forecast_only_climatology",
            "count": int(forecast.size),
            "year_counts": _year_counts(forecast),
            "first": _date_strings(forecast)[0],
            "last": _date_strings(forecast)[-1],
            "dates_sha256": initialization_dates_sha256(forecast),
            "truth_opened": False,
        },
        "scoring": {
            "name": "allseason_no_2025_truth",
            "count": int(score.size),
            "year_counts": _year_counts(score),
            "first": _date_strings(score)[0],
            "last": _date_strings(score)[-1],
            "dates_sha256": initialization_dates_sha256(score),
            "maximum_target_label": np.datetime_as_string(
                target_valid_dates(score)[:, -1, -1].max(), unit="D"
            ),
            "excluded_forecast_only_count": int(forecast.size - score.size),
        },
        "jjas_initialization": {
            "name": "jjas_initialization",
            "count": int(init_jjas.size),
            "year_counts": _year_counts(init_jjas),
            "first": _date_strings(init_jjas)[0],
            "last": _date_strings(init_jjas)[-1],
            "dates_sha256": initialization_dates_sha256(init_jjas),
            "not_valid_midpoint_jjas": True,
        },
        "jjas_valid_midpoint": {
            "name": "jjas_valid_midpoint",
            "case_count_by_lead": {
                lead: int(values.size) for lead, values in midpoint.items()
            },
            "dates_sha256_by_lead": {
                lead: initialization_dates_sha256(values)
                for lead, values in midpoint.items()
            },
        },
        "target_day_offsets": list(TARGET_DAY_OFFSETS),
        "last_allowed_target_date": np.datetime_as_string(
            LAST_ALLOWED_TARGET_DATE, unit="D"
        ),
        "sealed_2025_target_opened": False,
    }
    if strict:
        expected = {
            "forecast_count": 517,
            "forecast_year_counts": EXPECTED_FORECAST_COUNTS,
            "scoring_count": 505,
            "scoring_year_counts": EXPECTED_SCORING_COUNTS,
            "jjas_count": 170,
            "jjas_year_counts": EXPECTED_INITIALIZATION_JJAS_COUNTS,
        }
        actual = {
            "forecast_count": receipt["forecast_only"]["count"],
            "forecast_year_counts": receipt["forecast_only"]["year_counts"],
            "scoring_count": receipt["scoring"]["count"],
            "scoring_year_counts": receipt["scoring"]["year_counts"],
            "jjas_count": receipt["jjas_initialization"]["count"],
            "jjas_year_counts": receipt["jjas_initialization"]["year_counts"],
        }
        if actual != expected:
            raise BridgeContractError(f"canonical cohort counts differ: {actual}")
        observed_hashes = {
            "forecast_only_517": receipt["forecast_only"]["dates_sha256"],
            "scoring_505": receipt["scoring"]["dates_sha256"],
            "jjas_initialization_170": receipt["jjas_initialization"]["dates_sha256"],
        }
        if observed_hashes != EXPECTED_DATE_HASHES:
            raise BridgeContractError(
                f"canonical initialization hashes differ: {observed_hashes}"
            )
        if set(receipt["jjas_valid_midpoint"]["case_count_by_lead"].values()) != {169}:
            raise BridgeContractError("valid-midpoint JJAS must contain 169 cases per lead")
    return receipt


def load_canonical_initializations() -> np.ndarray:
    loader = build_benchmark_loader()
    yearly = []
    for year in YEARS:
        dataset = loader.open_forecast_dataset(
            model="fuxi_s2s",
            variable="tp",
            year=year,
            grid="common_1p5",
            experiment_id=FUXI_EXPERIMENT_ID,
        )
        try:
            values = np.asarray(dataset.init.values, dtype="datetime64[D]")
        finally:
            dataset.close()
        if values.size != EXPECTED_FORECAST_COUNTS[year]:
            raise BridgeContractError(
                f"FuXi {year} initialization count differs: {values.size}"
            )
        yearly.append(values)
    values = np.concatenate(yearly)
    values.sort()
    return values


def archive_dependencies_match_interpreter(archive_root: Path) -> bool:
    """Whether the archive's compiled numcodecs extension matches Python."""

    codec_root = Path(archive_root) / "_deps" / "numcodecs"
    return any(
        (codec_root / f"blosc{suffix}").is_file()
        for suffix in importlib.machinery.EXTENSION_SUFFIXES
    )


def build_benchmark_loader() -> Any:
    """Avoid activating archive wheels built for a different CPython ABI."""

    if str(BENCHMARK_CODE_ROOT) not in sys.path:
        sys.path.insert(0, str(BENCHMARK_CODE_ROOT))
    from s2s_verification.loader import S2SDataLoader

    loader = S2SDataLoader(BENCHMARK_DATA_CONFIG)
    if archive_dependencies_match_interpreter(loader.catalog.archive_root):
        return loader

    class EnvironmentZarrLoader(S2SDataLoader):
        def _open_store(self, store: str | Path) -> Any:
            import xarray as xr

            return xr.open_zarr(str(store), consolidated=True, chunks=None)

    return EnvironmentZarrLoader(BENCHMARK_DATA_CONFIG)


def _resolve_child(root: Path, relative: str) -> Path:
    if Path(relative).is_absolute():
        raise BridgeContractError(f"artifact path must be relative: {relative}")
    candidate = (root / relative).resolve()
    if root.resolve() not in candidate.parents:
        raise BridgeContractError(f"artifact path escapes parent run: {relative}")
    return candidate


def validate_parent_manifest(path: Path) -> dict[str, Any]:
    manifest_path = Path(path).resolve()
    run_root = manifest_path.parent
    if PARENT_ROOT.resolve() not in run_root.parents:
        raise BridgeContractError("parent manifest is not under the aligned resultsv3 root")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "experiment": PARENT_EXPERIMENT,
        "status": "complete",
        "mode": "full",
        "smoke": False,
        "configurations": ["location_spread"],
        "seeds": [42, 43, 44],
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise BridgeContractError(
                f"parent {key} differs: {manifest.get(key)!r} != {expected!r}"
            )
    contract = manifest.get("contract", {})
    if contract.get("target_day_offsets") != list(TARGET_DAY_OFFSETS):
        raise BridgeContractError("parent target offsets are not 1..42")
    if contract.get("initialization_day_included_in_target") is not False:
        raise BridgeContractError("parent includes initialization day in target")
    if contract.get("sealed_2025_target_opened") is not False:
        raise BridgeContractError("parent opened the sealed 2025 target")
    if contract.get("legacy_v1_zero_offset_checkpoint_loaded") is not False:
        raise BridgeContractError("parent loaded a legacy v1 checkpoint")

    selection = manifest.get("deployment_selection", {})
    candidates = selection.get("candidates", [])
    if [item.get("seed") for item in candidates] != [42, 43, 44]:
        raise BridgeContractError("parent deployment candidates are incomplete")
    expected_selected = min(
        candidates,
        key=lambda item: (float(item["best_validation_crps"]), int(item["seed"])),
    )
    if selection.get("selected_seed") != expected_selected["seed"]:
        raise BridgeContractError("parent deployable seed is not validation-selected")
    if selection.get("test_metrics_used_for_selection") is not False:
        raise BridgeContractError("parent selection consulted retrospective metrics")
    if selection.get("prediction_averaging") is not False or selection.get(
        "parameter_averaging"
    ) is not False:
        raise BridgeContractError("parent averages predictions or parameters across seeds")

    artifact_hashes = manifest.get("artifact_sha256", {})
    for candidate_record in candidates:
        checkpoint = _resolve_child(run_root, candidate_record["checkpoint"])
        if not checkpoint.is_file():
            raise BridgeContractError(f"parent checkpoint is missing: {checkpoint}")
        actual = sha256_file(checkpoint)
        if actual != candidate_record["checkpoint_sha256"]:
            raise BridgeContractError(f"checkpoint hash differs: {checkpoint}")
        if artifact_hashes.get(candidate_record["checkpoint"]) != actual:
            raise BridgeContractError("checkpoint is not bound by parent artifact hashes")

    selection_path = run_root / "models" / "deployable_selection.json"
    if json.loads(selection_path.read_text(encoding="utf-8")) != selection:
        raise BridgeContractError("selection artifact differs from parent manifest")
    selection_relative = str(selection_path.relative_to(run_root))
    if artifact_hashes.get(selection_relative) != sha256_file(selection_path):
        raise BridgeContractError("selection artifact hash differs")

    for relative, expected_hash in manifest.get("source_snapshot_sha256", {}).items():
        source = _resolve_child(run_root, relative)
        if not source.is_file() or sha256_file(source) != expected_hash:
            raise BridgeContractError(f"parent source snapshot differs: {relative}")
        if artifact_hashes.get(relative) != expected_hash:
            raise BridgeContractError(f"parent source is not artifact-bound: {relative}")
    return {
        "parent_manifest": str(manifest_path),
        "parent_manifest_sha256": sha256_file(manifest_path),
        "experiment": manifest["experiment"],
        "contract_revision": contract.get("revision"),
        "target_day_offsets": contract["target_day_offsets"],
        "selected_seed": selection["selected_seed"],
        "selected_checkpoint": selection["checkpoint"],
        "selected_checkpoint_sha256": selection["checkpoint_sha256"],
        "candidate_validation_crps": [
            {
                "seed": item["seed"],
                "best_validation_crps": item["best_validation_crps"],
            }
            for item in candidates
        ],
        "sealed_2025_target_opened": False,
    }


def _artifact_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in {"manifest.json", "failure.json"}
    }


def run_preflight(parent_manifest: Path, output: Path) -> None:
    requested = Path(output).resolve()
    if DEFAULT_OUTPUT_ROOT.resolve() not in requested.parents:
        raise BridgeContractError("bridge output must be under the resultsv3 bridge root")
    if requested.exists():
        raise FileExistsError(f"fresh output path required: {requested}")
    staging = requested.parent / f".{requested.name}.incomplete-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"staging path already exists: {staging}")
    staging.mkdir(parents=True)
    try:
        parent = validate_parent_manifest(parent_manifest)
        cohorts = build_cohort_receipt(load_canonical_initializations(), strict=True)
        write_json(staging / "parent_receipt.json", parent)
        write_json(staging / "cohort_receipt.json", cohorts)
        source_destination = staging / "code" / "src" / Path(__file__).name
        source_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(__file__).resolve(), source_destination)
        plan_destination = staging / "code" / "plan" / PLAN_PATH.name
        plan_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PLAN_PATH, plan_destination)
        manifest = {
            "experiment": EXPERIMENT,
            "status": "preflight_complete",
            "scientific_status": "contract preflight only; no forecast inference or target scoring",
            "created_utc": utc_now(),
            "output_path": str(requested),
            "parent_receipt": parent,
            "cohort_contract": {
                "forecast_only_count": cohorts["forecast_only"]["count"],
                "scoring_count": cohorts["scoring"]["count"],
                "jjas_initialization_count": cohorts["jjas_initialization"]["count"],
                "jjas_valid_midpoint_count_by_lead": cohorts["jjas_valid_midpoint"][
                    "case_count_by_lead"
                ],
            },
            "sealed_2025_target_opened": False,
            "artifact_sha256": _artifact_hashes(staging),
        }
        write_json(staging / "manifest.json", manifest)
        requested.parent.mkdir(parents=True, exist_ok=True)
        os.rename(staging, requested)
    except Exception as error:
        write_json(
            staging / "failure.json",
            {
                "experiment": EXPERIMENT,
                "status": "failed",
                "failed_utc": utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "sealed_2025_target_opened": False,
            },
        )
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preflight the corrected FuXi-to-India-S2S bridge contract."
    )
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_preflight(args.parent_manifest, args.output)
    print(f"PASS: bridge preflight published at {Path(args.output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
