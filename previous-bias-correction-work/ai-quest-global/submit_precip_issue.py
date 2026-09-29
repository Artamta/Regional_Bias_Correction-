#!/usr/bin/env python3
"""Create package-native Quest files and optionally submit with hard gates."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import re

import numpy as np
import xarray as xr

from precip_contract import GRID_SHAPE, ISSUE_DATE


PACKAGE_VERSION = "3.29"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_component(value: str, name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError(f"{name} contains unsafe filename characters")
    return value


def verify_permission(path: Path) -> dict:
    """Require a real, hashed written-permission record; never synthesize one."""

    record = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "status": "granted",
        "model": "FuXi-S2S",
        "covers_competition_submission": True,
        "covers_derived_probabilities": True,
    }
    for key, expected in required.items():
        if record.get(key) != expected:
            raise PermissionError(f"permission record must set {key}={expected!r}")
    evidence = Path(record.get("written_evidence_path", ""))
    if not evidence.is_file():
        raise PermissionError("written FuXi permission evidence file is missing")
    recorded = str(record.get("written_evidence_sha256", "")).lower()
    if recorded != sha256_file(evidence):
        raise PermissionError("written FuXi permission evidence hash does not match")
    return record


def _validate_package_array(data: xr.DataArray, expected: np.ndarray) -> None:
    if data.shape != (5, *GRID_SHAPE):
        raise ValueError(f"official package template has unexpected shape {data.shape}")
    values = np.asarray(data.values, dtype=np.float64)
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("forecast template contains invalid probabilities")
    if float(np.max(np.abs(values.sum(axis=0) - 1.0))) > 1.0e-6:
        raise ValueError("probability-sum error exceeds 1e-6")
    np.testing.assert_allclose(values, expected, atol=0, rtol=0)
    # The package, rather than this repository, owns coordinate values.  We
    # only reject missing/non-finite/duplicate coordinates after construction.
    if len(data.dims) != 3:
        raise ValueError("package template must have quintile/latitude/longitude dimensions")
    for dimension in data.dims:
        coordinate = np.asarray(data[dimension].values)
        if len(coordinate) != data.sizes[dimension] or len(np.unique(coordinate)) != len(coordinate):
            raise ValueError(f"package coordinate {dimension!r} is invalid")
        if np.issubdtype(coordinate.dtype, np.number) and not np.isfinite(coordinate).all():
            raise ValueError(f"package coordinate {dimension!r} is non-finite")


def _reopen_exact(path: Path, expected: xr.DataArray) -> None:
    with xr.open_dataarray(path) as reopened:
        if reopened.dims != expected.dims:
            raise ValueError(f"serialized dimensions changed for {path}")
        for name in expected.dims:
            np.testing.assert_array_equal(reopened[name].values, expected[name].values)
        np.testing.assert_array_equal(reopened.values, expected.values)


def run(args: argparse.Namespace) -> None:
    if args.issue_date != ISSUE_DATE:
        raise ValueError("this workflow is frozen to issue 20260813")
    permission = verify_permission(args.permission_record)
    try:
        installed = version("AI-WQ-package")
    except Exception:
        installed = version("AI_WQ_package")
    if installed != PACKAGE_VERSION:
        raise RuntimeError(
            f"official AI-WQ-package {PACKAGE_VERSION} is required; found {installed}"
        )
    from AI_WQ_package import forecast_submission

    password = os.environ.get(args.password_env)
    if not password:
        raise RuntimeError(f"set {args.password_env} to the registered submission password")
    team = safe_component(args.team, "team")
    model = safe_component(args.model, "model")
    with np.load(args.forecast, allow_pickle=False) as archive:
        probability = np.asarray(archive["probabilities"], dtype=np.float64)
        if str(archive["issue_date"].item()) != ISSUE_DATE:
            raise ValueError("local forecast issue date does not match 20260813")
    if probability.shape != (2, 5, *GRID_SHAPE):
        raise ValueError("local forecast probabilities must be [2,5,121,240]")
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("local forecast probabilities are invalid")
    if float(np.max(np.abs(probability.sum(axis=1) - 1.0))) > 1.0e-6:
        raise ValueError("local probability-sum error exceeds 1e-6")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    staged: list[tuple[int, xr.DataArray, Path]] = []
    for period in (1, 2):
        # This is intentionally the official constructor.  No dimensions,
        # coordinates, or NetCDF schema are maintained by hand here.
        data = forecast_submission.AI_WQ_create_empty_dataarray(
            "pr", ISSUE_DATE, period, team, model, password
        )
        data.values = probability[period - 1]
        _validate_package_array(data, probability[period - 1])
        path = args.output_dir / f"pr_{ISSUE_DATE}_p{period}_{team}_{model}.nc"
        data.to_netcdf(path)
        _reopen_exact(path, data)
        staged.append((period, data, path))

    manifest = {
        "status": "local_package_native_files_complete",
        "issue_date": ISSUE_DATE,
        "variable": "pr",
        "team": team,
        "model": model,
        "ai_wq_package_version": installed,
        "permission_record": str(args.permission_record),
        "permission_record_sha256": sha256_file(args.permission_record),
        "permission_evidence_sha256": permission["written_evidence_sha256"],
        "source_forecast": str(args.forecast),
        "source_forecast_sha256": sha256_file(args.forecast),
        "files": [
            {"period": period, "path": str(path), "sha256": sha256_file(path)}
            for period, _data, path in staged
        ],
        "probability_sum_max_error": float(
            np.max(np.abs(probability.sum(axis=1) - 1.0))
        ),
        "upload_attempted": False,
        "created_utc": utc_now(),
    }
    manifest_path = args.output_dir / "submission_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    if args.confirm_submit is None:
        print(json.dumps(manifest, indent=2))
        return
    if args.confirm_submit != ISSUE_DATE:
        raise PermissionError(f"upload requires literal --confirm-submit {ISSUE_DATE}")
    responses = []
    for period, data, _path in staged:
        response = forecast_submission.AI_WQ_forecast_submission(
            data, "pr", ISSUE_DATE, period, team, model, password
        )
        # The official function is the authoritative remote acceptance check.
        if response is False:
            raise RuntimeError(f"official package rejected precipitation period {period}")
        responses.append({"period": period, "official_response": repr(response)})
    receipt = {
        "status": "official_submission_function_accepted_both_periods",
        "issue_date": ISSUE_DATE,
        "responses": responses,
        "manifest_sha256_before_upload": sha256_file(manifest_path),
        "submitted_utc": utc_now(),
    }
    (args.output_dir / "remote_submission_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(receipt, indent=2))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--forecast", type=Path, required=True)
    result.add_argument("--permission-record", type=Path, required=True)
    result.add_argument("--team", required=True)
    result.add_argument("--model", required=True)
    result.add_argument("--issue-date", default=ISSUE_DATE)
    result.add_argument("--password-env", default="AI_WQ_PASSWORD")
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--confirm-submit")
    return result


if __name__ == "__main__":
    run(parser().parse_args())
