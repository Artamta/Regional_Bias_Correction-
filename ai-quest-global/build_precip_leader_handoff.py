#!/usr/bin/env python3
"""Export a verified probabilities-only handoff for the registered team leader.

These files deliberately do not imitate the password-dependent AI-WQ package
template.  They preserve the forecast values and coordinates so that the team
leader can create the official files with the registered identity and package.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import xarray as xr

from precip_contract import GRID_SHAPE, ISSUE_DATE


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_probabilities(path: Path) -> tuple[np.ndarray, str, str]:
    with np.load(path, allow_pickle=False) as archive:
        probability = np.asarray(archive["probabilities"], dtype=np.float32)
        issue = str(archive["issue_date"].item())
        winner = str(archive["winner"].item())
    if issue != ISSUE_DATE:
        raise ValueError(f"forecast issue {issue!r} does not match {ISSUE_DATE}")
    if probability.shape != (2, 5, *GRID_SHAPE):
        raise ValueError(f"forecast probabilities have wrong shape {probability.shape}")
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("forecast probabilities must be finite and in [0, 1]")
    sum_error = float(np.max(np.abs(probability.sum(axis=1, dtype=np.float64) - 1.0)))
    if sum_error > 1.0e-6:
        raise ValueError(f"probability-sum error {sum_error} exceeds 1e-6")
    return probability, issue, winner


def _reopen_exact(path: Path, expected: xr.DataArray) -> None:
    with xr.open_dataarray(path) as reopened:
        if reopened.dims != expected.dims:
            raise ValueError(f"serialized dimensions changed for {path}")
        for name in expected.dims:
            np.testing.assert_array_equal(reopened[name].values, expected[name].values)
        np.testing.assert_array_equal(reopened.values, expected.values)


def run(args: argparse.Namespace) -> None:
    probability, issue, winner = _load_probabilities(args.forecast)
    prediction_manifest = json.loads(args.prediction_manifest.read_text(encoding="utf-8"))
    if prediction_manifest.get("output_sha256") != sha256_file(args.forecast):
        raise ValueError("prediction manifest does not match the forecast archive")
    climatology = prediction_manifest.get("climatology", {})
    if climatology.get("kind") != "official_ai_wq_downloads" and not args.allow_proxy:
        raise PermissionError(
            "forecast does not use official AI-WQ climatology; pass --allow-proxy only "
            "for a clearly labelled leader preview"
        )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    latitude = np.linspace(90.0, -90.0, GRID_SHAPE[0], dtype=np.float32)
    longitude = np.arange(GRID_SHAPE[1], dtype=np.float32) * np.float32(1.5)
    records = []
    for period in (1, 2):
        data = xr.DataArray(
            probability[period - 1],
            name="probability",
            dims=("quintile", "latitude", "longitude"),
            coords={
                "quintile": np.arange(1, 6, dtype=np.int8),
                "latitude": latitude,
                "longitude": longitude,
            },
            attrs={
                "issue_date": issue,
                "period": period,
                "variable": "pr",
                "status": "leader_handoff_probabilities_only_not_official_ai_wq_template",
                "warning": "Do not upload directly; package with registered team/model/password.",
            },
        )
        path = args.output_dir / f"pr_{issue}_p{period}_probabilities_only.nc"
        data.to_netcdf(path)
        _reopen_exact(path, data)
        values = np.asarray(data.values, dtype=np.float64)
        records.append(
            {
                "period": period,
                "filename": path.name,
                "path": str(path),
                "sha256": sha256_file(path),
                "shape": list(values.shape),
                "minimum_probability": float(values.min()),
                "maximum_probability": float(values.max()),
                "probability_sum_max_error": float(
                    np.max(np.abs(values.sum(axis=0) - 1.0))
                ),
            }
        )

    performance = {
        "scope": "validation_only; not an official competition score",
        "test_status": "sealed_not_evaluated",
        "live_forecast_score_status": "unavailable_until_observations_and_official_evaluation",
    }
    comparison_path = getattr(args, "validation_comparison", None)
    if comparison_path is not None:
        with comparison_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        selected = next((row for row in rows if row.get("candidate") == winner), None)
        if selected is None:
            raise ValueError(f"winner {winner!r} is missing from validation comparison")
        performance.update(
            validation_rps_d19_25=float(selected["validation_rps_d19_25"]),
            validation_rps_d26_32=float(selected["validation_rps_d26_32"]),
            validation_rps_pooled=float(selected["validation_rps_pooled"]),
            validation_rpss_d19_25=float(selected["validation_rpss_d19_25"]),
            validation_rpss_d26_32=float(selected["validation_rpss_d26_32"]),
            validation_rpss_pooled=float(selected["validation_rpss_pooled"]),
            comparison_file=str(comparison_path),
            comparison_file_sha256=sha256_file(comparison_path),
        )
    checkpoint_path = getattr(args, "checkpoint", None)
    checkpoint = (
        {"path": str(checkpoint_path), "sha256": sha256_file(checkpoint_path)}
        if checkpoint_path is not None
        else None
    )
    manifest = {
        "schema": "ai-wq-precip-leader-handoff-v1",
        "status": "leader_handoff_probabilities_only_requires_official_packaging",
        "submission_ready": False,
        "not_uploadable_reasons": [
            "NetCDF files are probabilities-only handoff files, not official AI-WQ package templates.",
            "The current forecast uses a public ERA5 2003-2022 proxy climatology, not the official downloads.",
            "Registered team/model identity, password, permission record and remote acceptance are absent.",
        ],
        "issue_date": issue,
        "variable": "pr",
        "winner": winner,
        "performance": performance,
        "checkpoint_provenance_not_an_upload_file": checkpoint,
        "climatology": climatology,
        "source_forecast": str(args.forecast),
        "source_forecast_sha256": sha256_file(args.forecast),
        "source_prediction_manifest": str(args.prediction_manifest),
        "source_prediction_manifest_sha256": sha256_file(args.prediction_manifest),
        "files": records,
        "leader_must": [
            "Replace proxy climatology and rerun prediction if climatology.kind is not official_ai_wq_downloads.",
            "Create the official AI-WQ templates using the registered team, model and password.",
            "Copy these probability values into those official templates and run package validation.",
            "Upload both periods and retain the remote acceptance receipt.",
        ],
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = args.output_dir / "leader_handoff_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    score_line = (
        f"Validation RPS: {performance['validation_rps_pooled']:.6f}; "
        f"RPSS vs uniform: {100.0 * performance['validation_rpss_pooled']:.2f}%"
        if "validation_rps_pooled" in performance
        else "Validation score: not bundled"
    )
    readme = f"""# AI Quest precipitation leader handoff — {issue}

This directory contains the two probability payloads and a checksum/QC manifest.
The files are **not official AI-WQ upload templates** and must not be uploaded directly.

Model: `{winner}`

{score_line} (validation only; not an official competition score).

Climatology kind: `{climatology.get('kind', 'unknown')}`

The registered leader must follow every item in `leader_handoff_manifest.json` under
`leader_must`, using the official AI-WQ package, registered team/model identity,
password, and permission record. The checkpoint is provenance, not an upload file.

Before finalization, obtain the official precipitation climatology files for valid
Mondays `20260831` and `20260907`. The freely available land-sea mask is a different
evaluation input and does not replace these climatology boundaries.
"""
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--forecast", type=Path, required=True)
    result.add_argument("--prediction-manifest", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--checkpoint", type=Path)
    result.add_argument("--validation-comparison", type=Path)
    result.add_argument("--allow-proxy", action="store_true")
    return result


if __name__ == "__main__":
    run(parser().parse_args())
