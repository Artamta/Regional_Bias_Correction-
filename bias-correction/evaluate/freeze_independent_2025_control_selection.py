#!/usr/bin/env python3
"""Freeze a validation-only TP/T2M control before independent 2025 access.

This utility reads only the completed 2002--2019 validation experiment.  It
does not import, inspect, or open any 2025 forecast/observation store.  The
explicit attestation flag is required so the resulting manifest records that
the freeze occurred before independent-test access.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

import evaluate_independent_2025_control as evaluation


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def freeze_selection(
    validation_run: Path,
    output: Path,
    *,
    configuration: str = "physical_control",
    attest_no_2025_access: bool,
) -> Mapping[str, Any]:
    """Hash-lock one three-seed control using validation artifacts only."""

    if not attest_no_2025_access:
        raise ValueError(
            "refusing to freeze without an explicit attestation that 2025 has "
            "not been accessed for this selection"
        )
    validation_run = Path(validation_run).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"selection output already exists: {output}")
    manifest_path = validation_run / "manifest.json"
    normalization_path = validation_run / "normalization.json"
    anchor_path = validation_run / "models" / "log_bias_anchor.npz"
    ranking_path = validation_run / "metrics" / "ranked_configurations.csv"
    for path in (manifest_path, normalization_path, anchor_path, ranking_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"required validation artifact is missing: {path}")
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "complete" or manifest.get("smoke") is not False:
        raise ValueError("only a completed non-smoke validation run can be frozen")
    if manifest.get("test_predictions_created") is not False:
        raise ValueError("validation run does not attest absence of test predictions")
    if tuple(manifest.get("train_years", ())) != tuple(range(2002, 2018)):
        raise ValueError("training years are not exactly 2002-2017")
    if tuple(manifest.get("validation_years", ())) != (2018, 2019):
        raise ValueError("validation years are not exactly 2018-2019")
    candidate = evaluation._candidate_from_manifest(manifest, configuration)
    policy = {
        "selected_configuration": configuration,
        "independent_input_contract": evaluation.EXPECTED_CONTROL_CONTRACT,
    }
    evaluation._validate_control_candidate(candidate, policy)
    normalization = _read_json(normalization_path)
    control_normalization = evaluation.project_control_normalization(normalization)
    code_hashes = evaluation.live_code_hashes()
    data_contract = evaluation.build_control_data_contract(
        candidate,
        control_normalization,
        manifest,
        normalization_sha256=evaluation.sha256_file(normalization_path),
        anchor_sha256=evaluation.sha256_file(anchor_path),
    )

    ranking = pd.read_csv(ranking_path)
    rows = ranking.loc[ranking.configuration.astype(str).eq(configuration)]
    if len(rows) != 1:
        raise ValueError(
            f"validation ranking has {len(rows)} rows for {configuration!r}"
        )
    ranking_snapshot = {
        str(key): (None if pd.isna(value) else value.item() if hasattr(value, "item") else value)
        for key, value in rows.iloc[0].items()
    }
    checkpoints = []
    for seed in sorted(int(value) for value in manifest.get("seeds", ())):
        record_path = (
            validation_run
            / "models"
            / configuration
            / f"seed_{seed}"
            / "run_record.json"
        )
        record = _read_json(record_path)
        checkpoint = validation_run / str(record["checkpoint"])
        if (
            record.get("status") != "complete"
            or int(record.get("seed", -1)) != seed
            or str(record.get("candidate", {}).get("name")) != configuration
            or not checkpoint.is_file()
        ):
            raise ValueError(f"incomplete control seed record: {record_path}")
        checkpoint_hash = evaluation.sha256_file(checkpoint)
        if checkpoint_hash != record.get("checkpoint_sha256"):
            raise ValueError(f"checkpoint hash differs from run record: {checkpoint}")
        checkpoints.append(
            {
                "seed": seed,
                "path": str(checkpoint.relative_to(validation_run)),
                "sha256": checkpoint_hash,
                "best_epoch_zero_based": int(record["best_epoch_zero_based"]),
                "best_validation_loss": float(record["best_validation_loss"]),
            }
        )
    if len(checkpoints) != 3:
        raise ValueError("independent control requires exactly three frozen seeds")

    selection: dict[str, Any] = {
        "status": "frozen",
        "scientific_role": "predeclared TP/T2M-compatible independent-test control",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_before_2025_access": True,
        "selection_data_end_year": 2019,
        "selected_configuration": configuration,
        "independent_input_contract": evaluation.EXPECTED_CONTROL_CONTRACT,
        "validation_run": str(validation_run),
        "validation_run_manifest_sha256": evaluation.sha256_file(manifest_path),
        "normalization_sha256": evaluation.sha256_file(normalization_path),
        "log_bias_anchor_sha256": evaluation.sha256_file(anchor_path),
        "ranking_sha256": evaluation.sha256_file(ranking_path),
        "code_sha256": code_hashes,
        "data_contract": data_contract,
        "data_contract_sha256": evaluation.canonical_json_sha256(data_contract),
        "validation_ranking_snapshot": ranking_snapshot,
        "checkpoints": checkpoints,
        "policy": {
            "selection_data": "blocked 2018-2019 validation only",
            "test_year": 2025,
            "test_access_at_freeze": "none, explicitly attested by caller",
            "post_test_actions_forbidden": [
                "training",
                "checkpoint selection",
                "normalization fitting",
                "anchor fitting",
                "alpha tuning",
                "threshold tuning",
                "architecture reselection",
            ],
            "physical_input_policy": (
                "physical/member candidates are incompatible with the saved "
                "strict00z 2025 TP/T2M archive and must fail before data access"
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.partial-", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(selection, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # Re-run the same strict preflight used by the evaluator. This still
        # reads only the validation run and proves the complete payload is
        # executable before it becomes visible at the requested final path.
        evaluation.validate_frozen_control(validation_run, temporary)
        try:
            os.link(temporary, output)
        except FileExistsError as exc:
            raise FileExistsError(
                f"selection output already exists: {output}"
            ) from exc
    finally:
        temporary.unlink(missing_ok=True)
    return selection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--configuration", default="physical_control")
    parser.add_argument(
        "--attest-no-2025-access",
        required=True,
        action="store_true",
        help="attest that 2025 was not inspected or used to choose this control",
    )
    args = parser.parse_args()
    freeze_selection(
        args.validation_run,
        args.output,
        configuration=args.configuration,
        attest_no_2025_access=args.attest_no_2025_access,
    )
    print(f"PASS: frozen independent-test selection: {args.output.resolve()}")


if __name__ == "__main__":
    main()
