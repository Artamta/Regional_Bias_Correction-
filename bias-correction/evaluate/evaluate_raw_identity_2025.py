#!/usr/bin/env python3
"""Approval-gated dispatcher for one sealed raw-identity 2025 evaluation.

Only the selection and approval receipt are caller supplied. The preflight
receipt, output, access ledger, failure/status record, and data locators are
all derived from the authenticated frozen selection. Store-capable code is
imported only after the one-time ledger is durably committed.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

import raw_identity_2025_contract as contract


_STAGE_RANK = {
    "approval_validated_access_not_consumed": 0,
    "access_ledger_commit": 1,
    "ledger_committed_runtime_not_imported": 2,
    "runtime_imported_before_store_open": 3,
    "pre_staging_validated": 4,
    "staging_allocated_before_store_open": 5,
    "source_stores_loaded": 6,
    "artifacts_staged": 7,
    "complete_staging_fsynced_before_publication": 8,
    "published_before_parent_fsync": 9,
    "published_after_parent_fsync": 10,
}


def _furthest_stage(first: str, second: str) -> str:
    first_rank = _STAGE_RANK.get(first, -1)
    second_rank = _STAGE_RANK.get(second, -1)
    return second if second_rank >= first_rank else first


def _exec_authenticated_module(name: str, filename: str, content: bytes) -> ModuleType:
    module = ModuleType(name)
    module.__file__ = str(Path(__file__).with_name(filename))
    sys.modules[name] = module
    try:
        exec(compile(content, module.__file__, "exec"), module.__dict__)
    except BaseException:
        if sys.modules.get(name) is module:
            del sys.modules[name]
        raise
    return module


def _load_authenticated_runtime(
    frozen: contract.FrozenSelection,
) -> ModuleType:
    """Execute the exact workflow bytes authenticated before ledger commit."""

    code = frozen.workflow_code_bytes
    assets_module = _exec_authenticated_module(
        "raw_identity_2025_assets",
        "raw_identity_2025_assets.py",
        code["raw_identity_2025_assets.py"],
    )
    try:
        return _exec_authenticated_module(
            "raw_identity_2025_runtime",
            "raw_identity_2025_runtime.py",
            code["raw_identity_2025_runtime.py"],
        )
    except BaseException:
        if sys.modules.get("raw_identity_2025_assets") is assets_module:
            del sys.modules["raw_identity_2025_assets"]
        raise


def _status_payload(
    frozen: contract.FrozenSelection,
    *,
    status: str,
    stage: str,
    ledger_committed: bool,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": contract.ATTEMPT_SCHEMA_VERSION,
        "attempt": 1,
        "status": status,
        "stage": stage,
        "updated_utc": contract.utc_now(),
        "selection_manifest_sha256": frozen.sha256,
        "canonical_execution_paths_sha256": frozen.canonical_paths_sha256,
        "access_ledger_committed": ledger_committed,
        "attempt_consumed": ledger_committed,
        "retry_permitted": False,
    }
    if details:
        payload["details"] = dict(details)
    return payload


def _validate_local_boundaries(
    frozen: contract.FrozenSelection, approval_receipt: Path
) -> None:
    paths = {name: Path(value) for name, value in frozen.canonical_paths.items()}
    if len({str(value) for value in paths.values()}) != len(paths):
        raise contract.SealContractError("canonical execution paths overlap")
    output = paths["final_output"]
    for name, value in paths.items():
        contract.assert_secure_directory(value.parent, f"{name} parent")
        if name != "final_output" and output in value.parents:
            raise contract.SealContractError("control file is nested in final output")
    approval = contract.require_preflight_safe_path(
        approval_receipt, "approval receipt"
    )
    protected = {frozen.path, *paths.values()}
    if (
        approval in protected
        or output in approval.parents
        or approval in output.parents
    ):
        raise contract.SealContractError(
            "approval receipt overlaps sealed execution paths"
        )
    for name in ("final_output", "access_ledger", "failure_record"):
        if contract.entry_exists_no_follow(paths[name]):
            raise FileExistsError(f"fresh canonical {name} required: {paths[name]}")


def dispatch(
    *,
    selection_manifest: Path,
    approval_receipt: Path,
    require_cuda: bool = True,
) -> Path:
    """Consume exactly one approved access attempt and run its sealed runtime."""

    if "raw_identity_2025_runtime" in sys.modules:
        raise contract.SealContractError(
            "store-capable runtime was imported before the durable access ledger"
        )
    frozen = contract.load_frozen_selection(selection_manifest)
    preflight_path = Path(frozen.canonical_paths["preflight_receipt"])
    preflight_payload, preflight_sha256 = contract.validate_preflight_receipt(
        preflight_path, frozen
    )
    _, approval_sha256 = contract.validate_approval_receipt(
        approval_receipt,
        frozen,
        preflight_receipt_sha256=preflight_sha256,
        preflight_created_utc=str(preflight_payload["created_utc"]),
    )
    _validate_local_boundaries(frozen, approval_receipt)
    if require_cuda:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "final allocation has no CUDA GPU; access remains unconsumed"
            )

    ledger_path = Path(frozen.canonical_paths["access_ledger"])
    status_path = Path(frozen.canonical_paths["failure_record"])
    contract.write_json_exclusive(
        status_path,
        _status_payload(
            frozen,
            status="prepared",
            stage="approval_validated_access_not_consumed",
            ledger_committed=False,
        ),
    )
    ledger_committed = False
    stage = "access_ledger_commit"
    try:
        contract.create_access_ledger(
            ledger_path,
            frozen=frozen,
            preflight_receipt_sha256=preflight_sha256,
            approval_receipt_path=approval_receipt,
            approval_receipt_sha256=approval_sha256,
        )
        ledger_committed = True
        stage = "ledger_committed_runtime_not_imported"
        contract.atomic_replace_json(
            status_path,
            _status_payload(
                frozen,
                status="running",
                stage=stage,
                ledger_committed=True,
            ),
        )

        # DO NOT move this import above create_access_ledger.
        runtime = _load_authenticated_runtime(frozen)
        stage = "runtime_imported_before_store_open"
        contract.atomic_replace_json(
            status_path,
            _status_payload(
                frozen,
                status="running",
                stage=stage,
                ledger_committed=True,
            ),
        )
        return runtime.run_evaluation(
            frozen=frozen,
            preflight_receipt_sha256=preflight_sha256,
            approval_receipt_sha256=approval_sha256,
        )
    except BaseException:
        failure_type = sys.exc_info()[0]
        failure_traceback = traceback.format_exc()
        if not ledger_committed:
            try:
                ledger_committed = contract.entry_exists_no_follow(ledger_path)
            except Exception:
                # Unknown ledger state is conservatively treated as consumed.
                ledger_committed = True
        try:
            current_status = contract.read_json(status_path)
            failure_stage = _furthest_stage(
                stage, str(current_status.get("stage", stage))
            )
            current_details = current_status.get("details", {})
        except Exception:
            failure_stage = stage
            current_details = {}
        contract.atomic_replace_json(
            status_path,
            _status_payload(
                frozen,
                status="failed",
                stage=failure_stage,
                ledger_committed=ledger_committed,
                details={
                    "failure_type": (
                        failure_type.__name__ if failure_type else "unknown"
                    ),
                    "traceback": failure_traceback,
                    "attempt_remains_consumed": ledger_committed,
                    "runtime_details": current_details,
                },
            ),
        )
        raise


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--approval-receipt", type=Path, required=True)
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        required=True,
        help="refuse before access consumption if the allocation lacks CUDA",
    )
    args = parser.parse_args(argv)
    result = dispatch(
        selection_manifest=args.selection_manifest,
        approval_receipt=args.approval_receipt,
        require_cuda=args.require_cuda,
    )
    print(f"PASS: published sealed independent-2025 result: {result}", flush=True)


if __name__ == "__main__":
    main()
