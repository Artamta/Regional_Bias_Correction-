#!/usr/bin/env python3
"""Storage-incapable synthetic-CUDA preflight for the sealed 2025 test.

The command accepts no forecast, observation, output-result, or ledger path.
It validates only home-directory frozen artifacts, strictly loads all three
checkpoints, runs a synthetic 27x27 CUDA inference, and writes one exclusive
proof receipt for later human approval.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

import numpy as np

import raw_identity_2025_contract as contract


def _load_frozen_models(path: Path) -> ModuleType:
    content = contract.capture_authenticated_bytes(
        path, contract.EXPECTED_MODEL_SOURCE_SHA256, "sealed model source"
    )
    name = f"_raw_identity_2025_frozen_models_{contract.sha256_bytes(content)[:16]}"
    module = ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(content, str(path), "exec"), module.__dict__)
    return module


def synthetic_cuda_proof(
    frozen: contract.FrozenSelection, *, torch_module: Any | None = None
) -> Mapping[str, Any]:
    torch = torch_module
    if torch is None:
        import torch as imported_torch

        torch = imported_torch
    if not torch.cuda.is_available():
        raise RuntimeError("synthetic preflight requires a visible CUDA GPU")
    models = _load_frozen_models(frozen.model_source_path)
    # Deterministic values exercise every channel without reading a data store.
    values = np.linspace(
        -1.0,
        1.0,
        num=contract.EXPECTED_LEADS * 29 * 27 * 27,
        dtype=np.float32,
    ).reshape(1, contract.EXPECTED_LEADS, 29, 27, 27)
    batch = torch.from_numpy(values).to("cuda")
    outputs: list[np.ndarray] = []
    parameter_counts: list[int] = []
    checkpoint_metadata: list[Mapping[str, Any]] = []
    with torch.no_grad():
        for seed, checkpoint_path, expected_hash in frozen.checkpoints:
            checkpoint_content = contract.read_bytes_no_follow(checkpoint_path)
            if contract.sha256_bytes(checkpoint_content) != expected_hash:
                raise contract.SealContractError(
                    f"checkpoint seed {seed} changed during preflight"
                )
            model = models.FixedClimatologyAllLeadUNet(
                input_channels=29,
                base_channels=16,
                dropout=0.30,
            )
            count = int(sum(parameter.numel() for parameter in model.parameters()))
            if count != 144_689:
                raise contract.SealContractError(
                    f"frozen parameter count changed: {count}"
                )
            checkpoint = torch.load(
                io.BytesIO(checkpoint_content), map_location="cpu", weights_only=False
            )
            if not isinstance(checkpoint, Mapping):
                raise contract.SealContractError(
                    f"checkpoint root is not a mapping: {checkpoint_path}"
                )
            if "model_state_dict" not in checkpoint:
                raise contract.SealContractError(
                    f"checkpoint lacks model_state_dict: seed {seed}"
                )
            if int(checkpoint.get("seed", seed)) != seed:
                raise contract.SealContractError(
                    f"checkpoint metadata seed changed: expected {seed}"
                )
            best_validation_loss = float(
                checkpoint.get("best_validation_loss", float("nan"))
            )
            if not np.isfinite(best_validation_loss):
                raise contract.SealContractError(
                    f"checkpoint validation loss is non-finite: seed {seed}"
                )
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            model.to("cuda").eval()
            with torch.cuda.amp.autocast(enabled=True):
                residual = model(batch).float().cpu().numpy().astype(np.float32)
            if residual.shape != (1, 6, 27, 27) or not np.isfinite(residual).all():
                raise contract.SealContractError(
                    f"invalid synthetic residual for seed {seed}: {residual.shape}"
                )
            outputs.append(residual)
            parameter_counts.append(count)
            checkpoint_metadata.append(
                {
                    "seed": seed,
                    "sha256": expected_hash,
                    "best_epoch": int(checkpoint.get("best_epoch", -1)),
                    "best_validation_loss": best_validation_loss,
                }
            )
            del model
    torch.cuda.synchronize()
    ensemble = np.mean(np.stack(outputs), axis=0, dtype=np.float64).astype(np.float32)
    if not np.isfinite(ensemble).all():
        raise contract.SealContractError("synthetic ensemble is non-finite")
    properties = torch.cuda.get_device_properties(0)
    return {
        "synthetic_cuda_inference": True,
        "synthetic_input_shape": list(values.shape),
        "synthetic_residual_shape": list(ensemble.shape),
        "synthetic_ensemble_residual_sha256": contract.array_sha256(ensemble, "<f4"),
        "checkpoint_seeds_loaded": list(contract.EXPECTED_SEEDS),
        "checkpoint_metadata": checkpoint_metadata,
        "parameter_counts": parameter_counts,
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda),
        "gpu_name": str(torch.cuda.get_device_name(0)),
        "gpu_total_memory_bytes": int(properties.total_memory),
        "storage_paths_opened": [],
        "access_ledger_created": False,
        "result_created": False,
    }


def run_preflight(selection_manifest: Path) -> Path:
    frozen = contract.load_frozen_selection(selection_manifest)
    proof_receipt = Path(frozen.canonical_paths["preflight_receipt"])
    for key in ("final_output", "access_ledger", "failure_record"):
        if contract.entry_exists_no_follow(frozen.canonical_paths[key]):
            raise FileExistsError(
                f"preflight requires an unused frozen execution: {key}"
            )
    proof = synthetic_cuda_proof(frozen)
    payload: dict[str, Any] = {
        "schema_version": contract.PREFLIGHT_SCHEMA_VERSION,
        "status": "passed",
        "created_utc": contract.utc_now(),
        "test_year": contract.TEST_YEAR,
        "selection_manifest": str(frozen.path),
        "selection_manifest_sha256": frozen.sha256,
        "method_hierarchy": list(contract.METHOD_HIERARCHY),
        "code_sha256": contract.live_code_hashes(),
        "proof": proof,
    }
    payload["proof_payload_sha256"] = contract.canonical_json_sha256(payload)
    return contract.write_json_exclusive(proof_receipt, payload)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = run_preflight(args.selection_manifest)
    print(
        "PASS: storage-incapable synthetic-CUDA preflight; no /storage path "
        "opened, no access ledger created, no evaluation result created; "
        f"proof={receipt} sha256={contract.sha256_file(receipt)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
