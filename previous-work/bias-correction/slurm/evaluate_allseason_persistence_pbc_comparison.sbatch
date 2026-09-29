#!/usr/bin/env bash
#SBATCH --job-name=fuxi-pbc-cmp
#SBATCH --partition=gpu_prio
# cn3 is MIG; cn2, cn4, cn15, cn16, and cn17 are unreliable for S2S work.
#SBATCH --exclude=cn2,cn3,cn4,cn15,cn16,cn17
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/logs/%x-%j.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/logs/%x-%j.err

set -euo pipefail
umask 027

usage() {
    cat >&2 <<'EOF'
usage:
  sbatch slurm/evaluate_allseason_persistence_pbc_comparison.sbatch \
    /absolute/path/completed-persistence-full/manifest.json \
    /absolute/path/completed-joint-mask-control-full/manifest.json \
    /absolute/path/accepted-pbc-v2-full/manifest.json \
    /absolute/path/fresh-comparison-output

The persistence input must be the successful three-seed full run with its
Slurm gate receipt and receipted smoke parent. The joint-selection input must
be the successful externally receipted full mask-control addendum that binds
that exact persistence run and accepted PBC V2. The PBC input must be the exact
accepted V2 full result pinned by manifest, receipt, observation, lag, and fit
artifact hashes. The output must not exist. A v1-only descriptive run is never
launched.
EOF
}

if [[ $# -ne 4 ]]; then
    usage
    exit 2
fi
if [[ $1 != /* || $2 != /* || $3 != /* || $4 != /* ]]; then
    echo "all three manifests and output must be explicit absolute paths" >&2
    exit 2
fi
if [[ ! -f $1 || $(basename "$1") != manifest.json ]]; then
    echo "completed persistence full manifest is unavailable: $1" >&2
    exit 2
fi
if [[ ! -f $2 || $(basename "$2") != manifest.json ]]; then
    echo "completed joint mask-control full manifest is unavailable: $2" >&2
    exit 2
fi
if [[ ! -f $3 || $(basename "$3") != manifest.json ]]; then
    echo "accepted PBC V2 manifest is unavailable: $3" >&2
    exit 2
fi

WORK_ROOT=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction
PYTHON_BIN=/home/raj.ayush/.conda/envs/s2s_core/bin/python
PERSISTENCE_MANIFEST=$(readlink -f "$1")
JOINT_SELECTION_MANIFEST=$(readlink -f "$2")
PBC_MANIFEST=$(readlink -f "$3")
OUTPUT=$(readlink -m "$4")
OUTPUT_PARENT=$(dirname "${OUTPUT}")
OUTPUT_NAME=$(basename "${OUTPUT}")
EXPECTED_PLAN_SHA256=2a1c6d16957cbceedc0bcf62cb2a7518ca1e5e161f8df36aab0f600e28302699
EXPECTED_ADDENDUM_PLAN_SHA256=c429ee96bfd9b9fc1ddc8fa88da653eeb4923ea037d516d8d6a42071dd3c3ba0
EXPECTED_PBC_MANIFEST_SHA256=c8c8bbb840d4624df9b2f514d26e8dceb72586ab7de32ff7847a91034812e5f6
EXPECTED_PBC_RECEIPT_SHA256=1c0ecf41e3344fb1c7569e50bb289974b9f5d1c149ad62a14fddd8c76d51f1d2
EXPECTED_PBC_OBSERVATION_SHA256=3123b32075f1c4a211d294ae07a91a70c16128154e17441e6653ccbf33f7ae49
EXPECTED_PBC_LAG_SHA256=9555fc672628a90c39708550d85153e8787ff1a3a1abb4ef99625e347ab2c4ba
EXPECTED_PBC_FIT_SHA256=0dea66a543573d64943ec3447682a7698b9bbeb60239b6498b81af77a756fc36

if [[ ! -x ${PYTHON_BIN} ]]; then
    echo "required s2s_core Python is unavailable: ${PYTHON_BIN}" >&2
    exit 2
fi
if [[ -e ${OUTPUT} ]]; then
    echo "fresh output path required; refusing to overwrite: ${OUTPUT}" >&2
    exit 2
fi
if ! command -v srun >/dev/null 2>&1; then
    echo "srun is unavailable" >&2
    exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is unavailable" >&2
    exit 2
fi
if [[ ! ${SLURM_JOB_ID:-} =~ ^[0-9]+([_][0-9]+)?$ ]]; then
    echo "invalid or missing Slurm job id: ${SLURM_JOB_ID:-unset}" >&2
    exit 2
fi

REQUIRED_SOURCES=(
    src/fuxi_allseason_persistence_pbc_comparison.py
    tests/test_allseason_persistence_pbc_comparison.py
    src/fuxi_allseason_persistence_augmented.py
    src/fuxi_persistence_context.py
    src/fuxi_allseason_persistence_mask_control.py
    src/fuxi_persistence_mask_control.py
    src/fuxi_allseason_categorical_comparison.py
    src/fuxi_allseason_pbc_baseline.py
    src/fuxi_pbc_core.py
    src/fuxi_allseason_ensemble_calibration.py
    src/fuxi_ensemble_calibration_core.py
    src/fuxi_allseason_member_cache.py
    src/project_paths.py
    plan/PERSISTENCE_AUGMENTED_NEURAL_20260823.md
    plan/PERSISTENCE_MASK_CONTROL_ADDENDUM_20260823.md
    slurm/evaluate_allseason_persistence_pbc_comparison.sbatch
)
for relative in "${REQUIRED_SOURCES[@]}"; do
    if [[ ! -f ${WORK_ROOT}/${relative} ]]; then
        echo "required comparison source/plan is unavailable: ${relative}" >&2
        exit 2
    fi
done

ACTUAL_PLAN_SHA256=$(
    "${PYTHON_BIN}" - "${WORK_ROOT}/plan/PERSISTENCE_AUGMENTED_NEURAL_20260823.md" <<'PY'
import hashlib
import sys
from pathlib import Path

print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())
PY
)
if [[ ${ACTUAL_PLAN_SHA256} != "${EXPECTED_PLAN_SHA256}" ]]; then
    echo "frozen pre-score plan SHA-256 differs: ${ACTUAL_PLAN_SHA256}" >&2
    exit 2
fi
ACTUAL_ADDENDUM_PLAN_SHA256=$(
    "${PYTHON_BIN}" - "${WORK_ROOT}/plan/PERSISTENCE_MASK_CONTROL_ADDENDUM_20260823.md" <<'PY'
import hashlib
import sys
from pathlib import Path

print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())
PY
)
if [[ ${ACTUAL_ADDENDUM_PLAN_SHA256} != "${EXPECTED_ADDENDUM_PLAN_SHA256}" ]]; then
    echo "frozen mask-control addendum SHA-256 differs: ${ACTUAL_ADDENDUM_PLAN_SHA256}" >&2
    exit 2
fi

# Reject unsafe nesting before creating even the output parent or lock. This
# keeps invalid requests read-only with respect to all immutable input runs.
"${PYTHON_BIN}" - "${OUTPUT}" "${PERSISTENCE_MANIFEST}" \
    "${JOINT_SELECTION_MANIFEST}" "${PBC_MANIFEST}" <<'PY'
import sys
from pathlib import Path

output = Path(sys.argv[1]).resolve()
for manifest_arg in sys.argv[2:]:
    input_root = Path(manifest_arg).resolve().parent
    try:
        output.relative_to(input_root)
    except ValueError:
        continue
    raise SystemExit(f"output may not equal or nest in immutable input: {input_root}")
PY

mkdir -p "${WORK_ROOT}/logs" "${OUTPUT_PARENT}"
if [[ -e ${OUTPUT} ]]; then
    echo "output appeared during preflight; refusing to overwrite: ${OUTPUT}" >&2
    exit 2
fi

JOB_TMP=$(mktemp -d "/tmp/fuxi-pbc-cmp-${SLURM_JOB_ID}.XXXXXX")
OUTPUT_LOCK=${OUTPUT_PARENT}/.${OUTPUT_NAME}.pbc-comparison-launch.lock
LOCK_HELD=false
cleanup() {
    if [[ -d ${JOB_TMP} && ${JOB_TMP} == /tmp/fuxi-pbc-cmp-* ]]; then
        rm -rf -- "${JOB_TMP}"
    fi
    if [[ ${LOCK_HELD} == true ]]; then
        rmdir -- "${OUTPUT_LOCK}" 2>/dev/null || true
    fi
}
trap cleanup EXIT
if ! mkdir -- "${OUTPUT_LOCK}"; then
    echo "another launch owns this output path: ${OUTPUT}" >&2
    exit 2
fi
LOCK_HELD=true

mkdir -p "${JOB_TMP}/mpl" "${JOB_TMP}/xdg" "${JOB_TMP}/cuda"
export PYTHONPATH=${WORK_ROOT}/src:${WORK_ROOT}/evaluate:${WORK_ROOT}/hpc_compat
export MPLCONFIGDIR=${JOB_TMP}/mpl
export XDG_CACHE_HOME=${JOB_TMP}/xdg
export CUDA_CACHE_PATH=${JOB_TMP}/cuda
export TMPDIR=${JOB_TMP}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

cd "${WORK_ROOT}"
PREFLIGHT_RECEIPT=$(
    "${PYTHON_BIN}" - "${PERSISTENCE_MANIFEST}" \
        "${JOINT_SELECTION_MANIFEST}" "${PBC_MANIFEST}" \
        "${OUTPUT}" "${WORK_ROOT}" "${EXPECTED_PLAN_SHA256}" \
        "${EXPECTED_ADDENDUM_PLAN_SHA256}" \
        "${EXPECTED_PBC_MANIFEST_SHA256}" "${EXPECTED_PBC_RECEIPT_SHA256}" \
        "${EXPECTED_PBC_OBSERVATION_SHA256}" "${EXPECTED_PBC_LAG_SHA256}" \
        "${EXPECTED_PBC_FIT_SHA256}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import fuxi_allseason_persistence_pbc_comparison as comparison

persistence_path = Path(sys.argv[1]).resolve()
joint_path = Path(sys.argv[2]).resolve()
pbc_path = Path(sys.argv[3]).resolve()
output = Path(sys.argv[4]).resolve()
work_root = Path(sys.argv[5]).resolve()
expected_plan_sha = sys.argv[6]
expected_addendum_plan_sha = sys.argv[7]
expected_pbc_manifest_sha = sys.argv[8]
expected_pbc_receipt_sha = sys.argv[9]
expected_observation_sha = sys.argv[10]
expected_lag_sha = sys.argv[11]
expected_fit_sha = sys.argv[12]
blacklist = {"cn2", "cn3", "cn4", "cn15", "cn16", "cn17"}

def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"comparison preflight failed: {message}")

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def source_bundle_sha256(paths: dict[str, Path]) -> str:
    digest = hashlib.sha256()
    for relative, path in sorted(paths.items()):
        require(path.is_file(), f"source missing: {relative}")
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(sha256_file(path).encode("ascii") + b"\0")
    return digest.hexdigest()

for receipt_root in (persistence_path.parent, joint_path.parent, pbc_path.parent):
    try:
        output.relative_to(receipt_root)
    except ValueError:
        pass
    else:
        require(False, f"fresh output may not modify an input result: {receipt_root}")

plan = work_root / "plan/PERSISTENCE_AUGMENTED_NEURAL_20260823.md"
addendum_plan = work_root / "plan/PERSISTENCE_MASK_CONTROL_ADDENDUM_20260823.md"
require(sha256_file(plan) == expected_plan_sha, "frozen plan SHA-256")
require(
    sha256_file(addendum_plan) == expected_addendum_plan_sha,
    "frozen mask-control addendum SHA-256",
)
persistence = comparison.validate_persistence_receipt(persistence_path)
pbc = comparison.validate_accepted_pbc_receipt(pbc_path)
joint = comparison.validate_joint_selection_receipt(joint_path, persistence, pbc)
require(pbc.manifest_sha256 == expected_pbc_manifest_sha, "accepted PBC manifest SHA-256")
joint_snapshot = dict(joint.manifest.get("source_snapshot_sha256", {}))
addendum_snapshot_key = "code/plan/PERSISTENCE_MASK_CONTROL_ADDENDUM_20260823.md"
require(
    joint_snapshot.get(addendum_snapshot_key) == expected_addendum_plan_sha,
    "joint receipt does not bind the frozen addendum plan",
)
joint_source_binding = hashlib.sha256(
    json.dumps(joint_snapshot, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()
require(
    joint.gate_receipt.get("source_binding_sha256") == joint_source_binding,
    "joint source binding receipt",
)

persistence_gate = persistence.gate_receipt
require(persistence_gate.get("schema_version") == comparison.PERSISTENCE_RECEIPT_SCHEMA, "persistence receipt schema")
require(persistence_gate.get("status") == "post_run_audit_passed", "persistence post-run status")
require(persistence_gate.get("mode") == "full", "persistence full receipt")
require(persistence_gate.get("partition") == "gpu_prio", "persistence receipt partition")
require(persistence_gate.get("node") not in blacklist, "persistence receipt excluded node")
require(bool(persistence_gate.get("gpu")), "persistence receipt GPU")
require(persistence.manifest.get("software", {}).get("cuda_available") is True, "persistence CUDA evidence")

pbc_receipt_path = pbc_path.parent / "slurm_gate_receipt.json"
require(pbc_receipt_path.is_file(), "accepted PBC Slurm receipt")
require(sha256_file(pbc_receipt_path) == expected_pbc_receipt_sha, "accepted PBC receipt SHA-256")
pbc_manifest = pbc.manifest
require(
    pbc_manifest.get("observation_bundle_provenance", {}).get("sha256")
    == expected_observation_sha,
    "accepted PBC observation bundle SHA-256",
)
require(
    pbc_manifest.get("persistence_lag_provenance", {}).get("sha256")
    == expected_lag_sha,
    "accepted PBC lag SHA-256",
)
require(
    pbc_manifest.get("artifact_sha256", {}).get("models/pbc_fit.npz")
    == expected_fit_sha,
    "accepted PBC fit SHA-256",
)

source_paths = {
    "src/fuxi_allseason_persistence_pbc_comparison.py": work_root / "src/fuxi_allseason_persistence_pbc_comparison.py",
    "tests/test_allseason_persistence_pbc_comparison.py": work_root / "tests/test_allseason_persistence_pbc_comparison.py",
    "src/fuxi_allseason_persistence_augmented.py": work_root / "src/fuxi_allseason_persistence_augmented.py",
    "src/fuxi_persistence_context.py": work_root / "src/fuxi_persistence_context.py",
    "src/fuxi_allseason_persistence_mask_control.py": work_root / "src/fuxi_allseason_persistence_mask_control.py",
    "src/fuxi_persistence_mask_control.py": work_root / "src/fuxi_persistence_mask_control.py",
    "src/fuxi_allseason_categorical_comparison.py": work_root / "src/fuxi_allseason_categorical_comparison.py",
    "src/fuxi_allseason_pbc_baseline.py": work_root / "src/fuxi_allseason_pbc_baseline.py",
    "src/fuxi_pbc_core.py": work_root / "src/fuxi_pbc_core.py",
    "src/fuxi_allseason_ensemble_calibration.py": work_root / "src/fuxi_allseason_ensemble_calibration.py",
    "src/fuxi_ensemble_calibration_core.py": work_root / "src/fuxi_ensemble_calibration_core.py",
    "src/fuxi_allseason_member_cache.py": work_root / "src/fuxi_allseason_member_cache.py",
    "src/project_paths.py": work_root / "src/project_paths.py",
    "plan/PERSISTENCE_AUGMENTED_NEURAL_20260823.md": plan,
    "plan/PERSISTENCE_MASK_CONTROL_ADDENDUM_20260823.md": addendum_plan,
    "slurm/evaluate_allseason_persistence_pbc_comparison.sbatch": work_root / "slurm/evaluate_allseason_persistence_pbc_comparison.sbatch",
}
source_bundle = source_bundle_sha256(source_paths)
print(
    persistence.manifest_sha256,
    persistence.gate_receipt_sha256,
    joint.manifest_sha256,
    joint.gate_receipt_sha256,
    joint_source_binding,
    pbc.manifest_sha256,
    sha256_file(pbc_receipt_path),
    source_bundle,
)
PY
)
read -r PERSISTENCE_MANIFEST_SHA256 PERSISTENCE_RECEIPT_SHA256 \
    JOINT_SELECTION_MANIFEST_SHA256 JOINT_SELECTION_RECEIPT_SHA256 \
    JOINT_SELECTION_SOURCE_BINDING_SHA256 \
    PBC_MANIFEST_SHA256 PBC_RECEIPT_SHA256 \
    PREFLIGHT_SOURCE_BUNDLE_SHA256 <<<"${PREFLIGHT_RECEIPT}"
for digest in \
    "${PERSISTENCE_MANIFEST_SHA256}" \
    "${PERSISTENCE_RECEIPT_SHA256}" \
    "${JOINT_SELECTION_MANIFEST_SHA256}" \
    "${JOINT_SELECTION_RECEIPT_SHA256}" \
    "${JOINT_SELECTION_SOURCE_BINDING_SHA256}" \
    "${PBC_MANIFEST_SHA256}" \
    "${PBC_RECEIPT_SHA256}" \
    "${PREFLIGHT_SOURCE_BUNDLE_SHA256}"; do
    if [[ ! ${digest} =~ ^[0-9a-f]{64}$ ]]; then
        echo "preflight returned an invalid digest: ${digest}" >&2
        exit 2
    fi
done
if [[ ${PBC_MANIFEST_SHA256} != "${EXPECTED_PBC_MANIFEST_SHA256}" \
    || ${PBC_RECEIPT_SHA256} != "${EXPECTED_PBC_RECEIPT_SHA256}" ]]; then
    echo "accepted PBC manifest/receipt pins changed" >&2
    exit 2
fi

echo "accepted persistence full manifest: ${PERSISTENCE_MANIFEST}"
echo "persistence manifest SHA-256: ${PERSISTENCE_MANIFEST_SHA256}"
echo "accepted joint mask-control full manifest: ${JOINT_SELECTION_MANIFEST}"
echo "joint manifest SHA-256: ${JOINT_SELECTION_MANIFEST_SHA256}"
echo "accepted immutable PBC V2 manifest: ${PBC_MANIFEST}"
echo "accepted PBC manifest SHA-256: ${PBC_MANIFEST_SHA256}"
echo "preflight source bundle SHA-256: ${PREFLIGHT_SOURCE_BUNDLE_SHA256}"
echo "job=${SLURM_JOB_ID} node=${SLURMD_NODENAME:-unknown} partition=${SLURM_JOB_PARTITION:-unknown}"
echo "python=${PYTHON_BIN}"
"${PYTHON_BIN}" --version
nvidia-smi
GPU_RECEIPT=$(
    nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version \
        --format=csv,noheader,nounits
)
if [[ -z ${GPU_RECEIPT} || ${GPU_RECEIPT} == *$'\n'* ]]; then
    echo "expected exactly one nvidia-smi GPU receipt" >&2
    exit 2
fi
if [[ ${GPU_RECEIPT^^} != *A30* ]]; then
    echo "canonical comparison requires one NVIDIA A30; found: ${GPU_RECEIPT}" >&2
    exit 2
fi
"${PYTHON_BIN}" - <<'PY'
import sys

import matplotlib
import numpy
import pandas
import torch
import xarray

if "/s2s_core/" not in sys.executable:
    raise SystemExit(f"expected s2s_core Python, found {sys.executable}")
if not torch.cuda.is_available():
    raise SystemExit("Slurm assigned no visible CUDA GPU")
if torch.cuda.device_count() != 1:
    raise SystemExit(f"expected exactly one visible CUDA GPU, found {torch.cuda.device_count()}")
if "A30" not in torch.cuda.get_device_name(0).upper():
    raise SystemExit(f"canonical comparison requires NVIDIA A30, found {torch.cuda.get_device_name(0)}")
probe = torch.ones(1, device="cuda")
if not bool(torch.isfinite(probe).all()):
    raise SystemExit("CUDA tensor probe is non-finite")
print("sys.executable:", sys.executable)
print("Python:", sys.version.replace("\n", " "))
print("NumPy:", numpy.__version__)
print("Pandas:", pandas.__version__)
print("Xarray:", xarray.__version__)
print("Matplotlib:", matplotlib.__version__)
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
PY

if [[ -e ${OUTPUT} ]]; then
    echo "output appeared before evaluation; refusing to overwrite: ${OUTPUT}" >&2
    exit 2
fi

srun "${PYTHON_BIN}" src/fuxi_allseason_persistence_pbc_comparison.py \
    --persistence-manifest "${PERSISTENCE_MANIFEST}" \
    --joint-selection-manifest "${JOINT_SELECTION_MANIFEST}" \
    --pbc-manifest "${PBC_MANIFEST}" \
    --output "${OUTPUT}" \
    --device cuda \
    --batch-size 8 \
    --evaluation-batch-size 8 \
    --cdf-chunk-size 8 \
    --num-workers 0 \
    --bootstrap-samples 2000 \
    --block-length 13 \
    --bootstrap-seed 20260823

"${PYTHON_BIN}" - "${OUTPUT}" "${PERSISTENCE_MANIFEST}" \
    "${PERSISTENCE_MANIFEST_SHA256}" "${PERSISTENCE_RECEIPT_SHA256}" \
    "${JOINT_SELECTION_MANIFEST}" "${JOINT_SELECTION_MANIFEST_SHA256}" \
    "${JOINT_SELECTION_RECEIPT_SHA256}" \
    "${JOINT_SELECTION_SOURCE_BINDING_SHA256}" \
    "${PBC_MANIFEST}" "${PBC_MANIFEST_SHA256}" "${PBC_RECEIPT_SHA256}" \
    "${WORK_ROOT}" "${EXPECTED_PLAN_SHA256}" \
    "${EXPECTED_ADDENDUM_PLAN_SHA256}" \
    "${EXPECTED_PBC_MANIFEST_SHA256}" "${EXPECTED_PBC_RECEIPT_SHA256}" \
    "${EXPECTED_PBC_OBSERVATION_SHA256}" "${EXPECTED_PBC_LAG_SHA256}" \
    "${EXPECTED_PBC_FIT_SHA256}" "${PREFLIGHT_SOURCE_BUNDLE_SHA256}" \
    "${SLURM_JOB_ID}" "${SLURMD_NODENAME:-unknown}" \
    "${SLURM_JOB_PARTITION:-unknown}" "${GPU_RECEIPT}" <<'PY'
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import fuxi_allseason_persistence_pbc_comparison as comparison

root = Path(sys.argv[1]).resolve()
persistence_path = Path(sys.argv[2]).resolve()
expected_persistence_manifest_sha = sys.argv[3]
expected_persistence_receipt_sha = sys.argv[4]
joint_path = Path(sys.argv[5]).resolve()
expected_joint_manifest_sha = sys.argv[6]
expected_joint_receipt_sha = sys.argv[7]
expected_joint_source_binding_sha = sys.argv[8]
pbc_path = Path(sys.argv[9]).resolve()
expected_pbc_manifest_sha = sys.argv[10]
expected_pbc_receipt_sha = sys.argv[11]
work_root = Path(sys.argv[12]).resolve()
expected_plan_sha = sys.argv[13]
expected_addendum_plan_sha = sys.argv[14]
pinned_pbc_manifest_sha = sys.argv[15]
pinned_pbc_receipt_sha = sys.argv[16]
pinned_observation_sha = sys.argv[17]
pinned_lag_sha = sys.argv[18]
pinned_fit_sha = sys.argv[19]
expected_source_bundle_sha = sys.argv[20]
job_id, node, partition, gpu = sys.argv[21:25]
manifest_path = root / "manifest.json"
receipt_path = root / "slurm_gate_receipt.json"
arms = ["base_42k", "zero_lag_45k", "persistence_lag12_45k"]
pbc_baselines = ["combined_pbc", "persistence_plus_plus"]
categorical_methods = [*arms, *pbc_baselines]
continuous_methods = ["raw_fuxi", *arms]
seeds = [42, 43, 44]
scopes = ["W1-W6", "W1", "W2", "W3", "W4", "W5", "W6"]
blacklist = {"cn2", "cn3", "cn4", "cn15", "cn16", "cn17"}

def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"comparison post-run gate failed: {message}")

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def is_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )

def finite_json(value: object, location: str) -> None:
    if isinstance(value, float):
        require(math.isfinite(value), f"non-finite JSON number at {location}")
    elif isinstance(value, dict):
        for key, item in value.items():
            finite_json(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            finite_json(item, f"{location}[{index}]")

def source_bundle_sha256(paths: dict[str, Path]) -> str:
    digest = hashlib.sha256()
    for relative, path in sorted(paths.items()):
        require(path.is_file(), f"live source missing: {relative}")
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(sha256_file(path).encode("ascii") + b"\0")
    return digest.hexdigest()

require(manifest_path.is_file(), "manifest exists")
require(not (root / "failure.json").exists(), "failure receipt present")
require(not receipt_path.exists(), "Slurm receipt already exists before audit")
with manifest_path.open("r", encoding="utf-8") as stream:
    manifest = json.load(stream)
finite_json(manifest, "manifest")
require(manifest.get("experiment") == "fuxi_allseason_persistence_pbc_comparison_v1", "experiment")
require(manifest.get("status") == "complete", "status")
require(Path(manifest.get("output_path", "")).resolve() == root, "output identity")
require(
    manifest.get("evidence_status")
    == "reused 2020-2021 retrospective development comparison; not independent confirmation",
    "evidence status",
)

# Re-run every immutable receipt validator after evaluation. This proves no
# parent result, artifact, source snapshot, or external receipt changed while
# development scoring was in progress.
persistence = comparison.validate_persistence_receipt(persistence_path)
pbc = comparison.validate_accepted_pbc_receipt(pbc_path)
joint = comparison.validate_joint_selection_receipt(joint_path, persistence, pbc)
require(persistence.manifest_sha256 == expected_persistence_manifest_sha, "persistence manifest changed")
require(persistence.gate_receipt_sha256 == expected_persistence_receipt_sha, "persistence receipt changed")
require(joint.manifest_sha256 == expected_joint_manifest_sha, "joint selection manifest changed")
require(joint.gate_receipt_sha256 == expected_joint_receipt_sha, "joint selection receipt changed")
joint_source_snapshot = dict(joint.manifest.get("source_snapshot_sha256", {}))
joint_source_binding = hashlib.sha256(
    json.dumps(joint_source_snapshot, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()
require(
    joint_source_binding == expected_joint_source_binding_sha
    == joint.gate_receipt.get("source_binding_sha256"),
    "joint selection source binding changed",
)
require(pbc.manifest_sha256 == expected_pbc_manifest_sha == pinned_pbc_manifest_sha, "accepted PBC manifest changed")
pbc_receipt_path = pbc_path.parent / "slurm_gate_receipt.json"
require(sha256_file(pbc_receipt_path) == expected_pbc_receipt_sha == pinned_pbc_receipt_sha, "accepted PBC receipt changed")

contract = manifest.get("contract", {})
require(contract.get("training_performed") is False, "training firewall")
require(contract.get("selection_performed") is False, "selection firewall")
require(
    contract.get("selection_data")
    == "2018-2019 validation only in the externally receipted joint mask-control addendum",
    "selection boundary",
)
require(contract.get("evaluation_initialization_years") == [2020, 2021], "development years")
require(contract.get("evaluation_initialization_count") == 208, "development case count")
for key in (
    "development_scores_can_change_selection",
    "forecasts_probabilities_or_parameters_averaged_across_seeds",
    "forecasts_probabilities_or_parameters_written",
    "crps_significance_claimed",
):
    require(contract.get(key) is False, f"firewall {key}")
for key in (
    "superiority_claims_limited_to_joint_selected_arm",
    "original_v1_selection_is_not_claim_authority",
    "crps_improvement_language_requires_strictly_lower_mean",
    "no_2022_plus_initialization_cohort",
    "all_51_members_evaluated",
    "all_six_leads_grouped",
    "only_per_seed_scores_averaged",
):
    require(contract.get(key) is True, f"contract {key}")
require(contract.get("sealed_2025_target_opened") is False, "2025 seal")
original_selected_arm = contract.get("original_v1_validation_locked_selected_arm")
selected_arm = contract.get("joint_validation_locked_selected_arm")
require(
    original_selected_arm == persistence.selected_arm
    and original_selected_arm in {"base_42k", "persistence_lag12_45k"},
    "original v1 locked selected arm",
)
require(
    selected_arm == joint.selected_arm
    and selected_arm in {"base_42k", "persistence_lag12_45k"},
    "joint locked selected arm",
)
require(
    contract.get("nonselected_persistence_candidate_is_descriptive_only")
    is (selected_arm != "persistence_lag12_45k"),
    "nonselected candidate language",
)
require(contract.get("categorical_score_contract") == "normalized_informative_positive_cut_v2", "score contract")
require(
    contract.get("promotion_interpretation")
    == "validation RPS improvement with CRPS no-harm",
    "promotion interpretation",
)
require(
    contract.get("superiority_claim_metric")
    == "equality-aware quintile RPS only",
    "superiority claim metric",
)

input_receipts = manifest.get("input_receipts", {})
persistence_input = input_receipts.get("persistence", {})
require(Path(persistence_input.get("manifest_path", "")).resolve() == persistence_path, "persistence input path")
require(persistence_input.get("manifest_sha256") == expected_persistence_manifest_sha, "persistence input SHA")
require(persistence_input.get("gate_receipt_sha256") == expected_persistence_receipt_sha, "persistence gate SHA")
require(persistence_input.get("full_post_run_audit_passed") is True, "persistence post-run gate")
require(persistence_input.get("selected_arm") == original_selected_arm, "persistence selected arm")
joint_input = input_receipts.get("joint_mask_control_selection", {})
require(Path(joint_input.get("manifest_path", "")).resolve() == joint_path, "joint input path")
require(joint_input.get("manifest_sha256") == expected_joint_manifest_sha, "joint input SHA")
require(
    Path(joint_input.get("gate_receipt_path", "")).resolve()
    == joint.gate_receipt_path,
    "joint receipt path",
)
require(joint_input.get("gate_receipt_sha256") == expected_joint_receipt_sha, "joint gate SHA")
require(joint_input.get("full_post_run_audit_passed") is True, "joint post-run gate")
require(joint_input.get("selected_arm") == selected_arm, "joint selected arm")
require(joint_input.get("claim_authority") is True, "joint claim authority")
require(
    joint_input.get("joint_selector_recomputation") == joint.recomputation_evidence,
    "joint selector recomputation evidence",
)
require(
    joint_input.get("matching_environment") == joint.matching_environment_evidence,
    "joint matching-environment evidence",
)
require(
    joint_input.get("smoke_parent") == joint.smoke_parent_evidence,
    "joint smoke-parent evidence",
)
require(
    joint.recomputation_evidence.get("all_recomputed_gate_fields_exact") is True
    and joint.recomputation_evidence.get("selected_arm_exact") is True
    and joint.recomputation_evidence.get("development_indices_accessed") is False,
    "joint selection independent recomputation firewall",
)
for key in (
    "same_gpu_model_exact",
    "same_python_and_library_stack_exact",
    "same_amp_and_determinism_contract_exact",
    "same_seed_initial_states_exact_vs_v1_candidate",
    "same_seed_stochastic_streams_exact_vs_v1",
):
    require(joint.matching_environment_evidence.get(key) is True, f"joint environment {key}")
require(joint.matching_environment_evidence.get("mask_checkpoint_count") == 3, "joint mask checkpoint grid")
require(joint.smoke_parent_evidence.get("post_run_audit_passed") is True, "joint smoke parent audit")
pbc_input = input_receipts.get("accepted_pbc_v2", {})
require(Path(pbc_input.get("manifest_path", "")).resolve() == pbc_path, "PBC input path")
require(pbc_input.get("manifest_sha256") == pinned_pbc_manifest_sha, "pinned PBC manifest")
require(pbc_input.get("pinned_manifest_sha256") == pinned_pbc_manifest_sha, "PBC manifest pin receipt")
require(pbc_input.get("pinned_observation_bundle_sha256") == pinned_observation_sha, "PBC observation pin")
require(pbc_input.get("pinned_persistence_lag_sha256") == pinned_lag_sha, "PBC lag pin")
require(pbc_input.get("pinned_pbc_fit_artifact_sha256") == pinned_fit_sha, "PBC fit pin")
require(pbc_input.get("immutable_accepted_identity") is True, "immutable PBC identity")

identity = manifest.get("identity_gates", {})
for key in (
    "all_required_gates_passed_before_development_indexing",
    "support_exactly_shared",
    "thresholds_fit_on_exact_1652_training_indices",
    "normalization_artifact_exact",
    "development_indices_accessed",
    "development_unlocked_after_all_required_gates",
):
    require(identity.get(key) is True, f"identity gate {key}")
require(identity.get("sealed_2025_opened") is False, "identity 2025 seal")

def validate_array_provenance(
    value: object, *, bundle_name: str, array_names: set[str]
) -> dict[str, object]:
    require(isinstance(value, dict), f"{bundle_name} provenance mapping")
    provenance = dict(value)
    require(
        provenance.get("schema_version") == "named_c_order_array_bundle_sha256_v1",
        f"{bundle_name} provenance schema",
    )
    require(provenance.get("bundle_name") == bundle_name, f"{bundle_name} identity")
    require(is_digest(provenance.get("sha256")), f"{bundle_name} digest")
    arrays = provenance.get("arrays", {})
    require(isinstance(arrays, dict) and set(arrays) == array_names, f"{bundle_name} arrays")
    for name, metadata in arrays.items():
        require(isinstance(metadata, dict), f"{bundle_name} metadata {name}")
        require(isinstance(metadata.get("dtype"), str) and metadata["dtype"], f"{bundle_name} dtype {name}")
        shape = metadata.get("shape")
        require(
            isinstance(shape, list)
            and all(isinstance(size, int) and size >= 0 for size in shape),
            f"{bundle_name} shape {name}",
        )
    return provenance

base_context_provenance = validate_array_provenance(
    identity.get("fresh_full_base_context_provenance"),
    bundle_name="fresh_full_base_neural_context",
    array_names={
        "normalized_climatology",
        "climatology_mean_by_lead",
        "climatology_std_by_lead",
        "latitude_scaled",
        "longitude_scaled",
        "season_sin",
        "season_cos",
        "lead_scaled",
        "support",
    },
)
weekly_climatology_provenance = validate_array_provenance(
    identity.get("fresh_full_weekly_climatology_provenance"),
    bundle_name="fresh_full_weekly_training_climatology",
    array_names={"weekly_climatology"},
)
combined_context_provenance = validate_array_provenance(
    identity.get("joint_combined_base_context_and_weekly_climatology_provenance"),
    bundle_name="full_base_context_and_weekly_climatology",
    array_names={
        "weekly_climatology",
        "normalized_climatology",
        "climatology_mean_by_lead",
        "climatology_std_by_lead",
        "latitude_scaled",
        "longitude_scaled",
        "season_sin",
        "season_cos",
        "lead_scaled",
        "support",
    },
)
require(
    combined_context_provenance
    == joint.manifest.get("base_context_and_weekly_climatology_provenance"),
    "joint combined base-context/climatology provenance",
)
fresh_derived = manifest.get("fresh_derived_input_provenance", {})
require(isinstance(fresh_derived, dict), "fresh derived input receipt")
require(
    fresh_derived.get("full_base_context") == base_context_provenance
    and fresh_derived.get("full_weekly_climatology")
    == weekly_climatology_provenance
    and fresh_derived.get("joint_combined_base_context_and_weekly_climatology")
    == combined_context_provenance,
    "fresh derived input provenance duplication",
)
validation_replay = identity.get("validation_adjustment_replay", {})
require(isinstance(validation_replay, dict), "validation replay receipt")
require(validation_replay.get("validation_initialization_count") == 196, "validation replay cases")
require(validation_replay.get("validation_years") == [2018, 2019], "validation replay years")
require(validation_replay.get("checkpoint_count") == 9, "validation replay checkpoints")
for key in (
    "all_nine_stored_adjustments_reconstructed",
    "all_initialization_arrays_exact",
    "all_arrays_within_tight_tolerance",
):
    require(validation_replay.get(key) is True, f"validation replay {key}")
require(validation_replay.get("development_indices_accessed") is False, "validation replay development firewall")
require(
    float(validation_replay.get("comparison_rtol", -1.0)) == comparison.VALIDATION_REPLAY_RTOL
    and float(validation_replay.get("comparison_atol", -1.0)) == comparison.VALIDATION_REPLAY_ATOL,
    "validation replay tolerances",
)
replay_cells = validation_replay.get("cells", [])
require(isinstance(replay_cells, list) and len(replay_cells) == 9, "validation replay cell count")
replay_keys = set()
for cell in replay_cells:
    require(isinstance(cell, dict), "validation replay cell mapping")
    arm, seed = cell.get("arm"), cell.get("seed")
    require(arm in arms and seed in seeds, "validation replay cell identity")
    require(cell.get("initializations_exact") is True, f"validation replay initialization {arm}/{seed}")
    require(cell.get("all_arrays_within_tight_tolerance") is True, f"validation replay tolerance {arm}/{seed}")
    require(is_digest(cell.get("checkpoint_sha256")), f"validation replay checkpoint {arm}/{seed}")
    require(is_digest(cell.get("validation_adjustment_sha256")), f"validation replay adjustment {arm}/{seed}")
    comparisons = cell.get("comparisons", {})
    require(
        isinstance(comparisons, dict)
        and set(comparisons) == {"delta_log_location", "log_spread", "spread_factor"},
        f"validation replay arrays {arm}/{seed}",
    )
    for label, evidence in comparisons.items():
        require(isinstance(evidence, dict), f"validation replay evidence {arm}/{seed}/{label}")
        require(evidence.get("within_tight_tolerance") is True, f"validation replay match {arm}/{seed}/{label}")
        maximum = float(evidence.get("max_absolute_difference", float("nan")))
        require(math.isfinite(maximum) and maximum >= 0.0, f"validation replay delta {arm}/{seed}/{label}")
    replay_keys.add((arm, seed))
require(replay_keys == {(arm, seed) for arm in arms for seed in seeds}, "validation replay grid")

accepted_inputs = manifest.get("accepted_pbc_inputs", {})
require(accepted_inputs.get("accepted_pbc_manifest_sha256") == pinned_pbc_manifest_sha, "accepted input manifest")
require(accepted_inputs.get("development_case_count") == 208, "accepted input cases")
require(accepted_inputs.get("development_years") == [2020, 2021], "accepted input years")
for key in (
    "all_six_leads_grouped",
    "quintile_observed_cdf_valid_for_thresholds",
    "semidecile_observed_cdf_valid_for_thresholds",
    "threshold_fit_indices_equal_exact_training_split",
    "no_2022_plus_initialization_cohort",
):
    require(accepted_inputs.get(key) is True, f"accepted input gate {key}")
require(accepted_inputs.get("sealed_2025_opened") is False, "accepted input 2025 seal")
for key, value in manifest.get("pbc_raw_identity", {}).items():
    require(math.isfinite(float(value)) and 0.0 <= float(value) <= 2.0e-7, f"PBC raw identity {key}")

require(manifest.get("arms") == arms, "arm order")
require(manifest.get("seeds") == seeds, "seed order")
require(manifest.get("all_nine_checkpoints_loaded") is True, "nine-checkpoint receipt")
reconstructions = manifest.get("reconstruction_receipts", [])
require(len(reconstructions) == 9, "reconstruction receipt count")
reconstruction_keys = set()
for row in reconstructions:
    arm, seed = row.get("arm"), row.get("seed")
    require(arm in arms and seed in seeds, "reconstruction identity")
    require(row.get("development_initializations") == 208, f"reconstruction cases {arm}/{seed}")
    require(row.get("members_per_ensemble") == 51, f"reconstruction members {arm}/{seed}")
    for key in (
        "adjustments_finite",
        "reconstructed_members_finite_nonnegative",
        "quintile_cdf_valid_for_exact_pbc_thresholds",
        "continuous_scores_finite",
        "categorical_scores_finite",
    ):
        require(row.get(key) is True, f"reconstruction {key} {arm}/{seed}")
    require(row.get("forecasts_or_parameters_written") is False, f"write firewall {arm}/{seed}")
    require(row.get("forecasts_or_parameters_averaged_across_seeds") is False, f"averaging firewall {arm}/{seed}")
    require(is_digest(row.get("checkpoint_sha256")), f"checkpoint digest {arm}/{seed}")
    reconstruction_keys.add((arm, seed))
require(reconstruction_keys == {(arm, seed) for arm in arms for seed in seeds}, "reconstruction grid")

bootstrap_contract = manifest.get("bootstrap", {})
require(bootstrap_contract.get("samples") == 2000, "bootstrap samples")
require(bootstrap_contract.get("seed") == 20260823, "bootstrap seed")
require(bootstrap_contract.get("block_length_initializations") == 13, "bootstrap block")
require(
    bootstrap_contract.get("scheme")
    == "two_stage_year_resample_then_within_year_circular_blocks",
    "bootstrap scheme",
)
require(bootstrap_contract.get("year_resampling") is True, "bootstrap year resampling")
require(bootstrap_contract.get("all_six_leads_grouped") is True, "bootstrap grouping")
require(bootstrap_contract.get("reported_scopes") == scopes, "bootstrap scopes")
require(bootstrap_contract.get("comparison_methods") == arms, "bootstrap methods")
require(bootstrap_contract.get("baselines") == pbc_baselines, "bootstrap baselines")
require(bootstrap_contract.get("claim_eligible_methods") == [selected_arm], "claim eligibility")
require(
    bootstrap_contract.get("nonselected_methods_descriptive_only")
    == [arm for arm in arms if arm != selected_arm],
    "descriptive-only methods",
)

required = {
    "README.md",
    "claims.json",
    "inputs/persistence_manifest.json",
    "inputs/persistence_slurm_gate_receipt.json",
    "inputs/persistence_selection.json",
    "inputs/joint_selection_manifest.json",
    "inputs/joint_selection_slurm_gate_receipt.json",
    "inputs/joint_selection.json",
    "inputs/accepted_pbc_manifest.json",
    "inputs/accepted_pbc_slurm_gate_receipt.json",
    "metrics/continuous_seed_case_metrics.csv",
    "metrics/continuous_case_metrics.csv",
    "metrics/continuous_weekwise_metrics.csv",
    "metrics/continuous_pooled_metrics.csv",
    "metrics/continuous_seasonal_metrics.csv",
    "metrics/categorical_seed_case_scores.csv",
    "metrics/categorical_case_scores.csv",
    "metrics/categorical_weekwise_metrics.csv",
    "metrics/categorical_pooled_metrics.csv",
    "metrics/paired_pbc_bootstrap.csv",
}
source_map = {
    "code/src/fuxi_allseason_persistence_pbc_comparison.py": work_root / "src/fuxi_allseason_persistence_pbc_comparison.py",
    "code/tests/test_allseason_persistence_pbc_comparison.py": work_root / "tests/test_allseason_persistence_pbc_comparison.py",
    "code/src/fuxi_allseason_persistence_augmented.py": work_root / "src/fuxi_allseason_persistence_augmented.py",
    "code/src/fuxi_persistence_context.py": work_root / "src/fuxi_persistence_context.py",
    "code/src/fuxi_allseason_persistence_mask_control.py": work_root / "src/fuxi_allseason_persistence_mask_control.py",
    "code/src/fuxi_persistence_mask_control.py": work_root / "src/fuxi_persistence_mask_control.py",
    "code/src/fuxi_allseason_categorical_comparison.py": work_root / "src/fuxi_allseason_categorical_comparison.py",
    "code/src/fuxi_allseason_pbc_baseline.py": work_root / "src/fuxi_allseason_pbc_baseline.py",
    "code/src/fuxi_pbc_core.py": work_root / "src/fuxi_pbc_core.py",
    "code/src/fuxi_allseason_ensemble_calibration.py": work_root / "src/fuxi_allseason_ensemble_calibration.py",
    "code/src/fuxi_ensemble_calibration_core.py": work_root / "src/fuxi_ensemble_calibration_core.py",
    "code/src/fuxi_allseason_member_cache.py": work_root / "src/fuxi_allseason_member_cache.py",
    "code/src/project_paths.py": work_root / "src/project_paths.py",
    "code/plan/PERSISTENCE_AUGMENTED_NEURAL_20260823.md": work_root / "plan/PERSISTENCE_AUGMENTED_NEURAL_20260823.md",
    "code/plan/PERSISTENCE_MASK_CONTROL_ADDENDUM_20260823.md": work_root / "plan/PERSISTENCE_MASK_CONTROL_ADDENDUM_20260823.md",
    "code/slurm/evaluate_allseason_persistence_pbc_comparison.sbatch": work_root / "slurm/evaluate_allseason_persistence_pbc_comparison.sbatch",
}
required.update(source_map)
artifact_hashes = manifest.get("artifact_sha256", {})
require(isinstance(artifact_hashes, dict), "artifact hash map")
actual_artifacts = {
    str(path.relative_to(root))
    for path in root.rglob("*")
    if path.is_file() and str(path.relative_to(root)) not in {"manifest.json", "slurm_gate_receipt.json"}
}
require(set(artifact_hashes) == actual_artifacts, "complete artifact hash inventory")
require(required.issubset(actual_artifacts), "required artifact inventory")
require("slurm_gate_receipt.json" not in artifact_hashes, "external receipt is not driver-hashed")
for relative, digest in artifact_hashes.items():
    require(is_digest(digest), f"artifact digest {relative}")
    require(sha256_file(root / relative) == digest, f"artifact hash {relative}")
require(
    not any(path.suffix.lower() in {".npy", ".npz", ".pt", ".pth"} for path in root.rglob("*") if path.is_file()),
    "forecast/model array written to score-only output",
)

source_snapshot = manifest.get("source_snapshot_sha256", {})
require(set(source_snapshot) == set(source_map), "source snapshot inventory")
require(len(source_snapshot) == 16, "exact 16-file source snapshot")
for relative, live in source_map.items():
    digest = source_snapshot.get(relative)
    require(is_digest(digest), f"source digest {relative}")
    require(artifact_hashes.get(relative) == digest, f"source/artifact hash {relative}")
    require(sha256_file(root / relative) == digest, f"snapshot source hash {relative}")
    require(sha256_file(live) == digest, f"live source changed during run {relative}")
require(
    source_snapshot.get("code/plan/PERSISTENCE_AUGMENTED_NEURAL_20260823.md")
    == expected_plan_sha,
    "frozen plan snapshot",
)
require(
    source_snapshot.get("code/plan/PERSISTENCE_MASK_CONTROL_ADDENDUM_20260823.md")
    == expected_addendum_plan_sha,
    "frozen addendum plan snapshot",
)
launcher_path = work_root / "slurm/evaluate_allseason_persistence_pbc_comparison.sbatch"
preflight_source_paths = {
    relative.removeprefix("code/"): live for relative, live in source_map.items()
}
preflight_source_paths["slurm/evaluate_allseason_persistence_pbc_comparison.sbatch"] = launcher_path
require(source_bundle_sha256(preflight_source_paths) == expected_source_bundle_sha, "source changed after preflight")

require(sha256_file(root / "inputs/persistence_manifest.json") == expected_persistence_manifest_sha, "copied persistence manifest")
require(sha256_file(root / "inputs/persistence_slurm_gate_receipt.json") == expected_persistence_receipt_sha, "copied persistence receipt")
require(sha256_file(root / "inputs/joint_selection_manifest.json") == expected_joint_manifest_sha, "copied joint manifest")
require(sha256_file(root / "inputs/joint_selection_slurm_gate_receipt.json") == expected_joint_receipt_sha, "copied joint receipt")
require(sha256_file(root / "inputs/accepted_pbc_manifest.json") == pinned_pbc_manifest_sha, "copied PBC manifest")
require(sha256_file(root / "inputs/accepted_pbc_slurm_gate_receipt.json") == pinned_pbc_receipt_sha, "copied PBC receipt")
selection_path = persistence_path.parent / "selection.json"
require(sha256_file(root / "inputs/persistence_selection.json") == sha256_file(selection_path), "copied selection")
joint_selection_path = joint_path.parent / "joint_selection.json"
require(sha256_file(root / "inputs/joint_selection.json") == sha256_file(joint_selection_path), "copied joint selection")

seed_continuous = pd.read_csv(root / "metrics/continuous_seed_case_metrics.csv")
continuous_case = pd.read_csv(root / "metrics/continuous_case_metrics.csv")
require(len(seed_continuous) == 3 * 3 * 208 * 6, "continuous seed row count")
require(set(seed_continuous.method) == set(arms), "continuous seed methods")
require(set(seed_continuous.seed.astype(int)) == set(seeds), "continuous seeds")
require(not seed_continuous.duplicated(["method", "seed", "init", "lead_week"]).any(), "continuous seed pairing")
require(len(continuous_case) == 4 * 208 * 6, "continuous case row count")
require(set(continuous_case.method) == set(continuous_methods), "continuous methods")
require(not continuous_case.duplicated(["method", "init", "lead_week"]).any(), "continuous case pairing")
for frame, label in ((seed_continuous, "seed continuous"), (continuous_case, "continuous")):
    require(set(frame.year.astype(int)) == {2020, 2021}, f"{label} years")
    require(set(frame.lead_week.astype(int)) == set(range(1, 7)), f"{label} leads")
    require(not frame.init.astype(str).str.startswith(("2022-", "2023-", "2024-", "2025-")).any(), f"{label} date firewall")
    require(np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)).all(), f"finite {label} metrics")

seed_categorical = pd.read_csv(root / "metrics/categorical_seed_case_scores.csv")
categorical_case = pd.read_csv(root / "metrics/categorical_case_scores.csv")
require(len(seed_categorical) == 3 * 3 * 208 * 6, "categorical seed row count")
require(set(seed_categorical.method) == set(arms), "categorical seed methods")
require(set(seed_categorical.seed.astype(int)) == set(seeds), "categorical seeds")
require(not seed_categorical.duplicated(["method", "seed", "initialization", "lead_week"]).any(), "categorical seed pairing")
require(len(categorical_case) == 5 * 208 * 6, "categorical case row count")
require(set(categorical_case.method) == set(categorical_methods), "categorical methods")
require(not categorical_case.duplicated(["method", "initialization", "lead_week"]).any(), "categorical case pairing")
for frame, label in ((seed_categorical, "seed categorical"), (categorical_case, "categorical")):
    require(set(frame.year.astype(int)) == {2020, 2021}, f"{label} years")
    require(set(frame.lead_week.astype(int)) == set(range(1, 7)), f"{label} leads")
    require(set(frame.score_contract) == {"normalized_informative_positive_cut_v2"}, f"{label} score contract")
    require(np.isfinite(frame.rps.to_numpy(dtype=np.float64)).all() and frame.rps.between(0.0, 1.0).all(), f"finite bounded {label} RPS")
    require(not frame.initialization.astype(str).str.startswith(("2022-", "2023-", "2024-", "2025-")).any(), f"{label} date firewall")

seed_means = (
    seed_categorical.groupby(["method", "initialization", "lead_week"], as_index=False)
    .rps.mean()
    .sort_values(["method", "initialization", "lead_week"])
    .reset_index(drop=True)
)
stored_neural = (
    categorical_case.loc[categorical_case.method.isin(arms), ["method", "initialization", "lead_week", "rps"]]
    .sort_values(["method", "initialization", "lead_week"])
    .reset_index(drop=True)
)
require(len(seed_means) == len(stored_neural), "neural categorical aggregation rows")
require(
    seed_means[["method", "initialization", "lead_week"]].equals(
        stored_neural[["method", "initialization", "lead_week"]]
    ),
    "neural categorical aggregation keys",
)
require(np.allclose(seed_means.rps, stored_neural.rps, rtol=0.0, atol=1.0e-12), "mean of per-seed categorical scores")

categorical_weekwise = pd.read_csv(root / "metrics/categorical_weekwise_metrics.csv")
categorical_pooled = pd.read_csv(root / "metrics/categorical_pooled_metrics.csv")
require(len(categorical_weekwise) == 5 * 6, "categorical weekwise rows")
require(len(categorical_pooled) == 5, "categorical pooled rows")
require(set(categorical_weekwise.method) == set(categorical_methods), "weekwise methods")
require(set(categorical_pooled.method) == set(categorical_methods), "pooled methods")
recomputed_weekwise = categorical_case.groupby(["method", "lead_week"], as_index=False).rps.mean()
joined_weekwise = categorical_weekwise.merge(recomputed_weekwise, on=["method", "lead_week"], suffixes=("_stored", "_recomputed"), validate="one_to_one")
require(np.allclose(joined_weekwise.mean_rps, joined_weekwise.rps, rtol=0.0, atol=1.0e-12), "weekwise RPS identity")
recomputed_pooled = categorical_case.groupby("method", as_index=False).rps.mean()
joined_pooled = categorical_pooled.merge(recomputed_pooled, on="method", suffixes=("_stored", "_recomputed"), validate="one_to_one")
require(np.allclose(joined_pooled.mean_rps, joined_pooled.rps, rtol=0.0, atol=1.0e-12), "pooled RPS identity")

bootstrap = pd.read_csv(root / "metrics/paired_pbc_bootstrap.csv")
require(len(bootstrap) == 42, "exact 42 bootstrap rows")
require(set(bootstrap.method) == set(arms), "bootstrap arm inventory")
require(set(bootstrap.baseline) == set(pbc_baselines), "bootstrap baseline inventory")
require(set(bootstrap.lead_scope) == set(scopes), "bootstrap scope inventory")
require(
    not bootstrap.duplicated(["method", "baseline", "lead_scope"]).any(),
    "bootstrap comparison uniqueness",
)
require(set(bootstrap.bootstrap_samples.astype(int)) == {2000}, "bootstrap sample receipt")
require(set(bootstrap.block_length_initializations.astype(int)) == {13}, "bootstrap block receipt")
require(set(bootstrap.bootstrap_seed.astype(int)) == {20260823}, "bootstrap seed receipt")
require(set(bootstrap.paired_initializations.astype(int)) == {208}, "bootstrap paired cases")
require(set(bootstrap.development_years.astype(str)) == {"[2020, 2021]"}, "bootstrap development years")
require(np.isfinite(bootstrap.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)).all(), "finite bootstrap values")
require((bootstrap.ci_lower_95 <= bootstrap.ci_upper_95).all(), "ordered bootstrap intervals")
require(bootstrap.bootstrap_probability_improvement.between(0.0, 1.0).all(), "bootstrap probability bounds")
require(
    np.allclose(
        bootstrap.rps_reduction_fraction,
        1.0 - bootstrap.method_rps / bootstrap.baseline_rps,
        rtol=1.0e-12,
        atol=1.0e-12,
    ),
    "bootstrap point-effect identity",
)

recomputed_bootstrap = comparison.paired_pbc_bootstrap(
    categorical_case,
    arms,
    pbc_baselines,
    samples=2000,
    block_length=13,
    seed=20260823,
)
require(list(bootstrap.columns) == list(recomputed_bootstrap.columns), "bootstrap column schema")
for column in bootstrap.columns:
    if pd.api.types.is_numeric_dtype(recomputed_bootstrap[column]):
        require(
            np.allclose(
                bootstrap[column].to_numpy(dtype=np.float64),
                recomputed_bootstrap[column].to_numpy(dtype=np.float64),
                rtol=1.0e-12,
                atol=1.0e-12,
            ),
            f"recomputed bootstrap column {column}",
        )
    else:
        require(
            bootstrap[column].astype(str).tolist()
            == recomputed_bootstrap[column].astype(str).tolist(),
            f"recomputed bootstrap labels {column}",
        )

with (root / "claims.json").open("r", encoding="utf-8") as stream:
    claims_artifact = json.load(stream)
claims = manifest.get("claims", {})
require(claims_artifact == claims, "claims artifact/manifest agreement")
require(set(claims) == {"joint_validation_selected_arm"}, "claims limited to joint selected arm")
require(claims["joint_validation_selected_arm"].get("method") == selected_arm, "claim method")
recomputed_claims = comparison.classify_pbc_claims(
    bootstrap,
    {"joint_validation_selected_arm": selected_arm},
)
require(comparison._json_equivalent(claims, recomputed_claims), "claim classification identity")

require(job_id.replace("_", "").isdigit(), "Slurm job ID")
require(partition == "gpu_prio", "Slurm partition")
require(node not in blacklist and node != "unknown", "Slurm node")
require(bool(gpu) and "A30" in gpu.upper(), "NVIDIA A30 GPU receipt")
require(manifest.get("software", {}).get("cuda_available") is True, "manifest CUDA")
require("A30" in str(manifest.get("software", {}).get("cuda_device", "")).upper(), "manifest A30 CUDA device")

source_binding_sha = hashlib.sha256(
    json.dumps(source_snapshot, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()
artifact_binding_sha = hashlib.sha256(
    json.dumps(artifact_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()
receipt = {
    "schema_version": "fuxi_persistence_pbc_comparison_slurm_gate_receipt_v1",
    "experiment": manifest["experiment"],
    "status": "post_run_audit_passed",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "output_path": str(root),
    "manifest_path": str(manifest_path),
    "manifest_sha256": sha256_file(manifest_path),
    "job_id": job_id,
    "node": node,
    "partition": partition,
    "gpu": gpu,
    "cuda_device": manifest.get("software", {}).get("cuda_device"),
    "persistence_manifest_sha256": expected_persistence_manifest_sha,
    "persistence_slurm_receipt_sha256": expected_persistence_receipt_sha,
    "persistence_selection_sha256": sha256_file(selection_path),
    "original_v1_selected_arm": original_selected_arm,
    "joint_selection_manifest_path": str(joint_path),
    "joint_selection_manifest_sha256": expected_joint_manifest_sha,
    "joint_selection_slurm_receipt_sha256": expected_joint_receipt_sha,
    "joint_selection_artifact_sha256": sha256_file(joint_selection_path),
    "joint_selection_source_snapshot_sha256": joint_source_snapshot,
    "joint_selection_source_binding_sha256": joint_source_binding,
    "joint_validation_selected_arm": selected_arm,
    "joint_selector_recomputation": joint.recomputation_evidence,
    "joint_matching_environment": joint.matching_environment_evidence,
    "joint_smoke_parent": joint.smoke_parent_evidence,
    "accepted_pbc_manifest_sha256": pinned_pbc_manifest_sha,
    "accepted_pbc_slurm_receipt_sha256": pinned_pbc_receipt_sha,
    "accepted_pbc_observation_bundle_sha256": pinned_observation_sha,
    "accepted_pbc_persistence_lag_sha256": pinned_lag_sha,
    "accepted_pbc_fit_artifact_sha256": pinned_fit_sha,
    "source_snapshot_sha256": source_snapshot,
    "source_binding_sha256": source_binding_sha,
    "artifact_sha256": artifact_hashes,
    "artifact_binding_sha256": artifact_binding_sha,
    "preflight_source_bundle_sha256": expected_source_bundle_sha,
    "launcher_path": str(launcher_path),
    "launcher_sha256": sha256_file(launcher_path),
    "plan_binding": {
        "path": str(work_root / "plan/PERSISTENCE_AUGMENTED_NEURAL_20260823.md"),
        "expected_sha256": expected_plan_sha,
        "snapshot_sha256": source_snapshot["code/plan/PERSISTENCE_AUGMENTED_NEURAL_20260823.md"],
        "live_sha256": sha256_file(work_root / "plan/PERSISTENCE_AUGMENTED_NEURAL_20260823.md"),
    },
    "addendum_plan_binding": {
        "path": str(work_root / "plan/PERSISTENCE_MASK_CONTROL_ADDENDUM_20260823.md"),
        "expected_sha256": expected_addendum_plan_sha,
        "snapshot_sha256": source_snapshot["code/plan/PERSISTENCE_MASK_CONTROL_ADDENDUM_20260823.md"],
        "joint_snapshot_sha256": joint_source_snapshot["code/plan/PERSISTENCE_MASK_CONTROL_ADDENDUM_20260823.md"],
        "live_sha256": sha256_file(work_root / "plan/PERSISTENCE_MASK_CONTROL_ADDENDUM_20260823.md"),
    },
    "fresh_full_base_context_provenance": base_context_provenance,
    "fresh_full_weekly_climatology_provenance": weekly_climatology_provenance,
    "joint_combined_base_context_and_weekly_climatology_provenance": (
        combined_context_provenance
    ),
    "validation_adjustment_replay": {
        "sha256": hashlib.sha256(
            json.dumps(validation_replay, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "validation_initialization_count": 196,
        "validation_years": [2018, 2019],
        "checkpoint_count": 9,
        "all_nine_stored_adjustments_reconstructed": True,
        "all_initialization_arrays_exact": True,
        "all_arrays_within_tight_tolerance": True,
        "development_indices_accessed": False,
    },
    "bootstrap_artifact_sha256": artifact_hashes["metrics/paired_pbc_bootstrap.csv"],
    "bootstrap_rows": 42,
    "firewalls": {
        "training_performed": False,
        "selection_performed": False,
        "development_scores_can_change_selection": False,
        "superiority_claims_limited_to_joint_selected_arm": True,
        "original_v1_selection_is_not_claim_authority": True,
        "no_2022_plus_initialization_cohort": True,
        "sealed_2025_target_opened": False,
        "forecast_or_parameter_arrays_written": False,
    },
}
with receipt_path.open("x", encoding="utf-8") as stream:
    json.dump(receipt, stream, indent=2, sort_keys=True)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
print(receipt_path)
PY

echo "complete and post-run audited: ${OUTPUT}"
