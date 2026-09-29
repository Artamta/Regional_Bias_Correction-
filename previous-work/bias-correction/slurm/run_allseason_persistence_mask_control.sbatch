#!/usr/bin/env bash
#SBATCH --job-name=fuxi-mask-ctl
#SBATCH --partition=gpu_prio
# cn3 is MIG; cn2, cn4, cn15, cn16, and cn17 are unreliable for S2S work.
#SBATCH --exclude=cn2,cn3,cn4,cn15,cn16,cn17
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/logs/%x-%j.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/logs/%x-%j.err

set -euo pipefail
umask 027

usage() {
    cat >&2 <<'EOF'
usage:
  sbatch --nodelist=cn7 slurm/run_allseason_persistence_mask_control.sbatch \
    smoke /absolute/path/full-cache.npy /absolute/path/fresh-smoke-output \
    /absolute/path/original-v1-smoke/manifest.json \
    /absolute/path/original-v1-smoke/slurm_gate_receipt.json

  sbatch --nodelist=cn11 slurm/run_allseason_persistence_mask_control.sbatch \
    full /absolute/path/full-cache.npy /absolute/path/fresh-full-output \
    /absolute/path/original-v1-full/manifest.json \
    /absolute/path/original-v1-full/slurm_gate_receipt.json \
    /absolute/path/completed-mask-smoke/manifest.json \
    /absolute/path/completed-mask-smoke/slurm_gate_receipt.json

The selected Slurm node must exactly equal the node in the supplied original-v1
receipt (currently cn7 for smoke and cn11 for full). Both modes require an A30,
the canonical full cache, immutable accepted PBC V2, fresh output, and the exact
frozen source/plan bundle. Full additionally requires an externally audited mask
smoke made from that same 14-file source bundle and v1-smoke ancestry.
EOF
}

if [[ $# -lt 1 ]]; then
    usage
    exit 2
fi
MODE=$1
case ${MODE} in
    smoke)
        if [[ $# -ne 5 ]]; then
            usage
            exit 2
        fi
        ;;
    full)
        if [[ $# -ne 7 ]]; then
            usage
            exit 2
        fi
        ;;
    *)
        echo "mode must be smoke or full: ${MODE}" >&2
        usage
        exit 2
        ;;
esac

CACHE_INPUT=$2
OUTPUT_INPUT=$3
ORIGINAL_MANIFEST_INPUT=$4
ORIGINAL_RECEIPT_INPUT=$5
for value in "${CACHE_INPUT}" "${OUTPUT_INPUT}" \
    "${ORIGINAL_MANIFEST_INPUT}" "${ORIGINAL_RECEIPT_INPUT}"; do
    if [[ ${value} != /* ]]; then
        echo "cache, output, and original-v1 inputs must be absolute paths" >&2
        exit 2
    fi
done
if [[ ! -f ${CACHE_INPUT} || ! -f ${ORIGINAL_MANIFEST_INPUT} \
    || ! -f ${ORIGINAL_RECEIPT_INPUT} ]]; then
    echo "cache or original-v1 input is missing" >&2
    exit 2
fi

MASK_SMOKE_MANIFEST=
MASK_SMOKE_RECEIPT=
if [[ ${MODE} == full ]]; then
    if [[ $6 != /* || $7 != /* || ! -f $6 || ! -f $7 ]]; then
        echo "full mode requires absolute existing mask-smoke manifest/receipt" >&2
        exit 2
    fi
    MASK_SMOKE_MANIFEST=$(readlink -f "$6")
    MASK_SMOKE_RECEIPT=$(readlink -f "$7")
fi

WORK_ROOT=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction
PYTHON_BIN=/home/raj.ayush/.conda/envs/s2s_core/bin/python
CACHE=$(readlink -f "${CACHE_INPUT}")
OUTPUT=$(readlink -m "${OUTPUT_INPUT}")
ORIGINAL_MANIFEST=$(readlink -f "${ORIGINAL_MANIFEST_INPUT}")
ORIGINAL_RECEIPT=$(readlink -f "${ORIGINAL_RECEIPT_INPUT}")
OUTPUT_PARENT=$(dirname "${OUTPUT}")
OUTPUT_NAME=$(basename "${OUTPUT}")
PBC_ROOT=${WORK_ROOT}/resultsv2/fuxi_allseason_pbc_baseline_v2/full_20260822T173656Z
PBC_MANIFEST=${PBC_ROOT}/manifest.json
PBC_RECEIPT=${PBC_ROOT}/slurm_gate_receipt.json
PBC_FIT=${PBC_ROOT}/models/pbc_fit.npz
EXPECTED_CACHE_SHA256=2e0b4f93503c1de94428483bcd50122ab058a4f7e1bb606314e0f68896329a70
EXPECTED_SOURCE_FINGERPRINT=655ee4b82597daf150a8c28b2ed7b474ba6ce878d00836a6db8c3e75cb7a9dae
EXPECTED_PLAN_SHA256=2a1c6d16957cbceedc0bcf62cb2a7518ca1e5e161f8df36aab0f600e28302699
EXPECTED_ADDENDUM_PLAN_SHA256=c429ee96bfd9b9fc1ddc8fa88da653eeb4923ea037d516d8d6a42071dd3c3ba0
EXPECTED_PBC_MANIFEST_SHA256=c8c8bbb840d4624df9b2f514d26e8dceb72586ab7de32ff7847a91034812e5f6
EXPECTED_PBC_RECEIPT_SHA256=1c0ecf41e3344fb1c7569e50bb289974b9f5d1c149ad62a14fddd8c76d51f1d2
EXPECTED_PBC_FIT_SHA256=0dea66a543573d64943ec3447682a7698b9bbeb60239b6498b81af77a756fc36
EXPECTED_OBSERVATION_SHA256=3123b32075f1c4a211d294ae07a91a70c16128154e17441e6653ccbf33f7ae49
EXPECTED_LAG_SHA256=9555fc672628a90c39708550d85153e8787ff1a3a1abb4ef99625e347ab2c4ba

if [[ ${MODE} == smoke ]]; then
    EXPECTED_ORIGINAL_MANIFEST_SHA256=ba1e5530a17ccc43bb124ccdd257f27ff16233a5647406e2d7aa4c7a0d892804
    EXPECTED_ORIGINAL_RECEIPT_SHA256=c7a4bc8c53d5365d18d77f5226abf7062d4359532b97bc016295f762ff13bc8a
    EXPECTED_PARENT_NODE=cn7
    SEEDS=42
    MAX_EPOCHS=2
    PATIENCE=1
else
    EXPECTED_ORIGINAL_MANIFEST_SHA256=da4be561bca3c0ffa1a14fcf8f28f8a77e05bb6a60240edc4a48aed65948d642
    EXPECTED_ORIGINAL_RECEIPT_SHA256=6d07680e162eb6bb330bf59ccc92d4c0b24b1228c8f2291fadd2bc7d2403eea4
    EXPECTED_PARENT_NODE=cn11
    SEEDS=42,43,44
    MAX_EPOCHS=100
    PATIENCE=15
fi

if [[ ! -x ${PYTHON_BIN} ]]; then
    echo "required s2s_core Python is unavailable: ${PYTHON_BIN}" >&2
    exit 2
fi
if [[ -e ${OUTPUT} ]]; then
    echo "fresh output path required; refusing to overwrite: ${OUTPUT}" >&2
    exit 2
fi
for executable in srun nvidia-smi; do
    if ! command -v "${executable}" >/dev/null 2>&1; then
        echo "required executable is unavailable: ${executable}" >&2
        exit 2
    fi
done
if [[ ! ${SLURM_JOB_ID:-} =~ ^[0-9]+([_][0-9]+)?$ ]]; then
    echo "invalid or missing Slurm job id: ${SLURM_JOB_ID:-unset}" >&2
    exit 2
fi
if [[ ${SLURMD_NODENAME:-} != "${EXPECTED_PARENT_NODE}" ]]; then
    echo "${MODE} must run on original-v1 node ${EXPECTED_PARENT_NODE}; found ${SLURMD_NODENAME:-unset}" >&2
    exit 2
fi

REQUIRED_SOURCES=(
    src/fuxi_allseason_persistence_mask_control.py
    src/fuxi_persistence_mask_control.py
    tests/test_allseason_persistence_mask_control.py
    src/fuxi_allseason_persistence_augmented.py
    src/fuxi_persistence_context.py
    src/fuxi_allseason_ensemble_calibration.py
    src/fuxi_ensemble_calibration_core.py
    src/fuxi_allseason_pbc_baseline.py
    src/fuxi_pbc_core.py
    src/fuxi_allseason_member_cache.py
    src/project_paths.py
    slurm/run_allseason_persistence_mask_control.sbatch
    plan/PERSISTENCE_MASK_CONTROL_ADDENDUM_20260823.md
    plan/PERSISTENCE_AUGMENTED_NEURAL_20260823.md
)
for relative in "${REQUIRED_SOURCES[@]}"; do
    if [[ ! -f ${WORK_ROOT}/${relative} ]]; then
        echo "required mask-control source/plan is missing: ${relative}" >&2
        exit 2
    fi
done
for pinned in "${PBC_MANIFEST}" "${PBC_RECEIPT}" "${PBC_FIT}"; do
    if [[ ! -f ${pinned} ]]; then
        echo "accepted PBC input is missing: ${pinned}" >&2
        exit 2
    fi
done

# Reject unsafe nesting before creating even the output parent or lock. This
# keeps invalid requests completely read-only with respect to immutable inputs.
"${PYTHON_BIN}" - "${OUTPUT}" "${ORIGINAL_MANIFEST}" "${PBC_MANIFEST}" \
    "${MASK_SMOKE_MANIFEST:-none}" <<'PY'
import sys
from pathlib import Path

output = Path(sys.argv[1]).resolve()
roots = [Path(sys.argv[2]).resolve().parent, Path(sys.argv[3]).resolve().parent]
if sys.argv[4] != "none":
    roots.append(Path(sys.argv[4]).resolve().parent)
for root in roots:
    try:
        output.relative_to(root)
    except ValueError:
        continue
    raise SystemExit(f"output may not equal or nest in immutable input: {root}")
PY

mkdir -p "${WORK_ROOT}/logs" "${OUTPUT_PARENT}"
if [[ -e ${OUTPUT} ]]; then
    echo "output appeared during preflight; refusing to overwrite: ${OUTPUT}" >&2
    exit 2
fi
JOB_TMP=$(mktemp -d "/tmp/fuxi-mask-control-${SLURM_JOB_ID}.XXXXXX")
OUTPUT_LOCK=${OUTPUT_PARENT}/.${OUTPUT_NAME}.mask-control-launch.lock
LOCK_HELD=false
cleanup() {
    if [[ -d ${JOB_TMP} && ${JOB_TMP} == /tmp/fuxi-mask-control-* ]]; then
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
MASK_SMOKE_MANIFEST_ARG=${MASK_SMOKE_MANIFEST:-none}
MASK_SMOKE_RECEIPT_ARG=${MASK_SMOKE_RECEIPT:-none}
PREFLIGHT_RECEIPT=$(
    "${PYTHON_BIN}" - "${MODE}" "${CACHE}" "${OUTPUT}" \
        "${ORIGINAL_MANIFEST}" "${ORIGINAL_RECEIPT}" \
        "${MASK_SMOKE_MANIFEST_ARG}" "${MASK_SMOKE_RECEIPT_ARG}" \
        "${PBC_MANIFEST}" "${PBC_RECEIPT}" "${PBC_FIT}" "${WORK_ROOT}" \
        "${EXPECTED_CACHE_SHA256}" "${EXPECTED_SOURCE_FINGERPRINT}" \
        "${EXPECTED_PLAN_SHA256}" "${EXPECTED_ADDENDUM_PLAN_SHA256}" \
        "${EXPECTED_ORIGINAL_MANIFEST_SHA256}" \
        "${EXPECTED_ORIGINAL_RECEIPT_SHA256}" \
        "${EXPECTED_PBC_MANIFEST_SHA256}" "${EXPECTED_PBC_RECEIPT_SHA256}" \
        "${EXPECTED_PBC_FIT_SHA256}" "${EXPECTED_OBSERVATION_SHA256}" \
        "${EXPECTED_LAG_SHA256}" "${EXPECTED_PARENT_NODE}" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

import fuxi_allseason_persistence_mask_control as driver

mode = sys.argv[1]
cache = Path(sys.argv[2]).resolve()
output = Path(sys.argv[3]).resolve()
parent_manifest_path = Path(sys.argv[4]).resolve()
parent_receipt_path = Path(sys.argv[5]).resolve()
mask_smoke_manifest_arg = sys.argv[6]
mask_smoke_receipt_arg = sys.argv[7]
pbc_manifest_path = Path(sys.argv[8]).resolve()
pbc_receipt_path = Path(sys.argv[9]).resolve()
pbc_fit_path = Path(sys.argv[10]).resolve()
work_root = Path(sys.argv[11]).resolve()
expected_cache_sha = sys.argv[12]
expected_source_fingerprint = sys.argv[13]
expected_plan_sha = sys.argv[14]
expected_addendum_plan_sha = sys.argv[15]
expected_parent_manifest_sha = sys.argv[16]
expected_parent_receipt_sha = sys.argv[17]
expected_pbc_manifest_sha = sys.argv[18]
expected_pbc_receipt_sha = sys.argv[19]
expected_pbc_fit_sha = sys.argv[20]
expected_observation_sha = sys.argv[21]
expected_lag_sha = sys.argv[22]
expected_parent_node = sys.argv[23]

def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"mask-control preflight failed: {message}")

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

require(mode in {"smoke", "full"}, "mode")
require(sha256_file(parent_manifest_path) == expected_parent_manifest_sha, "original-v1 manifest pin")
require(sha256_file(parent_receipt_path) == expected_parent_receipt_sha, "original-v1 receipt pin")
parent_manifest, parent_receipt, parent_root, _ = driver.audit_parent_envelope(
    parent_manifest_path,
    parent_receipt_path,
    smoke=mode == "smoke",
)
require(parent_receipt.get("node") == expected_parent_node, "original-v1 node pin")
require(os.environ.get("SLURMD_NODENAME") == expected_parent_node, "current physical node")
require("A30" in str(parent_receipt.get("gpu", "")).upper(), "original-v1 A30 receipt")
require(
    parent_manifest.get("software", {}).get("cuda_device") == "NVIDIA A30",
    "original-v1 CUDA model",
)

for input_root in (parent_root, pbc_manifest_path.parent):
    try:
        output.relative_to(input_root)
    except ValueError:
        pass
    else:
        require(False, f"fresh output may not modify an input result: {input_root}")

pbc = driver.audit_accepted_pbc(pbc_manifest_path, pbc_fit_path)
require(pbc["manifest_sha256"] == expected_pbc_manifest_sha, "accepted PBC manifest pin")
require(pbc["fit_sha256"] == expected_pbc_fit_sha, "accepted PBC fit pin")
require(pbc["observation_bundle_sha256"] == expected_observation_sha, "PBC observation pin")
require(pbc["persistence_lag_sha256"] == expected_lag_sha, "PBC lag pin")
require(sha256_file(pbc_receipt_path) == expected_pbc_receipt_sha, "accepted PBC receipt pin")

metadata_path = cache.with_suffix("").with_suffix(".metadata.json")
require(cache.is_file() and metadata_path.is_file(), "cache and metadata")
require(sha256_file(cache) == expected_cache_sha, "canonical cache SHA-256")
metadata = driver.load_json(metadata_path)
require(metadata.get("data_sha256") == expected_cache_sha, "cache metadata SHA")
require(metadata.get("source_fingerprint") == expected_source_fingerprint, "cache source fingerprint")
require(metadata.get("scope") == "full_archive" and metadata.get("full_archive") is True, "full cache scope")
require(metadata.get("shape") == [2080, 51, 6, 27, 27], "cache shape")

plan = work_root / "plan/PERSISTENCE_AUGMENTED_NEURAL_20260823.md"
addendum = work_root / "plan/PERSISTENCE_MASK_CONTROL_ADDENDUM_20260823.md"
require(sha256_file(plan) == expected_plan_sha, "original frozen plan")
require(sha256_file(addendum) == expected_addendum_plan_sha, "frozen addendum plan")
require(driver.ADDENDUM_PLAN_SHA256 == expected_addendum_plan_sha, "driver addendum pin")

expected_source_paths = {
    "src/fuxi_allseason_persistence_mask_control.py",
    "src/fuxi_persistence_mask_control.py",
    "tests/test_allseason_persistence_mask_control.py",
    "src/fuxi_allseason_persistence_augmented.py",
    "src/fuxi_persistence_context.py",
    "src/fuxi_allseason_ensemble_calibration.py",
    "src/fuxi_ensemble_calibration_core.py",
    "src/fuxi_allseason_pbc_baseline.py",
    "src/fuxi_pbc_core.py",
    "src/fuxi_allseason_member_cache.py",
    "src/project_paths.py",
    "slurm/run_allseason_persistence_mask_control.sbatch",
    "plan/PERSISTENCE_MASK_CONTROL_ADDENDUM_20260823.md",
    "plan/PERSISTENCE_AUGMENTED_NEURAL_20260823.md",
}
require(set(driver.SOURCE_PATHS) == expected_source_paths, "exact 14-source inventory")
snapshot = {
    f"code/{relative}": sha256_file(Path(path))
    for relative, path in driver.SOURCE_PATHS.items()
}
source_binding = driver.mapping_sha256(snapshot)
mask_smoke_manifest_sha = "none"
mask_smoke_receipt_sha = "none"
if mode == "smoke":
    require(mask_smoke_manifest_arg == mask_smoke_receipt_arg == "none", "smoke mask parent absence")
else:
    mask_smoke_manifest_path = Path(mask_smoke_manifest_arg).resolve()
    mask_smoke_receipt_path = Path(mask_smoke_receipt_arg).resolve()
    try:
        output.relative_to(mask_smoke_manifest_path.parent)
    except ValueError:
        pass
    else:
        require(False, "fresh full output may not modify mask smoke")
    original_smoke = parent_receipt.get("smoke_manifest_path")
    require(isinstance(original_smoke, str), "original full smoke ancestry")
    original_smoke_path = Path(original_smoke).resolve()
    original_smoke_receipt = original_smoke_path.parent / "slurm_gate_receipt.json"
    driver.audit_mask_smoke_parent(
        mask_smoke_manifest_path,
        mask_smoke_receipt_path,
        current_source_snapshot=snapshot,
        expected_original_v1_smoke_manifest_sha256=sha256_file(original_smoke_path),
        expected_original_v1_smoke_receipt_sha256=sha256_file(original_smoke_receipt),
        accepted_pbc=pbc,
    )
    mask_smoke_manifest_sha = sha256_file(mask_smoke_manifest_path)
    mask_smoke_receipt_sha = sha256_file(mask_smoke_receipt_path)

print(
    expected_parent_manifest_sha,
    expected_parent_receipt_sha,
    mask_smoke_manifest_sha,
    mask_smoke_receipt_sha,
    source_binding,
    expected_parent_node,
)
PY
)
read -r ORIGINAL_MANIFEST_SHA256 ORIGINAL_RECEIPT_SHA256 \
    MASK_SMOKE_MANIFEST_SHA256 MASK_SMOKE_RECEIPT_SHA256 \
    PREFLIGHT_SOURCE_BINDING_SHA256 PARENT_NODE <<<"${PREFLIGHT_RECEIPT}"
for digest in "${ORIGINAL_MANIFEST_SHA256}" "${ORIGINAL_RECEIPT_SHA256}" \
    "${PREFLIGHT_SOURCE_BINDING_SHA256}"; do
    if [[ ! ${digest} =~ ^[0-9a-f]{64}$ ]]; then
        echo "preflight returned an invalid digest: ${digest}" >&2
        exit 2
    fi
done
if [[ ${MODE} == full ]]; then
    for digest in "${MASK_SMOKE_MANIFEST_SHA256}" "${MASK_SMOKE_RECEIPT_SHA256}"; do
        if [[ ! ${digest} =~ ^[0-9a-f]{64}$ ]]; then
            echo "full preflight returned an invalid mask-smoke digest: ${digest}" >&2
            exit 2
        fi
    done
elif [[ ${MASK_SMOKE_MANIFEST_SHA256} != none || ${MASK_SMOKE_RECEIPT_SHA256} != none ]]; then
    echo "smoke preflight unexpectedly bound a mask-smoke parent" >&2
    exit 2
fi
if [[ ${PARENT_NODE} != "${EXPECTED_PARENT_NODE}" \
    || ${SLURMD_NODENAME:-} != "${EXPECTED_PARENT_NODE}" ]]; then
    echo "physical-node lineage changed after preflight" >&2
    exit 2
fi

echo "mode=${MODE} job=${SLURM_JOB_ID} node=${SLURMD_NODENAME} partition=${SLURM_JOB_PARTITION:-unknown}"
echo "original-v1 manifest=${ORIGINAL_MANIFEST} sha256=${ORIGINAL_MANIFEST_SHA256}"
if [[ ${MODE} == full ]]; then
    echo "mask-smoke manifest=${MASK_SMOKE_MANIFEST} sha256=${MASK_SMOKE_MANIFEST_SHA256}"
fi
echo "source binding=${PREFLIGHT_SOURCE_BINDING_SHA256}"
nvidia-smi
GPU_RECEIPT=$(
    nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version \
        --format=csv,noheader,nounits
)
if [[ -z ${GPU_RECEIPT} || ${GPU_RECEIPT} == *$'\n'* \
    || ${GPU_RECEIPT^^} != *A30* ]]; then
    echo "expected exactly one NVIDIA A30; found: ${GPU_RECEIPT:-none}" >&2
    exit 2
fi

"${PYTHON_BIN}" - "${ORIGINAL_MANIFEST}" "${ORIGINAL_RECEIPT}" \
    "${GPU_RECEIPT}" <<'PY'
import sys
from pathlib import Path

import torch
import fuxi_allseason_persistence_mask_control as driver

if "/s2s_core/" not in sys.executable:
    raise SystemExit(f"expected s2s_core Python, found {sys.executable}")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("expected exactly one visible CUDA GPU")
if "A30" not in torch.cuda.get_device_name(0).upper():
    raise SystemExit(f"expected NVIDIA A30, found {torch.cuda.get_device_name(0)}")
probe = torch.ones(1, device="cuda")
if not bool(torch.isfinite(probe).all()):
    raise SystemExit("CUDA tensor probe failed")
parent = driver.load_json(Path(sys.argv[1]))
receipt = driver.load_json(Path(sys.argv[2]))
current_gpu = [value.strip() for value in sys.argv[3].split(",")]
parent_gpu = [value.strip() for value in str(receipt.get("gpu", "")).split(",")]
if len(current_gpu) != 5 or len(parent_gpu) != 5:
    raise SystemExit("current/original GPU receipt schema differs")
if current_gpu[2:] != parent_gpu[2:]:
    raise SystemExit(
        f"GPU model/memory/driver differs: current={current_gpu[2:]}, "
        f"parent={parent_gpu[2:]}"
    )
evidence = driver.audit_matching_environment(
    parent, receipt, torch.device("cuda")
)
if evidence.get("matches_original") is not True:
    raise SystemExit("hardware/software lineage did not match original v1")
print("matched environment:", evidence)
PY

if [[ -e ${OUTPUT} ]]; then
    echo "output appeared before training; refusing to overwrite: ${OUTPUT}" >&2
    exit 2
fi
RUN_ARGS=(
    --cache "${CACHE}"
    --output "${OUTPUT}"
    --original-v1-manifest "${ORIGINAL_MANIFEST}"
    --original-v1-receipt "${ORIGINAL_RECEIPT}"
    --pbc-manifest "${PBC_MANIFEST}"
    --pbc-fit "${PBC_FIT}"
    --device cuda
    --seeds "${SEEDS}"
    --max-epochs "${MAX_EPOCHS}"
    --patience "${PATIENCE}"
    --batch-size 8
    --member-subsample 16
    --learning-rate 0.0002
    --weight-decay 0.0001
    --num-workers 0
    --evaluation-batch-size 8
    --cdf-chunk-size 8
)
if [[ ${MODE} == smoke ]]; then
    RUN_ARGS+=(--smoke)
else
    RUN_ARGS+=(
        --mask-smoke-manifest "${MASK_SMOKE_MANIFEST}"
        --mask-smoke-receipt "${MASK_SMOKE_RECEIPT}"
    )
fi
srun "${PYTHON_BIN}" src/fuxi_allseason_persistence_mask_control.py "${RUN_ARGS[@]}"

"${PYTHON_BIN}" - "${MODE}" "${OUTPUT}" "${CACHE}" \
    "${ORIGINAL_MANIFEST}" "${ORIGINAL_MANIFEST_SHA256}" \
    "${ORIGINAL_RECEIPT}" "${ORIGINAL_RECEIPT_SHA256}" \
    "${MASK_SMOKE_MANIFEST_ARG}" "${MASK_SMOKE_MANIFEST_SHA256}" \
    "${MASK_SMOKE_RECEIPT_ARG}" "${MASK_SMOKE_RECEIPT_SHA256}" \
    "${PBC_MANIFEST}" "${PBC_RECEIPT}" "${PBC_FIT}" "${WORK_ROOT}" \
    "${PREFLIGHT_SOURCE_BINDING_SHA256}" "${EXPECTED_PLAN_SHA256}" \
    "${EXPECTED_ADDENDUM_PLAN_SHA256}" "${EXPECTED_PBC_MANIFEST_SHA256}" \
    "${EXPECTED_PBC_RECEIPT_SHA256}" "${EXPECTED_PBC_FIT_SHA256}" \
    "${EXPECTED_OBSERVATION_SHA256}" "${EXPECTED_LAG_SHA256}" \
    "${EXPECTED_CACHE_SHA256}" "${EXPECTED_SOURCE_FINGERPRINT}" \
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
import torch

import fuxi_allseason_persistence_mask_control as driver

mode = sys.argv[1]
root = Path(sys.argv[2]).resolve()
cache = Path(sys.argv[3]).resolve()
parent_manifest_path = Path(sys.argv[4]).resolve()
expected_parent_manifest_sha = sys.argv[5]
parent_receipt_path = Path(sys.argv[6]).resolve()
expected_parent_receipt_sha = sys.argv[7]
mask_smoke_manifest_arg = sys.argv[8]
expected_mask_smoke_manifest_sha = sys.argv[9]
mask_smoke_receipt_arg = sys.argv[10]
expected_mask_smoke_receipt_sha = sys.argv[11]
pbc_manifest_path = Path(sys.argv[12]).resolve()
pbc_receipt_path = Path(sys.argv[13]).resolve()
pbc_fit_path = Path(sys.argv[14]).resolve()
work_root = Path(sys.argv[15]).resolve()
expected_source_binding = sys.argv[16]
expected_plan_sha = sys.argv[17]
expected_addendum_plan_sha = sys.argv[18]
expected_pbc_manifest_sha = sys.argv[19]
expected_pbc_receipt_sha = sys.argv[20]
expected_pbc_fit_sha = sys.argv[21]
expected_observation_sha = sys.argv[22]
expected_lag_sha = sys.argv[23]
expected_cache_sha = sys.argv[24]
expected_source_fingerprint = sys.argv[25]
job_id, node, partition, gpu = sys.argv[26:30]
manifest_path = root / "manifest.json"
receipt_path = root / "slurm_gate_receipt.json"
expected_seeds = [42] if mode == "smoke" else [42, 43, 44]
selected_counts = (
    {"train": 32, "validation": 16}
    if mode == "smoke"
    else {"train": 1652, "validation": 196}
)
validation_count = selected_counts["validation"]

def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"mask-control post-run gate failed: {message}")

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def finite_json(value: object, location: str = "root") -> None:
    if isinstance(value, float):
        require(math.isfinite(value), f"finite JSON at {location}")
    elif isinstance(value, dict):
        for key, item in value.items():
            finite_json(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            finite_json(item, f"{location}[{index}]")

def is_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )

require(mode in {"smoke", "full"}, "mode")
require(manifest_path.is_file(), "manifest exists")
require(not (root / "failure.json").exists(), "failure artifact absent")
require(not receipt_path.exists(), "external receipt does not preexist")
manifest = driver.load_json(manifest_path)
finite_json(manifest)
require(manifest.get("schema_version") == "fuxi_persistence_mask_control_manifest_v1", "manifest schema")
require(manifest.get("experiment") == driver.EXPERIMENT, "experiment")
require(manifest.get("status") == "complete", "completion")
require(manifest.get("mode") == mode, "manifest mode")
require(manifest.get("smoke") is (mode == "smoke"), "smoke flag")
require(Path(str(manifest.get("output_path", ""))).resolve() == root, "output identity")

parent_manifest, parent_receipt, parent_root, parent_artifacts = (
    driver.audit_parent_envelope(
        parent_manifest_path,
        parent_receipt_path,
        smoke=mode == "smoke",
    )
)
require(sha256_file(parent_manifest_path) == expected_parent_manifest_sha, "original-v1 manifest unchanged")
require(sha256_file(parent_receipt_path) == expected_parent_receipt_sha, "original-v1 receipt unchanged")
require(node == parent_receipt.get("node") and node != "unknown", "same physical parent node")
require(partition == "gpu_prio", "gpu_prio partition")
require("A30" in gpu.upper(), "A30 nvidia-smi receipt")
current_gpu_parts = [value.strip() for value in gpu.split(",")]
parent_gpu_parts = [
    value.strip() for value in str(parent_receipt.get("gpu", "")).split(",")
]
require(
    len(current_gpu_parts) == len(parent_gpu_parts) == 5
    and current_gpu_parts[2:] == parent_gpu_parts[2:],
    "GPU model/memory/driver exact parent match",
)
require(str(job_id).replace("_", "").isdigit(), "Slurm job identity")
environment = driver.audit_matching_environment(
    parent_manifest, parent_receipt, torch.device("cuda")
)
require(environment.get("matches_original") is True, "matching environment")
require(manifest.get("hardware_software_match") == environment, "manifest environment receipt")

pbc = driver.audit_accepted_pbc(pbc_manifest_path, pbc_fit_path)
require(pbc["manifest_sha256"] == expected_pbc_manifest_sha, "PBC manifest pin")
require(pbc["fit_sha256"] == expected_pbc_fit_sha, "PBC fit pin")
require(pbc["observation_bundle_sha256"] == expected_observation_sha, "PBC observation pin")
require(pbc["persistence_lag_sha256"] == expected_lag_sha, "PBC lag pin")
require(sha256_file(pbc_receipt_path) == expected_pbc_receipt_sha, "PBC receipt pin")
require(manifest.get("accepted_pbc") == pbc, "manifest PBC binding")

metadata_path = cache.with_suffix("").with_suffix(".metadata.json")
metadata = driver.load_json(metadata_path)
require(sha256_file(cache) == expected_cache_sha, "cache unchanged")
require(metadata.get("data_sha256") == expected_cache_sha, "cache metadata hash")
require(metadata.get("source_fingerprint") == expected_source_fingerprint, "cache fingerprint")
require(manifest.get("cache", {}).get("actual_data_sha256") == expected_cache_sha, "manifest cache hash")

current_snapshot = {
    f"code/{relative}": sha256_file(Path(path))
    for relative, path in driver.SOURCE_PATHS.items()
}
require(len(current_snapshot) == 14, "exact 14-source snapshot")
require(manifest.get("source_snapshot_sha256") == current_snapshot, "source snapshot")
require(manifest.get("source_binding_sha256") == expected_source_binding == driver.mapping_sha256(current_snapshot), "source binding")
artifacts = driver.verify_artifact_inventory(root, manifest)
require(manifest.get("artifact_binding_sha256") == driver.mapping_sha256(artifacts), "artifact binding")
for relative, digest in current_snapshot.items():
    require(artifacts.get(relative) == digest, f"source artifact binding {relative}")
    require(sha256_file(root / relative) == digest, f"source snapshot bytes {relative}")

plan = manifest.get("addendum_plan", {})
plan_snapshot = "code/plan/PERSISTENCE_MASK_CONTROL_ADDENDUM_20260823.md"
require(
    plan.get("expected_sha256") == plan.get("live_sha256")
    == plan.get("snapshot_sha256") == expected_addendum_plan_sha,
    "addendum plan binding",
)
require(plan.get("snapshot_path") == plan_snapshot, "addendum snapshot path")
require(current_snapshot.get(plan_snapshot) == expected_addendum_plan_sha, "addendum source hash")
require(
    current_snapshot.get("code/plan/PERSISTENCE_AUGMENTED_NEURAL_20260823.md")
    == expected_plan_sha,
    "original plan hash",
)

required = {
    "README.md",
    "joint_selection.json",
    "history/training_history.csv",
    "evaluation/observation_bundle_provenance.json",
    "evaluation/persistence_lag_provenance.json",
    "evaluation/persistence_context_provenance.json",
    "evaluation/base_context_and_weekly_climatology_provenance.json",
    "evaluation/mask_only_context_provenance.json",
    "evaluation/scoring_support.npz",
    "evaluation/persistence_normalization.npz",
    "evaluation/validation_quintile_fit.npz",
    "metrics/original_validation_case_metrics.csv",
    "metrics/original_validation_case_metrics_reproduced.csv",
    "metrics/mask_validation_case_metrics.csv",
    "metrics/joint_validation_case_metrics.csv",
    "metrics/joint_validation_summary.csv",
    "metrics/joint_validation_by_year.csv",
    "metrics/joint_validation_by_seed.csv",
    "metrics/joint_validation_by_lead.csv",
}
for seed in expected_seeds:
    required.add(f"models/mask_only_45k/seed_{seed}/checkpoints/best.pt")
    required.add(f"models/mask_only_45k/seed_{seed}/validation_adjustments.npz")
require(required <= set(artifacts), "required artifact inventory")

original = manifest.get("original_v1", {})
require(Path(str(original.get("manifest_path", ""))).resolve() == parent_manifest_path, "original manifest path")
require(original.get("manifest_sha256") == expected_parent_manifest_sha, "original manifest hash")
require(Path(str(original.get("receipt_path", ""))).resolve() == parent_receipt_path, "original receipt path")
require(original.get("receipt_sha256") == expected_parent_receipt_sha, "original receipt hash")
require(original.get("source_snapshot_sha256") == parent_manifest.get("source_snapshot_sha256"), "original source map")
require(original.get("artifact_sha256") == parent_artifacts, "original artifact map")

mask_smoke_binding = None
if mode == "smoke":
    require(manifest.get("mask_smoke_parent") is None, "smoke has no mask-smoke parent")
    require(mask_smoke_manifest_arg == mask_smoke_receipt_arg == "none", "smoke parent args")
    require(expected_mask_smoke_manifest_sha == expected_mask_smoke_receipt_sha == "none", "smoke parent digests")
else:
    mask_smoke_manifest_path = Path(mask_smoke_manifest_arg).resolve()
    mask_smoke_receipt_path = Path(mask_smoke_receipt_arg).resolve()
    require(sha256_file(mask_smoke_manifest_path) == expected_mask_smoke_manifest_sha, "mask smoke manifest unchanged")
    require(sha256_file(mask_smoke_receipt_path) == expected_mask_smoke_receipt_sha, "mask smoke receipt unchanged")
    original_smoke = original.get("smoke_parent", {})
    require(isinstance(original_smoke, dict), "original-v1 smoke ancestry")
    mask_smoke_binding = driver.audit_mask_smoke_parent(
        mask_smoke_manifest_path,
        mask_smoke_receipt_path,
        current_source_snapshot=current_snapshot,
        expected_original_v1_smoke_manifest_sha256=str(original_smoke.get("manifest_sha256")),
        expected_original_v1_smoke_receipt_sha256=str(original_smoke.get("receipt_sha256")),
        accepted_pbc=pbc,
    )
    require(manifest.get("mask_smoke_parent") == mask_smoke_binding, "full mask-smoke binding")

contract = manifest.get("contract", {})
require(contract.get("train_years") == list(range(2002, 2018)), "train years")
require(contract.get("validation_years") == [2018, 2019], "validation years")
require(contract.get("test_metrics_consulted") is False, "test firewall")
require(contract.get("development_years_not_scored") == [2020, 2021], "development firewall")
require(contract.get("sealed_2025_target_opened") is False, "sealed 2025")
require(contract.get("mask_only_selectable") is False, "mask-only nonselectable")
require(manifest.get("seeds") == expected_seeds, "seed inventory")
require(manifest.get("split_counts_selected") == selected_counts, "selected split counts")

selection_path = root / str(manifest.get("joint_selection_artifact"))
selection = driver.load_json(selection_path)
require(selection == manifest.get("joint_selection"), "joint selection artifact")
require(sha256_file(selection_path) == manifest.get("joint_selection_sha256"), "joint selection hash")
require(selection.get("status") == driver.JOINT_SELECTION_STATUS, "joint selection lock")
require(selection.get("scientific_selection") is (mode == "full"), "selection scientific flag")
require(selection.get("test_metrics_consulted") is False, "selection test firewall")
require(selection.get("development_indices_accessed") is False, "selection development firewall")
require(selection.get("mask_only_selectable") is False, "selection mask nonselectability")
require(selection.get("selected_arm") in {"base_42k", "persistence_lag12_45k"}, "selected arm")

parent_metrics = pd.read_csv(root / "metrics/original_validation_case_metrics.csv")
reproduced_parent = pd.read_csv(root / "metrics/original_validation_case_metrics_reproduced.csv")
mask_metrics = pd.read_csv(root / "metrics/mask_validation_case_metrics.csv")
joint_metrics = pd.read_csv(root / "metrics/joint_validation_case_metrics.csv")
expected_parent_rows = 3 * len(expected_seeds) * validation_count * 6
expected_mask_rows = len(expected_seeds) * validation_count * 6
require(len(parent_metrics) == len(reproduced_parent) == expected_parent_rows, "parent validation rows")
require(len(mask_metrics) == expected_mask_rows, "mask validation rows")
require(len(joint_metrics) == expected_parent_rows + expected_mask_rows, "joint validation rows")
require(set(mask_metrics.arm) == {"mask_only_45k"}, "mask metric arm")
require(set(mask_metrics.seed.astype(int)) == set(expected_seeds), "mask metric seeds")
require(set(mask_metrics.year.astype(int)) == {2018, 2019}, "mask metric years")
require(set(mask_metrics.lead_week.astype(int)) == set(range(1, 7)), "mask metric leads")
require(not mask_metrics.duplicated(["arm", "seed", "init", "lead_week"]).any(), "mask metric keys")
require(np.isfinite(mask_metrics[["crps", "quintile_rps"]].to_numpy(dtype=np.float64)).all(), "finite mask scores")
bool_columns = {
    "cdf_shape_matches_thresholds",
    "forecast_cdf_finite_where_threshold_defined",
    "observed_cdf_finite_where_threshold_defined",
    "forecast_cdf_valid_for_thresholds",
    "observed_cdf_valid_for_thresholds",
}
require(bool_columns <= set(mask_metrics), "CDF evidence columns")
for column in bool_columns:
    values = mask_metrics[column].astype(str).str.lower()
    require(set(values) == {"true"}, f"CDF evidence {column}")
require(not joint_metrics.init.astype(str).str.startswith(("2020-", "2021-", "2022-", "2023-", "2024-", "2025-")).any(), "validation-only date firewall")

expected_dates = manifest.get("retained_initializations", {}).get("validation", [])
recomputed = driver.select_joint_arm(
    parent_metrics,
    mask_metrics,
    expected_seeds=expected_seeds,
    expected_initializations=expected_dates,
)
stored_stable = {
    key: value
    for key, value in selection.items()
    if key not in {"written_utc", "scientific_selection"}
}
require(recomputed == stored_stable, "joint selection recomputation")

reproduction = manifest.get("validation_reproduction", {})
require(reproduction.get("status") == "all_original_validation_artifacts_reproduced", "parent validation reproduction")
require(reproduction.get("all_available_parent_checkpoints_reproduced") is True, "checkpoint reproduction")
require(reproduction.get("all_adjustment_arrays_reproduced_byte_exact") is True, "adjustment reproduction")
require(reproduction.get("original_selection_recomputed_exact") is True, "original selection reproduction")
require(reproduction.get("persisted_score_reproduction", {}).get("passes") is True, "score reproduction")

training = manifest.get("training", {})
require(training.get("same_hyperparameters_as_original_v1") is True, "training recipe")
require(training.get("same_seed_initial_state_as_original_45k_arms") is True, "initial state gate")
require(training.get("same_seed_stochastic_training_stream_as_all_original_arms") is True, "RNG stream gate")
runs = training.get("runs", [])
require(isinstance(runs, list) and len(runs) == len(expected_seeds), "training run count")
parent_runs = {
    (str(row.get("arm")), int(row.get("seed", -1))): row
    for row in parent_manifest.get("training", {}).get("runs", [])
}
run_artifacts = []
for row in runs:
    seed = int(row.get("seed", -1))
    require(row.get("arm") == "mask_only_45k" and seed in expected_seeds, "run identity")
    require(row.get("parameter_count") == 45026 and row.get("context_channels") == 11, "run architecture")
    require(row.get("same_seed_initial_state_as_original_45k_arms") is True, "run initial state")
    require(row.get("same_seed_training_rng_as_all_original_arms") is True, "run RNG")
    parent_rng_receipts = {
        parent_runs[(arm, seed)].get("training_rng_state_sha256")
        for arm in ("base_42k", "zero_lag_45k", "persistence_lag12_45k")
    }
    require(
        len(parent_rng_receipts) == 1
        and row.get("training_rng_state_sha256") in parent_rng_receipts,
        f"run RNG receipt exact parent equality {seed}",
    )
    parent_45k_initial_states = {
        parent_runs[(arm, seed)].get("initial_state_sha256")
        for arm in ("zero_lag_45k", "persistence_lag12_45k")
    }
    require(
        len(parent_45k_initial_states) == 1
        and row.get("initial_state_sha256") in parent_45k_initial_states,
        f"run 45k initialization exact parent equality {seed}",
    )
    for path_key, hash_key in (("checkpoint", "checkpoint_sha256"), ("validation_adjustment", "validation_adjustment_sha256")):
        relative = str(row.get(path_key))
        digest = row.get(hash_key)
        require(is_digest(digest) and artifacts.get(relative) == digest, f"run artifact binding {seed}/{path_key}")
        require(sha256_file(root / relative) == digest, f"run artifact bytes {seed}/{path_key}")
    run_artifacts.append({
        "seed": seed,
        "checkpoint": row["checkpoint"],
        "checkpoint_sha256": row["checkpoint_sha256"],
        "validation_adjustment": row["validation_adjustment"],
        "validation_adjustment_sha256": row["validation_adjustment_sha256"],
        "initial_state_sha256": row["initial_state_sha256"],
        "training_rng_state_sha256": row["training_rng_state_sha256"],
    })

require(manifest.get("temporal_evidence", {}).get("development_indices_accessed") is False, "temporal development firewall")
require(manifest.get("observation_bundle_provenance", {}).get("sha256") == expected_observation_sha, "observation provenance")
require(manifest.get("persistence_lag_provenance", {}).get("sha256") == expected_lag_sha, "lag provenance")
base_context = manifest.get("base_context_and_weekly_climatology_provenance", {})
require(is_digest(base_context.get("sha256")), "base context/climatology provenance")
mask_context = manifest.get("mask_only_context_provenance", {})
mask_context_artifact = driver.load_json(
    root / "evaluation/mask_only_context_provenance.json"
)
require(mask_context_artifact == mask_context, "mask-only context provenance artifact")
require(
    mask_context.get("schema_version") == "named_c_order_array_bundle_sha256_v1"
    and mask_context.get("bundle_name") == "mask_only_added_issue_time_fields"
    and is_digest(mask_context.get("sha256")),
    "mask-only context provenance identity",
)
mask_context_arrays = mask_context.get("arrays", {})
require(
    isinstance(mask_context_arrays, dict)
    and set(mask_context_arrays)
    == {
        "forced_zero_normalized_rain",
        "support_masked_lag_available",
        "added_channel_names",
    },
    "mask-only context field inventory",
)
require(
    mask_context_arrays["forced_zero_normalized_rain"].get("dtype") == "<f4"
    and mask_context_arrays["support_masked_lag_available"].get("dtype") == "<f4"
    and mask_context_arrays["forced_zero_normalized_rain"].get("shape")
    == mask_context_arrays["support_masked_lag_available"].get("shape")
    and mask_context_arrays["added_channel_names"].get("shape") == [4],
    "mask-only zero-rain/true-mask geometry",
)
categorical = manifest.get("categorical_validation", {}).get("finite_and_cdf_validity_evidence", {})
for key in (
    "every_scoring_chunk_shape_matched_thresholds",
    "every_forecast_cdf_finite_where_threshold_defined",
    "every_observed_cdf_finite_where_threshold_defined",
    "every_forecast_cdf_valid_for_thresholds",
    "every_observed_cdf_valid_for_thresholds",
    "every_validation_score_finite",
):
    require(categorical.get(key) is True, f"categorical gate {key}")

evaluation = manifest.get("evaluation", {})
for path_key, hash_key in (
    ("mask_validation_case_metrics_artifact", "mask_validation_case_metrics_sha256"),
    ("joint_validation_case_metrics_artifact", "joint_validation_case_metrics_sha256"),
    ("scoring_support_artifact", "scoring_support_sha256"),
):
    relative = str(evaluation.get(path_key))
    require(artifacts.get(relative) == evaluation.get(hash_key), f"evaluation artifact {path_key}")
require(evaluation.get("persisted_metrics_recompute_joint_selection_exact") is True, "persisted selection gate")

source_binding = driver.mapping_sha256(current_snapshot)
artifact_binding = driver.mapping_sha256(artifacts)
launcher_path = work_root / "slurm/run_allseason_persistence_mask_control.sbatch"
receipt = {
    "schema_version": driver.RECEIPT_SCHEMA,
    "experiment": driver.EXPERIMENT,
    "status": "post_run_audit_passed",
    "mode": mode,
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "output_path": str(root),
    "manifest_path": str(manifest_path),
    "manifest_sha256": sha256_file(manifest_path),
    "job_id": job_id,
    "node": node,
    "partition": partition,
    "gpu": gpu,
    "current_gpu_receipt": gpu,
    "parent_gpu_receipt": parent_receipt.get("gpu"),
    "gpu_model_memory_driver_exact": True,
    "cuda_device": manifest.get("software", {}).get("cuda_device"),
    "mask_smoke_manifest_path": None if mode == "smoke" else str(Path(mask_smoke_manifest_arg).resolve()),
    "mask_smoke_manifest_sha256": None if mode == "smoke" else expected_mask_smoke_manifest_sha,
    "mask_smoke_receipt_path": None if mode == "smoke" else str(Path(mask_smoke_receipt_arg).resolve()),
    "mask_smoke_receipt_sha256": None if mode == "smoke" else expected_mask_smoke_receipt_sha,
    "original_v1_manifest_path": str(parent_manifest_path),
    "original_v1_manifest_sha256": expected_parent_manifest_sha,
    "original_v1_receipt_path": str(parent_receipt_path),
    "original_v1_receipt_sha256": expected_parent_receipt_sha,
    "original_v1_smoke_parent": original.get("smoke_parent"),
    "accepted_pbc_manifest_sha256": expected_pbc_manifest_sha,
    "accepted_pbc_slurm_receipt_sha256": expected_pbc_receipt_sha,
    "accepted_pbc_fit_sha256": expected_pbc_fit_sha,
    "cache_sha256": expected_cache_sha,
    "observation_bundle_sha256": expected_observation_sha,
    "persistence_lag_sha256": expected_lag_sha,
    "persistence_context_sha256": manifest.get("persistence_context_provenance", {}).get("sha256"),
    "base_context_and_weekly_climatology_sha256": base_context.get("sha256"),
    "mask_only_context_provenance_sha256": mask_context.get("sha256"),
    "mask_only_context_contract": {
        "normalized_rainfall_channels_forced_exact_zero": True,
        "availability_channels_are_issue_safe_support_masked_true_masks": True,
        "added_channel_order": [
            "lag_1week_log1p_z_forced_zero",
            "lag_2week_log1p_z_forced_zero",
            "lag_1week_available",
            "lag_2week_available",
        ],
    },
    "source_snapshot_sha256": current_snapshot,
    "source_binding_sha256": source_binding,
    "artifact_sha256": artifacts,
    "artifact_binding_sha256": artifact_binding,
    "launcher_path": str(launcher_path),
    "launcher_sha256": sha256_file(launcher_path),
    "addendum_plan_binding": plan,
    "joint_selection_path": str(selection_path),
    "joint_selection_sha256": manifest.get("joint_selection_sha256"),
    "joint_selected_arm": selection.get("selected_arm"),
    "mask_validation_case_metrics_sha256": evaluation.get("mask_validation_case_metrics_sha256"),
    "joint_validation_case_metrics_sha256": evaluation.get("joint_validation_case_metrics_sha256"),
    "training_run_artifacts": run_artifacts,
    "hardware_software_match": environment,
    "firewalls": {
        "test_metrics_consulted": False,
        "development_indices_accessed": False,
        "mask_only_selectable": False,
        "sealed_2025_target_opened": False,
        "same_physical_parent_node": True,
        "nvidia_a30": True,
        "automatic_mixed_precision": True,
        "deterministic_recipe": True,
    },
}
require(receipt["source_binding_sha256"] == expected_source_binding, "receipt source binding")
with receipt_path.open("x", encoding="utf-8") as stream:
    json.dump(receipt, stream, indent=2, sort_keys=True)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
print(receipt_path)
PY

echo "complete and post-run audited: ${OUTPUT}"
