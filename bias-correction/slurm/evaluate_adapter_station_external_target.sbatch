#!/usr/bin/env bash
#SBATCH --job-name=adapter-station-e3
# The local iiser partition is restricted to the `limit` account.  This
# account must use the general GPU-node partition even for a CPU-only task.
#SBATCH --partition=gpu
# No GPU is requested; avoid the cluster's known unsafe/unreliable nodes.
#SBATCH --exclude=cn2,cn3,cn4,cn15,cn16,cn17
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/logs/%x-%j.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/logs/%x-%j.err

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: sbatch $0 COMPLETED_E2_ROOT FRESH_E3_OUTPUT" >&2
    exit 2
fi

WORK_ROOT=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction
PYTHON_BIN=/home/raj.ayush/.conda/envs/s2s_core/bin/python
EXTENDED_ROOT=$(readlink -f "$1")
OUTPUT=$(readlink -m "$2")

if [[ ! -f "${EXTENDED_ROOT}/manifest.json" || ! -d "${EXTENDED_ROOT}/predictions.zarr" ]]; then
    echo "completed E2 root must contain manifest.json and predictions.zarr: ${EXTENDED_ROOT}" >&2
    exit 2
fi
if [[ -e "${OUTPUT}" ]]; then
    echo "fresh E3 output required: ${OUTPUT}" >&2
    exit 2
fi

JOB_TMP=$(mktemp -d "/tmp/adapter-station-e3-${SLURM_JOB_ID}-XXXXXX")
cleanup() {
    rm -rf "${JOB_TMP}"
}
trap cleanup EXIT

mkdir -p "${WORK_ROOT}/logs"
export PYTHONPATH=${WORK_ROOT}/src:${WORK_ROOT}/evaluate:${WORK_ROOT}/hpc_compat
export TMPDIR=${JOB_TMP}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

cd "${WORK_ROOT}"

echo "job=${SLURM_JOB_ID} node=${SLURMD_NODENAME:-unknown} partition=${SLURM_JOB_PARTITION:-unknown}"
echo "completed_e2=${EXTENDED_ROOT}"
echo "fresh_e3_output=${OUTPUT}"

srun "${PYTHON_BIN}" evaluate/evaluate_adapter_station_external_target.py \
    --extended-predictions "${EXTENDED_ROOT}" \
    --require-extended \
    --output "${OUTPUT}"

"${PYTHON_BIN}" - "${OUTPUT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
assert manifest["status"] == "complete_frozen_external_target_sensitivity"
assert manifest["canonical_artifact"] is True
assert manifest["execution_tier"] == "canonical_five_method_e3"
assert manifest["dates"]["initialization_years"] == [2024]
assert manifest["dates"]["case_leads"] == 180
assert manifest["2025_prediction_opened"] is False
assert manifest["2025_station_value_selected"] is False
assert manifest["2025_metric_computed"] is False
assert manifest["methods"] == [
    "raw_fuxi",
    "log_bias",
    "selected_adapter",
    "raw_identity",
    "raw_identity_raw_mean_preserved",
]
assert manifest["primary_estimand"]["circular_block_length_initializations"] == 13
assert manifest["primary_estimand"]["bootstrap_draws"] == 2000
assert manifest["extended_projection_verified_without_target_coverage"] is True
assert manifest["extended_prediction_canonical_contract_verified"] is True
assert manifest["bootstrap"]["equal_marginal_date_inclusion"] is True
assert abs(manifest["bootstrap"]["date_multiplicity"]["13"]["mean"] - 1.0) < 1e-12
print(root / "manifest.json")
PY

echo "E3 external-target robustness evaluation complete; no 2025 target was scored"
