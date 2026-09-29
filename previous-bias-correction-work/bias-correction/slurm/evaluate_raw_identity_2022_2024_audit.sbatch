#!/usr/bin/env bash
#SBATCH --job-name=imd-rawid-audit
#SBATCH --partition=gpu_prio
# cn3 is MIG; cn2, cn4, cn15, cn16, and cn17 are unreliable for S2S work.
#SBATCH --exclude=cn2,cn3,cn4,cn15,cn16,cn17
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/logs/%x-%j.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/logs/%x-%j.err

set -euo pipefail
umask 027

if [[ $# -ne 1 ]]; then
    echo "usage: sbatch $0 /absolute/path/to/fresh-output-directory" >&2
    exit 2
fi
if [[ $1 != /* ]]; then
    echo "output must be an absolute path: $1" >&2
    exit 2
fi

WORK_ROOT=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction
NEURAL_SRC=/home/raj.ayush/s2s/s2s_anlysis/clean/neural_adapter/src
PYTHON_BIN=/home/raj.ayush/.conda/envs/s2s_core/bin/python
OUTPUT=$1
JOB_TMP=/tmp/imd-rawid-audit-${SLURM_JOB_ID}

if [[ -e ${OUTPUT} ]]; then
    echo "fresh output path required: ${OUTPUT}" >&2
    exit 2
fi

cleanup() {
    rm -rf "${JOB_TMP}"
}
trap cleanup EXIT

mkdir -p \
    "${WORK_ROOT}/logs" \
    "${JOB_TMP}/mpl" \
    "${JOB_TMP}/xdg" \
    "${JOB_TMP}/cuda"
export PYTHONPATH=${WORK_ROOT}/src:${WORK_ROOT}/evaluate:${WORK_ROOT}/hpc_compat:${NEURAL_SRC}
export MPLCONFIGDIR=${JOB_TMP}/mpl
export XDG_CACHE_HOME=${JOB_TMP}/xdg
export CUDA_CACHE_PATH=${JOB_TMP}/cuda
export TMPDIR=${JOB_TMP}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONUNBUFFERED=1

cd "${WORK_ROOT}"

# Hash/selection/checkpoint preflight is read-only and never opens a 2025 store.
"${PYTHON_BIN}" evaluate/evaluate_raw_identity_2022_2024_audit.py --preflight-only

echo "job=${SLURM_JOB_ID} node=${SLURMD_NODENAME:-unknown} partition=${SLURM_JOB_PARTITION:-unknown}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
"${PYTHON_BIN}" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("Slurm assigned no visible CUDA GPU")
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
PY

srun "${PYTHON_BIN}" evaluate/evaluate_raw_identity_2022_2024_audit.py \
    --require-cuda \
    --bootstrap-draws 10000 \
    --bootstrap-block-length 13 \
    --bootstrap-seed 20260822 \
    --output "${OUTPUT}"

"${PYTHON_BIN}" - "${OUTPUT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest = json.loads((root / "manifest.json").read_text())
assert manifest["status"] == "complete"
assert manifest["canonical"] is True
assert manifest["scientific_eligible"] is True
assert manifest["smoke"] is False
assert manifest["audit_years"] == [2022, 2023, 2024]
assert manifest["final_2025_store_opened"] is False
assert manifest["methods"][-2:] == [
    "raw_identity",
    "raw_identity_raw_mean_preserved",
]
bootstrap = manifest["bootstrap"]
assert bootstrap["draws"] == 10_000
assert bootstrap["block_length_initializations"] == 13
assert bootstrap["seed"] == 20_260_822
assert bootstrap["canonical_contract"] == {
    "draws": 10_000,
    "block_length_initializations": 13,
    "seed": 20_260_822,
}
diagnostics = bootstrap["diagnostics"]
assert diagnostics["year_stratified"] is True
assert diagnostics["circular_within_year"] is True
assert diagnostics["equal_marginal_inclusion_by_design"] is True
assert diagnostics["no_year_crossing"] is True
assert diagnostics["all_six_leads_retained_per_start"] is True
assert abs(diagnostics["mean_multiplicity_across_initializations"] - 1.0) <= 1.0e-12
projection = manifest["projection"]
assert (
    projection["weighting_contract"]
    == "fixed_forecast_time_india_area_x_frozen_adapter_support"
)
assert projection["uses_weekly_imd_coverage"] is False
assert projection["maximum_float32_closure_mm_day"] <= 2.0e-6
print(root / "manifest.json")
PY

echo "complete: retrospective 2022-2024 development audit only; 2025 remained unopened"
