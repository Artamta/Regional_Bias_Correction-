#!/usr/bin/env bash
#SBATCH --job-name=imd-freeze-test
#SBATCH --partition=GPU-AI_prio
# Known avoid-list: cn3 is MIG; cn2, cn15, and cn17 are unreliable here.
#SBATCH --exclude=cn2,cn3,cn15,cn17
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/logs/%x-%j.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/logs/%x-%j.err

set -euo pipefail

WORK_ROOT=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction
NEURAL_SRC=/home/raj.ayush/s2s/s2s_anlysis/clean/neural_adapter/src
PYTHON_BIN=/home/raj.ayush/.conda/envs/s2s_core/bin/python
PHYSICAL_CACHE=${WORK_ROOT}/cache/fuxi_physical_weekly_2020_2021_jjas_exploratory_v2.npz
JOB_TMP=/tmp/imd-freeze-test-${SLURM_JOB_ID}

if [[ $# -lt 3 ]]; then
    echo "usage: sbatch $0 CONFIRM_SWEEP SELECTION_MANIFEST FRESH_OUTPUT_DIRECTORY [extra evaluator args]" >&2
    exit 2
fi

CONFIRM_SWEEP=$(readlink -f "$1")
SELECTION_PARENT=$(readlink -f "$(dirname "$2")")
SELECTION_MANIFEST=${SELECTION_PARENT}/$(basename "$2")
OUTPUT_PARENT=$(readlink -f "$(dirname "$3")")
OUTPUT_DIRECTORY=${OUTPUT_PARENT}/$(basename "$3")
shift 3

if [[ ! -d "${CONFIRM_SWEEP}" ]]; then
    echo "completed confirmation sweep does not exist: ${CONFIRM_SWEEP}" >&2
    exit 2
fi
if [[ -e "${SELECTION_MANIFEST}" ]]; then
    echo "fresh selection manifest is required: ${SELECTION_MANIFEST}" >&2
    exit 2
fi
if [[ -e "${OUTPUT_DIRECTORY}" ]]; then
    echo "fresh output is required: ${OUTPUT_DIRECTORY}" >&2
    exit 2
fi
if [[ ! -f "${PHYSICAL_CACHE}" ]]; then
    echo "finalized 2020-2021 physical cache is missing: ${PHYSICAL_CACHE}" >&2
    exit 2
fi

cleanup() {
    rm -rf "${JOB_TMP}"
}
trap cleanup EXIT

mkdir -p "${WORK_ROOT}/logs" "${JOB_TMP}/mpl" "${JOB_TMP}/xdg" "${JOB_TMP}/cuda"
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

echo "host=$(hostname) job=${SLURM_JOB_ID} cpus=${SLURM_CPUS_PER_TASK}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# This first command can read only completed validation artifacts.  The second
# command is the first point at which the evaluator is allowed to open the
# exploratory 2020-2021 forecasts and IMD targets.
srun "${PYTHON_BIN}" evaluate/evaluate_locked_physical_hindcast.py freeze \
    "${CONFIRM_SWEEP}" \
    "${SELECTION_MANIFEST}"
srun "${PYTHON_BIN}" evaluate/evaluate_locked_physical_hindcast.py evaluate \
    "${SELECTION_MANIFEST}" \
    "${OUTPUT_DIRECTORY}" \
    --physical-cache "${PHYSICAL_CACHE}" \
    "$@"
