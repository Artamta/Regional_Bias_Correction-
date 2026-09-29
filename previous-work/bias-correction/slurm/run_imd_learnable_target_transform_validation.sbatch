#!/usr/bin/env bash
#SBATCH --job-name=imd-learn-xform
#SBATCH --partition=GPU-AI_prio
# Known avoid-list: cn3 is MIG; cn2/cn15/cn17 and gpu2 are unreliable here.
#SBATCH --exclude=cn2,cn3,cn15,cn17,gpu2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/logs/%x-%j.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/logs/%x-%j.err

set -euo pipefail

SWEEP_ROOT=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction
NEURAL_SRC=/home/raj.ayush/s2s/s2s_anlysis/clean/neural_adapter/src
PYTHON_BIN=/home/raj.ayush/.conda/envs/s2s_core/bin/python

if [[ $# -lt 1 ]]; then
    echo "usage: sbatch $0 FRESH_OUTPUT_DIRECTORY [driver arguments]" >&2
    exit 2
fi

OUTPUT_PARENT=$(readlink -f "$(dirname "$1")")
OUTPUT_DIRECTORY=${OUTPUT_PARENT}/$(basename "$1")
shift

if [[ -e "${OUTPUT_DIRECTORY}" ]]; then
    echo "fresh output is required; refusing existing path: ${OUTPUT_DIRECTORY}" >&2
    exit 2
fi

JOB_TMP=$(mktemp -d "/tmp/imd-learn-xform-${SLURM_JOB_ID}-XXXXXX")
cleanup() {
    rm -rf "${JOB_TMP}"
}
trap cleanup EXIT

mkdir -p "${SWEEP_ROOT}/logs" "${JOB_TMP}/mpl" "${JOB_TMP}/xdg" "${JOB_TMP}/cuda"
export PYTHONPATH=${SWEEP_ROOT}/src:${SWEEP_ROOT}/evaluate:${SWEEP_ROOT}/hpc_compat:${NEURAL_SRC}
export MPLCONFIGDIR=${JOB_TMP}/mpl
export XDG_CACHE_HOME=${JOB_TMP}/xdg
export CUDA_CACHE_PATH=${JOB_TMP}/cuda
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1

cd "${SWEEP_ROOT}"

echo "job=${SLURM_JOB_ID} node=${SLURMD_NODENAME:-unknown} partition=${SLURM_JOB_PARTITION:-unknown} output=${OUTPUT_DIRECTORY}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
srun "${PYTHON_BIN}" src/fuxi_imd_learnable_target_transform_validation.py \
    "$@" \
    --output "${OUTPUT_DIRECTORY}" \
    --workers 2
