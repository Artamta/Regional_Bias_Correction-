#!/usr/bin/env bash
#SBATCH --job-name=rawid-2025-preflight
#SBATCH --partition=gpu_prio
#SBATCH --exclude=cn2,cn3,cn4,cn15,cn16,cn17
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/logs/%x-%j.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/logs/%x-%j.err

set -euo pipefail

EVAL_ROOT=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction
NEURAL_SRC=/home/raj.ayush/s2s/s2s_anlysis/clean/neural_adapter/src
PYTHON_BIN=/home/raj.ayush/.conda/envs/s2s_core/bin/python
PREFLIGHT_TMP=$(mktemp -d /tmp/rawid-2025-preflight.XXXXXX)

cleanup() {
    rm -rf "${PREFLIGHT_TMP}"
}
trap cleanup EXIT

mkdir -p "${EVAL_ROOT}/logs" "${PREFLIGHT_TMP}/cuda"
export PYTHONPATH=${EVAL_ROOT}/src:${EVAL_ROOT}/evaluate:${EVAL_ROOT}/hpc_compat:${NEURAL_SRC}
export CUDA_CACHE_PATH=${PREFLIGHT_TMP}/cuda
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

cd "${EVAL_ROOT}"

echo "host=$(hostname) partition=${SLURM_JOB_PARTITION} cpus=${SLURM_CPUS_PER_TASK}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# Only --selection-manifest is accepted. The canonical proof path comes from
# that sealed selection; no data, ledger, result, or proof path is caller-set.
srun "${PYTHON_BIN}" evaluate/preflight_raw_identity_2025.py "$@"
