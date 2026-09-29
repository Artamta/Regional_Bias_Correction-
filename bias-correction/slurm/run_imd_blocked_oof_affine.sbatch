#!/usr/bin/env bash
#SBATCH --job-name=imd-oof-affine
#SBATCH --partition=gpu
# cn3 is MIG; cn2/cn15/cn17 are unreliable. gpu2 is unreliable on GPU-AI.
#SBATCH --exclude=cn2,cn3,cn15,cn17,gpu2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/logs/%x-%j.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/logs/%x-%j.err

set -euo pipefail

EXPERIMENT_ROOT=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction
NEURAL_SRC=/home/raj.ayush/s2s/s2s_anlysis/clean/neural_adapter/src
PYTHON_BIN=/home/raj.ayush/.conda/envs/s2s_core/bin/python
JOB_TMP=/tmp/imd-oof-affine-${SLURM_JOB_ID}

cleanup() {
    rm -rf "${JOB_TMP}"
}
trap cleanup EXIT

mkdir -p "${EXPERIMENT_ROOT}/logs" "${JOB_TMP}/mpl" "${JOB_TMP}/xdg" "${JOB_TMP}/cuda"
export PYTHONPATH=${EXPERIMENT_ROOT}/src:${EXPERIMENT_ROOT}/evaluate:${EXPERIMENT_ROOT}/hpc_compat:${NEURAL_SRC}
export MPLCONFIGDIR=${JOB_TMP}/mpl
export XDG_CACHE_HOME=${JOB_TMP}/xdg
export CUDA_CACHE_PATH=${JOB_TMP}/cuda
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1

cd "${EXPERIMENT_ROOT}"

echo "job=${SLURM_JOB_ID} node=${SLURMD_NODENAME:-unknown} partition=${SLURM_JOB_PARTITION:-unknown}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
srun "${PYTHON_BIN}" src/fuxi_imd_blocked_oof_affine.py "$@"
