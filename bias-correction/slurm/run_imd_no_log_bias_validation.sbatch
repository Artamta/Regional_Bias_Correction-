#!/usr/bin/env bash
#SBATCH --job-name=imd-no-log-bias
#SBATCH --partition=gpu_prio
# Known avoid-list: cn3 is MIG; cn2, cn4, cn15, cn16, and cn17 are unreliable.
#SBATCH --exclude=cn2,cn3,cn4,cn15,cn16,cn17
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/logs/%x-%j.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/logs/%x-%j.err

set -euo pipefail

EXPERIMENT_ROOT=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction
NEURAL_SRC=/home/raj.ayush/s2s/s2s_anlysis/clean/neural_adapter/src
PYTHON_BIN=/home/raj.ayush/.conda/envs/s2s_core/bin/python
JOB_TMP=/tmp/imd-no-log-bias-${SLURM_JOB_ID}

cleanup() {
    rm -rf "${JOB_TMP}"
}
trap cleanup EXIT

mkdir -p "${EXPERIMENT_ROOT}/logs" "${JOB_TMP}/mpl" "${JOB_TMP}/xdg" "${JOB_TMP}/cuda"
export PYTHONPATH=${EXPERIMENT_ROOT}/src:${EXPERIMENT_ROOT}/evaluate:${EXPERIMENT_ROOT}/hpc_compat:${NEURAL_SRC}
export MPLCONFIGDIR=${JOB_TMP}/mpl
export XDG_CACHE_HOME=${JOB_TMP}/xdg
export CUDA_CACHE_PATH=${JOB_TMP}/cuda
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}
export PYTHONUNBUFFERED=1

cd "${EXPERIMENT_ROOT}"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
srun "${PYTHON_BIN}" src/fuxi_imd_no_log_bias_validation.py "$@"
