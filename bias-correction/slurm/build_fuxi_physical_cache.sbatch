#!/usr/bin/env bash
#SBATCH --job-name=fuxi-phys-cache
#SBATCH --partition=gpu_prio
# Known avoid-list: cn3 is MIG; cn2, cn15, and cn17 are unreliable here.
#SBATCH --exclude=cn2,cn3,cn15,cn17
#SBATCH --array=0-629%8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/logs/%x-%A_%a.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/logs/%x-%A_%a.err

set -euo pipefail

SWEEP_ROOT=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction
PYTHON_BIN=/home/raj.ayush/.conda/envs/s2s_core/bin/python
CACHE_TASK_COUNT=630
JOB_TMP=/tmp/fuxi-phys-cache-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}

cleanup() {
    rm -rf "${JOB_TMP}"
}
trap cleanup EXIT

mkdir -p "${SWEEP_ROOT}/logs" "${JOB_TMP}"
export TMPDIR=${JOB_TMP}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export BLOSC_NTHREADS=${SLURM_CPUS_PER_TASK}
export PYTHONUNBUFFERED=1

cd "${SWEEP_ROOT}"

echo "host=$(hostname) task=${SLURM_ARRAY_TASK_ID}/${CACHE_TASK_COUNT} cpus=${SLURM_CPUS_PER_TASK}"
srun "${PYTHON_BIN}" src/fuxi_physical_feature_cache.py build \
    --task-index "${SLURM_ARRAY_TASK_ID}" \
    --task-count "${CACHE_TASK_COUNT}" \
    "$@"
