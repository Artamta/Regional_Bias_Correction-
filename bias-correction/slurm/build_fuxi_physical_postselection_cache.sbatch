#!/usr/bin/env bash
#SBATCH --job-name=fuxi-phys-post
#SBATCH --partition=gpu_prio
# CPU-only extraction; cn3 is MIG and cn2/cn15/cn17 are unreliable here.
#SBATCH --exclude=cn2,cn3,cn15,cn17
#SBATCH --array=0-69%8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/logs/%x-%A_%a.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction/logs/%x-%A_%a.err

set -euo pipefail

POST_ROOT=/home/raj.ayush/s2s/s2s_anlysis/clean/bias-correction
PYTHON_BIN=/home/raj.ayush/.conda/envs/s2s_core/bin/python
POST_TASK_COUNT=70
POST_JOB_TMP=/tmp/fuxi-phys-post-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}

cleanup() {
    rm -rf "${POST_JOB_TMP}"
}
trap cleanup EXIT

mkdir -p "${POST_ROOT}/logs" "${POST_JOB_TMP}"
export TMPDIR=${POST_JOB_TMP}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export BLOSC_NTHREADS=${SLURM_CPUS_PER_TASK}
export PYTHONUNBUFFERED=1

cd "${POST_ROOT}"
echo "host=$(hostname) task=${SLURM_ARRAY_TASK_ID}/${POST_TASK_COUNT} cpus=${SLURM_CPUS_PER_TASK} gpu_requested=no"
srun "${PYTHON_BIN}" src/fuxi_physical_postselection_cache.py build \
    --task-index "${SLURM_ARRAY_TASK_ID}" \
    --task-count "${POST_TASK_COUNT}" \
    "$@"
