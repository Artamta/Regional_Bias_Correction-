#!/bin/bash
# Build 2,080 atomic precipitation-cache parts, at most eight tasks at once.

#SBATCH --job-name=quest-pr-cache
#SBATCH --partition=gpu_prio
#SBATCH --exclude=cn2,cn3,cn4,cn15,cn17
#SBATCH --array=0-63%8
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/clean/ai-quest-global/logs/precip_cache_%A_%a.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/clean/ai-quest-global/logs/precip_cache_%A_%a.err

set -euo pipefail
: "${QUEST_ERA5_STORE:?export the local WeatherBench2 ERA5 six-hour Zarr path}"

ROOT=/home/raj.ayush/s2s/s2s_anlysis/clean/ai-quest-global
PYTHON=/home/raj.ayush/.conda/envs/fuxi/bin/python
mkdir -p "${ROOT}/logs"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

"${PYTHON}" -u "${ROOT}/build_precip_cache.py" part \
  --era5-store "${QUEST_ERA5_STORE}" \
  --task-id "${SLURM_ARRAY_TASK_ID}" \
  --task-count 64
