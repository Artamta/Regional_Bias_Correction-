#!/bin/bash
# Two-case aggregation/leakage smoke before the full cache array.

#SBATCH --job-name=quest-pr-smoke
#SBATCH --partition=gpu_prio
#SBATCH --exclude=cn2,cn3,cn4,cn15,cn17
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/clean/ai-quest-global/logs/precip_smoke_%j.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/clean/ai-quest-global/logs/precip_smoke_%j.err

set -euo pipefail
: "${QUEST_ERA5_STORE:?export the local WeatherBench2 ERA5 six-hour Zarr path}"

ROOT=/home/raj.ayush/s2s/s2s_anlysis/clean/ai-quest-global
PYTHON=/home/raj.ayush/.conda/envs/fuxi/bin/python
mkdir -p "${ROOT}/logs"
"${PYTHON}" -u "${ROOT}/build_precip_cache.py" part \
  --era5-store "${QUEST_ERA5_STORE}" \
  --indices 0,1664 \
  --parts-root /storage/raj.ayush/s2s_final_data/final_iteration/ai_quest_precip_20260813/smoke_parts
