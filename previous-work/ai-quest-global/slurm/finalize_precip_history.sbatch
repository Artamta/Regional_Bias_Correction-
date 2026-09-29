#!/bin/bash

#SBATCH --job-name=quest-pr-final
#SBATCH --partition=gpu_prio
#SBATCH --exclude=cn2,cn3,cn4,cn15,cn17
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/clean/ai-quest-global/logs/precip_finalize_%j.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/clean/ai-quest-global/logs/precip_finalize_%j.err

set -euo pipefail
: "${QUEST_ERA5_STORE:?export the local WeatherBench2 ERA5 six-hour Zarr path}"
: "${QUEST_LAND_FILE:?export the evaluation land-fraction NetCDF path}"

ROOT=/home/raj.ayush/s2s/s2s_anlysis/clean/ai-quest-global
PYTHON=/home/raj.ayush/.conda/envs/fuxi/bin/python
mkdir -p "${ROOT}/logs"
"${PYTHON}" -u "${ROOT}/build_precip_cache.py" finalize \
  --era5-store "${QUEST_ERA5_STORE}" \
  --land-fraction "${QUEST_LAND_FILE}" \
  ${QUEST_PARTS_CONTRACT_SHA256:+--parts-contract-sha256 "${QUEST_PARTS_CONTRACT_SHA256}"} \
  ${QUEST_PRECIP_OUTPUT:+--output "${QUEST_PRECIP_OUTPUT}"}
