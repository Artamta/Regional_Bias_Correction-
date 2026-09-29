#!/bin/bash
# Validation-only control scoring and promotion decision after the seed-42 screen.

#SBATCH --job-name=quest-pr-select
#SBATCH --partition=gpu_prio
#SBATCH --exclude=cn2,cn3,cn4,cn15,cn17
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/clean/ai-quest-global/logs/precip_select_%j.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/clean/ai-quest-global/logs/precip_select_%j.err

set -euo pipefail
: "${QUEST_PRECIP_CACHE:?export the finalized 17/2/1 precipitation cache path}"
: "${QUEST_SCREEN_TAG:?export the immutable screen run tag}"

ROOT=/home/raj.ayush/s2s/s2s_anlysis/clean/ai-quest-global
PYTHON=${QUEST_PYTHON:-/home/raj.ayush/.conda/envs/diffusion_env/bin/python}
RUN_ROOT=/storage/raj.ayush/s2s_final_data/final_iteration/ai_quest_precip_20260813/runs
OUTPUT_ROOT=/storage/raj.ayush/s2s_final_data/final_iteration/ai_quest_precip_20260813/selections
export MPLCONFIGDIR=/tmp/quest-pr-mpl-${SLURM_JOB_ID}
mkdir -p "${ROOT}/logs" "${MPLCONFIGDIR}"

"${PYTHON}" -u "${ROOT}/select_precip_candidate.py" \
  --cache "${QUEST_PRECIP_CACHE}" \
  --neural-run "${RUN_ROOT}/${QUEST_SCREEN_TAG}_target_only_w16_seed42" \
  --neural-run "${RUN_ROOT}/${QUEST_SCREEN_TAG}_tp_only_w16_seed42" \
  --neural-run "${RUN_ROOT}/${QUEST_SCREEN_TAG}_full_w16_seed42" \
  --output-dir "${OUTPUT_ROOT}/${QUEST_SCREEN_TAG}_validation_only" \
  --device cpu
