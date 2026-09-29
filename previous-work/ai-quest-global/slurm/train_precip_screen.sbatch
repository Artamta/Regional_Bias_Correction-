#!/bin/bash
# Seed-42 screen: target-only, TP-only, and full physical context.

#SBATCH --job-name=quest-pr-train
#SBATCH --partition=gpu_prio
#SBATCH --exclude=cn2,cn3,cn4,cn15,cn17
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/clean/ai-quest-global/logs/precip_train_%A_%a.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/clean/ai-quest-global/logs/precip_train_%A_%a.err

set -euo pipefail
: "${QUEST_PRECIP_CACHE:?export the finalized precipitation cache path}"
: "${QUEST_RUN_TAG:?export a unique immutable run tag}"

ROOT=/home/raj.ayush/s2s/s2s_anlysis/clean/ai-quest-global
PYTHON=${QUEST_PYTHON:-/home/raj.ayush/.conda/envs/fuxi/bin/python}
VARIANTS=(target_only tp_only full)
VARIANT=${VARIANTS[${SLURM_ARRAY_TASK_ID}]}
RUN_ROOT=/storage/raj.ayush/s2s_final_data/final_iteration/ai_quest_precip_20260813/runs
mkdir -p "${ROOT}/logs"
export MPLCONFIGDIR=/tmp/quest-pr-mpl-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}
mkdir -p "${MPLCONFIGDIR}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

"${PYTHON}" -u "${ROOT}/train_precip.py" \
  --cache "${QUEST_PRECIP_CACHE}" \
  --run-dir "${RUN_ROOT}/${QUEST_RUN_TAG}_${VARIANT}_w16_seed42" \
  --variant "${VARIANT}" \
  --width 16 --seed 42 --max-epochs 12 --patience 3 \
  --batch-size 2 --effective-batch-size 8 --device cuda
