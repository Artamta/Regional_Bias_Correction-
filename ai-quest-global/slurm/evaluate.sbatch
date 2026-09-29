#!/usr/bin/env bash
#SBATCH --job-name=quest-tp-eval
#SBATCH --partition=gpu_prio
#SBATCH --exclude=cn2,cn3,cn4,cn15,cn17
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/home/raj.ayush/s2s/s2s_anlysis/clean/ai-quest-global/slurm-%x-%j.out
#SBATCH --error=/home/raj.ayush/s2s/s2s_anlysis/clean/ai-quest-global/slurm-%x-%j.err

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: sbatch slurm/evaluate.sbatch CHECKPOINT OUTPUT_DIRECTORY" >&2
    exit 2
fi

PROJECT_ROOT=/home/raj.ayush/s2s/s2s_anlysis/clean/ai-quest-global
PYTHON_BIN=/home/raj.ayush/.conda/envs/fuxi/bin/python
CACHE_STORE=/storage/raj.ayush/s2s_final_data/final_iteration/ai_quest_global/cache/fuxi_tp_2017_2021.zarr
CHECKPOINT=$1
OUTPUT_DIRECTORY=$2

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PYTHONUNBUFFERED=1

cd "${PROJECT_ROOT}"
echo "job_id=${SLURM_JOB_ID} node=${SLURMD_NODENAME:-unknown} partition=${SLURM_JOB_PARTITION:-unknown}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
srun "${PYTHON_BIN}" evaluate.py \
    --cases "${CACHE_STORE}" \
    --years 2020 2021 \
    --checkpoint "${CHECKPOINT}" \
    --output-dir "${OUTPUT_DIRECTORY}" \
    --device cuda \
    --batch-size 8 \
    --thursday-only
