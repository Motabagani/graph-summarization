#!/bin/bash
#SBATCH --job-name=pipe_grapher
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --output=pipeline/logs/%x_%A_%a.out
# Stage 2 on a GPU. PART=dev|test; submitted as an array (one task per shard) by submit.sh.
source pipeline/slurm/common.sh
U=${UNIT:-sentence}
python grapher.py "$WORK/$DATASET/$PART.$U.jsonl" --out "$WORK/$DATASET/$PART.$U.${ENCODER:-bert}" \
       --encoder "${ENCODER:-bert}" --beta-min "${BETA_MIN:-0.3}" \
       --shard "${SLURM_ARRAY_TASK_ID:-0}" --num-shards "${SHARDS:-1}"
