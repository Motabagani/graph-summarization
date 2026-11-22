#!/bin/bash
#SBATCH --job-name=pipe_preprocess
#SBATCH --cpus-per-task=16
#SBATCH --mem=48GB
#SBATCH --time=04:00:00
#SBATCH --output=pipeline/logs/%x_%j.out
# Stage 1 for one dataset: dev (first 1,000 tuning documents) and test documents.
source pipeline/slurm/common.sh
for part in dev test; do
  lim=(); [ "$part" = test ] && [ "${TEST_LIMIT:-0}" -gt 0 ] && lim=(--limit "$TEST_LIMIT")
  python preprocessor.py --dataset "$DATASET" --part $part --unit "${UNIT:-sentence}" "${lim[@]}" \
         --data-root "$DATA_ROOT" --out "$WORK/$DATASET/$part.${UNIT:-sentence}.jsonl" || exit 1
done
