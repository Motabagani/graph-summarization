#!/bin/bash
# Sweep every ranker / setting on ONE dataset per array task (6 tasks = 6 datasets).
#   sbatch --export=ALL,DATA=/path/to/jsonl_folder,TAG=dev ranking/hpc/sweep.sbatch
# DATA = folder with cnndm.jsonl pubmed.jsonl arxiv.jsonl billsum.jsonl wikihow.jsonl tldr.jsonl
#        (built with ranking/tools/prepare_data.py). Missing files are skipped.
# Optional: LIMIT=1000 (first N docs per file), ROUGE=official
# Output: ranking/results/sweep_<TAG>_<dataset>.csv, logs in ranking/hpc/logs/
#
#SBATCH --job-name=pt-sweep
#SBATCH --array=0-5
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=ranking/hpc/logs/%x_%A_%a.out
##SBATCH --account=YOUR_ACCOUNT        # uncomment if your cluster requires one
##SBATCH --partition=YOUR_PARTITION

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"                          # submit from the repo root
export NLTK_DATA=$PWD/.nltk_data OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
DATASETS=(cnndm pubmed arxiv billsum wikihow tldr)
D=${DATASETS[$SLURM_ARRAY_TASK_ID]}
: "${DATA:?set DATA=/path/to/jsonl_folder}"
TAG=${TAG:-dev}

.venv/bin/python ranking/tools/sweep.py --data "$DATA" --only "$D" \
    --rouge "${ROUGE:-lite}" ${LIMIT:+--limit $LIMIT} \
    --out "ranking/results/sweep_${TAG}_${D}.csv"
