#!/bin/bash
# Submit the whole pipeline for one dataset as a chain of dependent Slurm jobs:
#   preprocess (CPU) -> grapher dev + grapher test shards (GPU) -> ranker (CPU)
# From the repository root:
#   source baselines/config.sh
#   bash pipeline/slurm/submit.sh cnn_dailymail
#   SHARDS=8 BETA_MIN=0.3 bash pipeline/slurm/submit.sh pubmed          # split test over 8 GPUs
#   TEST_LIMIT=10000 bash pipeline/slurm/submit.sh tldr17               # sample the test set
#   UNIT=clause RANKER_ARGS="--lams 0 0.5 1" bash pipeline/slurm/submit.sh cnn_dailymail
#   ENCODER=tfidf RANKER_ARGS="--direction position" bash pipeline/slurm/submit.sh cnn_dailymail   # CPU only
# Extra sbatch options for the GPU jobs (partition, GPU type): GPU_SBATCH="--partition=..."
set -euo pipefail
export DATASET=${1:?usage: bash pipeline/slurm/submit.sh <dataset>}
[ -d pipeline/slurm ] || { echo "Run this from the repository root."; exit 1; }
: "${DATA_ROOT:?source baselines/config.sh first}"
export UNIT=${UNIT:-sentence} ENCODER=${ENCODER:-bert} SHARDS=${SHARDS:-1}
mkdir -p pipeline/logs
gpu=(${GPU_SBATCH:-}); [ "$ENCODER" = bert ] || gpu=(--gres=none)
p=$(sbatch --parsable --export=ALL pipeline/slurm/preprocess.sh)
gd=$(sbatch --parsable --export=ALL,PART=dev,SHARDS=1 --dependency=afterok:$p "${gpu[@]}" pipeline/slurm/grapher.sh)
gt=$(sbatch --parsable --export=ALL,PART=test --array=0-$((SHARDS - 1)) --dependency=afterok:$p "${gpu[@]}" pipeline/slurm/grapher.sh)
r=$(sbatch --parsable --export=ALL --dependency=afterok:$gd:$gt pipeline/slurm/ranker.sh)
echo "$DATASET: preprocess $p -> grapher dev $gd, test $gt (x$SHARDS) -> ranker $r"
echo "results: baselines/results/ours/$DATASET/  (then: python baselines/baselines/collect_results.py baselines/results)"
