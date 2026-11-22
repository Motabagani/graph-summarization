#!/bin/bash
#SBATCH --job-name=pipe_ranker
#SBATCH --cpus-per-task=16
#SBATCH --mem=48GB
#SBATCH --time=12:00:00
#SBATCH --output=pipeline/logs/%x_%j.out
# Stage 3: tune on dev, evaluate on test, write baselines/results/ours/<dataset>/<NAME>/.
# RANKER_ARGS passes extra options, e.g. "--ranker pagerank" or "--direction position --ranker degree".
source pipeline/slurm/common.sh
U=${UNIT:-sentence}; E=${ENCODER:-bert}
python ranker.py --dataset "$DATASET" \
       --dev-docs "$WORK/$DATASET/dev.$U.jsonl" --dev-graphs "$WORK/$DATASET/dev.$U.$E" \
       --test-docs "$WORK/$DATASET/test.$U.jsonl" --test-graphs "$WORK/$DATASET/test.$U.$E" \
       --out "$REPO/baselines/results/ours/$DATASET/${NAME:-${E}_${U}_pagetrust}" ${RANKER_ARGS:-}
