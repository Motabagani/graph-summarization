#!/bin/bash
# Quick look at how the pipeline is doing on every dataset, without tuning and without
# re-running any baseline. For each dataset, one GPU job runs the first N test documents through
# preprocessor -> grapher -> ranker (five variants); then a CPU job compares them with the baselines
# on exactly the same documents, using the baselines' saved per-document scores.
#
#   source baselines/config.sh                      # DATA_ROOT, VENV, SBATCH_ACCOUNT
#   bash pipeline/slurm/submit_pilot.sh             # N=300 documents per dataset
#   N=100 bash pipeline/slurm/submit_pilot.sh       # faster
#   BASELINE_RESULTS=/scratch/$USER/NLP/results     # where the baseline predictions.jsonl files are (default)
#   GPU_SBATCH="--partition=<gpu partition>"        # if GPU jobs need a partition
#   ENCODER=tfidf bash pipeline/slurm/submit_pilot.sh   # CPU-only dry run (no NSP: only the position/none variants work)
set -euo pipefail
[ -d pipeline/slurm ] || { echo "Run this from the repository root."; exit 1; }
: "${DATA_ROOT:?source baselines/config.sh first}"
export N=${N:-300} WORK=${WORK:-$PWD/work}
BASELINE_RESULTS=${BASELINE_RESULTS:-/scratch/$USER/NLP/results}
mkdir -p pipeline/logs
gpu=(${GPU_SBATCH:-}); [ "${ENCODER:-bert}" = bert ] || gpu=(--gres=none)
ids=()
for ds in ${DATASETS:-cnn_dailymail pubmed arxiv billsum wikihow tldr17}; do
    jid=$(sbatch --parsable --job-name="pilot_$ds" --export=ALL,DATASET="$ds" "${gpu[@]}" pipeline/slurm/pilot.sh)
    echo "pilot $ds -> job $jid"; ids+=("$jid")
done
dep=$(IFS=:; echo "${ids[*]}")
cid=$(sbatch --parsable --job-name=pilot_compare --dependency="afterany:$dep" --cpus-per-task=1 --mem=8GB \
      --time=01:00:00 --output="pipeline/logs/%x_%j.out" \
      --wrap="${VENV:+source $VENV/bin/activate && }cd pipeline && python compare_pilot.py --pilot $WORK/pilot --baselines $BASELINE_RESULTS")
echo "comparison -> job $cid; table: $WORK/pilot/pilot_summary.md"
