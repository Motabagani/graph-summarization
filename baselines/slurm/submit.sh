#!/bin/bash
# Launch a run on every dataset (one Slurm job per dataset, in parallel), then a small job that
# gathers all results into one table once they finish.
#
#   source config.sh
#   bash slurm/submit.sh                  # main-table baselines        -> results/all/
#   bash slurm/submit.sh sensitivity      # robustness variants         -> results/sensitivity/
#   bash slurm/submit.sh pacsum_cosine    # PacSum on the cosine graph  -> results/sensitivity/
#   DATASETS="cnn_dailymail pubmed" bash slurm/submit.sh     # only some datasets
#   SAMPLE_DOCS=100000 bash slurm/submit.sh                  # sample WikiHow / TL;DR
#
# Run from the repository root (logs go to ./logs/).
set -euo pipefail
MODE=${1:-main}
case "$MODE" in main|sensitivity|pacsum_cosine) ;; *) echo "MODE must be main | sensitivity | pacsum_cosine"; exit 1 ;; esac
[ -f slurm/run_baselines.sh ] || { echo "Run this from the repository root."; exit 1; }
: "${DATA_ROOT:?set DATA_ROOT first (source config.sh)}"
export MODE SAMPLE_DOCS=${SAMPLE_DOCS:-0}
mkdir -p logs

ids=()
for ds in ${DATASETS:-cnn_dailymail pubmed arxiv billsum wikihow tldr17}; do
    jid=$(sbatch --parsable --job-name="${MODE}_${ds}" --export=ALL,DATASET="$ds" slurm/run_baselines.sh)
    echo "submitted $MODE / $ds -> job $jid"
    ids+=("$jid")
done

dep=$(IFS=:; echo "${ids[*]}")
cid=$(sbatch --parsable --job-name="collect_${MODE}" --dependency="afterany:$dep" \
      --cpus-per-task=1 --mem=4GB --time=00:15:00 --output="logs/%x_%j.out" \
      --wrap="${VENV:+source $VENV/bin/activate && }python baselines/collect_results.py && python baselines/compare_published.py")
echo "submitted collector -> job $cid (runs after: $dep)"
echo "results: ${RESULTS_DIR:-results}/summary.md and alignment.md"
