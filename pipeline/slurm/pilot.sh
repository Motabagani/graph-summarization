#!/bin/bash
#SBATCH --job-name=pipe_pilot
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=08:00:00
#SBATCH --output=pipeline/logs/%x_%j.out
# Pilot for ONE dataset ($DATASET): the first $N test documents through all three stages with fixed
# settings (no tuning), then five ranker variants on the same saved graphs. Submitted by submit_pilot.sh.
source pipeline/slurm/common.sh
N=${N:-300}; BETA=${BETA:-0.3}; D=$WORK/pilot/$DATASET
mkdir -p "$D"
python preprocessor.py --dataset "$DATASET" --part test --limit "$N" --data-root "$DATA_ROOT" \
       --out "$D/test.jsonl" || exit 1
python grapher.py "$D/test.jsonl" --out "$D/graphs" --encoder "${ENCODER:-bert}" --beta-min "$BETA" || exit 1
FIXED=(--dataset "$DATASET" --test-docs "$D/test.jsonl" --test-graphs "$D/graphs"
       --betas "$BETA" --gammas "${GAMMA:-1}" --memories "${MEMORY:-0}" --lams 0)
run () { name=$1; shift; echo "== $name"; python ranker.py "${FIXED[@]}" "$@" --out "$D/$name" || echo "[FAILED] $name"; }
run nsp_pagetrust                                        # the proposed method
run nsp_pagetrust_no_negative   --no-negative            # without negative edges
run nsp_pagerank                --ranker pagerank        # NSP direction, plain PageRank
run position_pagetrust          --direction position --forward-weights -0.5   # PacSum-style direction instead of NSP
run undirected_pagetrust        --direction none         # no direction
