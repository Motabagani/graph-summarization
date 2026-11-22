#!/bin/bash
#SBATCH --job-name=position_analysis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48GB
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Where does summary material sit inside documents? On development data only (validation splits;
# BillSum train; WikiHow / TL;DR tuning portion), up to 5,000 docs per dataset:
#   - positions of Oracle sentences (reference-based)
#   - positions of the most central sentences (reference-free)
# Output: $RESULTS_DIR/position_analysis/{position_summary.md, positions.json, positions.png}
#
#   source config.sh && sbatch slurm/position_analysis.sh      (from the repository root)
set -uo pipefail
REPO_DIR=${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}
DATA_ROOT=${DATA_ROOT:?set DATA_ROOT (see config.example.sh)}
RESULTS_DIR=${RESULTS_DIR:-$REPO_DIR/results}
export NLTK_DATA=${NLTK_DATA:-$REPO_DIR/.nltk_data} PYTHONUNBUFFERED=1
if [ -n "${VENV:-}" ]; then source "$VENV/bin/activate"; fi
if [ -z "${SKIP_PIP:-}" ]; then pip install -q -r "$REPO_DIR/requirements.txt"; fi
cd "$REPO_DIR/baselines"
python position_analysis.py --data-root "$DATA_ROOT" --out "$RESULTS_DIR/position_analysis" --docs 5000
