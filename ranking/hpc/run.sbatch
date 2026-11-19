#!/bin/bash
# Run ONE command (any ranking/run.py call) as a batch job:
#   sbatch ranking/hpc/run.sbatch ranking/run.py /path/test/cnndm.jsonl --dataset cnndm --ranker degree --rouge official
# Output: ranking/hpc/logs/pt-run_<jobid>.out
#
#SBATCH --job-name=pt-run
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=ranking/hpc/logs/%x_%j.out
##SBATCH --account=YOUR_ACCOUNT
##SBATCH --partition=YOUR_PARTITION

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export NLTK_DATA=$PWD/.nltk_data OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python "$@"
