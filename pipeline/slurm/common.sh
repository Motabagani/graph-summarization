# Sourced by the job scripts. Run everything from the repository root after `source baselines/config.sh`
# (DATA_ROOT, VENV, SBATCH_ACCOUNT); WORK is where the stages exchange files.
set -uo pipefail
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
DATA_ROOT=${DATA_ROOT:?set DATA_ROOT (see baselines/config.example.sh)}
WORK=${WORK:-$REPO/work}
export NLTK_DATA=${NLTK_DATA:-$REPO/.nltk_data} PYTHONUNBUFFERED=1
if [ -n "${VENV:-}" ]; then source "$VENV/bin/activate"; fi
cd "$REPO/pipeline"
