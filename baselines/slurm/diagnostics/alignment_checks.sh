#!/bin/bash
#SBATCH --job-name=alignment_checks
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48GB
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# The extra experiments used to explain differences from published results (not needed for the
# main tables). Results go to $RESULTS_DIR/diagnostics/.
#   1. Lead with vs without ROUGE stemming on PubMed/arXiv (HipoRank's numbers match "without")
#   2. Oracle variants on CNN/DM (which objective / sentence cap best matches PacSum's Oracle)
#   3. Off-the-shelf sumy LexRank vs our LexPageRank (1,000 test docs each, PubMed and arXiv)
#   4. Continuous (weighted, no threshold) LexPageRank on PubMed/arXiv
#
#   source config.sh && sbatch slurm/diagnostics/alignment_checks.sh     (from the repository root)
set -uo pipefail
REPO_DIR=${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}
DATA_ROOT=${DATA_ROOT:?set DATA_ROOT (see config.example.sh)}
OUT=${RESULTS_DIR:-$REPO_DIR/results}/diagnostics
export NLTK_DATA=${NLTK_DATA:-$REPO_DIR/.nltk_data} PYTHONUNBUFFERED=1
if [ -n "${VENV:-}" ]; then source "$VENV/bin/activate"; fi
if [ -z "${SKIP_PIP:-}" ]; then pip install -q -r "$REPO_DIR/requirements.txt" sumy; fi
cd "$REPO_DIR/baselines"

FAILED=()
run () { echo -e "\n==================== $1 ====================  $(date)"
         "${@:2}" || { echo "[FAILED] $1"; FAILED+=("$1"); }; }
LONG () { echo --dataset "$1" --split test --data-root "$DATA_ROOT" --num-sentences 0 --word-limit "$2"; }

# 1. Lead: ROUGE with and without stemming
for spec in "pubmed 203" "arxiv 220"; do set -- $spec
  run "Lead $1 (no stemming)"   python lead3.py $(LONG $1 $2) --no-rouge-stemmer --out "$OUT/lead_stemming/$1_no_stemmer"
  run "Lead $1 (with stemming)" python lead3.py $(LONG $1 $2) --rouge-stemmer    --out "$OUT/lead_stemming/$1_stemmer"
done

# 2. Oracle variants on CNN/DM
CNN=(--dataset cnn_dailymail --split test --data-root "$DATA_ROOT" --num-sentences 3 --no-rouge-stemmer)
for obj in rouge1 rouge2 rouge12; do
  run "Oracle $obj max 3"     python oracle.py "${CNN[@]}" --search greedy --objective $obj --max-sentences 3  --out "$OUT/oracle_variants/${obj}_max3"
  run "Oracle $obj no limit"  python oracle.py "${CNN[@]}" --search greedy --objective $obj --max-sentences 1000 --out "$OUT/oracle_variants/${obj}_nolimit"
done

# 3. sumy LexRank vs ours on the same 1,000 test documents; 4. continuous LexPageRank
for spec in "pubmed 203 0.2" "arxiv 220 0.1"; do set -- $spec
  run "sumy LexRank $1"            python lexpagerank.py $(LONG $1 $2) --no-rouge-stemmer --limit 1000 --impl sumy --order document \
                                   --out "$OUT/lexrank_sumy/$1_sumy_1k"
  run "LexPageRank $1 (ours, 1k)"  python lexpagerank.py $(LONG $1 $2) --no-rouge-stemmer --limit 1000 --threshold $3 --idf-docs 20000 \
                                   --out "$OUT/lexrank_sumy/$1_ours_1k"
  run "LexPageRank $1 continuous"  python lexpagerank.py $(LONG $1 $2) --no-rouge-stemmer --continuous --idf-docs 20000 \
                                   --out "$OUT/lexpagerank_continuous/$1"
done

echo; if [ ${#FAILED[@]} -eq 0 ]; then echo "All diagnostics finished -> $OUT"; else echo "Failed: ${FAILED[*]}"; exit 1; fi
