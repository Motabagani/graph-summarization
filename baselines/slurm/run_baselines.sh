#!/bin/bash
#SBATCH --job-name=baselines
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Runs every baseline on ONE dataset. Normally submitted by slurm/submit.sh (one job per
# dataset). Submit from the repository root after `source config.sh`:
#     DATASET=pubmed sbatch slurm/run_baselines.sh
# MODE=main (default)          the main-table baselines -> $RESULTS_DIR/all/<dataset>/
# MODE=sensitivity             robustness variants      -> $RESULTS_DIR/sensitivity/<dataset>/
# MODE=pacsum_cosine           PacSum on the cosine graph -> $RESULTS_DIR/sensitivity/<dataset>/
# The sensitivity variants are:
#     TextRank / DegreeCentrality with length-normalised (cosine) tf-idf similarity,
#     TextRank exactly as in Mihalcea (2004) (word overlap, no stemming/stopwords),
#     LexPageRank with idf from the article's own sentences (ablation of the background idf).
#
# Settings per dataset (the ones validated against published results):
#   cnn_dailymail  3 sentences                          (PacSum)          tune on validation
#   pubmed         first 203 words budget               (HipoRank)        tune on validation
#   arxiv          220 words                            (HipoRank)        tune on validation
#   billsum        words = mean reference length (train split)            tune on train
#   wikihow        words = mean reference length (first 1,000 docs)       tune on first 1,000 docs,
#                                                                         evaluate on the rest
#   tldr17         1 sentence                                             tune on first 1,000 docs,
#                                                                         evaluate on the rest
# Everywhere: ROUGE F1 without stemming (matches PacSum and HipoRank, checked via Lead).
# Oracle: greedy (Nallapati et al. 2017), maximising ROUGE-1 F1 on CNN/DM (closest to PacSum's
# Oracle), ROUGE-2 F1 on PubMed/arXiv (as HipoRank states), ROUGE-1+2 F1 elsewhere.
# TextRank / DegreeCentrality use PacSum's tf-idf graph (the configuration validated on CNN/DM).

set -uo pipefail
DATASET=${DATASET:?usage: sbatch --export=ALL,DATASET=<name> baselines_job.sh}
SAMPLE_DOCS=${SAMPLE_DOCS:-0}     # wikihow/tldr17 only: evaluate this many docs (0 = all after the tuning docs)
MODE=${MODE:-main}                # main | sensitivity | pacsum_cosine

REPO_DIR=${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}     # submit from the repository root
CODE_DIR=$REPO_DIR/baselines
DATA_ROOT=${DATA_ROOT:?set DATA_ROOT (see config.example.sh)}
RESULTS_DIR=${RESULTS_DIR:-$REPO_DIR/results}
if [ "$MODE" = main ]; then OUT=$RESULTS_DIR/all/$DATASET
else OUT=$RESULTS_DIR/sensitivity/$DATASET; fi   # sensitivity and pacsum_cosine
export NLTK_DATA=${NLTK_DATA:-$REPO_DIR/.nltk_data}
export PYTHONUNBUFFERED=1
if [ -n "${VENV:-}" ]; then source "$VENV/bin/activate"; fi
if [ -z "${SKIP_PIP:-}" ]; then pip install -q -r "$REPO_DIR/requirements.txt"; fi
cd "$CODE_DIR"
mkdir -p "$OUT"

mean_ref_words () {   # mean reference length (words) over the first $2 docs of split $1
    python - "$DATASET" "$DATA_ROOT" "$1" "$2" <<'EOF'
import argparse, sys, numpy as np, summarize
ds, root, split, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
a = argparse.Namespace(dataset=ds, data_root=root, format=None, text_field=None, ref_field=None, id_field=None)
recs = summarize._limited(summarize.open_records(summarize.locate(a, split)), 0, n)
print(int(round(np.mean([len(" ".join(summarize.reference_sentences(r)).split()) for _, _, r in recs]))))
EOF
}

EVAL=(--split test); TUNE_SPLIT=validation; EXTRA=(); ORACLE_MAX=30; ORACLE_OBJ=rouge12
case "$DATASET" in
  cnn_dailymail) LEN=(--num-sentences 3); ORACLE_MAX=3; ORACLE_OBJ=rouge1 ;;   # closest to PacSum's Oracle
  pubmed)        LEN=(--num-sentences 0 --word-limit 203); ORACLE_OBJ=rouge2 ;;  # HipoRank: "Oracle (ROUGE-2, F1)"
  arxiv)         LEN=(--num-sentences 0 --word-limit 220); ORACLE_OBJ=rouge2 ;;
  billsum)       TUNE_SPLIT=train
                 LEN=(--num-sentences 0 --word-limit "$(mean_ref_words train 5000)") ;;
  wikihow)       EVAL=(--offset 1000); [ "$SAMPLE_DOCS" -gt 0 ] && EVAL+=(--limit "$SAMPLE_DOCS")
                 EXTRA=(--newline-boundaries)
                 LEN=(--num-sentences 0 --word-limit "$(mean_ref_words all 1000)") ;;   # tuning docs only
  tldr17)        EVAL=(--offset 1000); [ "$SAMPLE_DOCS" -gt 0 ] && EVAL+=(--limit "$SAMPLE_DOCS")
                 LEN=(--num-sentences 1); ORACLE_MAX=1 ;;
  *) echo "unknown dataset $DATASET"; exit 1 ;;
esac
# For wikihow/tldr17 (no official splits) tuning uses the first 1,000 docs and evaluation starts
# after them (--offset 1000), so the two never overlap.

COMMON=(--dataset "$DATASET" --data-root "$DATA_ROOT" --no-rouge-stemmer "${LEN[@]}" "${EVAL[@]}" "${EXTRA[@]}")
TFIDF=(--graph undirected --preprocess gensim --similarity tfidf_dot)
TUNE=(--tune --tune-split "$TUNE_SPLIT" --tune-docs 1000 --tune-metric rouge1)
echo "[$DATASET] summary length: ${LEN[*]}   evaluation: ${EVAL[*]}   tuning split: $TUNE_SPLIT"

FAILED=()
run () { echo -e "\n==================== $DATASET  $1 ====================  $(date)"
         "${@:2}" || { echo "[FAILED] $1"; FAILED+=("$1"); }; }

if [ "$MODE" = pacsum_cosine ]; then
  # PacSum's directed, position-weighted centrality on the length-normalised (cosine) tf-idf graph;
  # beta/lambda tuned on validation exactly as in the main run.
  run "PacSum (cosine)" python pacsum.py "${COMMON[@]}" "${TUNE[@]}" --rep tfidf --sim cosine --out "$OUT/pacsum_cosine"
  echo
  if [ ${#FAILED[@]} -eq 0 ]; then echo "[$DATASET] all baselines finished -> $OUT"
  else echo "[$DATASET] failed: ${FAILED[*]}"; exit 1; fi
  exit 0
fi

if [ "$MODE" = sensitivity ]; then
  COSINE=(--graph undirected --preprocess gensim --similarity tfidf_cosine)
  run "TextRank (cosine)"          python textrank.py         "${COMMON[@]}" "${COSINE[@]}" --out "$OUT/textrank_cosine"
  run "DegreeCentrality (cosine)"  python DegreeCentrality.py "${COMMON[@]}" "${COSINE[@]}" --out "$OUT/degree_centrality_cosine"
  run "TextRank (Mihalcea 2004)"   python textrank.py         "${COMMON[@]}"                --out "$OUT/textrank_mihalcea"
  run "LexPageRank (document idf)" python lexpagerank.py      "${COMMON[@]}" "${TUNE[@]}" --idf-source document \
                                   --out "$OUT/lexpagerank_docidf"
  echo
  if [ ${#FAILED[@]} -eq 0 ]; then echo "[$DATASET] all baselines finished -> $OUT"
  else echo "[$DATASET] failed: ${FAILED[*]}"; exit 1; fi
  exit 0
fi

run "Lead"              python lead3.py            "${COMMON[@]}"                  --out "$OUT/lead"
run "TextRank"          python textrank.py         "${COMMON[@]}" "${TFIDF[@]}"    --out "$OUT/textrank"
run "DegreeCentrality"  python DegreeCentrality.py "${COMMON[@]}" "${TFIDF[@]}"    --out "$OUT/degree_centrality"
run "LexPageRank"       python lexpagerank.py      "${COMMON[@]}" "${TUNE[@]}" --idf-docs 20000 --out "$OUT/lexpagerank"
run "PacSum (tf-idf)"   python pacsum.py           "${COMMON[@]}" "${TUNE[@]}" --rep tfidf --out "$OUT/pacsum_tfidf"
run "Oracle"            python oracle.py           "${COMMON[@]}" --search greedy --max-sentences "$ORACLE_MAX" \
                        --objective "$ORACLE_OBJ" --out "$OUT/oracle"

echo
if [ ${#FAILED[@]} -eq 0 ]; then echo "[$DATASET] all baselines finished -> $OUT"
else echo "[$DATASET] failed: ${FAILED[*]}"; exit 1; fi
