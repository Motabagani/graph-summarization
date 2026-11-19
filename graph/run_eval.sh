#!/usr/bin/env bash
# One-shot evaluation on a GPU machine (e.g. NYU Greene: run inside an sbatch job with --gres=gpu:1).
# Each run goes through scripts/run_logged.sh, which saves the exact command, versions and logs
# next to the results (scripts/analysis.py needs that to rebuild the same documents).
set -euo pipefail
cd "$(dirname "$0")"
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt_tab')"
python tests/logic.py && python tests/test_budget.py && python tests/test_team_compat.py

# Short documents: full β grid.
scripts/run_logged.sh results/short python evaluate.py --datasets cnn_dm tldr \
    --limit-dev 200 --limit-test 1000 --out results/short

# Long documents: cap length and skip β=0 (which runs NSP on every sentence pair).
scripts/run_logged.sh results/long python evaluate.py --datasets arxiv pubmed billsum \
    --limit-dev 100 --limit-test 500 --max-sentences 150 --betas 0.3 0.6 --out results/long

# WikiHow needs wikihowAll.csv in this folder (the original GitHub source is gone; ask Hashim).
if [ -f wikihowAll.csv ]; then
  scripts/run_logged.sh results/wikihow python evaluate.py --datasets wikihow --wikihow-csv wikihowAll.csv \
      --limit-dev 200 --limit-test 1000 --out results/wikihow
fi

# Paper tables (Table 1 main results, Table 2 ablations) from every run above.
python scripts/merge_tables.py results/short results/long results/wikihow --out results/paper_tables
