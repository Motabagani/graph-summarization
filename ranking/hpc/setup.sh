#!/bin/bash
# One-time setup on an HPC LOGIN node (compute nodes usually have no internet).
#   cd graph-summarization && bash ranking/hpc/setup.sh
# Optional: PYTHON=/path/to/python3.10 bash ranking/hpc/setup.sh
set -euo pipefail
cd "$(dirname "$0")/../.."                       # repo root
PY=${PYTHON:-python3}

"$PY" -c 'import sys; assert sys.version_info >= (3, 9), f"need Python >= 3.9, got {sys.version.split()[0]}: module load a newer python, or PYTHON=... bash ranking/hpc/setup.sh"'

"$PY" -m venv .venv
.venv/bin/pip install --upgrade pip -q
.venv/bin/pip install -r ranking/requirements.txt -q
# gensim gives the same stop-word list as the TextRank baseline. If it will not
# install, the code falls back to the HW4 closed-class list (and prints which one it used).
.venv/bin/pip install gensim -q || echo "WARNING: gensim did not install -> stop_list=hw4-closed-class"

# NLTK sentence splitter model, stored inside the repo so compute nodes can find it
mkdir -p .nltk_data
NLTK_DATA=$PWD/.nltk_data .venv/bin/python -c "import nltk; nltk.download('punkt_tab', quiet=True); nltk.download('punkt', quiet=True)"

NLTK_DATA=$PWD/.nltk_data .venv/bin/python -m unittest discover -s ranking/tests
NLTK_DATA=$PWD/.nltk_data .venv/bin/python -c "
import sys; sys.path.insert(0, 'ranking')
from preprocess import SPLITTER, STOP_LIST
print('setup OK  splitter =', SPLITTER, ' stop_list =', STOP_LIST)"
