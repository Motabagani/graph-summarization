#!/bin/bash
#SBATCH --job-name=download_datasets
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Downloads the six datasets into $DATA_ROOT with the layout the code expects:
#   cnn_dailymail/cnn_dailymail/{train,validation,test}.csv       Kaggle (needs Kaggle credentials)
#   long_summarization/{arxiv,pubmed}-dataset/{train,val,test}.txt  Cohan et al. 2018
#   billsum/data/*.parquet                                         Hugging Face (FiscalNote/billsum)
#   tldr17/corpus-webis-tldr-17.json                               Zenodo
#   wikihow/manual/wikihowAll.csv                                  MANUAL download (see below)
#
#   source config.sh && sbatch slurm/download_datasets.sh      (from the repository root)
# Then check everything with:  python baselines/check_datasets.py
#
# One failed dataset doesn't stop the others; the summary at the end lists any failures.
set -uo pipefail
DATA_ROOT=${DATA_ROOT:?set DATA_ROOT (see config.example.sh)}
if [ -n "${VENV:-}" ]; then source "$VENV/bin/activate"; fi
pip install -q kaggle huggingface_hub
export HF_HOME=${HF_HOME:-$DATA_ROOT/.cache/huggingface}
mkdir -p "$DATA_ROOT"/{cnn_dailymail,long_summarization,tldr17,billsum,wikihow/manual} "$HF_HOME"

FAILED=()
step () { echo -e "\n==================== $1 ====================  $(date)"
          if "${@:2}"; then echo "[OK] $1"; else echo "[FAILED] $1"; FAILED+=("$1"); fi; }

# 1. CNN/DailyMail (Kaggle). Needs ~/.kaggle/kaggle.json, ~/.kaggle/access_token or $KAGGLE_API_TOKEN.
dl_cnn () {
    if [ ! -f ~/.kaggle/kaggle.json ] && [ ! -f ~/.kaggle/access_token ] && [ -z "${KAGGLE_API_TOKEN:-}" ]; then
        echo "No Kaggle credentials found"; return 1; fi
    kaggle datasets download -d gowrishankarp/newspaper-text-summarization-cnn-dailymail \
        -p "$DATA_ROOT/cnn_dailymail" --unzip
}
step "CNN/DailyMail" dl_cnn

# 2. arXiv + PubMed (Cohan et al. 2018): archive.org mirror of the original zips, Hugging Face fallback.
dl_long () {
    local d="$DATA_ROOT/long_summarization" base="https://archive.org/download/armancohan-long-summarization-paper-code"
    local ok=0
    for name in arxiv-dataset pubmed-dataset; do
        if wget -c -q -O "$d/$name.zip" "$base/$name.zip" && unzip -q -o "$d/$name.zip" -d "$d"; then
            rm -f "$d/$name.zip"
        else
            echo "archive.org failed for $name, using the Hugging Face mirror"; rm -f "$d/$name.zip"
            python -c "from huggingface_hub import snapshot_download as s; s('ccdv/${name%-dataset}-summarization', repo_type='dataset', local_dir='$d/${name}_hf')" || ok=1
        fi
    done
    rm -rf "$d/__MACOSX"
    return $ok
}
step "arXiv / PubMed" dl_long

# 3. Webis-TLDR-17 (Zenodo; ~3 GB zipped, ~18 GB unzipped)
dl_tldr () {
    local d="$DATA_ROOT/tldr17"
    wget -c -q -O "$d/corpus-webis-tldr-17.zip" "https://zenodo.org/records/1043504/files/corpus-webis-tldr-17.zip?download=1" \
    && unzip -q -o "$d/corpus-webis-tldr-17.zip" -d "$d" && rm -f "$d/corpus-webis-tldr-17.zip"
}
step "Webis-TLDR-17" dl_tldr

# 4. BillSum (Hugging Face, parquet)
dl_billsum () {
    python -c "from huggingface_hub import snapshot_download as s; s('FiscalNote/billsum', repo_type='dataset', local_dir='$DATA_ROOT/billsum')"
}
step "BillSum" dl_billsum

# 5. WikiHow: wikihowAll.csv must be downloaded by hand (linked from
#    https://github.com/mahnazkoupaee/WikiHow-Dataset) and copied to $DATA_ROOT/wikihow/manual/.
dl_wikihow () {
    local f="$DATA_ROOT/wikihow/manual/wikihowAll.csv"
    [ -f "$f" ] || { echo "Missing $f: download it manually (see comment above)"; return 1; }
    echo "Found wikihowAll.csv ($(wc -l < "$f") lines)"
}
step "WikiHow" dl_wikihow

echo -e "\n==================== SUMMARY ====================  $(date)"
du -sh "$DATA_ROOT"/* 2>/dev/null
if [ ${#FAILED[@]} -eq 0 ]; then echo "All datasets ready."; else echo "Failed: ${FAILED[*]}"; exit 1; fi
