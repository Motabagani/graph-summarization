#!/bin/bash
# Run ON THE CLUSTER to package the results you already computed for the repository:
# every metrics.json / tuning.json / summary / analysis file, but NOT predictions.jsonl
# (those are gigabytes: TL;DR alone has 3.8M summaries).
#
#   bash export_results_from_hpc.sh /scratch/$USER/NLP/results
#   -> results_for_github.tar.gz in the current directory (a few MB)
#
# The old folders are renamed to the repository's layout:
#   all/, sensitivity/, position_analysis/  kept as they are
#   hiporank_long, lexpagerank_diagnose, oracle_variants, pacsum_cnndm  -> diagnostics/<name>/
set -euo pipefail
SRC=${1:?usage: bash export_results_from_hpc.sh /path/to/NLP/results}
TMP=$(mktemp -d)
mkdir -p "$TMP/results/diagnostics"
copy () {   # copy small files from $1 to $2, keeping the folder structure
    [ -d "$1" ] || return 0
    (cd "$1" && find . -type f \( -name '*.json' -o -name '*.md' -o -name '*.csv' -o -name '*.txt' -o -name '*.png' \) \
        ! -name 'predictions.jsonl' -print0 | xargs -0 -I{} cp --parents {} "$2")
}
for d in all sensitivity position_analysis; do mkdir -p "$TMP/results/$d"; copy "$SRC/$d" "$TMP/results/$d"; done
for d in hiporank_long lexpagerank_diagnose oracle_variants pacsum_cnndm; do
    mkdir -p "$TMP/results/diagnostics/$d"; copy "$SRC/$d" "$TMP/results/diagnostics/$d"; done
tar -czf results_for_github.tar.gz -C "$TMP" results
echo "Wrote $(pwd)/results_for_github.tar.gz ($(du -h results_for_github.tar.gz | cut -f1))"
tar -tzf results_for_github.tar.gz | head -20; echo "..."
rm -rf "$TMP"
