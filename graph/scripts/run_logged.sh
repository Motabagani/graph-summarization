#!/usr/bin/env bash
# Run an evaluate.py command and record provenance next to its results.
#   scripts/run_logged.sh results/smoke python evaluate.py ... --out results/smoke
# Writes <out>/RUN_INFO.txt (exact command, git commit or source hash, seed, versions, timing),
# <out>/log.txt (stderr) and <out>/stdout.md.
set -uo pipefail
out="$1"; shift
mkdir -p "$out"
cd "$(dirname "$0")/.."
seed=$(printf '%s\n' "$@" | awk 'p{print; exit} /^--seed$/{p=1}')
{
  echo "command: $(printf '%q ' "$@")"
  echo "cwd: $(pwd)"
  echo "seed: ${seed:-0 (evaluate.py default)}"
  if git rev-parse HEAD >/dev/null 2>&1; then
    echo "git commit: $(git rev-parse HEAD)$(git diff --quiet HEAD -- . 2>/dev/null || echo ' (+ uncommitted changes)')"
  else
    echo "git commit: none (not a git repository)"
  fi
  sha256() { if command -v sha256sum >/dev/null; then sha256sum; else shasum -a 256; fi; }   # Linux / macOS
  echo "source sha256 (*.py, sorted): $(cat $(ls *.py | sort) | sha256 | cut -d' ' -f1)"
  echo "host: $(hostname)  started: $(date -u +%FT%TZ)"
  python - <<'EOF'
import importlib.metadata as md, platform
mods = ["torch", "transformers", "datasets", "spacy", "rouge_score", "numpy"]
print("python", platform.python_version(), "|", " ".join(f"{m}={md.version(m.replace('_', '-'))}" for m in mods))
EOF
} > "$out/RUN_INFO.txt"
start=$(date +%s)
"$@" > "$out/stdout.md" 2> "$out/log.txt"
rc=$?
echo "finished: $(date -u +%FT%TZ)  wall seconds: $(( $(date +%s) - start ))  exit: $rc" >> "$out/RUN_INFO.txt"
exit $rc
