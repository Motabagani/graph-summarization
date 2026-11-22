#!/usr/bin/env python3
"""Sanity-check the downloaded summarization datasets.
Usage:  python baselines/check_datasets.py [DATA_ROOT]   (default: $DATA_ROOT, else current directory)
Prints each data file, its size, row count, columns, and compares counts
with the published sizes."""
import glob, json, os, sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("DATA_ROOT", ".")

# Published sizes (approximate where noted)
EXPECTED = {
    "cnn_dailymail/**/train.csv": 287_113, "cnn_dailymail/**/validation.csv": 13_368,
    "cnn_dailymail/**/test.csv": 11_490,
    "**/arxiv-dataset/train.txt": 203_037, "**/arxiv-dataset/val.txt": 6_436,
    "**/arxiv-dataset/test.txt": 6_440,
    "**/pubmed-dataset/train.txt": 119_924, "**/pubmed-dataset/val.txt": 6_633,
    "**/pubmed-dataset/test.txt": 6_658,
    "tldr17/**/corpus-webis-tldr-17.json": 3_848_330,
    "billsum/**/train-*.parquet": 18_949, "billsum/**/test-*.parquet": 3_269,
    "billsum/**/ca_test-*.parquet": 1_237,
    "wikihow/manual/wikihowAll.csv": 214_000,   # ~ after dropping empty rows
}


def human(n):
    for u in ["B", "K", "M", "G"]:
        if n < 1024:
            return f"{n:.0f}{u}"
        n /= 1024
    return f"{n:.1f}T"


def inspect(path):
    ext = os.path.splitext(path)[1]
    if ext in (".csv", ".parquet"):
        import pandas as pd
        df = pd.read_csv(path) if ext == ".csv" else pd.read_parquet(path)
        empty = int(df.isna().any(axis=1).sum())
        return len(df.dropna()), list(df.columns), f"{empty} rows with missing values"
    rows, cols, bad = 0, None, 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                cols = cols or list(r.keys())
                rows += 1
            except json.JSONDecodeError:
                bad += 1
    return rows, cols, f"{bad} unparseable lines"


print(f"Checking {os.path.abspath(ROOT)}\n")
seen = set()
for pattern, expected in EXPECTED.items():
    files = sorted(glob.glob(os.path.join(ROOT, pattern), recursive=True))
    if not files:
        print(f"MISSING  {pattern}")
        continue
    for path in files:
        seen.add(os.path.abspath(path))
        try:
            n, cols, note = inspect(path)
        except Exception as e:
            print(f"ERROR    {path}: {e}")
            continue
        ratio = n / expected
        status = "OK      " if 0.95 <= ratio <= 1.05 else "CHECK   "
        print(f"{status} {os.path.relpath(path, ROOT)}  [{human(os.path.getsize(path))}]  "
              f"rows={n:,} (expected ~{expected:,})  {note}")
        print(f"         columns: {cols}")

others = [p for p in glob.glob(os.path.join(ROOT, "**/*"), recursive=True)
          if os.path.isfile(p) and os.path.abspath(p) not in seen
          and p.endswith((".csv", ".txt", ".json", ".jsonl", ".parquet"))
          and "/results/" not in p]
if others:
    print("\nOther data files (fallback downloads, etc.):")
    for p in sorted(others):
        print(f"         {os.path.relpath(p, ROOT)}  [{human(os.path.getsize(p))}]")
