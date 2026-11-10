"""Turn hand-downloaded Hugging Face files into the same 100-doc JSONL samples that
fetch_samples.py writes (use this when the datasets-server API is down).

    python3 tools/convert_raw.py billsum ~/Downloads/test-00000-of-00001.parquet
    python3 tools/convert_raw.py tldr    ~/Downloads/0009.parquet
    python3 tools/convert_raw.py wikihow ~/Downloads/wikihowAll.csv

Needs pyarrow for .parquet files (pip install pyarrow); .csv uses the standard library.
Samples are spread over the file in chunks of 10 rows, like fetch_samples.py.
  billsum: the official test split (FiscalNote/billsum data/test-00000-of-00001.parquet)
  tldr   : the last parquet chunk of webis/tldr-17 (default/partial-train/0009.parquet),
           i.e. the end of the data, the proposal's 80/10/10 test part
  wikihow: wikihowAll.csv, rows from the last 5%
"""

import argparse
import csv
import json
import os
import sys

FIELDS = {  # name -> (article column, summary column, hf source label)
    "billsum": ("text", "summary", "FiscalNote/billsum/default/test"),
    "tldr": ("content", "summary", "webis/tldr-17/default/train (parquet 0009)"),
    "wikihow": ("text", "headline", "Lajvi/wikihowAll/default/train"),
}


def read_rows(path, cols):
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq
        t = pq.read_table(path, columns=list(cols))
        return t.to_pylist(), 0
    csv.field_size_limit(sys.maxsize)
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows, int(0.95 * len(rows))         # wikihow: sample from the last 5%


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", choices=list(FIELDS))
    ap.add_argument("path")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_sample"))
    a = ap.parse_args()
    af, sf, src = FIELDS[a.name]
    rows, start = read_rows(os.path.expanduser(a.path), (af, sf))
    chunks = max(1, a.n // 10)
    step = (len(rows) - start) // chunks
    out = []
    for c in range(chunks):
        for k in range(start + c * step, start + c * step + 10):
            if len(out) >= a.n or k >= len(rows):
                break
            art, summ = rows[k].get(af), rows[k].get(sf)
            if not art or not summ:
                continue
            if a.name == "wikihow":            # headline = one step per line
                summ = " ".join(s.strip().rstrip(",") for s in summ.split("\n") if s.strip())
            out.append({"id": f"{a.name}-{k}", "article": art, "summary": summ, "source": src})
    os.makedirs(a.out, exist_ok=True)
    dest = os.path.join(a.out, f"{a.name}.jsonl")
    with open(dest, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{a.name}: {len(out)} docs from {len(rows)} rows (from row {start}) -> {dest}")


if __name__ == "__main__":
    main()
