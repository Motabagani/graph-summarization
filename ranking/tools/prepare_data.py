"""Turn a full dataset file (any common format) into the JSONL that run.py / sweep.py read:
one line per document, {"id", "article", "summary"}.

Step 1 -- look at the file first (prints the format, column names and the first row):
    python ranking/tools/prepare_data.py --peek /path/to/cnndm/test.parquet

Step 2 -- convert it, naming the article and summary columns you saw:
    python ranking/tools/prepare_data.py cnndm /path/to/cnndm/test.parquet \\
        --article-col article --summary-col highlights --out /path/to/jsonl/test

    # a 1,000-doc dev subset for tuning, spread over the whole split
    python ranking/tools/prepare_data.py arxiv /path/to/arxiv/val.txt \\
        --article-col article_text --summary-col abstract_text --n 1000 --out /path/to/jsonl/dev

Writes <out>/<dataset>.jsonl. Reads .jsonl / .json / .txt (JSON lines) / .parquet / .csv / .tsv,
or a folder saved with Hugging Face datasets.save_to_disk (needs the `datasets` package).
List-valued columns are kept as lists (arXiv/PubMed sentences); <S> </S> tags are removed later.
"""

import argparse
import csv
import json
import os
import sys

NAMES = ["cnndm", "pubmed", "arxiv", "billsum", "wikihow", "tldr"]


def read_rows(path):
    path = os.path.expanduser(path)
    if os.path.isdir(path):
        from datasets import load_from_disk
        ds = load_from_disk(path)
        if hasattr(ds, "keys"):                # DatasetDict: pick a split explicitly
            sys.exit(f"{path} holds splits {list(ds.keys())}: pass {path}/<split> instead")
        return list(ds), "hf save_to_disk"
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pylist(), "parquet"
    if ext in (".csv", ".tsv"):
        csv.field_size_limit(sys.maxsize)
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f, delimiter="\t" if ext == ".tsv" else ",")), ext[1:]
    with open(path, encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":                       # one JSON array
            return json.load(f), "json array"
        return [json.loads(l) for l in f if l.strip()], "json lines"


def short(v, k=300):
    if isinstance(v, list):
        return f"list of {len(v)}: {[short(x, 80) for x in v[:3]]}"
    s = str(v)
    return repr(s[:k] + (" ..." if len(s) > k else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", nargs="?", choices=NAMES)
    ap.add_argument("path")
    ap.add_argument("--peek", action="store_true", help="print columns + first row, write nothing")
    ap.add_argument("--article-col")
    ap.add_argument("--summary-col")
    ap.add_argument("--id-col", help="optional; default: row number")
    ap.add_argument("--n", type=int, help="keep N docs spread evenly over the file (default: all)")
    ap.add_argument("--start-frac", type=float, default=0.0,
                    help="skip the first fraction of rows, e.g. 0.9 = use the last 10%% "
                         "(for WikiHow / TL;DR, which only have one split)")
    ap.add_argument("--end-frac", type=float, default=1.0)
    ap.add_argument("--out", help="output folder; writes <out>/<name>.jsonl")
    a = ap.parse_args()

    rows, fmt = read_rows(a.path)
    if a.peek:
        print(f"{a.path}\nformat: {fmt}, {len(rows)} rows\ncolumns: {list(rows[0])}\nfirst row:")
        for k, v in rows[0].items():
            print(f"  {k}: {short(v)}")
        return
    if not (a.name and a.article_col and a.summary_col and a.out):
        ap.error("converting needs: name, --article-col, --summary-col, --out (run --peek first)")
    for c in (a.article_col, a.summary_col):
        if c not in rows[0]:
            sys.exit(f"column {c!r} not in {list(rows[0])}")

    lo, hi = int(a.start_frac * len(rows)), int(a.end_frac * len(rows))
    idx = list(range(lo, hi))
    if a.n and a.n < len(idx):
        idx = [idx[int(i * len(idx) / a.n)] for i in range(a.n)]

    os.makedirs(a.out, exist_ok=True)
    dest = os.path.join(a.out, f"{a.name}.jsonl")
    kept = 0
    with open(dest, "w", encoding="utf-8") as f:
        for i in idx:
            r = rows[i]
            art, summ = r.get(a.article_col), r.get(a.summary_col)
            if not art or not summ:
                continue
            if isinstance(summ, list) and a.name not in ("arxiv", "pubmed"):
                summ = "\n".join(map(str, summ))
            if a.name == "wikihow" and isinstance(summ, str):   # same as the local samples
                summ = " ".join(s.strip().rstrip(",") for s in summ.split("\n") if s.strip())
            rid = r.get(a.id_col) if a.id_col else f"{a.name}-{i}"
            f.write(json.dumps({"id": str(rid), "article": art, "summary": summ}, ensure_ascii=False) + "\n")
            kept += 1
    print(f"{a.name}: wrote {kept} docs (rows {lo}-{hi} of {len(rows)}, {fmt}) -> {dest}")


if __name__ == "__main__":
    main()
