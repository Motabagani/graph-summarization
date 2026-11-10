"""Fetch ~100-document local samples of the 6 evaluation datasets from Hugging Face
(datasets-server API, JSON, no `datasets` library needed). For building / debugging
only -- reported numbers come from Finley's full copies on the HPC.

Usage:  python3 tools/fetch_samples.py [--n 100] [--out data_sample]
Writes one JSONL per dataset: {"id", "article", "summary", "source"}.
  article: str (raw text; split with NLTK like the baselines) or list[str] (pre-split)
"""

import argparse
import json
import os
import time
import urllib.parse
import urllib.request

API = "https://datasets-server.huggingface.co"

# name -> (hf dataset, config, split, article field, summary field, take from)
# "take from" = fraction into the split where the sample starts; train-only sets
# (TL;DR-17, WikiHow) sample from the last 10%, i.e. the proposal's 80/10/10 test part.
SOURCES = {
    "cnndm":   ("abisee/cnn_dailymail", "3.0.0", "test", "article", "highlights", 0.0),
    "arxiv":   ("ccdv/arxiv-summarization", "document", "test", "article", "abstract", 0.0),
    "pubmed":  ("ccdv/pubmed-summarization", "document", "test", "article", "abstract", 0.0),
    "billsum": ("FiscalNote/billsum", "default", "test", "text", "summary", 0.0),
    "wikihow": ("Lajvi/wikihowAll", "default", "train", "text", "headline", 0.95),
    "tldr":    ("webis/tldr-17", "default", "train", "content", "summary", 0.95),
}


def get(path, **params):
    url = f"{API}/{path}?{urllib.parse.urlencode(params)}"
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                out = json.load(r)
            time.sleep(0.5)                  # stay under the API's rate limit
            return out
        except Exception as e:  # rate limits (503), outages (502), timeouts
            if attempt == 3:
                raise SystemExit(f"Hugging Face API not responding ({e}). "
                                 "Try again in a few minutes; files already written are kept.")
            print(f"  ...{e}, retrying in {5 * (attempt + 1)} s", flush=True)
            time.sleep(5 * (attempt + 1))


def n_rows(ds, config, split):
    info = get("size", dataset=ds)
    for s in info["size"]["splits"]:
        if s["config"] == config and s["split"] == split:
            return s["num_rows"]
    raise KeyError((ds, config, split))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_sample"))
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for name, (ds, cfg, split, af, sf, frac) in SOURCES.items():
        if a.only and name not in a.only:
            continue
        total = n_rows(ds, cfg, split)
        start = int(frac * total)
        # spread the sample over the whole split in chunks of 10 rows: splits are often
        # ordered by source (CNN/DM test = all CNN first, then DailyMail)
        chunks = max(1, a.n // 10)
        step = (total - start) // chunks
        offset = start
        rows = []
        for c in range(chunks):
            print(f"{name}: chunk {c + 1}/{chunks}", flush=True)
            page = get("rows", dataset=ds, config=cfg, split=split, offset=start + c * step,
                       length=min(10, a.n - 10 * c))["rows"]
            for r in page:
                row = r["row"]
                art, summ = row.get(af), row.get(sf)
                if not art or not summ:
                    continue
                if name in ("arxiv", "pubmed"):      # pre-split: one sentence per line
                    art = [s for s in art.split("\n") if s.strip()]
                if name == "wikihow":                # headline = one step per line
                    summ = " ".join(s.strip().rstrip(",") for s in summ.split("\n") if s.strip())
                rows.append({"id": f"{name}-{r['row_idx']}", "article": art, "summary": summ,
                             "source": f"{ds}/{cfg}/{split}"})
        path = os.path.join(a.out, f"{name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)} docs from {ds} ({cfg}/{split}, rows {offset}-{total}) -> {path}")


if __name__ == "__main__":
    main()
