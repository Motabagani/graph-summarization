#!/usr/bin/env python3
"""
Compare our runs with the numbers published in PacSum (Zheng & Lapata 2019, Table 2) and
HipoRank (Dong et al. 2021, Tables 3-4). Writes results/alignment.md.

    python baselines/compare_published.py [RESULTS_DIR]      (default: $RESULTS_DIR or ./results)

PubMed/arXiv are compared on ROUGE-1/2 only: HipoRank's ROUGE-L setup cannot be reproduced ("n/c").
"""
import json
import os
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("RESULTS_DIR", "results")
# (baseline, run folder under results/all, dataset, source, published R-1, R-2, R-L or None)
PUBLISHED = [
    ("PacSum (unnormalized)",           "pacsum_tfidf",      "cnn_dailymail", "PacSum, Table 2",             39.2, 16.3, 35.3),
    ("TextRank (unnormalized)",         "textrank",          "cnn_dailymail", "PacSum, Table 2",             33.2, 11.8, 29.6),
    ("DegreeCentrality (unnormalized)", "degree_centrality", "cnn_dailymail", "PacSum, Table 2",             33.0, 11.7, 29.5),
    ("Lead",                            "lead",              "cnn_dailymail", "PacSum, Table 2",             40.5, 17.7, 36.7),
    ("Lead",                            "lead",              "pubmed",        "HipoRank, Table 3",           35.63, 12.28, None),
    ("Lead",                            "lead",              "arxiv",         "HipoRank, Table 4",           33.66, 8.94, None),
    ("LexPageRank (corpus idf)",        "lexpagerank",       "pubmed",        "HipoRank (LexRank), Table 3", 39.19, 13.89, None),
    ("LexPageRank (corpus idf)",        "lexpagerank",       "arxiv",         "HipoRank (LexRank), Table 4", 33.85, 10.73, None),
    ("Oracle",                          "oracle",            "cnn_dailymail", "PacSum, Table 2",             54.7, 30.4, 50.8),
    ("Oracle",                          "oracle",            "pubmed",        "HipoRank, Table 3",           55.05, 27.48, None),
    ("Oracle",                          "oracle",            "arxiv",         "HipoRank, Table 4",           53.88, 23.05, None),
]


def num(v):
    return "n/c" if v is None else f"{v:.2f}"


rows, absdiff = [], {}
for name, folder, ds, src, *pub in PUBLISHED:
    p = os.path.join(ROOT, "all", ds, folder, "metrics.json")
    ours = diff = None
    if os.path.exists(p):
        r = json.load(open(p))["rouge"]
        ours = [100 * r[k]["f1"] for k in ("rouge1", "rouge2", "rougeLsum")]
        ours = [o if q is not None else None for o, q in zip(ours, pub)]
        diff = [o - q if q is not None else None for o, q in zip(ours, pub)]
        absdiff.setdefault(name, []).extend(abs(d) for d in diff if d is not None)
    rows.append((name, ds, src, ours, pub, diff))

ranked = sorted(absdiff, key=lambda n: sum(absdiff[n]) / len(absdiff[n]))
ranked += [n for n in dict.fromkeys(r[0] for r in rows) if n not in absdiff]
md = ["# Alignment with published results (ours minus published; ROUGE F1 x100, no stemming)\n",
      "| Baseline | Dataset | Source | Ours (R-1 / R-2 / R-L) | Published | Difference | Mean abs. diff |",
      "|---|---|---|---|---|---|---|"]
for name in ranked:
    mean = f"{sum(absdiff[name]) / len(absdiff[name]):.2f}" if name in absdiff else "-"
    for n, ds, src, ours, pub, diff in rows:
        if n != name:
            continue
        o = "not run" if ours is None else " / ".join(num(x) for x in ours)
        d = "-" if diff is None else " / ".join("n/c" if x is None else f"{x:+.2f}" for x in diff)
        md.append(f"| {n} | {ds} | {src} | {o} | {' / '.join(num(x) for x in pub)} | {d} | {mean} |")

os.makedirs(ROOT, exist_ok=True)
with open(os.path.join(ROOT, "alignment.md"), "w") as fh:
    fh.write("\n".join(md) + "\n")
print("\n".join(md))
print(f"\nSaved {ROOT}/alignment.md")
