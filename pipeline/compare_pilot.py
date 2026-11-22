#!/usr/bin/env python3
"""
Pilot comparison: pipeline variants vs the baselines on exactly the same test documents,
without re-running any baseline.

Each baseline run saved per-document ROUGE in its predictions.jsonl, so the baselines' scores on
any subset of documents can be read back. This script takes the documents the pilot summarized,
looks up the same document ids in the baseline predictions, and averages both over the documents
they share.

    python compare_pilot.py --pilot ../work/pilot --baselines /scratch/$USER/NLP/results
    -> ../work/pilot/pilot_summary.md
"""
from __future__ import annotations

import argparse
import json
import os

DATASETS = ["cnn_dailymail", "pubmed", "arxiv", "billsum", "wikihow", "tldr17"]
NAMES = {"cnn_dailymail": "CNN/DM", "pubmed": "PubMed", "arxiv": "arXiv", "billsum": "BillSum",
         "wikihow": "WikiHow", "tldr17": "TL;DR-17"}
BASELINES = [   # (label, subfolder, run folder)
    ("Oracle", "all", "oracle"), ("Lead", "all", "lead"),
    ("PacSum (unnormalized)", "all", "pacsum_tfidf"), ("PacSum (normalized)", "sensitivity", "pacsum_cosine"),
    ("LexPageRank (corpus idf)", "all", "lexpagerank"), ("LexPageRank (per-article idf)", "sensitivity", "lexpagerank_docidf"),
    ("TextRank (normalized)", "sensitivity", "textrank_cosine"), ("TextRank (Mihalcea 2004)", "sensitivity", "textrank_mihalcea"),
    ("TextRank (unnormalized)", "all", "textrank"), ("DegreeCentrality (normalized)", "sensitivity", "degree_centrality_cosine"),
    ("DegreeCentrality (unnormalized)", "all", "degree_centrality"),
]
METRICS = ("rouge1", "rouge2", "rougeLsum")


def per_doc(path, ids=None):
    """{id: [R-1, R-2, R-Lsum] F1} from a predictions.jsonl; stops once all wanted ids are found."""
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            i = str(r["id"])
            if ids is not None and i not in ids:
                continue
            if "rouge" in r:
                out[i] = [r["rouge"][m][2] for m in METRICS]
            if ids is not None and len(out) == len(ids):
                break
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pilot", required=True, help="pilot work folder (<pilot>/<dataset>/<variant>/predictions.jsonl)")
    p.add_argument("--baselines", required=True, help="baseline results folder containing all/ and sensitivity/")
    a = p.parse_args(argv)

    table, docs = {}, {}
    for ds in DATASETS:
        d = os.path.join(a.pilot, ds)
        if not os.path.isdir(d):
            continue
        variants = sorted(v for v in os.listdir(d) if os.path.exists(os.path.join(d, v, "predictions.jsonl")))
        if not variants:
            continue
        ours = {v: per_doc(os.path.join(d, v, "predictions.jsonl")) for v in variants}
        ids = set.intersection(*(set(o) for o in ours.values()))
        base = {lab: per_doc(os.path.join(a.baselines, sub, ds, run, "predictions.jsonl"), ids)
                for lab, sub, run in BASELINES}
        shared = set(ids)
        for lab, b in base.items():
            if b:
                shared &= set(b)
        docs[ds] = len(shared)
        rows = [(f"Pipeline: {v}", ours[v]) for v in variants] + [(lab, b) for lab, b in base.items() if b]
        for lab, scores in rows:
            vals = [scores[i] for i in shared]
            table.setdefault(lab, {})[ds] = ([100 * sum(x[k] for x in vals) / len(vals) for k in range(3)]
                                             if vals else None)

    shown = [ds for ds in DATASETS if ds in docs]
    md = ["# Pilot: pipeline vs baselines on the same test documents",
          "ROUGE F1 x100 (R-1 / R-2 / R-Lsum), no stemming. Same documents in every row of a column; "
          "baseline scores read from their saved per-document results.\n",
          "| System | " + " | ".join(f"{NAMES[d]} (n={docs[d]})" for d in shown) + " |",
          "|---|" + "---|" * len(shown)]
    for lab, row in table.items():
        cells = [" / ".join(f"{x:.2f}" for x in row[d]) if row.get(d) else "-" for d in shown]
        md.append(f"| {lab} | " + " | ".join(cells) + " |")
    out = os.path.join(a.pilot, "pilot_summary.md")
    with open(out, "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
