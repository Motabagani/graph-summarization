#!/usr/bin/env python3
"""
Gather every run into one place:
  results/summary.md   main table (all 11 baselines x 6 datasets, ROUGE-1/2/L F1) + direction-effect table
  results/summary.csv  the same numbers in long format
Missing runs are shown as "not run", so this is safe to run at any time.

    python baselines/collect_results.py [RESULTS_DIR]      (default: $RESULTS_DIR or ./results)
"""
import csv
import json
import os
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("RESULTS_DIR", "results")
DATASETS = ["cnn_dailymail", "pubmed", "arxiv", "billsum", "wikihow", "tldr17"]
NAMES = {"cnn_dailymail": "CNN/DM", "pubmed": "PubMed", "arxiv": "arXiv", "billsum": "BillSum",
         "wikihow": "WikiHow", "tldr17": "TL;DR-17"}
# (label, type, subfolder of RESULTS_DIR, run folder)
BASELINES = [
    ("Oracle",                          "Upper bound",                  "all",         "oracle"),
    ("Lead",                            "Aligned to PacSum / HipoRank", "all",         "lead"),
    ("PacSum (unnormalized)",           "Aligned to PacSum",            "all",         "pacsum_tfidf"),
    ("PacSum (normalized)",             "Variant",                      "sensitivity", "pacsum_cosine"),
    ("LexPageRank (corpus idf)",        "Paper-faithful",               "all",         "lexpagerank"),
    ("LexPageRank (per-article idf)",   "Variant",                      "sensitivity", "lexpagerank_docidf"),
    ("TextRank (Mihalcea 2004)",        "Paper-faithful",               "sensitivity", "textrank_mihalcea"),
    ("TextRank (unnormalized)",         "Aligned to PacSum",            "all",         "textrank"),
    ("TextRank (normalized)",           "Variant",                      "sensitivity", "textrank_cosine"),
    ("DegreeCentrality (unnormalized)", "Aligned to PacSum",            "all",         "degree_centrality"),
    ("DegreeCentrality (normalized)",   "Variant",                      "sensitivity", "degree_centrality_cosine"),
]
KEYS = ("rouge1", "rouge2", "rougeLsum")


def load(sub, ds, folder):
    p = os.path.join(ROOT, sub, ds, folder, "metrics.json")
    if not os.path.exists(p):
        return None
    m = json.load(open(p))
    if "rouge1" not in m["rouge"]:
        return None
    m["_scores"] = [100 * m["rouge"][k]["f1"] for k in KEYS]
    return m


def fmt(s):
    return "not run" if s is None else " / ".join(f"{x:.2f}" for x in s)


runs = {(b[0], ds): load(b[2], ds, b[3]) for b in BASELINES for ds in DATASETS}
md = ["# Results: ROUGE F1 x100 (R-1 / R-2 / R-L), no stemming; R-L = summary-level (Lsum)\n",
      "| Baseline | Type | " + " | ".join(NAMES[d] for d in DATASETS) + " | Average |",
      "|---|---|" + "---|" * (len(DATASETS) + 1)]
rows = []
for label, typ, _, _ in BASELINES:
    scores = [runs[(label, ds)]["_scores"] if runs[(label, ds)] else None for ds in DATASETS]
    done = [s for s in scores if s]
    avg = [sum(s[i] for s in done) / len(done) for i in range(3)] if len(done) == len(DATASETS) else None
    md.append(f"| {label} | {typ} | " + " | ".join(fmt(s) for s in scores) + f" | {fmt(avg) if avg else '-'} |")
    for ds, s in zip(DATASETS, scores):
        m = runs[(label, ds)]
        if s:
            sc = m["summary_config"]
            rows.append({"baseline": label, "type": typ, "dataset": ds,
                         "rouge1": round(s[0], 2), "rouge2": round(s[1], 2), "rougeL": round(s[2], 2),
                         "documents": m["documents"],
                         "length": (f"{sc['num_sentences']} sentences" if sc["num_sentences"]
                                    else f"{sc['word_limit']} words"),
                         "tuned": json.dumps(m.get("tuned", {}).get("best", ""))})

# Direction effect: PacSum (normalized) vs DegreeCentrality (normalized) -- same graph, with/without direction
md += ["\n## Effect of PacSum's position-based direction on the same cosine graph (ROUGE-1)\n",
       "| Dataset | DegreeCentrality (normalized, no direction) | PacSum (normalized, directed) "
       "| Direction effect | Tuned (beta, lambda1) |",
       "|---|---|---|---|---|"]
for ds in DATASETS:
    deg, pac = runs[("DegreeCentrality (normalized)", ds)], runs[("PacSum (normalized)", ds)]
    if deg and pac:
        t = pac.get("tuned", {}).get("best", {})
        md.append(f"| {NAMES[ds]} | {deg['_scores'][0]:.2f} | {pac['_scores'][0]:.2f} | "
                  f"{pac['_scores'][0] - deg['_scores'][0]:+.2f} | ({t.get('beta')}, {t.get('lambda1')}) |")
    else:
        md.append(f"| {NAMES[ds]} | - | - | - | - |")

os.makedirs(ROOT, exist_ok=True)
with open(os.path.join(ROOT, "summary.md"), "w") as f:
    f.write("\n".join(md) + "\n")
if rows:
    with open(os.path.join(ROOT, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
print("\n".join(md))
print(f"\nSaved {ROOT}/summary.md and summary.csv")
