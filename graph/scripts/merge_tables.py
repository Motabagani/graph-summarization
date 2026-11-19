"""Merge every results/*/results.json into paper-ready tables (step 6).

    python scripts/merge_tables.py results/short results/long results/wikihow --out results/paper_tables

Writes <out>.md and <out>.tex:
  Table 1  main results: R-1 / R-2 / R-L per dataset, one row per system (bounds, baselines, ours)
  Table 2  ablations: full pipeline vs. each ablation, per dataset
Each table notes N, the number of dev/test docs, and the run folder each dataset came from.
If a dataset appears in more than one run, the last folder given wins.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ORDER = ["oracle", "random", "lead", "degree_tfidf", "textrank", "lexpagerank", "pacsum_tfidf",
         "full_pipeline", "ext_clauses"]
ABLATIONS = ["full_pipeline", "abl_position", "abl_positive_pagerank", "abl_degree"]
LABEL = {
    "oracle": "Oracle", "random": "Random", "lead": "Lead", "degree_tfidf": "Degree (TF-IDF)",
    "textrank": "TextRank", "lexpagerank": "LexPageRank", "pacsum_tfidf": "PacSum (TF-IDF)",
    "full_pipeline": "Ours: BERT-NSP + PageTrust", "ext_clauses": "Ours + clause vertices",
    "abl_position": "− NSP direction, + position heuristic",
    "abl_positive_pagerank": "− negative edges (PageRank)", "abl_degree": "− PageTrust (in-degree)",
}
DS_LABEL = {"cnn_dm": "CNN/DM", "arxiv": "arXiv", "pubmed": "PubMed", "tldr": "tldr-17",
            "wikihow": "WikiHow", "billsum": "BillSum", "synthetic": "Synthetic"}
DS_ORDER = ["cnn_dm", "tldr", "wikihow", "arxiv", "pubmed", "billsum", "synthetic"]


def load(folders):
    data, src = {}, {}
    for f in folders:
        p = Path(f) / "results.json"
        if not p.exists():
            print(f"skip {f}: no results.json")
            continue
        for ds, r in json.loads(p.read_text()).items():
            data[ds], src[ds] = r, str(f)
    return {k: data[k] for k in sorted(data, key=lambda d: DS_ORDER.index(d) if d in DS_ORDER else 99)}, src


def table(data, systems, best_from=None):
    """Rows = systems, column groups = datasets. Bold the best non-bound score per column."""
    dss = list(data)
    rows = [s for s in systems if any(s in data[d]["systems"] for d in dss)]
    best = {}
    for d in dss:
        cand = [s for s in rows if s in data[d]["systems"] and data[d]["systems"][s]["group"] != "bound"]
        for m in range(3):
            vals = [data[d]["systems"][s]["test"][m] for s in cand]
            best[(d, m)] = max(vals) if vals else None
    out = []
    for s in rows:
        cells = []
        for d in dss:
            r = data[d]["systems"].get(s)
            for m in range(3):
                if r is None:
                    cells.append(None)
                else:
                    v = r["test"][m]
                    cells.append((v, r["group"] != "bound" and v == best[(d, m)]))
        out.append((s, cells))
    return dss, out


def to_md(dss, rows, title):
    head = "| System | " + " | ".join(f"{DS_LABEL.get(d, d)} R-1 | R-2 | R-L" for d in dss) + " |"
    sep = "|---|" + "---:|" * (3 * len(dss))
    lines = [f"### {title}\n", head, sep]
    for s, cells in rows:
        txt = ["–" if c is None else (f"**{c[0]:.2f}**" if c[1] else f"{c[0]:.2f}") for c in cells]
        lines.append(f"| {LABEL.get(s, s)} | " + " | ".join(txt) + " |")
    return "\n".join(lines)


def to_tex(dss, rows, caption, label):
    cols = "l" + "rrr" * len(dss)
    top = " & ".join(rf"\multicolumn{{3}}{{c}}{{{DS_LABEL.get(d, d)}}}" for d in dss)
    sub = " & ".join(["R-1 & R-2 & R-L"] * len(dss))
    lines = [r"\begin{table*}[t]", r"\centering", r"\small", rf"\begin{{tabular}}{{{cols}}}", r"\toprule",
             rf"System & {top} \\", rf" & {sub} \\", r"\midrule"]
    prev_group = None
    for s, cells in rows:
        txt = ["--" if c is None else (rf"\textbf{{{c[0]:.2f}}}" if c[1] else f"{c[0]:.2f}") for c in cells]
        lines.append(f"{LABEL.get(s, s).replace('−', '$-$').replace('+', '$+$')} & " + " & ".join(txt) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table*}"]
    return "\n".join(lines)


def notes(data, src):
    out = []
    for d, r in data.items():
        ours = r["systems"].get("full_pipeline", {})
        g = ours.get("graph", {})
        params = ", ".join(f"{k}={v}" for k, v in ours.get("params", {}).items())
        out.append(f"- **{DS_LABEL.get(d, d)}**: N = {r['n']}, {r['n_dev']} dev / {r['n_test']} test docs "
                   f"(`{src[d]}`). Ours tuned: {params or '–'}. "
                   f"NSP directed {g.get('directed_share', '–')}, rejected {g.get('rejected_share', '–')}, "
                   f"all-tied docs {g.get('all_tied_docs', '–')}.")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="results/paper_tables")
    args = ap.parse_args()
    data, src = load(args.runs)
    if not data:
        raise SystemExit("no results found")
    cap = ("Test ROUGE F1 ($\\times$100, Google rouge-score, stemmed, ROUGE-Lsum). Parameters tuned on dev "
           "(mean of R-1/R-2/R-L). Best non-oracle score per column in bold.")
    t1 = table(data, ORDER)
    t2 = table(data, ABLATIONS)
    md = ["# Paper tables\n", to_md(*t1, "Table 1. Main results (test ROUGE F1 × 100)"), "",
          to_md(*t2, "Table 2. Ablations (test ROUGE F1 × 100)"), "", "Notes:", notes(data, src), ""]
    tex = [to_tex(*t1, "Main results. " + cap, "tab:main"), "",
           to_tex(*t2, "Ablations of the full pipeline. " + cap, "tab:ablation"), ""]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".md").write_text("\n".join(md))
    out.with_suffix(".tex").write_text("\n".join(tex))
    print("\n".join(md))
    print(f"wrote {out.with_suffix('.md')} and {out.with_suffix('.tex')}")


if __name__ == "__main__":
    main()
