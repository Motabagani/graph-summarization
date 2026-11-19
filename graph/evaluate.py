"""
Evaluate every system on each dataset: tune on dev, score on test, ROUGE-1/2/L F1.

    python evaluate.py --datasets cnn_dm billsum --limit-dev 200 --limit-test 1000
    python evaluate.py --datasets wikihow --wikihow-csv wikihowAll.csv
    python evaluate.py --synthetic --model /tmp/tinybert         # offline harness check

Outputs (in --out):
    results.md        one table per dataset (test F1 ×100, tuned params, N)
    results.json      the same, machine-readable, plus dev scores of every grid point
    per_doc.jsonl     test selections + ROUGE per system per document (error analysis)
    oracle_positions  where Oracle's sentences sit in each document (position analysis)

Summary length: by default the team's fixed per-dataset budgets (TEAM_BUDGETS, the same as
the baselines in the team repo's ranking/selector.py), so all systems are held to the same
length; `--length median` uses the median number of dev reference sentences instead, and
`--n K` forces K sentences.

Splitting: by default the team's (splitting.py): NLTK punkt sentences, arXiv/PubMed pre-split
at " ." tokens, references split like ranking/run.py, rule-based clauses, max over a
sentence's clause scores. `--splitter spacy --clause-splitter spacy --aggregate mean`
reproduces the original graphsum setup.

Tuning objective: mean of ROUGE-1, ROUGE-2 and ROUGE-L F1 on dev.
ROUGE is Google's rouge-score (Porter stemming, ROUGE-Lsum with one sentence per line),
which is close to but not identical to the Perl ROUGE-1.5.5 numbers in the literature.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import spacy
from rouge_score import rouge_scorer

from data import DATASETS, load_split, synthetic
from pipeline import Budget, build_systems, prepare, summary_text
from rankers import SOURCE as RANKER_SOURCE

METRICS = ("rouge1", "rouge2", "rougeLsum")
_scorer = rouge_scorer.RougeScorer(list(METRICS), use_stemmer=True)

# Per-dataset summary budgets shared with the team's baselines (ranking/selector.py).
# arXiv: the team notes say 200 in one place and 220 in another; selector.py uses 220.
TEAM_BUDGETS = {
    "cnn_dm": ("sentences", 3), "tldr": ("sentences", 1),
    "pubmed": ("words", 203), "arxiv": ("words", 220),
    "billsum": ("words", 184), "wikihow": ("words", 55),
}
ALIASES = {"cnndm": "cnn_dm"}                 # the team repo's dataset keys


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, file=sys.stderr, flush=True)


class RougeCache:
    def __init__(self, budget: Budget):
        self.c = {}
        self.budget = budget

    def __call__(self, doc, sel) -> tuple[float, float, float]:
        key = (doc.id, tuple(sel))
        if key not in self.c:
            pred = summary_text(doc.sentences, sel, self.budget)
            s = _scorer.score("\n".join(doc.ex.reference), pred)
            self.c[key] = tuple(100 * s[m].fmeasure for m in METRICS)
        return self.c[key]


def run_system(system, docs, budget, params, rouge):
    sels = [system.select(d, budget, **params) for d in docs]
    scores = np.array([rouge(d, s) for d, s in zip(docs, sels)]) if docs else np.zeros((0, 3))
    return sels, scores


def tune(system, dev, budget, rouge):
    if len(system.grid) == 1:
        return system.grid[0], []
    trials = []
    for k, params in enumerate(system.grid):
        _, sc = run_system(system, dev, budget, params, rouge)
        mean = sc.mean(0)
        trials.append({"params": params, "dev": mean.round(3).tolist(), "objective": float(mean.mean())})
        if (k + 1) % 12 == 0 or k + 1 == len(system.grid):
            log(f"    {system.name}: {k + 1}/{len(system.grid)} grid points")
    best = max(trials, key=lambda t: t["objective"])
    return best["params"], trials


def choose_budget(ds, dev_docs, args) -> Budget:
    if args.n:
        return Budget("sentences", args.n)
    if args.length == "team" and ds in TEAM_BUDGETS:
        kind, size = TEAM_BUDGETS[ds]
        return Budget(kind, size, args.overflow)
    return Budget("sentences", choose_n(dev_docs))


def choose_n(dev_docs) -> int:
    counts = [len(d.ex.reference) for d in dev_docs] or [3]
    return max(1, int(round(float(np.median(counts)))))


def oracle_positions(docs, sels) -> dict:
    rel = [i / max(len(d.sentences) - 1, 1) for d, s in zip(docs, sels) for i in s]
    hist, _ = np.histogram(rel, bins=10, range=(0, 1))
    return {"deciles": (hist / max(hist.sum(), 1)).round(3).tolist(),
            "mean_relative_position": round(float(np.mean(rel)), 3) if rel else None}


def fmt(p: dict) -> str:
    return ", ".join(f"{k}={'∞' if isinstance(v, float) and math.isinf(v) else v}" for k, v in p.items()) or "–"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["cnn_dm"], choices=list(DATASETS) + list(ALIASES))
    ap.add_argument("--synthetic", action="store_true", help="offline toy data instead of --datasets")
    ap.add_argument("--systems", nargs="+", default=None, help="subset of systems (default: all)")
    ap.add_argument("--model", default="bert-base-uncased")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--nsp-max-length", type=int, default=256,
                    help="token cap per NSP pair (BERT max 512); lower is much faster")
    ap.add_argument("--limit-dev", type=int, default=200)
    ap.add_argument("--limit-test", type=int, default=1000)
    ap.add_argument("--max-sentences", type=int, default=None,
                    help="truncate documents to their first K sentences (all systems alike)")
    ap.add_argument("--betas", type=float, nargs="+", default=None,
                    help="β grid (default 0 0.3 0.6). Dropping 0 avoids running NSP on every pair")
    ap.add_argument("--length", choices=["team", "median"], default="team",
                    help="team: fixed per-dataset budgets (TEAM_BUDGETS); median: median dev reference "
                         "sentences (always used for --synthetic)")
    ap.add_argument("--n", type=int, default=None, help="force a summary length of N sentences")
    ap.add_argument("--overflow", choices=["include", "skip", "truncate"], default="include",
                    help="word budgets: what to do with the sentence that crosses the limit")
    ap.add_argument("--clauses", action="store_true", help="also run the clause-vertex extension")
    ap.add_argument("--splitter", choices=["nltk", "spacy"], default="nltk",
                    help="sentence splitter for documents and references (team: nltk)")
    ap.add_argument("--clause-splitter", choices=["rules", "spacy"], default="rules",
                    help="clause vertices: team Preprocessor rules, or spaCy dependencies")
    ap.add_argument("--aggregate", choices=["max", "mean", "sum"], default="max",
                    help="clause scores -> sentence score (team selector: max)")
    ap.add_argument("--wikihow-csv", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    args.datasets = [ALIASES.get(d, d) for d in args.datasets]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    nlp = spacy.load("en_core_web_sm")
    systems = build_systems(clauses=args.clauses, seed=args.seed, betas=args.betas, aggregate=args.aggregate)
    if args.systems:
        systems = {k: v for k, v in systems.items() if k in args.systems or k == "oracle"}

    encoder = None
    if any(s.needs_bert for s in systems.values()):
        from bert_graph import FrozenBert
        encoder = FrozenBert(args.model, batch_size=args.batch_size, device=args.device,
                             max_length=args.nsp_max_length)
        log(f"loaded {args.model} on {encoder.device}")

    names = ["synthetic"] if args.synthetic else args.datasets
    all_results = {}
    per_doc = open(out / "per_doc.jsonl", "w")
    for ds in names:
        t0 = time.time()
        if args.synthetic:
            dev_ex, test_ex = synthetic(args.limit_dev, args.seed), synthetic(args.limit_test, args.seed + 1)
        else:
            log(f"[{ds}] loading dev/test")
            dev_ex = load_split(ds, "dev", args.limit_dev, nlp, args.seed, args.wikihow_csv, args.splitter)
            test_ex = load_split(ds, "test", args.limit_test, nlp, args.seed, args.wikihow_csv, args.splitter)
        log(f"[{ds}] preprocessing {len(dev_ex)} dev + {len(test_ex)} test docs")
        split_kw = dict(splitter=args.splitter, clause_splitter=args.clause_splitter)
        dev = [prepare(e, nlp, encoder, args.max_sentences, **split_kw) for e in dev_ex]
        test = [prepare(e, nlp, encoder, args.max_sentences, **split_kw) for e in test_ex]
        dev = [d for d in dev if d.sentences]
        test = [d for d in test if d.sentences]
        budget = choose_budget(ds, dev, args)
        log(f"[{ds}] budget = {budget}; avg sentences/doc = {np.mean([len(d.sentences) for d in test]):.1f}")

        rouge = RougeCache(budget)
        res = {"n": budget.size, "budget": str(budget), "n_dev": len(dev), "n_test": len(test),
               "splitting": dict(split_kw, aggregate=args.aggregate), "pagetrust": RANKER_SOURCE,
               "systems": {}}
        for name, sys_ in systems.items():
            ts = time.time()
            log(f"[{ds}] {name}: tuning over {len(sys_.grid)} setting(s)")
            params, trials = tune(sys_, dev, budget, rouge)
            sels, sc = run_system(sys_, test, budget, params, rouge)
            mean = sc.mean(0)
            res["systems"][name] = {"group": sys_.group, "params": params,
                                    "test": mean.round(2).tolist(), "dev_trials": trials,
                                    "seconds": round(time.time() - ts, 1)}
            for d, s, r in zip(test, sels, sc):
                per_doc.write(json.dumps({"dataset": ds, "system": name, "doc": d.id,
                                          "n_sentences": len(d.sentences), "selected": s,
                                          "rouge": r.round(2).tolist()}, default=float) + "\n")
            if sys_.needs_bert:
                infos = [d.cache.get("last_info") for d in test if d.cache.get("last_info")]
                kept = sum(i["kept"] for i in infos) or 1
                res["systems"][name]["graph"] = {
                    "directed_share": round(sum(i["directed"] for i in infos) / kept, 3),
                    "rejected_share": round(sum(i["rejected_pairs"] for i in infos) / kept, 3),
                    "all_tied_docs": round(float(np.mean([bool(d.cache.get("last_tied")) for d in test])), 3),
                }
            if name == "oracle":
                res["oracle_positions"] = oracle_positions(test, sels)
            log(f"[{ds}] {name}: R1/R2/RL = {mean[0]:.2f}/{mean[1]:.2f}/{mean[2]:.2f}  ({fmt(params)})")
        for d in dev + test:
            d.drop_bert_cache()
        res["seconds"] = round(time.time() - t0, 1)
        all_results[ds] = res
    per_doc.close()

    (out / "results.json").write_text(json.dumps(all_results, indent=2, default=lambda v: "inf" if v == np.inf else float(v)))
    md = ["# Results (test, ROUGE F1 × 100)\n",
          f"Tuned on dev (objective: mean of R-1/R-2/R-L F1). Model: `{args.model}`. "
          f"Docs truncated to {args.max_sentences or 'no'} sentence limit. "
          f"Splitting: sentences={args.splitter}, clauses={args.clause_splitter}, "
          f"clause aggregate={args.aggregate}. PageRank/PageTrust from `{RANKER_SOURCE}`.\n"]
    for ds, r in all_results.items():
        md += [f"\n## {ds}\n", f"Budget = {r.get('budget', str(r['n']) + ' sentences')} · {r['n_dev']} dev docs · {r['n_test']} test docs\n",
               "| System | Group | R-1 | R-2 | R-L | Tuned params |", "|---|---|---:|---:|---:|---|"]
        for name, s in r["systems"].items():
            r1, r2, rl = s["test"]
            md.append(f"| {name} | {s['group']} | {r1:.2f} | {r2:.2f} | {rl:.2f} | {fmt(s['params'])} |")
        g = {k: s["graph"] for k, s in r["systems"].items() if "graph" in s}
        if g:
            md += ["\nGraph diagnostics on test (share of kept cross-sentence pairs that NSP directed / "
                   "rejected both ways; share of docs where every sentence tied, which reduces to Lead):\n",
                   "| System | directed | rejected | all-tied docs |", "|---|---:|---:|---:|"]
            md += [f"| {k} | {v['directed_share']} | {v['rejected_share']} | {v['all_tied_docs']} |" for k, v in g.items()]
        if "oracle_positions" in r:
            op = r["oracle_positions"]
            md.append(f"\nOracle sentence positions by document decile (first → last): {op['deciles']}; "
                      f"mean relative position {op['mean_relative_position']}.")
    (out / "results.md").write_text("\n".join(md) + "\n")
    log(f"wrote {out}/results.md, results.json, per_doc.jsonl")
    print("\n".join(md))


if __name__ == "__main__":
    main()
