"""Sweep rankers / settings over the dataset samples and write one CSV row per config.

    python3 tools/sweep.py                      # all 6 samples in data_sample/ -> results/sweep_sample.csv
    python3 tools/sweep.py --only cnndm arxiv --tag mytest
    python3 tools/sweep.py --data /path/to/dev_jsonl --only arxiv --tag dev --rouge official

--data is a folder holding <dataset>.jsonl files (cnndm, pubmed, arxiv, billsum, wikihow,
tldr); missing ones are skipped. Documents are spread over all CPUs the job was given.

Configs per dataset:
  lead                                       first sentences
  degree    beta x lambda_bwd                PacSum centrality on the stand-in graph
  pagetrust beta x lambda_bwd x gamma x M    gamma = 0 is PageRank on the positive edges
(lambda_fwd = 1 throughout: positive weight on edges to LATER sentences, as in PacSum.)

The samples are test documents, so the best config here is for debugging only --
tune on dev splits on the HPC before reporting anything.
"""

import argparse
import csv
import itertools
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from pagetrust import degree_centrality, pacsum_signed_graph, pagetrust, threshold_edges  # noqa: E402
from pipeline import tfidf_similarity  # noqa: E402
from preprocess import preprocess  # noqa: E402
from preprocess import SPLITTER, STOP_LIST  # noqa: E402
from rouge_lite import get_scorer  # noqa: E402
from run import PRESPLIT, n_workers, presplit, ref_lines  # noqa: E402
from selector import BUDGETS, select, summary_text  # noqa: E402

BETAS = (0.0, 0.3, 0.6)
LAMBDA_BWD = (-2.0, -1.0, -0.5, 0.0)
PT_LAMBDA_BWD = (-2.0, -0.5)
GAMMAS = (0.0, 1.0, 2.0, np.inf)
MEMORIES = (0.0, 0.5, 0.9)


def score_doc(args):
    """All configs for one document -> {config key: (R1, R2, RL)}, or None if skipped."""
    name, row, scorer = args
    rouge = get_scorer(scorer)
    btype, budget = BUDGETS[name]
    art = presplit(row["article"]) if name in PRESPLIT else row["article"]
    doc = preprocess(art)
    if doc.n_units < 2:
        return None
    ref = ref_lines(row["summary"], name)
    cache: dict = {}
    out: dict = {}

    def add(key, scores):
        chosen = tuple(select(doc.sentences, scores, btype, budget))
        if chosen not in cache:
            cache[chosen] = np.array(rouge(summary_text(doc.sentences, list(chosen), btype, budget), ref))
        out[key] = cache[chosen]

    add(("lead",), -np.arange(doc.n_units, dtype=float))
    sim = tfidf_similarity(doc.unit_tokens)
    for beta in BETAS:
        e = threshold_edges(sim, beta)
        for lb in LAMBDA_BWD:
            W = pacsum_signed_graph(e, 1.0, lb)
            add(("degree", beta, lb), degree_centrality(W))
            if lb not in PT_LAMBDA_BWD:
                continue
            for g, m in itertools.product(GAMMAS, MEMORIES):
                if g == 0 and m != 0:      # gamma = 0 ignores blacklists: M is irrelevant
                    continue
                add(("pagetrust", beta, lb, g, m), pagetrust(W, alpha=0.85, gamma=g, memory=m))
    return out


def run_dataset(name, path, pool, limit=None, scorer="lite"):
    t0 = time.time()
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(l) for l, _ in zip(f, range(limit or 10**9))]
    totals: dict = {}
    n_scored = 0
    for i, res in enumerate(pool.imap_unordered(score_doc, ((name, r, scorer) for r in rows), chunksize=4), 1):
        if res is not None:
            n_scored += 1
            for k, v in res.items():
                totals[k] = totals.get(k, 0) + v
        if i % 500 == 0:
            print(f"  {name}: {i}/{len(rows)} docs, {time.time() - t0:.0f} s", flush=True)

    out = []
    for key, tot in totals.items():
        ranker = key[0]
        beta, lb, g, m = (list(key[1:]) + [""] * 4)[:4]
        r1, r2, rl = tot / max(n_scored, 1)
        out.append({"dataset": name, "ranker": ranker, "beta": beta, "lambda_bwd": lb,
                    "gamma": g, "memory": m, "n_docs": n_scored, "n_skipped": len(rows) - n_scored,
                    "R1": round(r1, 2), "R2": round(r2, 2), "RL": round(rl, 2)})
    print(f"{name}: {n_scored} docs scored ({len(rows) - n_scored} with <2 sentences skipped), "
          f"{len(out)} configs, {time.time() - t0:.0f} s", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data_sample"), help="folder of <dataset>.jsonl files")
    ap.add_argument("--only", nargs="*", default=list(BUDGETS), choices=list(BUDGETS))
    ap.add_argument("--limit", type=int, help="first N docs of each file")
    ap.add_argument("--rouge", default="lite", choices=["lite", "official"])
    ap.add_argument("--tag", default="sample", help="output name: results/sweep_<tag>.csv")
    ap.add_argument("--out", help="explicit output CSV path (overrides --tag)")
    ap.add_argument("--workers", type=int, help="default: CPUs given to the job")
    a = ap.parse_args()
    out_path = a.out or os.path.join(HERE, "results", f"sweep_{a.tag}.csv")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    workers = a.workers or n_workers()
    print(f"splitter={SPLITTER} stop_list={STOP_LIST} rouge={a.rouge} workers={workers}", flush=True)

    results = []
    with Pool(workers) as pool:
        for d in a.only:
            path = os.path.join(a.data, f"{d}.jsonl")
            if not os.path.exists(path):
                print(f"{d}: {path} not found, skipped", flush=True)
                continue
            results += run_dataset(d, path, pool, a.limit, a.rouge)
    if not results:
        sys.exit("no datasets found -- check --data")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)
    print(f"wrote {len(results)} rows -> {out_path}")


if __name__ == "__main__":
    main()
