"""Sweep rankers / settings over the dataset samples and write one CSV row per config.

    python3 tools/sweep.py                      # all 6 samples in data_sample/
    python3 tools/sweep.py --only cnndm arxiv --out results/sweep.csv

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
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from pagetrust import degree_centrality, pacsum_signed_graph, pagetrust, threshold_edges  # noqa: E402
from pipeline import tfidf_similarity  # noqa: E402
from preprocess import preprocess  # noqa: E402
from rouge_lite import rouge  # noqa: E402
from run import PRESPLIT, presplit, ref_lines  # noqa: E402
from selector import BUDGETS, select, summary_text  # noqa: E402

BETAS = (0.0, 0.3, 0.6)
LAMBDA_BWD = (-2.0, -1.0, -0.5, 0.0)
PT_LAMBDA_BWD = (-2.0, -0.5)
GAMMAS = (0.0, 1.0, 2.0, np.inf)
MEMORIES = (0.0, 0.5, 0.9)


def run_dataset(name, path, limit=None):
    t0 = time.time()
    rows = [json.loads(l) for l in open(path, encoding="utf-8")][:limit]
    btype, budget = BUDGETS[name]
    totals: dict[tuple, np.ndarray] = {}

    for r in rows:
        art = presplit(r["article"]) if name in PRESPLIT else r["article"]
        doc = preprocess(art)
        if doc.n_units < 2:
            continue
        ref = ref_lines(r["summary"], name)
        cache: dict[tuple, np.ndarray] = {}

        def score(scores):
            chosen = tuple(select(doc.sentences, scores, btype, budget))
            if chosen not in cache:
                cache[chosen] = np.array(rouge(summary_text(doc.sentences, list(chosen), btype, budget), ref))
            return cache[chosen]

        def add(key, scores):
            totals[key] = totals.get(key, 0) + score(scores)

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

    n = len(rows)
    out = []
    for key, tot in totals.items():
        ranker = key[0]
        beta, lb, g, m = (list(key[1:]) + [""] * 4)[:4]
        r1, r2, rl = tot / n
        out.append({"dataset": name, "ranker": ranker, "beta": beta, "lambda_bwd": lb,
                    "gamma": g, "memory": m, "n_docs": n,
                    "R1": round(r1, 2), "R2": round(r2, 2), "RL": round(rl, 2)})
    print(f"{name}: {n} docs, {len(out)} configs, {time.time() - t0:.0f} s", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data_sample"))
    ap.add_argument("--only", nargs="*", default=list(BUDGETS))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "sweep_sample.csv"))
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    jobs = [(d, os.path.join(a.data, f"{d}.jsonl"), a.limit) for d in a.only]
    with ProcessPoolExecutor(max_workers=min(len(jobs), os.cpu_count() or 1)) as ex:
        results = [row for rows in ex.map(run_dataset, *zip(*jobs)) for row in rows]
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)
    print(f"wrote {len(results)} rows -> {a.out}")


if __name__ == "__main__":
    main()
