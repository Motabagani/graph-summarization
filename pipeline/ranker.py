#!/usr/bin/env python3
"""
Stage 3 -- Ranker: document graphs -> sentence scores -> summaries -> ROUGE.

For each document it
  1. builds the signed, directed matrix W from the grapher's cached similarities and NSP scores,
     with Alfardil's graph/bert_graph.py: build_graph (beta threshold, NSP / position / no
     direction, negative edges when NSP rejects both orders, clause weight lambda),
  2. ranks the units with Hashim's ranking/pagetrust.py (pagetrust, pagerank or degree),
  3. turns unit scores into sentence scores (ranking/selector.py: sentence_scores; identity for
     sentence units, max over clauses by default),
  4. builds the summary and scores it exactly like the baselines (baselines/baselines/summarize.py:
     build_summary; rouge-score without stemming; same length per dataset, cut at N words).

Settings are chosen on the development documents (ROUGE-1 F1, like every tuned baseline), then the
best setting is run once on the test documents. Output (same format as the baselines, so
baselines/baselines/collect_results.py picks it up from results/ours/<dataset>/<name>/):
    tuning.json  every setting tried, its dev score, the one chosen
    metrics.json mean ROUGE-1/2/L/Lsum on test, settings, graph statistics
    predictions.jsonl  one summary per test document

    python ranker.py --dataset pubmed --dev-docs work/pubmed/dev.jsonl --dev-graphs work/pubmed/dev.bert \\
        --test-docs work/pubmed/test.jsonl --test-graphs work/pubmed/test.bert \\
        --out ../baselines/results/ours/pubmed/nsp_pagetrust
    # ablations: --ranker pagerank | degree, --direction position | none, --no-negative
    # sanity check (no graphs needed): --system lead must reproduce the baselines' Lead exactly
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

import pipeline_common as pc
import summarize
from pagetrust import degree_centrality, pagerank, pagetrust          # ranking/
from selector import sentence_scores                                  # ranking/


# --------------------------------------------------------------------------- graphs on disk
class GraphStore:
    """id -> (sim, nsp) from the grapher's .npz files, read lazily."""

    def __init__(self, folder: str):
        self.files = sorted(glob.glob(os.path.join(folder, "*.npz")))
        if not self.files:
            sys.exit(f"No graph files in {folder}: run grapher.py first")
        metas = [json.load(open(m)) for m in sorted(glob.glob(os.path.join(folder, "meta_shard*.json")))]
        self.meta = metas[0] if metas else {}
        self.index, self._open = {}, {}
        for f in self.files:
            with np.load(f) as z:
                for name in z.files:
                    if name.startswith("id_"):
                        self.index[str(z[name])] = (f, name[3:])

    def get(self, doc_id: str):
        f, k = self.index[doc_id]
        z = self._open.get(f)
        if z is None:
            z = self._open[f] = np.load(f)
        return z[f"sim_{k}"].astype(float), z[f"nsp_{k}"].astype(float)


class CachedScores:
    """Stands in for bert_graph.DocumentScores, with BERT outputs read from disk."""

    def __init__(self, vertices, sim, nsp):
        self.vertices, self.sim, self._L = vertices, sim, nsp

    @property
    def n(self):
        return len(self.vertices)

    def nsp(self, pairs):
        out = {}
        for i, j in pairs:
            a, b = self._L[i, j], self._L[j, i]
            if np.isnan(a) or np.isnan(b):
                raise KeyError(f"NSP score missing for pair {(i, j)}: rerun grapher.py with a lower --beta-min")
            out[(i, j)] = (float(a), float(b))
        return out


# --------------------------------------------------------------------------- one document
_W = {}


def _init(dataset, graph_dirs, graph_grid, rank_grid, aggregate, system, mode, stemmer):
    bg = pc.import_bert_graph()
    _W.update(bg=bg, cfg=pc.summary_config(dataset), stores={}, graph_dirs=graph_dirs,
              graph_grid=[bg.GraphConfig(**g) for g in graph_grid], rank_grid=rank_grid,
              aggregate=aggregate, system=system, mode=mode,
              rouge=pc.Rouge(("rouge1",) if mode == "tune" else ("rouge1", "rouge2", "rougeL", "rougeLsum"),
                             stemmer))


def _rank(W, r):
    if r["ranker"] == "pagetrust":
        return pagetrust(W, alpha=r["alpha"], gamma=r["gamma"], memory=r["memory"], distrust=r["distrust"])
    if r["ranker"] == "pagerank":
        return pagerank(W, alpha=r["alpha"])
    return degree_centrality(W)


def _summaries(doc, which):
    """Yield (config index, chosen sentence ids, summary sentences, graph info) for `doc`."""
    bg, sents = _W["bg"], doc["sentences"]
    if _W["system"] == "lead":
        scores = -np.arange(len(sents), dtype=float)
        _, chosen, out = summarize.build_summary(sents, scores, _W["cfg"])
        yield 0, chosen, out, {}
        return
    if doc["part"] not in _W["stores"]:
        _W["stores"][doc["part"]] = GraphStore(_W["graph_dirs"][doc["part"]])
    sim, nsp = _W["stores"][doc["part"]].get(doc["id"])
    scores = CachedScores(bg.vertices_from_units(doc["units"], doc["unit_sentence"]), sim, nsp)
    k = 0
    for gcfg in _W["graph_grid"]:
        graph = bg.build_graph(scores, gcfg)
        for r in _W["rank_grid"]:
            if which is None or k == which:
                unit_scores = _rank(graph.W, r)
                s = sentence_scores(unit_scores, doc["unit_sentence"], len(sents), _W["aggregate"])
                _, chosen, out = summarize.build_summary(sents, s, _W["cfg"])
                info = dict(graph.info)
                info["all_tied"] = bool(np.ptp(unit_scores) < 1e-12) if len(unit_scores) else True
                yield k, chosen, out, info
            k += 1


def _tune_doc(doc):
    res = {}
    for k, _, out, _ in _summaries(doc, None):
        res[k] = _W["rouge"].score(doc["reference"], out)["rouge1"][2] if out else 0.0
    return res


def _test_doc(args):
    doc, best = args
    _, chosen, out, info = next(_summaries(doc, best))
    rec = {"id": doc["id"], "summary": " ".join(out), "reference": " ".join(doc["reference"]),
           "selected": [int(c) for c in chosen], "num_sentences": len(doc["sentences"]),
           "num_units": len(doc["units"]), "graph": {k: v for k, v in info.items() if k != "threshold"}}
    if out:
        rec["rouge"] = {m: list(v) for m, v in _W["rouge"].score(doc["reference"], out).items()}
    return rec


# --------------------------------------------------------------------------- grids
def build_grids(a, beta_min, has_nsp):
    if a.system == "lead":
        return [{}], [{}]
    if a.direction == "nsp" and not has_nsp:
        sys.exit("These graphs have no NSP scores (tf-idf encoder): use --direction position or none")
    betas = [b for b in a.betas if b >= beta_min - 1e-9]
    if len(betas) < len(a.betas):
        print(f"[ranker] dropping betas below the grapher's --beta-min {beta_min}: "
              f"{sorted(set(a.betas) - set(betas))}", file=sys.stderr)
    fw = a.forward_weights if a.direction == "position" else [0.0]
    graph_grid = [dict(beta=b, lam=l, direction=a.direction, confidence=a.confidence,
                       allow_negative=not a.no_negative, forward_weight=f, backward_weight=1.0)
                  for b, l, f in itertools.product(betas, a.lams, fw)]
    if a.ranker == "pagetrust":
        rank_grid = [dict(ranker="pagetrust", alpha=a.alpha, gamma=g, memory=m, distrust=a.distrust)
                     for g, m in itertools.product(a.gammas, a.memories)]
    else:
        rank_grid = [dict(ranker=a.ranker, alpha=a.alpha)]
    return graph_grid, rank_grid


def label(graph_grid, rank_grid, k):
    g, r = graph_grid[k // len(rank_grid)], rank_grid[k % len(rank_grid)]
    return {**g, **r}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=list(pc.DATASETS))
    p.add_argument("--dev-docs"), p.add_argument("--dev-graphs")
    p.add_argument("--test-docs", required=True), p.add_argument("--test-graphs")
    p.add_argument("--out", required=True)
    p.add_argument("--system", choices=["graph", "lead"], default="graph",
                   help="lead = sanity check against the baselines (no graphs needed)")
    p.add_argument("--ranker", choices=["pagetrust", "pagerank", "degree"], default="pagetrust")
    p.add_argument("--direction", choices=["nsp", "position", "none"], default="nsp")
    p.add_argument("--confidence", choices=["logodds", "prob", "none"], default="logodds")
    p.add_argument("--no-negative", action="store_true", help="drop negative edges (ablation)")
    p.add_argument("--betas", type=float, nargs="+", default=[0.0, 0.3, 0.6])
    p.add_argument("--gammas", type=float, nargs="+", default=[0.0, 1.0, 2.0, float("inf")])
    p.add_argument("--memories", type=float, nargs="+", default=[0.0, 0.5, 0.9])
    p.add_argument("--lams", type=float, nargs="+", default=[0.0], help="clause units: try 0 0.5 1")
    p.add_argument("--forward-weights", type=float, nargs="+", default=[-2.0, -1.0, -0.5, 0.0],
                   help="--direction position: PacSum's lambda1 (backward weight fixed at 1)")
    p.add_argument("--alpha", type=float, default=0.85)
    p.add_argument("--distrust", choices=["binary", "weighted"], default="binary")
    p.add_argument("--aggregate", choices=["max", "mean", "sum"], default="max")
    p.add_argument("--rouge-stemmer", action="store_true", help="NOT the baselines' setting; for checks only")
    p.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)))
    a = p.parse_args(argv)

    graph_dirs = {"dev": a.dev_graphs, "test": a.test_graphs}
    meta = {}
    if a.system == "graph":
        if not a.test_graphs:
            sys.exit("--test-graphs is required (or use --system lead)")
        meta = GraphStore(a.test_graphs).meta
    graph_grid, rank_grid = build_grids(a, meta.get("beta_min", 0.0), meta.get("has_nsp", True))
    n_cfg = len(graph_grid) * len(rank_grid)
    os.makedirs(a.out, exist_ok=True)
    init = (a.dataset, graph_dirs, graph_grid, rank_grid, a.aggregate, a.system, None, a.rouge_stemmer)

    # ---- tune on dev
    best, tuned = 0, {}
    if n_cfg > 1:
        if not a.dev_docs or (a.system == "graph" and not a.dev_graphs):
            sys.exit(f"{n_cfg} settings to choose from: give --dev-docs and --dev-graphs (or single values)")
        t0, total, n = time.time(), np.zeros(n_cfg), 0
        with Pool(a.workers, initializer=_init, initargs=init[:6] + ("tune", a.rouge_stemmer)) as pool:
            for res in pool.imap(_tune_doc, pc.read_docs(a.dev_docs), chunksize=4):
                for k, v in res.items():
                    total[k] += v
                n += 1
        means = total / max(n, 1)
        best = int(np.argmax(means))
        order = np.argsort(-means, kind="stable")
        tuned = {"metric": "rouge1", "documents": n, "best": label(graph_grid, rank_grid, best),
                 "best_f1": float(means[best])}
        with open(os.path.join(a.out, "tuning.json"), "w") as f:
            json.dump({**tuned, "results": [{"params": label(graph_grid, rank_grid, int(k)), "f1": float(means[k])}
                                            for k in order]}, f, indent=2, default=str)
        print(f"[ranker] tuned {n_cfg} settings on {n} dev docs in {time.time() - t0:.0f}s; best "
              f"ROUGE-1 {means[best]:.4f}: {label(graph_grid, rank_grid, best)}", file=sys.stderr)

    # ---- evaluate on test
    t0, sums, n, stats = time.time(), {}, 0, {"neg": 0, "edges": 0, "tied": 0, "units": 0}
    with open(os.path.join(a.out, "predictions.jsonl"), "w", encoding="utf-8") as fout, \
            Pool(a.workers, initializer=_init, initargs=init[:6] + ("test", a.rouge_stemmer)) as pool:
        for rec in pool.imap(_test_doc, ((d, best) for d in pc.read_docs(a.test_docs)), chunksize=4):
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            g = rec["graph"]
            stats["neg"] += g.get("negative_edges", 0)
            stats["edges"] += g.get("negative_edges", 0) + g.get("directed", 0) + g.get("same_sentence", 0)
            stats["tied"] += g.get("all_tied", False)
            stats["units"] += rec["num_units"]
            for m, v in rec.get("rouge", {}).items():
                acc = sums.setdefault(m, [0.0, 0.0, 0.0, 0])
                for j in range(3):
                    acc[j] += v[j]
                acc[3] += 1
    scfg = pc.summary_config(a.dataset)
    metrics = {
        "method": os.path.basename(os.path.normpath(a.out)), "system": a.system, "dataset": a.dataset,
        "split": pc.DATASETS[a.dataset]["test"][0], "documents": n, "seconds": round(time.time() - t0, 2),
        "graphs": meta, "settings": label(graph_grid, rank_grid, best), "aggregate": a.aggregate,
        "summary_config": dict(splitter=scfg.splitter, newline_boundaries=scfg.newline_boundaries,
                               word_limit=scfg.word_limit, num_sentences=scfg.num_sentences,
                               truncate=scfg.truncate, order=scfg.order),
        "rouge_stemmer": a.rouge_stemmer,
        "graph_stats": {"mean_units": stats["units"] / max(n, 1),
                        "negative_edge_share": stats["neg"] / max(stats["edges"], 1),
                        "all_tied_docs": stats["tied"] / max(n, 1)} if a.system == "graph" else {},
        **({"tuned": tuned} if tuned else {}),
        "rouge": {m: {"precision": v[0] / v[3], "recall": v[1] / v[3], "f1": v[2] / v[3], "n": v[3]}
                  for m, v in sums.items()},
    }
    with open(os.path.join(a.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"[ranker] {a.dataset}: {n} test documents -> {a.out}")
    for m, v in metrics["rouge"].items():
        print(f"  {m:9s}  P={v['precision']:.4f}  R={v['recall']:.4f}  F1={v['f1']:.4f}")
    if metrics["graph_stats"]:
        print(f"  graphs: {metrics['graph_stats']}")


if __name__ == "__main__":
    main()
