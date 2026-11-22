#!/usr/bin/env python3
"""
Stage 2 -- Grapher: standard documents -> the expensive half of each document graph.

For every document from preprocessor.py this runs Alfardil's frozen-BERT code
(graph/bert_graph.py: FrozenBert, DocumentScores) on the document's units and stores
    sim  (n x n)  BERT similarity of every unit pair (dot product of mean-pooled vectors)
    nsp  (n x n)  NSP isNext log-odds, nsp[a, b] = "unit b follows unit a"; NaN where not computed
NSP is computed (both orders) for every pair the graph can need: units from different sentences
whose similarity passes PacSum's threshold at --beta-min (graph/bert_graph.py: build_graph).
The ranker turns these into the signed, directed matrix W for any beta >= --beta-min, any
direction rule and any clause weight, without running BERT again. That split matters because
BERT is >99% of the cost and needs a GPU, while tuning the ranker needs many cheap CPU passes.

Encoders:
    bert   frozen BERT (default bert-base-uncased): Alfardil's Grapher + NSP Directioner
    tfidf  tf-idf cosine on the cleaned tokens (ranking/pipeline.py: tfidf_similarity), no NSP:
           Hashim's stand-in graph, CPU only; use with --direction position / none in the ranker
    fake   hashed bag-of-words + toy NSP, for tests only

Output: a folder of .npz files (one per --docs-per-file documents) plus meta.json.

    python grapher.py work/pubmed/dev.jsonl  --out work/pubmed/dev.bert  --beta-min 0.3
    python grapher.py work/pubmed/test.jsonl --out work/pubmed/test.bert --beta-min 0.3 \\
           --shard $SLURM_ARRAY_TASK_ID --num-shards 8          # split one big file over GPUs

Cost: NSP runs BERT twice per kept pair. beta-min = 0 keeps every pair: ~40,000 BERT passes for a
200-sentence arXiv paper. Use --beta-min 0.3 (or 0.6) on PubMed / arXiv / BillSum.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib

import numpy as np

import pipeline_common as pc


class FakeEncoder:
    """Deterministic stand-in for FrozenBert (tests only): hashed bag-of-words vectors, and an
    NSP score that likes word overlap and penalizes the second text being shorter."""

    def __init__(self, dim: int = 64):
        self.dim = dim

    def _vec(self, t):
        v = np.zeros(self.dim)
        for w in t.lower().split():
            v[zlib.crc32(w.encode()) % self.dim] += 1.0
        return v

    def embed(self, texts):
        return np.array([self._vec(t) for t in texts]).reshape(len(texts), self.dim)

    def nsp_logodds(self, firsts, seconds):
        out = []
        for a, b in zip(firsts, seconds):
            sa, sb = set(a.lower().split()), set(b.lower().split())
            out.append(len(sa & sb) / max(1, len(sa | sb)) * 6 - 1 + 0.1 * (len(sb) - len(sa)))
        return np.array(out, dtype=float)


def nsp_pairs(sim: np.ndarray, sentence_ids, beta_min: float):
    """Pairs build_graph will ask NSP about for any beta >= beta_min (same threshold rule)."""
    n = sim.shape[0]
    if n < 2:
        return []
    iu = np.triu_indices(n, k=1)
    off = sim[iu]
    lo, hi = float(off.min()), float(off.max())
    thresh = lo + beta_min * (hi - lo)
    sid = np.asarray(sentence_ids)
    return [(int(i), int(j)) for i, j in zip(*iu) if sid[i] != sid[j] and sim[i, j] >= thresh]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("docs", help="standard .jsonl from preprocessor.py")
    p.add_argument("--out", required=True, help="output folder")
    p.add_argument("--encoder", choices=["bert", "tfidf", "fake"], default="bert")
    p.add_argument("--model", default="bert-base-uncased")
    p.add_argument("--pooling", choices=["mean", "cls"], default="mean")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", help="cuda / cpu (default: cuda if available)")
    p.add_argument("--beta-min", type=float, default=0.0,
                   help="smallest beta the ranker may use; NSP is only run for pairs that can matter")
    p.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--docs-per-file", type=int, default=200)
    p.add_argument("--limit", type=int, default=0, help="first N documents only (0 = all)")
    a = p.parse_args(argv)

    bg = pc.import_bert_graph(need_torch=(a.encoder == "bert"))
    if a.encoder == "bert":
        enc = bg.FrozenBert(a.model, pooling=a.pooling, batch_size=a.batch_size, device=a.device)
    elif a.encoder == "fake":
        enc = FakeEncoder()
    else:
        tfidf_similarity = pc.load_module("ranking_pipeline", os.path.join(pc.RANKING_DIR, "pipeline.py")).tfidf_similarity

    os.makedirs(a.out, exist_ok=True)
    meta = {"docs": os.path.abspath(a.docs), "encoder": a.encoder,
            "model": a.model if a.encoder == "bert" else None, "pooling": a.pooling,
            "beta_min": a.beta_min, "has_nsp": a.encoder != "tfidf",
            "num_shards": a.num_shards}
    with open(os.path.join(a.out, f"meta_shard{a.shard:03d}.json"), "w") as f:
        json.dump(meta, f, indent=2)

    buf, part, n_docs, n_pairs, t0 = {}, 0, 0, 0, time.time()

    def flush():
        nonlocal buf, part
        if buf:
            np.savez_compressed(os.path.join(a.out, f"shard{a.shard:03d}_{part:04d}.npz"), **buf)
            buf, part = {}, part + 1

    for k, doc in enumerate(pc.read_docs(a.docs)):
        if a.limit and k >= a.limit:
            break
        if k % a.num_shards != a.shard:
            continue
        units, sids = doc["units"], doc["unit_sentence"]
        n = len(units)
        if a.encoder == "tfidf":
            sim = tfidf_similarity(doc["unit_tokens"]).astype(np.float32)
            nsp = np.full((n, n), np.nan, dtype=np.float32)
        else:
            scores = bg.DocumentScores(bg.vertices_from_units(units, sids), enc)
            sim = np.asarray(scores.sim, dtype=np.float32)
            pairs = nsp_pairs(sim, sids, a.beta_min)
            scores.nsp(pairs)                                   # BERT runs here (both orders)
            nsp = np.full((n, n), np.nan, dtype=np.float32)
            for (i, j), v in scores._nsp.items():
                nsp[i, j] = v
            n_pairs += len(pairs)
        key = str(len(buf) // 3)
        buf[f"id_{key}"] = np.array(doc["id"])
        buf[f"sim_{key}"] = sim
        buf[f"nsp_{key}"] = nsp
        n_docs += 1
        if len(buf) // 3 >= a.docs_per_file:
            flush()
        if n_docs % 50 == 0:
            print(f"  {n_docs} docs, {n_pairs} NSP pairs, {n_docs / (time.time() - t0):.2f} docs/s",
                  file=sys.stderr, flush=True)
    flush()
    print(f"[grapher] shard {a.shard}/{a.num_shards}: {n_docs} documents, {n_pairs} NSP pairs "
          f"in {time.time() - t0:.0f}s -> {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
