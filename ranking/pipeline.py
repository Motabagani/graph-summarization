"""End-to-end glue for Hashim's part: Preprocessor -> [graph] -> PageTrust -> Selector.

Until the BERT Grapher / Directioner exist, the graph is a stand-in:
  * edge weights: tf-idf dot product between units (as in HW4 / PacSum tf-idf:
    TF = 1 + log(count), IDF over the units of the document, unit-length vectors)
  * direction + sign: PacSum's position rule (pacsum_signed_graph), whose negative
    coefficient yields negative edges.
This gives the "PacSum + PageTrust" ablation now; later, swap `build_graph` for the
Directioner's output (same W convention, see pagetrust.py).

Usage (one document from a text file):
    python3 pipeline.py doc.txt --dataset cnndm [--unit clause] [--gamma 1 --memory 1]
"""

from __future__ import annotations

import argparse
import math
from collections import Counter

import numpy as np

from pagetrust import degree_centrality, pacsum_signed_graph, pagerank, pagetrust, threshold_edges
from preprocess import Document, preprocess
from selector import summarize


def tfidf_similarity(unit_tokens: list[list[str]]) -> np.ndarray:
    """Cosine of tf-idf vectors (dot product of unit-length vectors), units = 'documents'."""
    n = len(unit_tokens)
    df = Counter(w for toks in unit_tokens for w in set(toks))
    vocab = {w: k for k, w in enumerate(df)}
    X = np.zeros((n, len(vocab)))
    for i, toks in enumerate(unit_tokens):
        for w, c in Counter(toks).items():
            X[i, vocab[w]] = (1 + math.log(c)) * math.log(n / df[w]) if n > 1 else 1.0
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.where(norms > 0, norms, 1.0)
    S = X @ X.T
    np.fill_diagonal(S, 0.0)
    return S


def build_graph(doc: Document, beta: float, lambda_fwd: float, lambda_bwd: float) -> np.ndarray:
    return pacsum_signed_graph(threshold_edges(tfidf_similarity(doc.unit_tokens), beta),
                               lambda_fwd, lambda_bwd)


def rank(W: np.ndarray, ranker: str, alpha=0.85, gamma=1.0, memory=1.0, eps=1e-6,
         max_iter=100, distrust="binary") -> np.ndarray:
    if ranker == "pagetrust":
        return pagetrust(W, alpha, gamma, memory, eps, max_iter, distrust)
    if ranker == "pagerank":
        return pagerank(W, alpha, eps, max_iter)
    if ranker == "degree":
        return degree_centrality(W)
    raise ValueError(ranker)


def run(doc_input, dataset: str, unit="sentence", beta=0.3, lambda_fwd=1.0, lambda_bwd=-0.5,
        ranker="pagetrust", aggregate="max", overflow="include", **rank_kw):
    doc = preprocess(doc_input, unit=unit)
    if doc.n_units == 0:
        return doc, [], ""
    W = build_graph(doc, beta, lambda_fwd, lambda_bwd)
    scores = rank(W, ranker, **rank_kw)
    chosen, text = summarize(doc.sentences, scores, doc.unit_sentence, dataset, aggregate, overflow)
    return doc, chosen, text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--dataset", default="cnndm")
    ap.add_argument("--unit", choices=["sentence", "clause"], default="sentence")
    ap.add_argument("--ranker", choices=["pagetrust", "pagerank", "degree"], default="pagetrust")
    ap.add_argument("--beta", type=float, default=0.3)
    ap.add_argument("--lambda-fwd", type=float, default=1.0)
    ap.add_argument("--lambda-bwd", type=float, default=-0.5)
    ap.add_argument("--alpha", type=float, default=0.85)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--memory", type=float, default=1.0)
    a = ap.parse_args()
    text = open(a.path, encoding="utf-8").read()
    kw = dict(alpha=a.alpha, gamma=a.gamma, memory=a.memory) if a.ranker == "pagetrust" else (
        dict(alpha=a.alpha) if a.ranker == "pagerank" else {})
    doc, chosen, summ = run(text, a.dataset, a.unit, a.beta, a.lambda_fwd, a.lambda_bwd, a.ranker, **kw)
    print(f"{len(doc.sentences)} sentences, {doc.n_units} {doc.unit_type} vertices "
          f"(splitter: {doc.meta['splitter']}); chose {chosen}\n")
    print(summ)


if __name__ == "__main__":
    main()
