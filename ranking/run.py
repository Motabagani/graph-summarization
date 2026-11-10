"""Run the pipeline over a JSONL dataset sample and report mean ROUGE-1/2/L F1.

    python3 run.py data_sample/cnndm.jsonl --dataset cnndm --ranker degree
    python3 run.py data_sample/cnndm.jsonl --dataset cnndm --ranker pagetrust --gamma 1 --memory 0
    python3 run.py data_sample/cnndm.jsonl --dataset cnndm --ranker lead

JSONL rows: {"id", "article": str | list[str], "summary": str}.
Rankers: lead (first sentences), degree (PacSum centrality on the stand-in graph),
pagerank, pagetrust. Scores use rouge_lite (local only; see its docstring).
"""

from __future__ import annotations

import argparse
import json
import re

import numpy as np

from pipeline import build_graph, rank
from preprocess import preprocess, split_sentences
from rouge_lite import rouge
from selector import summarize

PRESPLIT = {"arxiv", "pubmed"}


_TOKENIZED_SENT = re.compile(r"(?<= [.?!])\s+")


def presplit(text) -> list[str]:
    """arXiv / PubMed (Cohan et al. 2018) are lowercased, tokenized text whose sentences end
    in a separate " ." token. The original release stores them as a list of sentences; the
    Hugging Face copy joins them into one line, so cut at those tokens to get the list back."""
    if isinstance(text, list):
        text = " ".join(text)
    return [s.strip() for s in _TOKENIZED_SENT.split(text) if s.strip()]


def summarize_doc(article, dataset, ranker, beta, lf, lb, **kw):
    if dataset in PRESPLIT:
        article = presplit(article)
    doc = preprocess(article)
    if doc.n_units == 0:
        return ""
    if ranker == "lead":
        scores = -np.arange(doc.n_units, dtype=float)
    else:
        scores = rank(build_graph(doc, beta, lf, lb), ranker, **kw)
    return summarize(doc.sentences, scores, doc.unit_sentence, dataset)[1]


def ref_lines(ref: str, dataset: str = "") -> str:
    """Reference with one sentence per line (CNN/DM highlights already are)."""
    if dataset in PRESPLIT:
        return "\n".join(presplit(ref))
    lines = [l.strip() for l in ref.split("\n") if l.strip()]
    if len(lines) > 1:
        return "\n".join(lines)
    return "\n".join(split_sentences(ref))


def evaluate(rows, dataset, ranker="degree", beta=0.3, lf=1.0, lb=-0.5, **kw):
    s = np.array([rouge(summarize_doc(r["article"], dataset, ranker, beta, lf, lb, **kw), ref_lines(r["summary"], dataset))
                  for r in rows])
    return s.mean(axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ranker", default="pagetrust", choices=["lead", "degree", "pagerank", "pagetrust"])
    ap.add_argument("--beta", type=float, default=0.3)
    ap.add_argument("--lambda-fwd", type=float, default=1.0, help="weight on edges to LATER sentences")
    ap.add_argument("--lambda-bwd", type=float, default=-0.5, help="weight on edges to EARLIER sentences")
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--memory", type=float, default=0.0)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.path, encoding="utf-8")][: a.limit]
    kw = {"gamma": a.gamma, "memory": a.memory} if a.ranker == "pagetrust" else {}
    r1, r2, rl = evaluate(rows, a.dataset, a.ranker, a.beta, a.lambda_fwd, a.lambda_bwd, **kw)
    print(f"{a.dataset} {a.ranker} n={len(rows)}  R-1 {r1:.2f}  R-2 {r2:.2f}  R-L {rl:.2f}")


if __name__ == "__main__":
    main()
