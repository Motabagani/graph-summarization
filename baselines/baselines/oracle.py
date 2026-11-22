#!/usr/bin/env python3
"""
Extractive ORACLE (upper bound): picks the sentences that maximise ROUGE
against the reference summary, so it is the only baseline that looks at the
reference.

PacSum (Zheng & Lapata 2019, Sec. 5.1): "We used a greedy algorithm similar to
Nallapati et al. (2017) to generate an oracle summary for each document. The
algorithm explores different combinations of sentences and generates an oracle
consisting of multiple sentences which maximize the ROUGE score against the gold
summary." All systems were limited to three sentences on CNN/DM.

Nallapati et al. (2017): add one sentence at a time, each time the one that
maximises ROUGE of the selected set against the reference; stop when no
remaining sentence improves it.

Options (every knob is a Config field / CLI flag):
  --search greedy          Nallapati-style greedy (default)
  --search combinations    every set of up to --max-sentences sentences drawn from the
                           --top-k individually best sentences (closer to "explores
                           different combinations")
  --objective rouge12      ROUGE-1 F + ROUGE-2 F (default; common greedy-oracle choice)
              rouge1 | rouge2 | rouge12
  --max-sentences 3        PacSum's 3-sentence limit on CNN/DM
  --stop-no-gain / --no-stop-no-gain
                           stop early when nothing improves (Nallapati), or always take
                           --max-sentences

ROUGE inside the search uses the same tokenizer as the final evaluation
(rouge-score; stemming follows --rouge-stemmer). The summary is output in
document order.

    python oracle.py --dataset cnn_dailymail --split test --data-root ../datasets --num-sentences 3 --no-rouge-stemmer
"""

import argparse
import itertools
import sys
from collections import Counter
from dataclasses import dataclass

import numpy as np

import summarize

NEEDS_REFERENCE = True


@dataclass
class Config:
    search: str = "greedy"          # greedy | combinations
    objective: str = "rouge12"      # rouge1 | rouge2 | rouge12
    max_sentences: int = 3
    stop_no_gain: bool = True
    top_k: int = 10                 # candidates for --search combinations
    rouge_stemmer: bool = True      # set from --rouge-stemmer by prepare()


_tok = {}


def _tokenizer(stem):
    if stem not in _tok:
        from rouge_score import tokenizers
        _tok[stem] = tokenizers.DefaultTokenizer(use_stemmer=stem)
    return _tok[stem]


def _ngrams(tokens, n):
    return Counter(zip(*[tokens[i:] for i in range(n)]))


def _f1(pred, ref):
    if not pred or not ref:
        return 0.0
    overlap = sum((pred & ref).values())
    if overlap == 0:
        return 0.0
    p, r = overlap / sum(pred.values()), overlap / sum(ref.values())
    return 2 * p * r / (p + r)


def score_sentences(sentences, cfg, reference):
    tok = _tokenizer(cfg.rouge_stemmer)
    ref_tokens = tok.tokenize(" ".join(reference))
    ref1, ref2 = _ngrams(ref_tokens, 1), _ngrams(ref_tokens, 2)
    sent_tokens = [tok.tokenize(s) for s in sentences]

    def objective(selected):
        toks = [t for i in sorted(selected) for t in sent_tokens[i]]     # document order
        if cfg.objective == "rouge1":
            return _f1(_ngrams(toks, 1), ref1)
        if cfg.objective == "rouge2":
            return _f1(_ngrams(toks, 2), ref2)
        return _f1(_ngrams(toks, 1), ref1) + _f1(_ngrams(toks, 2), ref2)

    n = len(sentences)
    k = min(cfg.max_sentences, n)
    if cfg.search == "greedy":
        selected, best = [], 0.0
        for _ in range(k):
            cand = [(objective(selected + [i]), i) for i in range(n) if i not in selected]
            if not cand:
                break
            value, i = max(cand, key=lambda x: (x[0], -x[1]))           # ties -> earlier sentence
            if cfg.stop_no_gain and value <= best:
                break
            selected.append(i)
            best = value
    else:
        singles = sorted(range(n), key=lambda i: (-objective([i]), i))[:cfg.top_k]
        sizes = range(1, k + 1) if cfg.stop_no_gain else [k]
        best, selected = -1.0, []
        for size in sizes:
            for combo in itertools.combinations(sorted(singles), size):
                value = objective(list(combo))
                if value > best:
                    best, selected = value, list(combo)

    scores = np.full(n, -np.inf)
    for i in selected:
        scores[i] = n - i                     # selected sentences, in document order
    return scores, {"iterations": 0, "converged": True}


# =============================================================================
# Command-line hooks
# =============================================================================
def add_args(parser):
    d = Config()
    g = parser.add_argument_group("Oracle")
    g.add_argument("--search", choices=["greedy", "combinations"], default=d.search)
    g.add_argument("--objective", choices=["rouge1", "rouge2", "rouge12"], default=d.objective)
    g.add_argument("--max-sentences", type=int, default=d.max_sentences)
    g.add_argument("--stop-no-gain", action=argparse.BooleanOptionalAction, default=d.stop_no_gain)
    g.add_argument("--top-k", type=int, default=d.top_k)


def prepare(cfg, args=None):
    if args is not None:
        cfg.rouge_stemmer = args.rouge_stemmer
    _tokenizer(cfg.rouge_stemmer)


if __name__ == "__main__":
    summarize.cli(sys.modules[__name__], "oracle")
