#!/usr/bin/env python3
"""
LexPageRank (Erkan & Radev, 2004, "LexPageRank: Prestige in Multi-Document Text
Summarization"), applied to single documents.

Paper -> code (every knob is a Config field / CLI flag):
  * Vertices = sentences; similarity = cosine of tf*idf vectors          --idf-source
    ("idf-modified cosine"; idf from a background corpus, as MEAD's enidf)
  * Edge between two sentences iff cosine > threshold (0.1 / 0.2 / 0.3;  --threshold
    0.1 best in the paper); unweighted, undirected
  * PageRank on that graph, where every sentence also links to itself    --damping, --self-loops
    (cosine(s, s) = 1 > threshold), damping 0.85

  Checked against the paper's worked example (Fig. 1 cosine matrix): this graph
  reproduces its LexPageRank scores for threshold 0.1 (Fig. 3) to within 0.0006
  (i.e. rounding), and its degree table (Table 1, which counts no self-links)
  exactly for thresholds 0.1 and 0.3.
  * Rank sentences by score
  * (MEAD's LengthCutoff 9 was used in all of the paper's runs)          --min-sentence-words 9

  Not reproduced: the paper's multi-document MEAD pipeline also mixed in a
  Position feature and a CSIS/MMR redundancy reranker. Those are MEAD
  components, not LexPageRank, and are left out so the baseline measures
  LexPageRank centrality alone (as PacSum/HipoRank's LexRank baselines do).

Variants:
  --continuous      continuous LexRank (Erkan & Radev 2004, JAIR): weighted cosine edges, no threshold
  --impl sumy       the off-the-shelf `sumy` LexRankSummarizer (the usual tool behind published
                    "LexRank" baseline numbers, e.g. arXiv/PubMed in Cohan et al. 2018 / HipoRank):
                    idf over the document's own sentences (log N/(1+n)), max-normalised tf,
                    stop words removed, Snowball stemming, threshold 0.1, self-links, no damping,
                    power method stopped at epsilon 0.1. Scores come from sumy itself; summary
                    assembly and ROUGE stay in summarize.py. Needs: pip install sumy
  --tune            pick the threshold (0.1 / 0.2 / 0.3) on a validation split

    python lexpagerank.py --dataset cnn_dailymail --split test --data-root ../datasets --num-sentences 3

Loading, sentence splitting, summary assembly and ROUGE come from summarize.py.
"""

import argparse
import hashlib
import json
import math
import os
import re
import sys
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

import summarize
import textrank


@dataclass
class Config:
    impl: str = "paper"            # paper (this reimplementation) | sumy (off-the-shelf LexRank)
    threshold: float = 0.1         # cosine threshold for an edge (paper: 0.1, 0.2, 0.3)
    continuous: bool = False       # weighted cosine edges instead of thresholded 0/1
    damping: float = 0.85          # PageRank follow probability
    tol: float = 1e-4
    max_iter: int = 200
    init: float = 1.0
    self_loops: bool = True        # each sentence links to itself (reproduces the paper's Fig. 3)
    dangling: str = "uniform"      # only matters without self-loops: isolated sentences jump uniformly
    idf_source: str = "train"      # train (background corpus = training documents) | document (its own sentences)
    idf_docs: int = 50000          # training documents used to estimate idf
    idf_cache: str = ""            # set by prepare(); path of the cached idf table
    lowercase: bool = True
    stem: bool = False
    min_sentence_words: int = 0    # MEAD LengthCutoff: shorter sentences are never selected (paper: 9)


_TOKEN_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)
_IDF = {}            # word -> idf, filled by prepare() (inherited by worker processes)
_IDF_META = {}


def _tokens(text, cfg):
    toks = _TOKEN_RE.findall(text)
    if cfg.lowercase:
        toks = [t.lower() for t in toks]
    if cfg.stem:
        st = textrank._lazy("stemmer")
        toks = [st.stem(t) for t in toks]
    return toks


def _load_idf(cfg):
    if not _IDF and cfg.idf_cache:
        with open(cfg.idf_cache) as f:
            d = json.load(f)
        _IDF.update(d["idf"])
        _IDF_META.update(n_docs=d["n_docs"])
    return _IDF


# =============================================================================
# Similarity graph and scores
# =============================================================================
def cosine_matrix(sentences, cfg):
    """idf-modified cosine between all sentence pairs."""
    toks = [_tokens(s, cfg) for s in sentences]
    n = len(toks)
    vocab, rows, cols, vals = {}, [], [], []
    for i, ts in enumerate(toks):
        counts = {}
        for t in ts:
            counts[t] = counts.get(t, 0) + 1
        for t, c in counts.items():
            rows.append(i)
            cols.append(vocab.setdefault(t, len(vocab)))
            vals.append(float(c))
    if not vocab:
        return np.zeros((n, n))
    C = sp.csr_matrix((vals, (rows, cols)), shape=(n, len(vocab)))
    words = sorted(vocab, key=vocab.get)
    if cfg.idf_source == "document":
        df = np.asarray((C > 0).sum(axis=0)).ravel()
        idf = np.log(n / df)
    else:
        table = _load_idf(cfg)
        unseen = math.log(_IDF_META.get("n_docs", 1))          # as if seen in one document
        idf = np.array([table.get(w, unseen) for w in words])
    V = C.multiply(idf[None, :]).tocsr()
    S = (V @ V.T).toarray()
    norm = np.sqrt(np.asarray(V.multiply(V).sum(axis=1)).ravel())
    denom = norm[:, None] * norm[None, :]
    S = np.divide(S, denom, out=np.zeros_like(S), where=denom > 0)
    np.fill_diagonal(S, 0)
    return S


_SUMY = {}


def _sumy_scores(sentences, cfg):
    """Sentence scores exactly as sumy's LexRankSummarizer computes them."""
    if not _SUMY:
        from sumy.models.dom import Sentence
        from sumy.nlp.stemmers import Stemmer
        from sumy.nlp.tokenizers import Tokenizer
        from sumy.summarizers.lex_rank import LexRankSummarizer
        from sumy.utils import get_stop_words
        summ = LexRankSummarizer(Stemmer("english"))
        summ.stop_words = get_stop_words("english")
        _SUMY.update(summ=summ, tok=Tokenizer("english"), Sentence=Sentence)
    summ, tok = _SUMY["summ"], _SUMY["tok"]
    words = [summ._to_words_set(_SUMY["Sentence"](s, tok)) for s in sentences]
    tf, idf = summ._compute_tf(words), summ._compute_idf(words)
    M = summ._create_matrix(words, summ.threshold, tf, idf)
    return np.asarray(summ.power_method(M, summ.epsilon), dtype=float)


def score_sentences(sentences, cfg):
    if cfg.impl == "sumy":
        scores = _sumy_scores(sentences, cfg)
        if cfg.min_sentence_words:
            short = np.array([len(s.split()) < cfg.min_sentence_words for s in sentences])
            if not short.all():
                scores = np.where(short, -np.inf, scores)
        return scores, {"iterations": 0, "converged": True}
    S = cosine_matrix(sentences, cfg)
    A = S if cfg.continuous else (S > cfg.threshold).astype(float)
    if cfg.self_loops:
        np.fill_diagonal(A, 1.0)
    scores, iters, conv = textrank.weighted_pagerank(A, cfg)
    if cfg.min_sentence_words:
        short = np.array([len(s.split()) < cfg.min_sentence_words for s in sentences])
        if not short.all():                       # never leave a document with nothing to pick
            scores = np.where(short, -np.inf, scores)
    return scores, {"iterations": iters, "converged": conv}


def tune_grid(cfg):
    """The thresholds the paper compares."""
    if cfg.impl == "sumy":
        sys.exit("--impl sumy uses sumy's fixed threshold (0.1); drop --tune")
    return [{"threshold": t} for t in (0.1, 0.2, 0.3)]


# =============================================================================
# Command-line hooks
# =============================================================================
def add_args(parser):
    d = Config()
    g = parser.add_argument_group("LexPageRank")
    g.add_argument("--impl", choices=["paper", "sumy"], default=d.impl)
    g.add_argument("--threshold", type=float, default=d.threshold)
    g.add_argument("--continuous", action=argparse.BooleanOptionalAction, default=d.continuous)
    g.add_argument("--damping", type=float, default=d.damping)
    g.add_argument("--tol", type=float, default=d.tol)
    g.add_argument("--max-iter", type=int, default=d.max_iter)
    g.add_argument("--init", type=float, default=d.init)
    g.add_argument("--self-loops", action=argparse.BooleanOptionalAction, default=d.self_loops)
    g.add_argument("--dangling", choices=["none", "uniform"], default=d.dangling)
    g.add_argument("--idf-source", choices=["train", "document"], default=d.idf_source)
    g.add_argument("--idf-docs", type=int, default=d.idf_docs)
    g.add_argument("--idf-input", help="file/glob of background documents for idf (default: the dataset's train split)")
    g.add_argument("--lowercase", action=argparse.BooleanOptionalAction, default=d.lowercase)
    g.add_argument("--stem", action=argparse.BooleanOptionalAction, default=d.stem)
    g.add_argument("--min-sentence-words", type=int, default=d.min_sentence_words)


def prepare(cfg, args=None):
    """Build (or load the cached) idf table from the training documents."""
    if cfg.impl == "sumy":
        try:
            import sumy  # noqa: F401
        except ImportError:
            sys.exit("--impl sumy needs sumy: pip install sumy")
        summarize.ensure_nltk([("tokenizers/punkt", "punkt"), ("tokenizers/punkt_tab", "punkt_tab")])
        _sumy_scores(["Warm up the tokenizer.", "Second sentence."], cfg)
        return
    if cfg.stem:
        textrank._lazy("stemmer")
    if cfg.idf_source != "train":
        return
    loc = summarize.locate(args, "train", getattr(args, "idf_input", None))
    key = json.dumps([loc["files"], cfg.idf_docs, cfg.lowercase, cfg.stem], sort_keys=True)
    cache_dir = os.path.join(args.data_root, ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    cfg.idf_cache = os.path.join(cache_dir, f"idf_{hashlib.md5(key.encode()).hexdigest()[:12]}.json")

    if not os.path.exists(cfg.idf_cache):
        print(f"[LexPageRank] building idf from {cfg.idf_docs} documents of {loc['files']} ...",
              file=sys.stderr, flush=True)
        df, n = {}, 0
        for _, text, _ in summarize._limited(summarize.open_records(loc), 0, cfg.idf_docs):
            if isinstance(text, (list, tuple, np.ndarray)):
                text = " ".join(map(str, text))
            for w in set(_tokens(str(text), cfg)):
                df[w] = df.get(w, 0) + 1
            n += 1
        idf = {w: math.log(n / c) for w, c in df.items()}          # idf_w = log(N / n_w)
        with open(cfg.idf_cache, "w") as f:
            json.dump({"n_docs": n, "files": loc["files"], "idf": idf}, f)
        print(f"[LexPageRank] idf over {n} documents, {len(idf)} words -> {cfg.idf_cache}", file=sys.stderr)
    _load_idf(cfg)


if __name__ == "__main__":
    summarize.cli(sys.modules[__name__], "lexpagerank")
