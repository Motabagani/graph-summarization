#!/usr/bin/env python3
"""
TextRank sentence extraction (Mihalcea, 2004), weighted PageRank (Eq. 10):

    PR(V_i) = (1 - d) + d * sum_{j in In(i)}  w_ji * PR(V_j) / sum_{k in Out(j)} w_jk

    python textrank.py --dataset cnn_dailymail --split test --data-root ../datasets

Paper -> code (every knob is a Config field / CLI flag):
  * Vertices = whole sentences                                   (Sec. 3)
  * Edge weight = |shared words| / (log|S_i| + log|S_j|)         (Sec. 3)  --similarity overlap
  * Overlap on all tokens, or through a syntactic filter         (Sec. 3)  --pos-filter
  * Graph: undirected | directed forward | directed backward     (Sec. 3)  --graph
  * d = 0.85                                                     (fn. 1)   --damping
  * Iterate from arbitrary start until change < threshold        (Sec 2.3) --tol, --init
  * Top-scored sentences, ~100-word summary, first 100 evaluated (Sec 3-4) --word-limit, --truncate

PacSum's TextRank (tf-idf) baseline:
    --preprocess gensim --similarity tfidf_dot --num-sentences 3
Barrios et al. (2016) cosine tf-idf variant:  --similarity tfidf_cosine

The sentence graph (GraphConfig, build_graph, add_graph_args) is also used by
DegreeCentrality.py. Loading, splitting, summary assembly and ROUGE come from summarize.py.
"""

import math
import re
import sys
from dataclasses import dataclass

import argparse

import numpy as np
import scipy.sparse as sp

import summarize


# =============================================================================
# Configuration
# =============================================================================
@dataclass
class GraphConfig:
    """How the sentence graph is built (shared by TextRank and DegreeCentrality)."""
    graph: str = "undirected"      # undirected | forward | backward
    similarity: str = "overlap"    # overlap (Mihalcea 2004) | tfidf_dot (PacSum) | tfidf_cosine (Barrios 2016)
    preprocess: str = "basic"      # basic (flags below) | gensim (PacSum: gensim DEFAULT_FILTERS)
    lowercase: bool = True
    remove_stopwords: bool = False
    stem: bool = False             # Porter stemming of tokens
    pos_filter: str = "none"       # none | nouns_adjs | nouns
    length_mode: str = "tokens"    # |S_i| = number of tokens | number of distinct words ("types")
    log_base: float = math.e       # cancels out in Eq. 10; kept for completeness
    min_tokens: int = 1            # sentences with fewer (filtered) tokens get no edges


@dataclass
class Config(GraphConfig):
    """TextRank = sentence graph + weighted PageRank (Sec. 2.3, Eq. 10)."""
    damping: float = 0.85          # d
    tol: float = 1e-4              # convergence threshold (max |PR_t - PR_{t-1}|)
    max_iter: int = 200            # safety cap on iterations
    init: float = 1.0              # arbitrary initial score for every vertex
    dangling: str = "none"         # none (paper formula) | uniform (redistribute)


# =============================================================================
# Sentence -> tokens
# =============================================================================
_TOKEN_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)
_NOUNS = {"NN", "NNS", "NNP", "NNPS"}
_ADJS = {"JJ", "JJR", "JJS"}
_cache = {}


def _lazy(name):
    if name not in _cache:
        if name == "stopwords":
            from nltk.corpus import stopwords
            _cache[name] = set(stopwords.words("english"))
        elif name == "stemmer":
            from nltk.stem.porter import PorterStemmer
            _cache[name] = PorterStemmer()
        elif name == "pos_tag_sents":
            import nltk
            _cache[name] = nltk.pos_tag_sents
        elif name == "gensim":
            try:
                from gensim.parsing.preprocessing import preprocess_string, DEFAULT_FILTERS
            except ImportError:
                sys.exit("--preprocess gensim needs gensim: pip install --only-binary=:all: gensim")
            _cache[name] = lambda s: preprocess_string(s, DEFAULT_FILTERS)
    return _cache[name]


def tokenize_sentences(sentences, cfg):
    """Lexical representation of each sentence used for similarity."""
    if cfg.preprocess == "gensim":
        # Exactly PacSum's tf-idf preprocessing (their code/gensim_preprocess.py):
        # lowercase, strip tags, strip punctuation, collapse whitespace, strip digits,
        # remove gensim STOPWORDS, drop tokens < 3 chars, Porter-stem.
        # The lowercase/stopword/stem/POS flags are ignored in this mode.
        pp = _lazy("gensim")
        return [pp(s) for s in sentences]

    toks = [_TOKEN_RE.findall(s) for s in sentences]
    if cfg.pos_filter != "none":
        keep = _NOUNS | _ADJS if cfg.pos_filter == "nouns_adjs" else _NOUNS
        toks = [[w for w, t in sent if t in keep] for sent in _lazy("pos_tag_sents")(toks)]
    if cfg.lowercase:
        toks = [[w.lower() for w in s] for s in toks]
    if cfg.remove_stopwords:
        sw = _lazy("stopwords")
        toks = [[w for w in s if w.lower() not in sw] for s in toks]
    if cfg.stem:
        st = _lazy("stemmer")
        toks = [[st.stem(w) for w in s] for s in toks]
    return toks


# =============================================================================
# Similarity graph
# =============================================================================
def similarity_matrix(token_lists, cfg):
    if cfg.similarity == "overlap":
        return _overlap_similarity(token_lists, cfg)
    if cfg.similarity in ("tfidf_dot", "tfidf_cosine"):
        return _tfidf_similarity(token_lists, cfg)
    raise ValueError(f"unknown similarity: {cfg.similarity}")


def _overlap_similarity(token_lists, cfg):
    """W[i, j] = |S_i ∩ S_j| / (log|S_i| + log|S_j|), all pairs at once."""
    n = len(token_lists)
    vocab, rows, cols = {}, [], []
    lengths = np.zeros(n)
    for i, toks in enumerate(token_lists):
        types = set(toks)
        lengths[i] = len(toks) if cfg.length_mode == "tokens" else len(types)
        for t in types:
            rows.append(i)
            cols.append(vocab.setdefault(t, len(vocab)))
    if not vocab:
        return np.zeros((n, n))

    X = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, len(vocab)))
    overlap = (X @ X.T).toarray()                         # |S_i ∩ S_j|
    with np.errstate(divide="ignore"):
        loglen = np.log(lengths) / math.log(cfg.log_base)  # -inf for empty sentences
    denom = loglen[:, None] + loglen[None, :]

    W = np.zeros((n, n))
    # Edge only where the formula is defined: two one-word sentences give
    # log1 + log1 = 0 (division by zero), so they get no edge.
    ok = (overlap > 0) & (denom > 0)
    W[ok] = overlap[ok] / denom[ok]
    return _finish(W, lengths, cfg)


def _tfidf_similarity(token_lists, cfg):
    """
    tf-idf similarity with idf over the sentences of this document, as in
    PacSum's PacSumExtractorWithTfIdf:
        idf(w)       = log(N - df(w) + 0.5) - log(df(w) + 0.5)    (N = #sentences)
        tfidf_dot    = sum_w tf_i(w) * tf_j(w) * idf(w)^2          (PacSum's _idf_modified_dot)
        tfidf_cosine = tfidf_dot / (||tf_i*idf|| * ||tf_j*idf||)   (Barrios et al. 2016)
    """
    n = len(token_lists)
    vocab, rows, cols, vals = {}, [], [], []
    for i, toks in enumerate(token_lists):
        counts = {}
        for t in toks:
            counts[t] = counts.get(t, 0) + 1
        for t, c in counts.items():
            rows.append(i)
            cols.append(vocab.setdefault(t, len(vocab)))
            vals.append(float(c))
    if not vocab:
        return np.zeros((n, n))

    C = sp.csr_matrix((vals, (rows, cols)), shape=(n, len(vocab)))   # term counts
    df = np.asarray((C > 0).sum(axis=0)).ravel()
    idf = np.log(n - df + 0.5) - np.log(df + 0.5)                     # may be <= 0; squared below
    V = C.multiply(idf[None, :]).tocsr()
    W = (V @ V.T).toarray()
    if cfg.similarity == "tfidf_cosine":
        norm = np.sqrt(np.asarray(V.multiply(V).sum(axis=1)).ravel())
        denom = norm[:, None] * norm[None, :]
        W = np.divide(W, denom, out=np.zeros_like(W), where=denom > 0)
    W[W < 0] = 0
    return _finish(W, np.array([len(t) for t in token_lists]), cfg)


def _finish(W, lengths, cfg):
    short = lengths < cfg.min_tokens
    W[short, :] = 0
    W[:, short] = 0
    np.fill_diagonal(W, 0)        # no self-loops
    return W


def orient(W, graph):
    """W[j, i] = weight of edge j -> i."""
    if graph == "undirected":
        return W
    if graph == "forward":        # sentence -> sentences that follow it
        return np.triu(W, k=1)
    if graph == "backward":       # sentence -> sentences that precede it
        return np.tril(W, k=-1)
    raise ValueError(f"unknown graph type: {graph}")


# =============================================================================
# Weighted PageRank (Eq. 10)
# =============================================================================
def weighted_pagerank(W, cfg):
    """Iterate Eq. 10 until max change < tol. Returns (scores, iterations, converged)."""
    n = W.shape[0]
    out_w = W.sum(axis=1)                                  # sum_k w_jk
    M = np.divide(W, out_w[:, None], out=np.zeros_like(W), where=out_w[:, None] > 0)
    MT = np.ascontiguousarray(M.T)                         # MT[i, j] = w_ji / sum_k w_jk
    dangling = out_w == 0
    d = cfg.damping

    pr = np.full(n, cfg.init, dtype=np.float64)
    delta, it = np.inf, 0
    for it in range(1, cfg.max_iter + 1):
        new = (1.0 - d) + d * (MT @ pr)
        if cfg.dangling == "uniform" and dangling.any():
            new += d * pr[dangling].sum() / n
        delta = np.abs(new - pr).max()
        pr = new
        if delta < cfg.tol:
            break
    return pr, it, bool(delta < cfg.tol)


def build_graph(sentences, cfg):
    """Weighted adjacency matrix W, W[j, i] = weight of edge j -> i."""
    return orient(similarity_matrix(tokenize_sentences(sentences, cfg), cfg), cfg.graph)


def score_sentences(sentences, cfg):
    """TextRank score per sentence."""
    scores, iters, conv = weighted_pagerank(build_graph(sentences, cfg), cfg)
    return scores, {"iterations": iters, "converged": conv}


def textrank(doc, cfg=None, summary_cfg=None):
    """Summarise one document (string or list of sentences)."""
    return summarize.summarize_doc(doc, score_sentences, cfg or Config(),
                                   summary_cfg or summarize.SummaryConfig())


# =============================================================================
# Command-line hooks (used by summarize.cli)
# =============================================================================
def add_graph_args(parser):
    d = GraphConfig()
    t = parser.add_argument_group("sentence graph")
    t.add_argument("--graph", choices=["undirected", "forward", "backward"], default=d.graph)
    t.add_argument("--similarity", choices=["overlap", "tfidf_dot", "tfidf_cosine"], default=d.similarity)
    t.add_argument("--preprocess", choices=["basic", "gensim"], default=d.preprocess)
    t.add_argument("--lowercase", action=argparse.BooleanOptionalAction, default=d.lowercase)
    t.add_argument("--remove-stopwords", action=argparse.BooleanOptionalAction, default=d.remove_stopwords)
    t.add_argument("--stem", action=argparse.BooleanOptionalAction, default=d.stem)
    t.add_argument("--pos-filter", choices=["none", "nouns_adjs", "nouns"], default=d.pos_filter)
    t.add_argument("--length-mode", choices=["tokens", "types"], default=d.length_mode)
    t.add_argument("--log-base", type=float, default=d.log_base)
    t.add_argument("--min-tokens", type=int, default=d.min_tokens)


def add_args(parser):
    add_graph_args(parser)
    d = Config()
    t = parser.add_argument_group("PageRank")
    t.add_argument("--damping", type=float, default=d.damping)
    t.add_argument("--tol", type=float, default=d.tol)
    t.add_argument("--max-iter", type=int, default=d.max_iter)
    t.add_argument("--init", type=float, default=d.init)
    t.add_argument("--dangling", choices=["none", "uniform"], default=d.dangling)


def prepare(cfg, args=None):
    """Fail fast on missing packages; download the NLTK data the options need."""
    if cfg.preprocess == "gensim":
        _lazy("gensim")
        return
    need = []
    if cfg.remove_stopwords:
        need.append(("corpora/stopwords", "stopwords"))
    if cfg.pos_filter != "none":
        need.append(("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"))
    if need:
        summarize.ensure_nltk(need)


if __name__ == "__main__":
    summarize.cli(sys.modules[__name__], "textrank")
