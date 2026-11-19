"""
Non-BERT baselines and shared preprocessing.

Baselines work on the second copy of each sentence described in the proposal:
lowercased, Porter-stemmed, with stop words and punctuation removed.
All selectors return sentence indices; callers put them back in document order.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
from nltk.stem import PorterStemmer
from rouge_score.tokenizers import DefaultTokenizer

from rankers import in_degree, pagerank

_stemmer = PorterStemmer()
_rouge_tok = DefaultTokenizer(use_stemmer=True)   # same tokenization ROUGE scoring uses


def stem_sentence(sent) -> list[str]:
    """spaCy span -> lowercased Porter stems, no stop words / punctuation / whitespace."""
    return [
        _stemmer.stem(t.lower_) for t in sent
        if not (t.is_stop or t.is_punct or t.is_space or not t.text.strip())
    ]


# --------------------------------------------------------------------------- similarity

def tfidf_cosine(stems: list[list[str]]) -> np.ndarray:
    """Sentence-by-sentence TF-IDF cosine; IDF over the document's sentences."""
    n = len(stems)
    vocab = {w: k for k, w in enumerate(sorted({w for s in stems for w in s}))}
    if n == 0 or not vocab:
        return np.zeros((n, n))
    tf = np.zeros((n, len(vocab)))
    for i, s in enumerate(stems):
        for w, c in Counter(s).items():
            tf[i, vocab[w]] = c
    df = (tf > 0).sum(axis=0)
    idf = np.log((1 + n) / (1 + df)) + 1.0
    v = tf * idf
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    v = v / np.where(norm > 0, norm, 1.0)
    sim = v @ v.T
    np.fill_diagonal(sim, 0.0)
    return sim


def textrank_similarity(stems: list[list[str]]) -> np.ndarray:
    """Mihalcea & Tarau (2004): |Si ∩ Sj| / (log|Si| + log|Sj|)."""
    n = len(stems)
    sets = [set(s) for s in stems]
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            ov = len(sets[i] & sets[j])
            if ov:
                denom = math.log(max(len(stems[i]), 1)) + math.log(max(len(stems[j]), 1))
                sim[i, j] = sim[j, i] = ov / denom if denom > 0 else ov
    return sim


def threshold(sim: np.ndarray, beta: float) -> np.ndarray:
    """PacSum threshold: keep edges >= min + β(max - min) over off-diagonal entries."""
    n = sim.shape[0]
    if n < 2:
        return sim.copy()
    off = sim[np.triu_indices(n, 1)]
    t = off.min() + beta * (off.max() - off.min())
    out = np.where(sim >= t, sim, 0.0)
    np.fill_diagonal(out, 0.0)
    return out


# --------------------------------------------------------------------------- selectors

def top(scores: np.ndarray, n: int | None) -> list[int]:
    """n given: the top-n indices in document order. n=None: the full ranking, best first
    (ties broken by position), for the caller to cut to a word or sentence budget."""
    order = np.argsort(-np.asarray(scores), kind="stable").tolist()
    return order if n is None else sorted(order[:n])


def lead(n_sent: int, n: int) -> list[int]:
    return list(range(min(n, n_sent)))


def random_pick(n_sent: int, n: int, rng: np.random.Generator) -> list[int]:
    return sorted(rng.choice(n_sent, size=min(n, n_sent), replace=False).tolist())


def degree_tfidf(stems, n: int | None, beta: float) -> list[int]:
    return top(in_degree(threshold(tfidf_cosine(stems), beta)), n)


def textrank(stems, n: int | None) -> list[int]:
    return top(pagerank(textrank_similarity(stems)), n)


def lexpagerank(stems, n: int | None, cos_threshold: float = 0.1) -> list[int]:
    """Continuous LexRank (Erkan & Radev 2004): cosine edges above 0.1, PageRank."""
    sim = tfidf_cosine(stems)
    return top(pagerank(np.where(sim >= cos_threshold, sim, 0.0)), n)


def pacsum_tfidf(stems, n: int | None, beta: float, lambda1: float, lambda2: float = 1.0) -> list[int]:
    """PacSum (Zheng & Lapata 2019), TF-IDF variant:
    centrality(i) = λ1 Σ_{j<i} e_ij + λ2 Σ_{j>i} e_ij."""
    e = threshold(tfidf_cosine(stems), beta)
    k = e.shape[0]
    earlier = np.tril(np.ones((k, k)), -1)        # [i, j] = 1 if j < i
    scores = lambda1 * (e * earlier).sum(1) + lambda2 * (e * earlier.T).sum(1)
    return top(scores, n)


# --------------------------------------------------------------------------- oracle

def _ngrams(toks, k):
    return Counter(zip(*(toks[i:] for i in range(k))))


def _f1(cand: Counter, ref: Counter) -> float:
    ov = sum((cand & ref).values())
    if not ov:
        return 0.0
    p, r = ov / sum(cand.values()), ov / sum(ref.values())
    return 2 * p * r / (p + r)


def oracle(sentences: list[str], reference: list[str], n: int | None = None,
           max_words: int | None = None, overflow: str = "include") -> list[int]:
    """Greedy oracle: repeatedly add the sentence that most improves ROUGE-1 + ROUGE-2
    F1 against the reference; stop at n sentences, or once max_words words are used
    (same overflow rule as every other system, see pipeline.Budget), or when nothing
    improves. With "skip", sentences that would cross max_words are not candidates."""
    toks = [_rouge_tok.tokenize(s) for s in sentences]
    ref_t = _rouge_tok.tokenize(" ".join(reference))
    ref1, ref2 = _ngrams(ref_t, 1), _ngrams(ref_t, 2)
    s1 = [_ngrams(t, 1) for t in toks]
    s2 = [_ngrams(t, 2) for t in toks]
    words = [len(s.split()) for s in sentences]
    chosen: list[int] = []
    c1, c2, best, used = Counter(), Counter(), 0.0, 0
    while len(chosen) < min(n or len(sentences), len(sentences)):
        if max_words is not None and used >= max_words:
            break
        cand = None
        for i in range(len(sentences)):
            if i in chosen:
                continue
            if max_words is not None and overflow == "skip" and used + words[i] > max_words:
                continue
            sc = _f1(c1 + s1[i], ref1) + _f1(c2 + s2[i], ref2)
            if sc > best:
                best, cand = sc, i
        if cand is None:
            break
        chosen.append(cand)
        c1, c2 = c1 + s1[cand], c2 + s2[cand]
        used += words[cand]
    return sorted(chosen)
