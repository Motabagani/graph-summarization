"""Minimal ROUGE-1 / ROUGE-2 / ROUGE-Lsum F1, mirroring rouge-score's defaults
(lowercase, tokens = [a-z0-9]+, Porter stemming of tokens longer than 3 chars).
ROUGE-L is the summary-level "rougeLsum" (sentences = lines), which is what the
summarization papers we compare against report (plain rougeL is ~6-8 points lower).

For LOCAL comparisons only -- report numbers with the same scorer Finley used on HPC.
"""

import re
from collections import Counter

from porter import PorterStemmer

_st = PorterStemmer()
_cache: dict[str, str] = {}


def tokens(text: str) -> list[str]:
    out = []
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        if len(w) > 3:
            s = _cache.get(w)
            if s is None:
                s = _cache[w] = _st.stem(w)
            w = s
        out.append(w)
    return out


def _f1(overlap: int, n_pred: int, n_ref: int) -> float:
    if not overlap or not n_pred or not n_ref:
        return 0.0
    p, r = overlap / n_pred, overlap / n_ref
    return 2 * p * r / (p + r)


def _ngrams(t, n):
    return Counter(zip(*[t[i:] for i in range(n)]))


def _lcs(a, b) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def _lcs_set(a, b) -> set[int]:
    """Indexes of `a` that lie on one longest common subsequence with `b`."""
    m, n = len(a), len(b)
    T = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            T[i][j] = T[i - 1][j - 1] + 1 if a[i - 1] == b[j - 1] else max(T[i - 1][j], T[i][j - 1])
    out, i, j = set(), m, n
    while i and j:
        if a[i - 1] == b[j - 1]:
            out.add(i - 1); i -= 1; j -= 1
        elif T[i - 1][j] >= T[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return out


def _rouge_lsum(pred_sents: list[list[str]], ref_sents: list[list[str]]) -> float:
    """Summary-level ROUGE-L (union LCS per reference sentence), as rouge-score rougeLsum."""
    n_p, n_r = sum(map(len, pred_sents)), sum(map(len, ref_sents))
    if not n_p or not n_r:
        return 0.0
    p_cnt = Counter(t for s in pred_sents for t in s)
    r_cnt = Counter(t for s in ref_sents for t in s)
    hits = 0
    for r in ref_sents:
        union = set()
        for p in pred_sents:
            union |= _lcs_set(r, p)
        for i in sorted(union):
            t = r[i]
            if p_cnt[t] > 0 and r_cnt[t] > 0:
                hits += 1
                p_cnt[t] -= 1
                r_cnt[t] -= 1
    return _f1(hits, n_p, n_r)


def rouge(pred: str, ref: str) -> tuple[float, float, float]:
    """(R-1, R-2, R-Lsum) F1 x 100. Sentences = newline-separated lines, as rouge-score
    expects for rougeLsum: pass pred / ref with one sentence per line."""
    p, r = tokens(pred), tokens(ref)
    out = []
    for n in (1, 2):
        pn, rn = _ngrams(p, n), _ngrams(r, n)
        out.append(_f1(sum((pn & rn).values()), sum(pn.values()), sum(rn.values())))
    ps = [t for t in (tokens(l) for l in pred.split("\n")) if t]
    rs = [t for t in (tokens(l) for l in ref.split("\n")) if t]
    out.append(_rouge_lsum(ps, rs))
    return tuple(100 * x for x in out)
