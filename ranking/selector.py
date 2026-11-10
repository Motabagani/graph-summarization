"""Sentence selector: vertex scores -> summary.

Budgets are the ones the baselines use (keep them identical, or ROUGE F1 is not
comparable):
    CNN/DM   3 sentences       PubMed  203 words      arXiv  220 words
    BillSum  184 words         WikiHow 55 words       TL;DR  1 sentence
(arXiv: notes say 200 in one place and 220 in another -- use whatever the baseline
code uses.)

For word budgets, `overflow` decides the sentence that crosses the budget:
    "include"  -- keep it whole (summary may run slightly over)
    "truncate" -- keep it, cut the summary at exactly `budget` words
    "skip"     -- drop it and try the next-ranked sentence that still fits
Match whatever the baseline code does.

Clause graphs (extension): `aggregate` turns clause scores into sentence scores
("max", "mean" or "sum"; "sum" favours long sentences).
"""

from __future__ import annotations

import numpy as np

BUDGETS = {
    "cnndm": ("sentences", 3),
    "pubmed": ("words", 203),
    "arxiv": ("words", 220),
    "billsum": ("words", 184),
    "wikihow": ("words", 55),
    "tldr": ("sentences", 1),
}


def sentence_scores(unit_scores, unit_sentence, n_sentences: int, aggregate: str = "max") -> np.ndarray:
    """Combine vertex scores into one score per sentence. Sentences with no vertex get -inf."""
    unit_scores = np.asarray(unit_scores, dtype=float)
    owner = np.asarray(unit_sentence, dtype=int)
    out = np.full(n_sentences, -np.inf)
    if aggregate == "max":
        np.maximum.at(out, owner, unit_scores)
    elif aggregate in ("sum", "mean"):
        tot = np.zeros(n_sentences)
        np.add.at(tot, owner, unit_scores)
        cnt = np.bincount(owner, minlength=n_sentences)
        vals = tot if aggregate == "sum" else tot / np.maximum(cnt, 1)
        out = np.where(cnt > 0, vals, -np.inf)
    else:
        raise ValueError(aggregate)
    return out


def select(sentences: list[str], scores, budget_type: str, budget: int,
           overflow: str = "include", doc_order: bool = True) -> list[int]:
    """Indexes of the chosen sentences (ties broken by position, earlier first)."""
    scores = np.asarray(scores, dtype=float)
    ranked = sorted(range(len(sentences)), key=lambda i: (-scores[i], i))
    ranked = [i for i in ranked if np.isfinite(scores[i])]
    chosen: list[int] = []
    if budget_type == "sentences":
        chosen = ranked[:budget]
    elif budget_type == "words":
        used = 0
        for i in ranked:
            if used >= budget:
                break
            n = len(sentences[i].split())
            if overflow == "skip" and used + n > budget:
                continue
            chosen.append(i)
            used += n
    else:
        raise ValueError(budget_type)
    return sorted(chosen) if doc_order else chosen


def summary_text(sentences: list[str], chosen: list[int], budget_type: str, budget: int,
                 overflow: str = "include") -> str:
    """One sentence per line (ROUGE-Lsum needs the sentence breaks)."""
    lines = [sentences[i] for i in chosen]
    if budget_type == "words" and overflow == "truncate":
        out, left = [], budget
        for line in lines:
            w = line.split()
            if left <= 0:
                break
            out.append(" ".join(w[:left]))
            left -= len(w)
        lines = out
    return "\n".join(lines)


def summarize(sentences, unit_scores, unit_sentence, dataset: str, aggregate: str = "max",
              overflow: str = "include") -> tuple[list[int], str]:
    btype, b = BUDGETS[dataset]
    s = sentence_scores(unit_scores, unit_sentence, len(sentences), aggregate)
    chosen = select(sentences, s, btype, b, overflow)
    return chosen, summary_text(sentences, chosen, btype, b, overflow)
