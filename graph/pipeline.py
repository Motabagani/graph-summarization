"""
Full pipeline and every system compared in the evaluation, behind one interface:

    system.select(doc, budget, **params) -> sorted sentence indices

Each system ranks the sentences (best first); `apply_budget` then cuts the ranking to
the dataset's budget, a number of sentences or of words, so every system is held to the
same length. Oracle is the exception: it searches greedily under the budget itself.

`PreparedDoc` holds everything expensive about a document (spaCy parse, stems, BERT
vectors, NSP scores, built graphs), so a parameter grid can be swept on dev without
redoing any of it.

Single-document use:
    from pipeline import Summarizer
    s = Summarizer("bert-base-uncased")
    print(s.summarize(text, n=3))
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Callable

import numpy as np

import baselines as B
from bert_graph import (DocumentScores, FrozenBert, GraphConfig, build_graph,
                        make_vertices, sentence_scores)
from data import Example
from splitting import split_document
from rankers import in_degree, pagerank, pagetrust

ALPHA, EPS, MAX_ITER = 0.85, 1e-6, 100     # fixed: α from the literature, ε / cap as in PageRank


# --------------------------------------------------------------------------- documents

@dataclass
class PreparedDoc:
    ex: Example
    sentences: list[str]
    stems: list[list[str]]
    spans: list | None                            # spaCy sentence spans (splitter="spacy" only)
    encoder: FrozenBert | None = None
    nlp: object = None                            # for spaCy clause splitting of string sentences
    clause_splitter: str = "rules"
    cache: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.ex.id

    def scores(self, clauses: bool) -> DocumentScores:
        key = ("scores", clauses)
        if key not in self.cache:
            verts = make_vertices(self.spans if self.spans is not None else self.sentences, self.nlp,
                                  clauses=clauses, clause_splitter=self.clause_splitter)
            self.cache[key] = DocumentScores(verts, self.encoder)
        return self.cache[key]

    def graph(self, clauses: bool, cfg: GraphConfig):
        key = ("graph", clauses, tuple(sorted(cfg.__dict__.items())))
        if key not in self.cache:
            self.cache[key] = build_graph(self.scores(clauses), cfg)
        return self.cache[key]

    def drop_bert_cache(self):
        """Free BERT outputs once a document is done (keeps memory flat on big runs)."""
        self.cache.clear()


def prepare(ex: Example, nlp, encoder: FrozenBert | None, max_sentences: int | None = None,
            splitter: str = "nltk", clause_splitter: str = "rules") -> PreparedDoc:
    """Split a document into sentences (team default: NLTK punkt, arXiv/PubMed pre-split;
    see splitting.py). `nlp` (spaCy) is still used to tokenize sentences for the tf-idf
    baselines, and for splitter/clause_splitter="spacy"."""
    sentences, spans = split_document(ex.text, ex.dataset, splitter, nlp)
    if max_sentences:
        sentences = sentences[:max_sentences]
        spans = spans[:max_sentences] if spans is not None else None
    toks = spans if spans is not None else [nlp.make_doc(s) for s in sentences]
    return PreparedDoc(ex, sentences, [B.stem_sentence(t) for t in toks], spans, encoder,
                       nlp, clause_splitter)


# --------------------------------------------------------------------------- budgets

@dataclass(frozen=True)
class Budget:
    """Summary length. kind="sentences": the top `size` sentences. kind="words": add
    sentences best-first until `size` words (whitespace tokens) are used; `overflow`
    decides the sentence that crosses the limit, as in the team's ranking/selector.py:
        include   keep it whole (the summary may run a little over)
        skip      drop it and try the next-ranked sentence that still fits
        truncate  keep it, then cut the summary text at exactly `size` words"""
    kind: str = "sentences"
    size: int = 3
    overflow: str = "include"

    def __str__(self) -> str:
        return f"{self.size} {self.kind}" + (f" ({self.overflow})" if self.kind == "words" else "")

    def covers(self, sentences: list[str]) -> bool:
        """True if the whole document fits, so every system returns all of it."""
        if self.kind == "sentences":
            return len(sentences) <= self.size
        return sum(len(s.split()) for s in sentences) <= self.size


def apply_budget(ranking: list[int], sentences: list[str], budget: Budget) -> list[int]:
    if budget.kind == "sentences":
        return sorted(ranking[:budget.size])
    chosen, used = [], 0
    for i in ranking:
        if used >= budget.size:
            break
        w = len(sentences[i].split())
        if budget.overflow == "skip" and used + w > budget.size:
            continue
        chosen.append(i)
        used += w
    return sorted(chosen)


def summary_text(sentences: list[str], sel: list[int], budget: Budget) -> str:
    """One sentence per line (ROUGE-Lsum needs the breaks); cut to the budget for "truncate"."""
    lines = [sentences[i] for i in sel]
    if budget.kind == "words" and budget.overflow == "truncate":
        out, left = [], budget.size
        for line in lines:
            if left <= 0:
                break
            w = line.split()
            out.append(" ".join(w[:left]))
            left -= len(w)
        lines = out
    return "\n".join(lines)


# --------------------------------------------------------------------------- systems

@dataclass
class System:
    name: str
    group: str                                    # "baseline" / "ours" / "ablation" / "bound"
    fn: Callable                                  # fn(doc, **params) -> ranking, best first
    grid: list[dict] = field(default_factory=lambda: [{}])
    needs_bert: bool = False
    budget_aware: bool = False                    # fn(doc, budget, **params) -> selection

    def select(self, doc: PreparedDoc, budget: Budget, **params) -> list[int]:
        if budget.covers(doc.sentences):
            return list(range(len(doc.sentences)))
        if self.budget_aware:
            return self.fn(doc, budget, **params)
        return apply_budget(self.fn(doc, **params), doc.sentences, budget)


def _grid(**axes) -> list[dict]:
    keys = list(axes)
    return [dict(zip(keys, vals)) for vals in product(*(axes[k] for k in keys))]


BETAS = [0.0, 0.3, 0.6]                          # PacSum's grid
GAMMAS = [0.0, 1.0, 2.0, np.inf]                 # PageTrust's grid
MEMORY = [0.0, 0.5, 0.9, 1.0]                    # proposal's {0, 1} + the team's {0, .5, .9}
LAMBDA1 = [-2.0, -1.0, -0.5, 0.0]                # PacSum forward weight (λ2 fixed at 1)
CLAUSE_LAMBDA = [0.0, 0.5, 1.0]                  # same-sentence clause bonus


def _ours(doc, clauses=False, direction="nsp", ranker="pagetrust", allow_negative=True,
          beta=0.3, gamma=1.0, memory=0.0, lam=0.0, forward_weight=0.0, aggregate="max") -> list[int]:
    """Ranking of the document's sentences (best first; ties broken by position)."""
    cfg = GraphConfig(beta=beta, lam=lam, direction=direction, allow_negative=allow_negative,
                      forward_weight=forward_weight, backward_weight=1.0)
    G = doc.graph(clauses, cfg)
    doc.cache["last_info"] = G.info               # read by evaluate.py for graph diagnostics
    if ranker == "pagetrust":
        v = pagetrust(G.W, alpha=ALPHA, gamma=gamma, memory=memory, eps=EPS, max_iter=MAX_ITER)
    elif ranker == "pagerank":
        v = pagerank(G.W, alpha=ALPHA, eps=EPS, max_iter=MAX_ITER)
    else:
        v = in_degree(G.W)
    ss = sentence_scores(v, G.vertices, aggregate)
    doc.cache["last_tied"] = bool(np.ptp(ss) < 1e-9)  # all sentences scored equal -> Lead by tie-break
    return np.argsort(-ss, kind="stable").tolist()


def build_systems(clauses: bool = False, seed: int = 0, betas: list[float] | None = None,
                  aggregate: str = "max") -> dict[str, System]:
    """betas overrides the β grid: β = 0 keeps every pair, so NSP runs on all n² pairs.
    aggregate: how clause scores become sentence scores (team selector default: max)."""
    rng = np.random.default_rng(seed)
    BETAS = betas or globals()["BETAS"]
    def _oracle(d, budget):
        if budget.kind == "sentences":
            return B.oracle(d.sentences, d.ex.reference, n=budget.size)
        return B.oracle(d.sentences, d.ex.reference, max_words=budget.size, overflow=budget.overflow)

    def _random(d, budget):
        if budget.kind == "sentences":            # same draws as before word budgets existed
            return B.random_pick(len(d.sentences), budget.size, rng)
        return apply_budget(rng.permutation(len(d.sentences)).tolist(), d.sentences, budget)

    S = [
        System("random", "bound", _random, budget_aware=True),
        System("lead", "baseline", lambda d: list(range(len(d.sentences)))),
        System("oracle", "bound", _oracle, budget_aware=True),
        System("degree_tfidf", "baseline", lambda d, beta: B.degree_tfidf(d.stems, None, beta),
               _grid(beta=BETAS)),
        System("textrank", "baseline", lambda d: B.textrank(d.stems, None)),
        System("lexpagerank", "baseline", lambda d: B.lexpagerank(d.stems, None)),
        System("pacsum_tfidf", "baseline",
               lambda d, beta, lambda1: B.pacsum_tfidf(d.stems, None, beta, lambda1),
               _grid(beta=BETAS, lambda1=LAMBDA1)),
        # --- ours + ablations (rows of the proposal's ablation table)
        System("full_pipeline", "ours", lambda d, **p: _ours(d, **p),
               _grid(beta=BETAS, gamma=GAMMAS, memory=MEMORY), needs_bert=True),
        System("abl_position", "ablation",
               lambda d, **p: _ours(d, direction="position", **p),
               _grid(beta=BETAS, forward_weight=LAMBDA1, gamma=GAMMAS, memory=MEMORY), needs_bert=True),
        System("abl_positive_pagerank", "ablation",
               lambda d, **p: _ours(d, ranker="pagerank", allow_negative=False, **p),
               _grid(beta=BETAS), needs_bert=True),
        System("abl_degree", "ablation",
               lambda d, **p: _ours(d, ranker="degree", **p),
               _grid(beta=BETAS), needs_bert=True),
    ]
    if clauses:
        S.append(System("ext_clauses", "ours",
                        lambda d, **p: _ours(d, clauses=True, aggregate=aggregate, **p),
                        _grid(beta=BETAS, gamma=GAMMAS, memory=MEMORY, lam=CLAUSE_LAMBDA),
                        needs_bert=True))
    return {s.name: s for s in S}


# --------------------------------------------------------------------------- convenience

class Summarizer:
    """End-to-end summarizer with fixed parameters (use the ones dev tuning picked)."""

    def __init__(self, model_name="bert-base-uncased", nlp=None, beta=0.3, gamma=1.0,
                 memory=0.0, clauses=False, lam=0.0, splitter="nltk", clause_splitter="rules",
                 aggregate="max", **encoder_kw):
        import spacy
        self.nlp = nlp or spacy.load("en_core_web_sm")
        self.encoder = FrozenBert(model_name, **encoder_kw)
        self.split = dict(splitter=splitter, clause_splitter=clause_splitter)
        self.params = dict(beta=beta, gamma=gamma, memory=memory, clauses=clauses, lam=lam,
                           aggregate=aggregate)

    def summarize(self, text: str, n: int = 3, max_words: int | None = None) -> list[str]:
        """n sentences, or (if max_words is given) a word budget with the "include" rule."""
        doc = prepare(Example("doc", text, []), self.nlp, self.encoder, **self.split)
        budget = Budget("words", max_words) if max_words else Budget("sentences", n)
        if budget.covers(doc.sentences):
            return doc.sentences
        return [doc.sentences[i] for i in apply_budget(_ours(doc, **self.params), doc.sentences, budget)]
