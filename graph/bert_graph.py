"""
BERT side of the summarization pipeline: frozen BERT → directed, signed document graph.

    text ──spaCy──> vertices (sentences, or clauses in the extension)
         ──BERT encoder──> vectors ──dot product──> similarity, PacSum β threshold
         ──BERT NSP head (both orders)──> edge direction + confidence, negative edges
         ──> weight matrix W  (feed to PageTrust / PageRank / degree)

Matrix convention used everywhere in this file:
    W[i, j] != 0  means an edge i -> j, i.e. vertex i "endorses" vertex j.
    A vertex's score should therefore come from its *incoming* column W[:, j].

Direction semantics (NSP):
    p(A,B) = P(isNext | "[CLS] A [SEP] B [SEP]")  -- "B follows / builds on A".
    If A-then-B is the more plausible order, B builds on A, so the edge is B -> A
    (the later/referring vertex endorses the one it refers back to). This matches
    PacSum's finding that being built on by other sentences signals importance.
    If BERT rejects both orders (isNext log-odds < 0 both ways), the pair does not fit
    together and gets a negative edge in both directions.

Typical use:
    enc = FrozenBert("bert-base-uncased")
    verts = make_vertices(splitting.nltk_sentences(doc_text))   # or make_vertices(doc_text, nlp): spaCy split
    scores = DocumentScores(verts, enc)          # BERT runs here (cached per doc)
    W = build_graph(scores, GraphConfig(beta=0.3))  # cheap; re-run per β / λ when tuning
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch
from transformers import AutoTokenizer, BertForNextSentencePrediction


# --------------------------------------------------------------------------- vertices

@dataclass
class Vertex:
    text: str          # raw text: BERT is given the unstemmed, unfiltered text
    sentence_id: int   # index of the sentence it came from (same as idx without clauses)
    idx: int           # vertex index in document order


# Clause-level attachments that start a new clause. ccomp/xcomp are deliberately absent:
# a complement clause is an *argument* of its verb ("Critics said [the plan ignores roads]"),
# so it stays with that verb.
CLAUSE_DEPS = {"conj", "advcl", "relcl", "parataxis"}


def _clause_text(toks, sent_end: int) -> str:
    """Render a (possibly non-contiguous) clause: drop punctuation next to a gap left by a
    removed sub-clause, strip dangling conjunctions/commas at the edges, keep the
    sentence-final punctuation."""
    n = len(toks)
    keep = []
    for k, t in enumerate(toks):
        if t.is_punct and t.i != sent_end - 1:
            gap_before = k > 0 and toks[k - 1].i != t.i - 1
            gap_after = k < n - 1 and toks[k + 1].i != t.i + 1
            if gap_before or gap_after:
                continue
        keep.append(t)
    edge = lambda t: t.dep_ == "cc" or (t.is_punct and t.i != sent_end - 1)
    while keep and (edge(keep[0]) or keep[0].is_punct):
        keep.pop(0)
    while keep and edge(keep[-1]):
        keep.pop()
    parts = []
    for k, t in enumerate(keep):
        nxt = keep[k + 1] if k + 1 < len(keep) else None
        ws = t.whitespace_ if (nxt is not None and nxt.i == t.i + 1) else " "
        if nxt is not None and nxt.is_punct and nxt.i != t.i + 1:
            ws = ""
        parts.append(t.text + ws)
    return "".join(parts).strip()


def _split_clauses(sent, min_tokens: int = 3) -> list[str]:
    """Split one spaCy sentence span into clauses.

    A clause head is the sentence root, or a verb attached by a clause-level
    dependency (conj/advcl/relcl/parataxis). Each token joins the clause of its
    nearest clause-head ancestor, so a clause keeps its verb and arguments. Clauses
    with fewer than `min_tokens` non-punctuation tokens are merged into their parent.
    Clauses may be non-contiguous ("The man [who I saw] arrived" -> "The man arrived"
    + "who I saw").
    """
    heads = {sent.root} | {
        t for t in sent if t.dep_ in CLAUSE_DEPS and t.pos_ in ("VERB", "AUX")
    }

    def owner(tok):
        while tok not in heads and tok.head is not tok:
            tok = tok.head
        return tok if tok in heads else sent.root

    while True:
        groups: dict = {}
        for tok in sent:
            groups.setdefault(owner(tok), []).append(tok)
        small = [
            h for h, toks in groups.items()
            if h is not sent.root and sum(not t.is_punct for t in toks) < min_tokens
        ]
        if not small:
            break
        heads.difference_update(small)

    clauses = []
    for toks in sorted(groups.values(), key=lambda ts: ts[0].i):
        text = _clause_text(toks, sent.end)
        if text and any(not t.is_punct for t in toks):
            clauses.append(text)
    return clauses


def make_vertices(text_or_sents, nlp=None, clauses: bool = False, min_clause_tokens: int = 3,
                  clause_splitter: Literal["rules", "spacy"] = "spacy") -> list[Vertex]:
    """Vertices in document order. Accepts
      * raw text (spaCy-split; needs `nlp`),
      * a list of spaCy sentence spans, or
      * a list of sentence strings (e.g. NLTK punkt output, the team's default).
    With clauses=True each sentence becomes clause vertices: clause_splitter="rules" is the
    team Preprocessor's splitter (splitting.rule_clauses); "spacy" is the dependency-based
    one above (needs `nlp` when given strings; a string that spaCy itself splits into several
    sentences keeps one sentence_id)."""
    from splitting import rule_clauses
    if isinstance(text_or_sents, str):
        sents = [s for s in nlp(text_or_sents).sents if s.text.strip()]
    else:
        sents = list(text_or_sents)
    as_text = [s if isinstance(s, str) else s.text for s in sents]
    if clauses and clause_splitter == "spacy" and any(isinstance(s, str) for s in sents):
        parsed = list(nlp.pipe(as_text))
        sents = [s if not isinstance(s, str) else list(p.sents) for s, p in zip(sents, parsed)]
    verts: list[Vertex] = []
    for sid, sent in enumerate(sents):
        if not clauses:
            pieces = [as_text[sid].strip()]
        elif clause_splitter == "rules":
            pieces = rule_clauses(as_text[sid].strip(), min_clause_tokens)
        else:
            spans = sent if isinstance(sent, list) else [sent]
            pieces = [c for sp in spans for c in _split_clauses(sp, min_clause_tokens)] or [as_text[sid].strip()]
        for p in pieces:
            verts.append(Vertex(text=p, sentence_id=sid, idx=len(verts)))
    return verts


def vertices_from_units(units: list[str], unit_sentence: list[int]) -> list[Vertex]:
    """Vertices from the team Preprocessor's output (ranking/preprocess.py: Document.units /
    Document.unit_sentence), so the graph is built on exactly the ranker's units:
        doc = preprocess(text, unit="clause")
        W = build_graph(DocumentScores(vertices_from_units(doc.units, doc.unit_sentence), enc), cfg).W
    """
    return [Vertex(text=u, sentence_id=int(s), idx=k) for k, (u, s) in enumerate(zip(units, unit_sentence))]


# --------------------------------------------------------------------------- frozen BERT

class FrozenBert:
    """One frozen BERT checkpoint used for both sentence vectors and NSP.

    BertForNextSentencePrediction = encoder (`.bert`) + pretrained NSP head (`.cls`),
    so a single load serves both. Nothing here computes gradients.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        pooling: Literal["mean", "cls"] = "mean",
        batch_size: int = 32,
        max_length: int = 512,
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = BertForNextSentencePrediction.from_pretrained(model_name).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.pooling = pooling
        self.batch_size = batch_size
        self.max_length = max_length

    @torch.no_grad()
    def embed(self, texts: list[str]) -> np.ndarray:
        """(n, hidden) sentence vectors. Mean pooling by default: a frozen, un-fine-tuned
        [CLS] vector is a weak sentence representation."""
        out = []
        for k in range(0, len(texts), self.batch_size):
            enc = self.tok(
                texts[k:k + self.batch_size], padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
            ).to(self.device)
            h = self.model.bert(**enc).last_hidden_state            # (b, L, d)
            if self.pooling == "cls":
                v = h[:, 0]
            else:
                m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
                v = (h * m).sum(1) / m.sum(1).clamp(min=1)
            out.append(v.float().cpu())
        return torch.cat(out).numpy() if out else np.zeros((0, self.model.config.hidden_size))

    @torch.no_grad()
    def nsp_logodds(self, firsts: list[str], seconds: list[str]) -> np.ndarray:
        """isNext log-odds for each (first, second) pair: logit[isNext] - logit[notNext].
        > 0 means BERT thinks `second` plausibly follows `first`. Log-odds rather than
        probabilities, because NSP probabilities saturate near 1 for in-document pairs."""
        out = []
        for k in range(0, len(firsts), self.batch_size):
            enc = self.tok(
                firsts[k:k + self.batch_size], seconds[k:k + self.batch_size],
                padding=True, truncation="longest_first",
                max_length=self.max_length, return_tensors="pt",
            ).to(self.device)
            logits = self.model(**enc).logits.float()              # (b, 2): [isNext, notNext]
            out.append((logits[:, 0] - logits[:, 1]).cpu())
        return torch.cat(out).numpy() if out else np.zeros(0)


# --------------------------------------------------------------------------- per-document cache

class DocumentScores:
    """Expensive BERT outputs for one document, computed once and reused across the
    β / λ / direction grid. Similarities are computed for all pairs up front; NSP runs
    lazily only for pairs that some graph actually needs (those surviving the threshold),
    since it costs two BERT passes per pair."""

    def __init__(self, vertices: list[Vertex], encoder: FrozenBert):
        self.vertices = vertices
        self.encoder = encoder
        vecs = encoder.embed([v.text for v in vertices])
        self.sim = vecs @ vecs.T                                    # raw dot products (PacSum)
        self._nsp: dict[tuple[int, int], float] = {}                # (a, b) -> log-odds "b follows a"

    @property
    def n(self) -> int:
        return len(self.vertices)

    def nsp(self, pairs: list[tuple[int, int]]) -> dict[tuple[int, int], tuple[float, float]]:
        """For each i<j pair return (logodds(i then j), logodds(j then i)), computing misses."""
        need = [(a, b) for (i, j) in pairs for (a, b) in ((i, j), (j, i)) if (a, b) not in self._nsp]
        if need:
            T = [v.text for v in self.vertices]
            lo = self.encoder.nsp_logodds([T[a] for a, _ in need], [T[b] for _, b in need])
            self._nsp.update(zip(need, lo.tolist()))
        return {(i, j): (self._nsp[(i, j)], self._nsp[(j, i)]) for (i, j) in pairs}


# --------------------------------------------------------------------------- graph

@dataclass
class GraphConfig:
    beta: float = 0.3                 # PacSum threshold: keep sim >= min + β(max - min)
    lam: float = 0.0                  # extra weight on same-sentence clause edges (extension)
    direction: Literal["nsp", "position", "none"] = "nsp"
    confidence: Literal["logodds", "prob", "none"] = "logodds"
    allow_negative: bool = True       # False -> "Positive-weights" ablation (drop negative edges)
    # Position-heuristic ablation (PacSum style): edge from earlier j to later i gets
    # forward_weight, from later to earlier gets backward_weight. Tune on dev; PacSum's
    # tuned forward weight came out negative.
    forward_weight: float = 0.0
    backward_weight: float = 1.0


@dataclass
class Graph:
    W: np.ndarray                     # (n, n), W[i, j] = weight of edge i -> j
    vertices: list[Vertex]
    info: dict = field(default_factory=dict)


def _confidence(d_ab: float, d_ba: float, mode: str) -> float:
    if mode == "none":
        return 1.0
    if mode == "prob":                # literal gap in isNext probability (saturates)
        s = lambda x: 1.0 / (1.0 + np.exp(-x))
        return abs(s(d_ab) - s(d_ba))
    return float(np.tanh(abs(d_ab - d_ba) / 2.0))  # = 2σ(|Δ log-odds|) - 1, in [0, 1)


def build_graph(scores: DocumentScores, cfg: GraphConfig) -> Graph:
    n = scores.n
    W = np.zeros((n, n))
    info = {"n": n, "kept": 0, "directed": 0, "rejected_pairs": 0, "same_sentence": 0}
    if n < 2:
        return Graph(W, scores.vertices, info)

    iu = np.triu_indices(n, k=1)
    off = scores.sim[iu]
    lo, hi = float(off.min()), float(off.max())
    thresh = lo + cfg.beta * (hi - lo)
    # Similarity rescaled to [0, 1]. If every pair is equally similar (always the case
    # for a 2-vertex document) min-max would zero every edge, so treat them all as 1.
    norm = (scores.sim - lo) / (hi - lo) if hi > lo else np.ones_like(scores.sim)
    sid = np.array([v.sentence_id for v in scores.vertices])

    same, cross = [], []
    for i, j in zip(*iu):
        i, j = int(i), int(j)
        if sid[i] == sid[j]:
            same.append((i, j))               # always kept: holds a sentence's clauses together
        elif scores.sim[i, j] >= thresh:
            cross.append((i, j))

    # Same-sentence clause pairs: undirected, boosted by λ, never negative. NSP on two
    # fragments of one sentence is not meaningful and would often reject both orders.
    for i, j in same:
        W[i, j] = W[j, i] = max(norm[i, j], 0.0) + cfg.lam
    info["same_sentence"] = len(same)
    info["kept"] = len(cross)

    if cfg.direction == "none":
        for i, j in cross:
            W[i, j] = W[j, i] = norm[i, j]
    elif cfg.direction == "position":
        for i, j in cross:                    # i < j, so i is earlier
            W[i, j] = norm[i, j] * cfg.forward_weight   # earlier -> later
            W[j, i] = norm[i, j] * cfg.backward_weight  # later -> earlier
            if not cfg.allow_negative:
                W[i, j], W[j, i] = max(W[i, j], 0.0), max(W[j, i], 0.0)
    else:  # nsp
        lo_odds = scores.nsp(cross)
        for (i, j), (d_ij, d_ji) in lo_odds.items():
            s = norm[i, j]
            if d_ij < 0 and d_ji < 0:         # BERT rejects both orders: pair doesn't fit
                info["rejected_pairs"] += 1   # NSP rejected both orders
                if cfg.allow_negative:
                    W[i, j] = W[j, i] = -s
                continue
            w = s * _confidence(d_ij, d_ji, cfg.confidence)
            if d_ij >= d_ji:                  # "i then j" more plausible: j builds on i -> edge j -> i
                W[j, i] = w
            else:                             # i builds on j -> edge i -> j
                W[i, j] = w
            info["directed"] += 1
    info["threshold"] = thresh
    info["negative_edges"] = int((W < 0).sum())       # directed entries, any mode
    return Graph(W, scores.vertices, info)


# --------------------------------------------------------------------------- scoring helpers

def in_degree_scores(W: np.ndarray) -> np.ndarray:
    """Degree-only ablation / baseline: sum of incoming (signed) edge weights."""
    return W.sum(axis=0)


def sentence_scores(vertex_scores: np.ndarray, vertices: list[Vertex],
                    aggregate: Literal["max", "mean", "sum"] = "max") -> np.ndarray:
    """Combine vertex (clause) scores into one score per sentence. "max" is the team
    selector's default (ranking/selector.py); "mean" stops long sentences winning by
    having more clauses; "sum" favours them. Identity when vertices are full sentences."""
    n_sent = max(v.sentence_id for v in vertices) + 1
    sid = np.array([v.sentence_id for v in vertices])
    vs = np.asarray(vertex_scores, dtype=float)
    if aggregate == "max":
        out = np.full(n_sent, -np.inf)
        np.maximum.at(out, sid, vs)
        return np.where(np.isfinite(out), out, 0.0)
    tot = np.zeros(n_sent)
    np.add.at(tot, sid, vs)
    if aggregate == "sum":
        return tot
    if aggregate == "mean":
        return tot / np.maximum(np.bincount(sid, minlength=n_sent), 1)
    raise ValueError(aggregate)


def top_n(sent_scores: np.ndarray, n: int) -> list[int]:
    """Indices of the top-n sentences, returned in document order."""
    return sorted(np.argsort(-sent_scores, kind="stable")[:n].tolist())
