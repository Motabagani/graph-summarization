"""Preprocessor: raw document -> sentences (or clauses), raw and cleaned.

Two copies of every unit, as in the proposal's pipeline table:
  * raw      -- untouched text, for BERT (Grapher / Directioner)
  * tokens   -- gensim-style cleaned tokens, for the tf-idf baselines and the
                tf-idf stand-in graph: lowercase, strip tags / punctuation /
                digits, drop stop words, drop words under 3 chars, Porter-stem.
                This mirrors the TextRank baseline preprocessing so the model and
                baselines see identical units.

Splitting:
  * sentences -- NLTK punkt when available (same as the baselines for CNN/DM,
                 BillSum, TL;DR, WikiHow). arXiv / PubMed come pre-split: pass a
                 list of sentences instead of a string. A regex splitter is the
                 fallback when NLTK is missing (local testing only; use NLTK on HPC
                 so units match the baselines exactly).
  * clauses   -- (extension) rule-based, from the constituent-structure lecture:
                 subordinate conjunctions combine clauses "in special logical,
                 temporal or discourse relationships"; coordinate conjunctions
                 combine like constituents (XP -> XP and XP), so we only split
                 ", and/but/or/so/yet" when what follows looks like a new clause
                 (it starts with a subject-like word); non-restrictive relative
                 clauses (", which / who ...") and ";" also start a clause.
                 Each clause keeps the index of its sentence for the selector.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from porter import PorterStemmer

# ------------------------------------------------------------------ stop words
try:  # identical to the TextRank baseline when gensim is installed (it is on HPC)
    from gensim.parsing.preprocessing import STOPWORDS as _GENSIM_STOP

    STOP_WORDS = set(_GENSIM_STOP)
except ImportError:  # fallback: the course's closed-class list (HW4 stop_list.py)
    STOP_WORDS = set("""a the an and or but about above after along amid among as at by for
    from in into like minus near of off on onto out over past per plus since till to under
    until up via vs with that can cannot could may might must need ought shall should will
    would have had has having be is am are was were being been get gets got gotten getting
    seem seeming seems seemed enough both all your those this these their some our no neither
    my its his her every either each any another just mere such merely right not only sheer
    even especially namely more most less least so too pretty quite rather somewhat
    sufficiently same different when why where how what who whom which whether whose if
    anybody anyone anyplace anything anytime anywhere everybody everyday everyone everyplace
    everything everywhere whatever whenever wherever whichever whoever whomever he him she it
    they them we us you i me""".split())

_stemmer = PorterStemmer()
_stem_cache: dict[str, str] = {}

_TAG_RE = re.compile(r"<[^>]+>")
_WORD_RE = re.compile(r"[a-z]+")  # drops punctuation and digits in one step


def clean_tokens(text: str, stop_words: set[str] = STOP_WORDS, min_len: int = 3) -> list[str]:
    """Gensim-style cleaning (see module docstring)."""
    out = []
    for w in _WORD_RE.findall(_TAG_RE.sub(" ", text.lower())):
        if len(w) < min_len or w in stop_words:
            continue
        s = _stem_cache.get(w)
        if s is None:
            s = _stem_cache[w] = _stemmer.stem(w)
        out.append(s)
    return out


# ------------------------------------------------------------------ sentences
try:
    import nltk

    def split_sentences(text: str) -> list[str]:
        return [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]

    SPLITTER = "nltk"
except ImportError:
    _ABBREV = r"(?<!\bMr)(?<!\bMrs)(?<!\bMs)(?<!\bDr)(?<!\bSt)(?<!\bvs)(?<!\be\.g)(?<!\bi\.e)(?<!\b[A-Z])"
    _SENT_RE = re.compile(_ABBREV + r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9])")

    def split_sentences(text: str) -> list[str]:
        return [s.strip() for s in _SENT_RE.split(text.strip()) if s.strip()]

    SPLITTER = "regex-fallback"


# ------------------------------------------------------------------ clauses (extension)
SUBORD = ("because although though whereas unless while if when whenever since after "
          "before until once so that").split()
# subordinators that are also prepositions ("after the war") only split after a comma
_AMBIGUOUS_SUBORD = {"since", "after", "before", "until", "once", "so", "that", "while", "when", "if"}
COORD = ("and", "but", "or", "so", "yet")
RELATIVE = ("which", "who", "whom", "whose")
# words that typically open a clause's subject NP (pronouns, determiners, proper-name-ish)
SUBJECT_START = set("""i you he she it we they this that these those there the a an his her
its their our my your some many most all each every no one""".split())

_TOKEN_SPAN = re.compile(r"\S+")


def split_clauses(sentence: str, min_words: int = 3) -> list[str]:
    """Rule-based clause splitter (see module docstring). Never returns an empty list."""
    toks = [m.group(0) for m in _TOKEN_SPAN.finditer(sentence)]
    if len(toks) < 2 * min_words:
        return [sentence.strip()]
    cuts = []  # token indexes where a new clause starts
    for i in range(1, len(toks)):
        prev, w = toks[i - 1], toks[i].lower().strip("\"'(")
        nxt = toks[i + 1].lower().strip("\"'(,") if i + 1 < len(toks) else ""
        after_comma = prev.endswith(",")
        if prev.endswith((";", ":")):
            cuts.append(i)
        elif w in COORD and after_comma and (nxt in SUBJECT_START or toks[i + 1][:1].isupper()):
            cuts.append(i)
        elif w in RELATIVE and after_comma:
            cuts.append(i)
        elif w in SUBORD and (after_comma or w not in _AMBIGUOUS_SUBORD):
            cuts.append(i)
    # build clauses, merging fragments shorter than min_words into the previous clause
    bounds = [0] + cuts + [len(toks)]
    segs = [toks[a:b] for a, b in zip(bounds, bounds[1:]) if b > a]
    clauses: list[list[str]] = []
    for seg in segs:
        if clauses and len(seg) < min_words:
            clauses[-1].extend(seg)          # short fragment -> previous clause
        elif clauses and len(clauses[-1]) < min_words:
            clauses[-1].extend(seg)          # short first fragment -> absorb next
        else:
            clauses.append(list(seg))
    return [" ".join(c) for c in clauses]


# ------------------------------------------------------------------ document
@dataclass
class Document:
    sentences: list[str]                    # raw sentences (selector returns these)
    units: list[str]                        # raw graph vertices: sentences or clauses
    unit_tokens: list[list[str]]            # cleaned tokens per unit
    unit_sentence: list[int]                # sentence index of each unit
    unit_type: str = "sentence"
    meta: dict = field(default_factory=dict)

    @property
    def n_units(self) -> int:
        return len(self.units)


def preprocess(doc: str | list[str], unit: str = "sentence", min_tokens: int = 0) -> Document:
    """doc: raw text, or a list of sentences for pre-split corpora (arXiv / PubMed).

    min_tokens > 0 drops units whose cleaned token list is shorter (keeps them out of the
    graph); they are still kept in `sentences`, so output text is unaffected. Leave at 0
    to match the baselines unless they also filter.
    """
    sentences = list(doc) if isinstance(doc, list) else split_sentences(doc)
    sentences = [s.strip() for s in sentences if s and s.strip()]
    units, owner = [], []
    for si, s in enumerate(sentences):
        parts = split_clauses(s) if unit == "clause" else [s]
        units.extend(parts)
        owner.extend([si] * len(parts))
    toks = [clean_tokens(u) for u in units]
    if min_tokens:
        keep = [i for i, t in enumerate(toks) if len(t) >= min_tokens]
        units, owner, toks = [units[i] for i in keep], [owner[i] for i in keep], [toks[i] for i in keep]
    return Document(sentences, units, toks, owner, unit, {"splitter": SPLITTER})
