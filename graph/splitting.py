"""
Sentence, clause and reference splitting, matched to the team repo's ranking/ code so
graph/ and ranking/ see the same units (and ROUGE sees the same reference lines).

Team defaults (what every system here uses unless told otherwise):
    sentences   NLTK punkt                         (ranking/preprocess.py: split_sentences)
    arXiv/PubMed pre-split at " ." tokens           (ranking/run.py: presplit)
    references  one line per sentence: arXiv/PubMed pre-split, else the reference's own
                lines when it has more than one, else NLTK punkt   (ranking/run.py: ref_lines)
    clauses     rule-based splitter                 (ranking/preprocess.py: split_clauses)

`splitter="spacy"` and `clause_splitter="spacy"` keep the original graphsum behaviour
(spaCy sentences / dependency-based clauses in bert_graph.py) for comparison.
"""
from __future__ import annotations

import re

PRESPLIT = {"arxiv", "pubmed"}


# --------------------------------------------------------------------------- sentences

def nltk_sentences(text: str) -> list[str]:
    import nltk
    try:
        sents = nltk.sent_tokenize(text)
    except LookupError as e:
        raise LookupError("NLTK punkt data missing: python -c \"import nltk; nltk.download('punkt_tab')\"") from e
    return [s.strip() for s in sents if s.strip()]


_TOKENIZED_SENT = re.compile(r"(?<= [.?!])\s+")


def presplit(text) -> list[str]:
    """arXiv / PubMed (Cohan et al. 2018) are lowercased, tokenized text whose sentences end
    in a separate " ." token. Cut at those tokens to get the sentence list back (as the
    team's ranking/run.py does)."""
    if isinstance(text, list):
        text = " ".join(text)
    return [s.strip() for s in _TOKENIZED_SENT.split(text) if s.strip()]


def split_document(text: str, dataset: str | None = None, splitter: str = "nltk", nlp=None):
    """Document -> sentences. Returns (sentences, spaCy spans or None).
    Spans are only returned for splitter="spacy" (they're reused for clause splitting)."""
    if dataset in PRESPLIT and splitter == "nltk":
        return presplit(text), None
    if splitter == "nltk":
        return nltk_sentences(text), None
    if splitter == "spacy":
        spans = [s for s in nlp(text).sents if s.text.strip()]
        return [s.text.strip() for s in spans], spans
    raise ValueError(splitter)


def split_reference(ref: str, dataset: str | None = None, splitter: str = "nltk", nlp=None,
                    ref_split: str = "spacy") -> list[str]:
    """Reference -> one sentence per item (ROUGE-Lsum needs the breaks)."""
    ref = ref.replace("<S>", "").replace("</S>", "\n")      # arXiv/PubMed-style markers
    if splitter == "nltk":                                   # = team ranking/run.py: ref_lines
        if dataset in PRESPLIT:
            return presplit(ref)
        lines = [l.strip() for l in ref.split("\n") if l.strip()]
        return lines if len(lines) > 1 else nltk_sentences(ref)
    # original graphsum behaviour
    if ref_split == "newline" and "\n" in ref.strip():
        parts = [re.sub(r"\s+", " ", p).strip(" ,") for p in ref.split("\n")]
    else:
        parts = [s.text.strip() for s in nlp(re.sub(r"\s+", " ", ref)).sents]
    return [p for p in parts if p]


# --------------------------------------------------------------------------- clauses (rules)
# Verbatim port of the team's ranking/preprocess.py: split_clauses, so clause vertices
# match the Preprocessor's. tests/test_team_compat.py checks it against the original.

SUBORD = ("because although though whereas unless while if when whenever since after "
          "before until once so that").split()
_AMBIGUOUS_SUBORD = {"since", "after", "before", "until", "once", "so", "that", "while", "when", "if"}
COORD = ("and", "but", "or", "so", "yet")
RELATIVE = ("which", "who", "whom", "whose")
SUBJECT_START = set("""i you he she it we they this that these those there the a an his her
its their our my your some many most all each every no one""".split())

_TOKEN_SPAN = re.compile(r"\S+")


def rule_clauses(sentence: str, min_words: int = 3) -> list[str]:
    """Rule-based clause splitter (team Preprocessor). Never returns an empty list."""
    toks = [m.group(0) for m in _TOKEN_SPAN.finditer(sentence)]
    if len(toks) < 2 * min_words:
        return [sentence.strip()]
    cuts = []
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
    bounds = [0] + cuts + [len(toks)]
    segs = [toks[a:b] for a, b in zip(bounds, bounds[1:]) if b > a]
    clauses: list[list[str]] = []
    for seg in segs:
        if clauses and len(seg) < min_words:
            clauses[-1].extend(seg)
        elif clauses and len(clauses[-1]) < min_words:
            clauses[-1].extend(seg)
        else:
            clauses.append(list(seg))
    return [" ".join(c) for c in clauses]
