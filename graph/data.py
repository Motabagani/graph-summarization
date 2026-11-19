"""
Dataset loading for the six evaluation sets in the proposal.

Splits: CNN/DM, arXiv and PubMed keep their official train/validation/test splits.
tldr-17, WikiHow and BillSum are pooled and re-split 80/10/10 with a fixed seed,
which reproduces the proposal's split sizes (e.g. BillSum 18,764 / 2,345 / 2,346).

Only dev and test are needed (the pipeline uses a frozen BERT), and each is sampled
down to --limit-dev / --limit-test documents with the same seed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from splitting import split_reference


@dataclass
class Example:
    id: str
    text: str
    reference: list[str]          # reference summary, one sentence per item
    dataset: str | None = None    # decides arXiv/PubMed pre-splitting (splitting.PRESPLIT)


@dataclass
class Spec:
    hf_id: str | None
    config: str | None
    text: str
    ref: str
    splits: dict | None           # {"dev": hf_split, "test": hf_split}; None -> pooled 80/10/10
    pool: tuple = ("train",)      # HF splits to pool when splits is None
    ref_split: str = "spacy"      # how to split references: "newline" or "spacy"


DATASETS: dict[str, Spec] = {
    "cnn_dm":  Spec("abisee/cnn_dailymail", "3.0.0", "article", "highlights",
                    {"dev": "validation", "test": "test"}, ref_split="newline"),
    "arxiv":   Spec("ccdv/arxiv-summarization", "document", "article", "abstract",
                    {"dev": "validation", "test": "test"}),
    "pubmed":  Spec("ccdv/pubmed-summarization", "document", "article", "abstract",
                    {"dev": "validation", "test": "test"}),
    "tldr":    Spec("webis/tldr-17", None, "content", "summary", None),
    "wikihow": Spec(None, None, "text", "headline", None, ref_split="newline"),  # local CSV
    "billsum": Spec("FiscalNote/billsum", None, "text", "summary", None,
                    pool=("train", "test", "ca_test")),
}


def _hf_load(spec: Spec, split: str):
    from datasets import load_dataset
    try:
        return load_dataset(spec.hf_id, spec.config, split=split)
    except Exception as first:   # script-based repos: fall back to the auto-converted parquet
        try:
            return load_dataset(spec.hf_id, spec.config, split=split, revision="refs/convert/parquet")
        except Exception:
            raise first


def _split_reference(ref: str, how: str, nlp) -> list[str]:
    """Original graphsum reference splitting (spaCy); see splitting.split_reference."""
    return split_reference(ref, splitter="spacy", nlp=nlp, ref_split=how)


def _pooled_indices(n: int, seed: int) -> dict[str, np.ndarray]:
    perm = np.random.default_rng(seed).permutation(n)
    a, b = int(0.8 * n), int(0.9 * n)
    return {"train": perm[:a], "dev": perm[a:b], "test": perm[b:]}


def load_split(name: str, split: str, limit: int | None, nlp, seed: int = 0,
               wikihow_csv: str | None = None, splitter: str = "nltk") -> list[Example]:
    """split in {"dev", "test"}. Returns up to `limit` examples, sampled with `seed`.
    References are split like the team's ranking/run.py (splitter="nltk") or with spaCy."""
    spec = DATASETS[name]
    rng = np.random.default_rng(seed + (0 if split == "dev" else 1))

    if name == "wikihow":
        if not wikihow_csv:
            raise ValueError("WikiHow needs --wikihow-csv path/to/wikihowAll.csv "
                             "(https://github.com/mahnazkoupaee/WikiHow-Dataset)")
        import pandas as pd
        df = pd.read_csv(wikihow_csv).dropna(subset=["text", "headline"])
        idx = _pooled_indices(len(df), seed)[split]
        if limit:
            idx = rng.choice(idx, size=min(limit, len(idx)), replace=False)
        rows = [df.iloc[int(i)] for i in idx]
        get = lambda r, f: r[f]
        ids = [str(int(i)) for i in idx]
    else:
        if spec.splits:
            ds = _hf_load(spec, spec.splits[split])
            idx = np.arange(len(ds))
        else:
            from datasets import concatenate_datasets
            ds = concatenate_datasets([_hf_load(spec, s) for s in spec.pool])
            idx = _pooled_indices(len(ds), seed)[split]
        if limit:
            idx = rng.choice(idx, size=min(limit, len(idx)), replace=False)
        rows = ds.select([int(i) for i in idx])
        get = lambda r, f: r[f]
        ids = [str(int(i)) for i in idx]

    out = []
    for i, r in zip(ids, rows):
        text, ref = str(get(r, spec.text) or ""), str(get(r, spec.ref) or "")
        if not text.strip() or not ref.strip():
            continue
        refs = split_reference(ref, name, splitter, nlp, spec.ref_split)
        out.append(Example(f"{name}-{split}-{i}", text, refs, name))
    return out


# --------------------------------------------------------------------------- offline smoke data

def synthetic(n_docs: int, seed: int = 0) -> list[Example]:
    """Toy documents for testing the harness without network access. Each document has
    a few topic sentences, distractors, and a reference that paraphrases the topic ones."""
    rng = np.random.default_rng(seed)
    topics = ["council", "budget", "storm", "vaccine", "election", "bridge", "school", "river"]
    verbs = ["approved", "delayed", "funded", "blocked", "praised", "reviewed"]
    fillers = ["the weather was mild", "a local team won a match", "traffic was light downtown",
               "a bakery opened on main street", "the museum extended its hours"]
    docs = []
    for d in range(n_docs):
        t = rng.choice(topics, 2, replace=False)
        key = [f"Officials {rng.choice(verbs)} the {t[0]} plan after a long {t[1]} debate.",
               f"The {t[0]} decision affects the {t[1]} schedule for next year.",
               f"Critics said the {t[0]} plan ignores the {t[1]} problem."]
        sents = key + [f"Meanwhile, {f}." for f in rng.choice(fillers, 4, replace=False)]
        order = rng.permutation(len(sents))
        text = " ".join(sents[k] for k in order)
        ref = [f"The {t[0]} plan was {rng.choice(verbs)} amid the {t[1]} debate.",
               f"Critics say it ignores the {t[1]} problem."]
        docs.append(Example(f"synthetic-{d}", text, ref))
    return docs
