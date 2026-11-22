"""
Shared pieces of the team pipeline (preprocessor -> grapher -> ranker).

Nothing here re-implements the team's methods. It wires the three folders together:
    baselines/baselines/summarize.py   dataset loading, sentence splitting, summary assembly
                                       (the exact rules every baseline number uses)
    ranking/preprocess.py, pagetrust.py, selector.py   Hashim: units, tokens, rankers
    graph/bert_graph.py                 Alfardil: frozen BERT similarity, NSP direction, W
and holds the per-dataset settings that make results comparable with the baselines.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RANKING_DIR = os.environ.get("TEAM_RANKING", os.path.join(REPO, "ranking"))
GRAPH_DIR = os.environ.get("TEAM_GRAPH", os.path.join(REPO, "graph"))
BASELINES_DIR = os.environ.get("TEAM_BASELINES", os.path.join(REPO, "baselines", "baselines"))
for _p in (BASELINES_DIR, RANKING_DIR, GRAPH_DIR):     # appended: this folder's modules come first
    if _p not in sys.path:
        sys.path.append(_p)

import summarize                                        # noqa: E402  (baselines)

# --------------------------------------------------------------------------- datasets
# Identical to the baselines (slurm/run_baselines.sh): summary length, which documents are
# test, which are development (first 1,000 of the tuning split), sentence splitting.
DEV_DOCS = 1000
DATASETS = {
    #                budget                 test (split, offset)  dev (split, offset)
    "cnn_dailymail": dict(budget=("sentences", 3), test=("test", 0), dev=("validation", 0)),
    "pubmed":        dict(budget=("words", 203),   test=("test", 0), dev=("validation", 0)),
    "arxiv":         dict(budget=("words", 220),   test=("test", 0), dev=("validation", 0)),
    "billsum":       dict(budget=("words", 184),   test=("test", 0), dev=("train", 0)),
    "wikihow":       dict(budget=("words", 55),    test=("all", 1000), dev=("all", 0), newline=True),
    "tldr17":        dict(budget=("sentences", 1), test=("all", 1000), dev=("all", 0)),
}


def summary_config(dataset: str) -> "summarize.SummaryConfig":
    """The baselines' summary settings for `dataset` (length, truncation, splitting)."""
    spec = DATASETS[dataset]
    kind, size = spec["budget"]
    return summarize.SummaryConfig(
        newline_boundaries=spec.get("newline", False),
        num_sentences=size if kind == "sentences" else 0,
        word_limit=size if kind == "words" else 100,       # 100 = unused default, as in the baselines
        truncate=True, order="score")


# --------------------------------------------------------------------------- doc files
def read_docs(path: str):
    """Yield the standard documents written by preprocessor.py (one JSON object per line)."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_module(name: str, path: str):
    """Import a file under a private name (ranking/pipeline.py and graph/pipeline.py share a name)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def import_bert_graph(need_torch: bool = False):
    """graph/bert_graph.py. The ranker only needs its graph-building code, which is pure numpy,
    so on machines without PyTorch the torch/transformers imports are stubbed out. The grapher
    (need_torch=True) needs the real libraries."""
    if "bert_graph" in sys.modules:
        return sys.modules["bert_graph"]
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        stubbed = []
    except ImportError:
        if need_torch:
            raise SystemExit("The BERT grapher needs PyTorch and transformers: pip install -r graph/requirements.txt")
        torch_stub = types.ModuleType("torch")
        torch_stub.no_grad = lambda: (lambda f: f)
        torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
        tf_stub = types.ModuleType("transformers")
        tf_stub.AutoTokenizer = tf_stub.BertForNextSentencePrediction = None
        stubbed = [m for m, s in (("torch", torch_stub), ("transformers", tf_stub)) if m not in sys.modules]
        sys.modules.update({"torch": torch_stub, "transformers": tf_stub})
    try:
        import bert_graph
    finally:
        for m in stubbed:                 # other libraries (scipy, nltk) must not see the stubs
            sys.modules.pop(m, None)
    return bert_graph


# --------------------------------------------------------------------------- ROUGE
class Rouge:
    """ROUGE exactly as the baselines compute it: rouge-score F1, no stemming; ROUGE-1/2/L on
    space-joined text, ROUGE-Lsum on newline-joined sentences."""

    def __init__(self, metrics=("rouge1", "rouge2", "rougeL", "rougeLsum"), stemmer: bool = False):
        from rouge_score import rouge_scorer
        self.flat = [m for m in metrics if m != "rougeLsum"]
        self.lsum = "rougeLsum" in metrics
        self._flat = rouge_scorer.RougeScorer(self.flat, use_stemmer=stemmer) if self.flat else None
        self._lsum = rouge_scorer.RougeScorer(["rougeLsum"], use_stemmer=stemmer) if self.lsum else None

    def score(self, ref_sents, summ_sents) -> dict:
        out = {}
        if self._flat:
            out.update(self._flat.score(" ".join(ref_sents), " ".join(summ_sents)))
        if self._lsum:
            out.update(self._lsum.score("\n".join(ref_sents), "\n".join(summ_sents)))
        return {k: (v.precision, v.recall, v.fmeasure) for k, v in out.items()}
