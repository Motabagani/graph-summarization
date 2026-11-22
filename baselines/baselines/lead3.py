#!/usr/bin/env python3
"""
Lead-k baseline: the summary is the first k sentences of the document
(k = 3 by default, i.e. LEAD-3).

    python lead3.py --dataset cnn_dailymail --split test --data-root ../datasets
    python lead3.py ... --num-sentences 0 --word-limit 220     # first ~220 words (HipoRank's Lead)

Loading, sentence splitting, summary assembly and ROUGE come from summarize.py,
shared with every other baseline.
"""

import sys
from dataclasses import dataclass

import numpy as np

import summarize


@dataclass
class Config:
    pass        # no parameters: k comes from the shared --num-sentences / --word-limit


def add_args(parser):
    pass


def prepare(cfg, args=None):
    pass


def score_sentences(sentences, cfg=None):
    """Earlier sentence = higher score, so the shared selector takes the first k."""
    return np.arange(len(sentences), 0, -1, dtype=float), {"iterations": 0, "converged": True}


def lead(doc, k=3):
    """Summarise one document (string or list of sentences)."""
    return summarize.summarize_doc(doc, score_sentences, Config(),
                                   summarize.SummaryConfig(num_sentences=k))


if __name__ == "__main__":
    summarize.cli(sys.modules[__name__], "lead3", defaults={"num_sentences": 3})
