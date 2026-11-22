#!/usr/bin/env python3
"""
Degree-centrality baseline (Erkan & Radev 2004's "Degree"; PacSum Eq. 1):

    centrality(s_i) = sum_j w_ji

i.e. the sum of the weights of the edges into sentence i (its weighted
in-degree; the row sum when the graph is undirected). This is degree
*centrality* on the weighted graph, not the vertex degree (the number of edges).

on exactly the same sentence graph as TextRank (built by textrank.build_graph),
so the two differ only in PageRank vs. simple degree.

    python DegreeCentrality.py --dataset cnn_dailymail --split test --data-root ../datasets
PacSum's DEGREE (tf-idf) baseline:
    --preprocess gensim --similarity tfidf_dot --num-sentences 3

Loading, splitting, summary assembly and ROUGE come from summarize.py.
"""

import sys

import summarize
import textrank

Config = textrank.GraphConfig          # graph options only; no PageRank parameters
add_args = textrank.add_graph_args
prepare = textrank.prepare


def score_sentences(sentences, cfg):
    """Degree centrality of each sentence: sum of incoming edge weights in the TextRank graph."""
    W = textrank.build_graph(sentences, cfg)
    return W.sum(axis=0), {"iterations": 0, "converged": True}


def degree_centrality(doc, cfg=None, summary_cfg=None):
    """Summarise one document (string or list of sentences)."""
    return summarize.summarize_doc(doc, score_sentences, cfg or Config(),
                                   summary_cfg or summarize.SummaryConfig())


if __name__ == "__main__":
    summarize.cli(sys.modules[__name__], "degree_centrality")
