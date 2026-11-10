"""Tests for Hashim's modules.  Run:  python3 -m unittest discover -s tests -v"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pagetrust import (degree_centrality, pacsum_signed_graph, pagerank, pagetrust,  # noqa: E402
                       split_signed, threshold_edges)
from preprocess import clean_tokens, preprocess, split_clauses  # noqa: E402
from selector import select, sentence_scores, summary_text  # noqa: E402

RNG = np.random.default_rng(0)


def reference_pagerank(W, alpha=0.85, iters=5000):
    """Independent PageRank: dense power iteration with explicit loops (slow, obvious)."""
    n = len(W)
    A = np.array(W, float)
    np.fill_diagonal(A, 0)
    A[A < 0] = 0
    x = np.ones(n) / n
    for _ in range(iters):
        new = np.full(n, (1 - alpha) / n)
        for i in range(n):
            out = A[i].sum()
            for j in range(n):
                new[j] += alpha * x[i] * (A[i, j] / out if out > 0 else 1 / n)
        x = new
    return x


class TestPageRank(unittest.TestCase):
    def test_matches_loop_reference(self):
        W = RNG.random((7, 7)) * (RNG.random((7, 7)) > 0.4)
        np.testing.assert_allclose(pagerank(W, eps=1e-12, max_iter=2000), reference_pagerank(W), atol=1e-8)

    def test_dangling_and_sums_to_one(self):
        W = np.zeros((4, 4)); W[0, 1] = W[1, 2] = 1.0  # 2 and 3 dangling
        x = pagerank(W, eps=1e-12, max_iter=1000)
        self.assertAlmostEqual(x.sum(), 1.0)
        np.testing.assert_allclose(x, reference_pagerank(W), atol=1e-8)


class TestPageTrust(unittest.TestCase):
    def test_no_negative_edges_equals_pagerank(self):
        W = RNG.random((9, 9))
        for g in (0, 1, 2, np.inf):
            for m in (0, 1):
                np.testing.assert_allclose(pagetrust(W, gamma=g, memory=m, eps=1e-12, max_iter=2000),
                                           pagerank(W, eps=1e-12, max_iter=2000), atol=1e-9)

    def test_gamma_zero_equals_pagerank_of_positive_part(self):
        W = RNG.random((8, 8)) - 0.3
        np.testing.assert_allclose(pagetrust(W, gamma=0, eps=1e-12, max_iter=2000),
                                   pagerank(W, eps=1e-12, max_iter=2000), atol=1e-9)

    def test_distrusted_vertex_drops(self):
        # symmetric ring: everyone equal under PageRank; 0 and 1 distrust vertex 3
        n = 6
        W = np.zeros((n, n))
        for i in range(n):
            W[i, (i + 1) % n] = W[i, (i - 1) % n] = 1.0
        W[0, 3] = W[1, 3] = -1.0
        pr = pagerank(W, eps=1e-12, max_iter=2000)
        pt = pagetrust(W, gamma=1, eps=1e-12, max_iter=2000)
        np.testing.assert_allclose(pr, 1 / n, atol=1e-9)
        self.assertEqual(int(np.argmin(pt)), 3)
        self.assertLess(pt[3], pr[3])
        self.assertAlmostEqual(pt.sum(), 1.0)

    def test_more_conviction_more_penalty(self):
        n = 6
        W = np.zeros((n, n))
        for i in range(n):
            W[i, (i + 1) % n] = W[i, (i - 1) % n] = 1.0
        W[0, 3] = -1.0
        s = [pagetrust(W, gamma=g, eps=1e-12, max_iter=3000)[3] for g in (0, 1, 2, np.inf)]
        self.assertTrue(all(a >= b - 1e-12 for a, b in zip(s, s[1:])), s)
        self.assertLess(s[-1], s[0])

    def test_memory_off_weakens_penalty(self):
        n = 6
        W = np.zeros((n, n))
        for i in range(n):
            W[i, (i + 1) % n] = W[i, (i - 1) % n] = 1.0
        W[0, 3] = -1.0
        on = pagetrust(W, gamma=1, memory=1, eps=1e-12, max_iter=3000)[3]
        off = pagetrust(W, gamma=1, memory=0, eps=1e-12, max_iter=3000)[3]
        self.assertLessEqual(on, off + 1e-12)

    def test_full_memory_empties_distrusted_vertex(self):
        # paper Table 1: with M = 1 the distrusted node's score goes to 0
        n = 6
        W = np.zeros((n, n))
        for i in range(n):
            W[i, (i + 1) % n] = W[i, (i - 1) % n] = 1.0
        W[0, 3] = -0.01                          # binary distrust: strength ignored
        x = pagetrust(W, gamma=1, memory=1, eps=1e-13, max_iter=200000)
        self.assertLess(x[3], 1e-3)

    # ---- paper: de Kerchove & Van Dooren (2008), Figure 2 graph, Tables 1 and 2
    @staticmethod
    def fig2():
        W = np.zeros((5, 5))                     # nodes 1..5 -> 0..4
        for a, b in [(1, 0), (0, 2), (2, 0), (1, 2), (2, 1), (2, 3), (3, 4), (4, 3), (4, 1)]:
            W[a, b] = 1.0
        W[0, 3] = -1.0                           # node 1 distrusts node 4
        return W

    def test_paper_pagerank_column(self):
        np.testing.assert_allclose(pagerank(self.fig2(), alpha=0.9, eps=1e-12, max_iter=5000).round(2),
                                   [.18, .18, .27, .18, .18])

    def test_paper_tables_alpha_09(self):
        # (M, beta) -> the paper's printed PageTrust vector. The paper's own alpha=.5 PageRank
        # column does not match standard PageRank at .5, so its numbers are not exactly
        # reproducible; we require the same ranking, node 4 at 0 where the paper has 0,
        # and every value within 0.05.
        paper = {(0, 1): [.24, .19, .36, .10, .11], (1, 1): [.31, .20, .49, 0, .01],
                 (0, 2): [.30, .20, .47, .01, .02], (0, np.inf): [.31, .20, .49, 0, .01]}
        for (m, b), ref in paper.items():
            x = pagetrust(self.fig2(), alpha=0.9, gamma=b, memory=m, eps=1e-12, max_iter=100000)
            np.testing.assert_allclose(x, ref, atol=0.05, err_msg=f"M={m} beta={b}")
            self.assertEqual(list(np.argsort(-x)), list(np.argsort(-np.array(ref), kind="stable")),
                             f"ranking M={m} beta={b}")
            if ref[3] == 0:
                self.assertLess(x[3], 0.005)

    def test_valid_state_on_random_signed_graphs(self):
        for _ in range(20):
            n = int(RNG.integers(2, 25))
            W = RNG.normal(size=(n, n)) * (RNG.random((n, n)) > 0.5)
            x, P, _ = pagetrust(W, gamma=2, memory=0.5, return_state=True)
            self.assertTrue(np.all(x >= 0)); self.assertAlmostEqual(x.sum(), 1.0)
            self.assertTrue(np.all((P >= -1e-12) & (P <= 1 + 1e-12)))

    def test_all_negative_graph_does_not_crash(self):
        x = pagetrust(-np.ones((5, 5)), gamma=np.inf)
        np.testing.assert_allclose(x, 0.2, atol=1e-9)

    def test_split_signed(self):
        pos, D = split_signed(np.array([[5, 2, -4], [-2, 0, 1], [0, 0, 0]]), distrust="weighted")
        self.assertEqual(pos[0, 0], 0)          # diagonal ignored
        self.assertEqual(D[0, 2], 1.0); self.assertEqual(D[1, 0], 0.5)


class TestGraphBuilders(unittest.TestCase):
    def test_pacsum_graph_degree_equals_pacsum_formula(self):
        n = 6
        e = threshold_edges(RNG.random((n, n)) + RNG.random((n, n)).T, beta=0.3)
        lf, lb = -0.7, 1.3
        W = pacsum_signed_graph(e, lf, lb)
        expected = [lf * sum(e[i, j] for j in range(i + 1, n)) + lb * sum(e[i, j] for j in range(i))
                    for i in range(n)]
        np.testing.assert_allclose(degree_centrality(W), expected)

    def test_threshold(self):
        s = np.array([[0, .2, .8], [.2, 0, .5], [.8, .5, 0]])
        e = threshold_edges(s, beta=0.5)        # threshold = .2 + .5*.6 = .5
        np.testing.assert_allclose(e, [[0, 0, .3], [0, 0, 0], [.3, 0, 0]], atol=1e-12)


class TestPreprocess(unittest.TestCase):
    def test_clean_tokens(self):
        self.assertEqual(clean_tokens("<b>The</b> pressures were analyzed in 1958, by us!"),
                         ["pressur", "analyz"])

    def test_presplit_input_kept(self):
        d = preprocess(["first sentence here.", "  ", "second one."])
        self.assertEqual(d.sentences, ["first sentence here.", "second one."])
        self.assertEqual(d.unit_sentence, [0, 1])

    def test_clauses(self):
        s = "The model failed because the data was noisy, which surprised the team."
        cl = split_clauses(s)
        self.assertEqual(cl, ["The model failed", "because the data was noisy,",
                              "which surprised the team."])
        self.assertEqual(split_clauses("Salt and pepper were added to the soup today."),
                         ["Salt and pepper were added to the soup today."])  # NP coordination
        self.assertEqual(" ".join(split_clauses(s)), s)

    def test_clause_units_map_to_sentences(self):
        d = preprocess(["The home team won, but they protested loudly about it.", "Fine."], unit="clause")
        self.assertEqual(d.unit_sentence, [0, 0, 1])


class TestSelector(unittest.TestCase):
    S = ["a b c", "d e f g", "h i", "j k l m n"]

    def test_sentence_budget_doc_order(self):
        self.assertEqual(select(self.S, [.1, .9, .5, .7], "sentences", 2), [1, 3])

    def test_word_budget_modes(self):
        sc = [.1, .9, .5, .7]                    # rank: 1(4w) 3(5w) 2(2w) 0(3w)
        self.assertEqual(select(self.S, sc, "words", 6, "include"), [1, 3])
        self.assertEqual(select(self.S, sc, "words", 6, "skip"), [1, 2])
        ch = select(self.S, sc, "words", 6, "truncate")
        self.assertEqual(len(summary_text(self.S, ch, "words", 6, "truncate").split()), 6)

    def test_clause_aggregation(self):
        np.testing.assert_allclose(sentence_scores([1, 3, 2], [0, 0, 1], 3, "max")[:2], [3, 2])
        np.testing.assert_allclose(sentence_scores([1, 3, 2], [0, 0, 1], 3, "mean")[:2], [2, 2])
        self.assertTrue(np.isneginf(sentence_scores([1], [0], 2)[1]))


if __name__ == "__main__":
    unittest.main()
