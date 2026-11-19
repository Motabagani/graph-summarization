"""
Checks that graph/ agrees with the team repo's ranking/ code:
PageRank / PageTrust scores, W convention, budgets and selection, sentence / clause /
reference splitting, and clause -> sentence aggregation.

Finds ranking/ at $TEAM_RANKING, or ../ranking (graph/ inside the team repo). Without it,
only the self-contained checks run. With it, rankers.pagetrust is the team's own function,
and own_pagetrust (the standalone copy) is compared against it.

    python tests/test_team_compat.py
    TEAM_RANKING=/path/to/graph-summarization/ranking python tests/test_team_compat.py
"""
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import rankers  # noqa: E402
from bert_graph import DocumentScores, GraphConfig, Vertex, build_graph, sentence_scores  # noqa: E402
from evaluate import TEAM_BUDGETS  # noqa: E402
from pipeline import Budget, apply_budget  # noqa: E402
from splitting import nltk_sentences, presplit, rule_clauses, split_reference  # noqa: E402


def random_signed_graph(rng, n):
    """Shaped like bert_graph output: per pair one directed positive edge, a mutual negative
    edge (NSP rejected both orders), or nothing."""
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            r, v = rng.random(), rng.random()
            if r < 0.4:
                W[j, i] = v
            elif r < 0.7:
                W[i, j] = v
            elif r < 0.85:
                W[i, j] = W[j, i] = -v
    return W


# ------------------------------------------------------------------ self-contained
class Stub:
    def embed(self, texts): return np.eye(len(texts)) + 1.0
    def nsp_logodds(self, A, B): return np.array([-1.0 if {a, b} == {"a", "c"} else (2.0 if a < b else -0.5)
                                                  for a, b in zip(A, B)])


W = build_graph(DocumentScores([Vertex(t, i, i) for i, t in enumerate("abc")], Stub()), GraphConfig(beta=0.0)).W
assert W.shape == (3, 3) and np.all(np.diag(W) == 0)
assert W[0, 2] < 0 and W[2, 0] < 0                 # rejected both orders -> negative both ways
assert ((W > 0) & (W.T > 0)).sum() == 0            # positive edges carry one direction only
assert presplit("first sentence . second one ? third") == ["first sentence .", "second one ?", "third"]
assert nltk_sentences("Dr. Smith went home. He slept.") == ["Dr. Smith went home.", "He slept."]
assert split_reference("a b .\nc d .", "cnn_dm") == ["a b .", "c d ."]
assert split_reference("One here. Two there.", "tldr") == ["One here.", "Two there."]
assert split_reference("one here . two there .", "arxiv") == ["one here .", "two there ."]
print("self-contained compatibility checks pass")

# ------------------------------------------------------------------ against the team code
team_dir = os.environ.get("TEAM_RANKING") or os.path.join(os.path.dirname(HERE), "ranking")
if not os.path.exists(os.path.join(team_dir, "pagetrust.py")):
    print(f"team ranking/ not found at {team_dir}: skipped comparisons with the team code")
    sys.exit(0)


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(team_dir, file))
    mod = sys.modules[name] = importlib.util.module_from_spec(spec)   # dataclasses look it up
    spec.loader.exec_module(mod)
    return mod


# team modules import each other by top-level name; graph/ also has a pipeline.py, so swap
# the team's modules in only while loading them
saved = {k: sys.modules.get(k) for k in ("pipeline", "pagetrust", "preprocess", "selector", "porter",
                                         "rouge_lite", "team_run")}
sys.path.append(team_dir)
try:
    T_pt = load("pagetrust", "pagetrust.py")
    T_sel = load("selector", "selector.py")
    T_pre = load("preprocess", "preprocess.py")
    load("pipeline", "pipeline.py")
    T_run = load("team_run", "run.py")
finally:
    sys.path.remove(team_dir)
    for k, v in saved.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v

# 1. rankers: identical scores on bert_graph-shaped signed graphs, every grid point
rng = np.random.default_rng(0)
worst = 0.0
for _ in range(200):
    W = random_signed_graph(rng, int(rng.integers(2, 40)))
    worst = max(worst, np.abs(rankers.own_pagerank(W) - T_pt.pagerank(W)).max())
    for g in (0.0, 1.0, 2.0, np.inf):
        for m in (0.0, 0.5, 0.9, 1.0):
            worst = max(worst, np.abs(rankers.own_pagetrust(W, 0.85, g, m) - T_pt.pagetrust(W, 0.85, g, m)).max())
    assert np.allclose(rankers.in_degree(W), T_pt.degree_centrality(W))
assert worst < 1e-9, worst
print(f"own rankers match the team's (max |diff| {worst:.1e})")
if not os.environ.get("GRAPHSUM_OWN_RANKERS"):
    assert rankers.SOURCE == "ranking/pagetrust.py", rankers.SOURCE
    W = random_signed_graph(rng, 25)
    assert np.array_equal(rankers.pagetrust(W, 0.85, 1.0, 0.5), T_pt.pagetrust(W, 0.85, 1.0, 0.5))
    print("graph/ calls the team's pagetrust()")

# 2. budgets and selection
team_keys = {"cnndm": "cnn_dm"}
assert {team_keys.get(k, k): v for k, v in T_sel.BUDGETS.items()} == TEAM_BUDGETS
for _ in range(200):
    n = int(rng.integers(1, 30))
    sents = [" ".join(["w"] * int(rng.integers(1, 40))) for _ in range(n)]
    scores = rng.integers(0, 5, n).astype(float)               # integers -> plenty of ties
    ranking = np.argsort(-scores, kind="stable").tolist()
    for kind, size in set(TEAM_BUDGETS.values()):
        for ov in ("include", "skip", "truncate"):
            assert apply_budget(ranking, sents, Budget(kind, size, ov)) == T_sel.select(sents, scores, kind, size, ov)
print("budgets and selection match")

# 3. splitting
texts = ["Dr. Smith said the U.S. plan, which passed, will start soon. It costs $3.5 bn! Critics agreed; others didn't.",
         "The council approved the plan because roads were failing, and the mayor said it was overdue."]
for t in texts:
    assert nltk_sentences(t) == T_pre.split_sentences(t), t
    for s in nltk_sentences(t):
        assert rule_clauses(s) == T_pre.split_clauses(s), s
art = "we study graphs . results are good ? yes ."
assert presplit(art) == T_run.presplit(art)
for ref, ds in [("a b .\nc d .", "cnn_dm"), ("One. Two here.", "tldr"), ("x y . z w .", "arxiv")]:
    assert "\n".join(split_reference(ref, ds)) == T_run.ref_lines(ref, ds), (ref, ds)
print("splitting matches")

# 4. clause -> sentence aggregation
verts = [Vertex("u", s, k) for k, s in enumerate([0, 0, 1, 2, 2, 2])]
x = rng.random(6)
for agg in ("max", "mean", "sum"):
    assert np.allclose(sentence_scores(x, verts, agg), T_sel.sentence_scores(x, [0, 0, 1, 2, 2, 2], 3, agg))
print("aggregation matches\nall team compatibility checks pass")
