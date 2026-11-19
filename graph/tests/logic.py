import sys, numpy as np
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))))
from bert_graph import *

class Stub:
    """vectors chosen so all pairs pass beta=0; NSP log-odds hand-set."""
    def __init__(self, table): self.t = table
    def embed(self, texts): return np.eye(len(texts)) + 1.0
    def nsp_logodds(self, A, B): return np.array([self.t[(a, b)] for a, b in zip(A, B)])

V = [Vertex(t, i, i) for i, t in enumerate("abc")]
t = {("a","b"): 3.0, ("b","a"): -1.0,   # a-then-b plausible -> b builds on a -> edge b->a
     ("a","c"): -2.0, ("c","a"): -0.5,  # both rejected -> negative both ways
     ("b","c"): -1.0, ("c","b"): 2.0}   # c-then-b plausible -> b builds on c -> edge b->c
S = DocumentScores(V, Stub(t))
G = build_graph(S, GraphConfig(beta=0.0))
W = G.W
assert W[1,0] > 0 and W[0,1] == 0, W
assert W[0,2] < 0 and W[2,0] < 0
assert W[1,2] > 0 and W[2,1] == 0
assert np.isclose(W[1,0], np.tanh(4/2))      # norm sim = 1 here; confidence = tanh(|3-(-1)|/2)
assert (build_graph(S, GraphConfig(beta=0.0, allow_negative=False)).W >= 0).all()
# same-sentence clauses: undirected + lambda, never sent to NSP
V2 = [Vertex("x",0,0), Vertex("y",0,1), Vertex("z",1,2)]
S2 = DocumentScores(V2, Stub({("x","z"):1.,("z","x"):0.,("y","z"):1.,("z","y"):0.}))
G2 = build_graph(S2, GraphConfig(beta=0.0, lam=0.5))
assert G2.W[0,1] == G2.W[1,0] == 1.5 and ("x","y") not in [(V2[a].text,V2[b].text) for a,b in S2._nsp]
assert np.allclose(sentence_scores(np.array([1.,3.,5.]), V2, "mean"), [2.,5.])
assert np.allclose(sentence_scores(np.array([1.,3.,5.]), V2), [3.,5.])          # team default: max
assert np.allclose(sentence_scores(np.array([1.,3.,5.]), V2, "sum"), [4.,5.])
# string sentences, rule clauses (team Preprocessor) keep their sentence ids
VS = make_vertices(["The council approved the plan, and the mayor said it was overdue.", "Short one."],
                   clauses=True, clause_splitter="rules")
assert [v.sentence_id for v in VS] == [0, 0, 1] and VS[1].text.startswith("and the mayor"), VS
assert [v.text for v in vertices_from_units(["a b", "c"], [0, 0])] == ["a b", "c"]
print("logic tests pass:", G.info)
