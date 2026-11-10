"""Ranker: PageRank and PageTrust on a signed, directed graph (numpy, vectorized).

Graph convention (the interface with the Directioner):
    W is an (n, n) float array.
    W[i, j] > 0  : positive edge i -> j  (i "endorses" / points to j)
    W[i, j] < 0  : negative edge i -> j  (i "distrusts" j: BERT NSP rejected both orders)
    W[i, j] == 0 : no edge.  The diagonal is ignored.

PageTrust = Algorithm 4.1 of de Kerchove & Van Dooren, "The PageTrust algorithm: How
to rank web pages when negative links are allowed?", SIAM SDM 2008 (eqs. 3.3-3.6).
  Infinitely many random walkers move exactly as in PageRank (follow a positive
  out-link with prob. alpha, zap to a uniform random vertex with prob. 1 - alpha).
  Each walker carries a list of distrusted vertices.
      x[i]     : proportion of walkers in i who trust i            (the score)
      P[i, k]  : proportion of walkers in i who distrust k
  Rules (paper Sec. 3.2):
   1. walkers keep their opinions when they move (P~ = T P, eq. 3.4);
   2. a walker in i adopts i's negative links (P[i, k] = 1 if (i, k) in L-, eq. 3.5);
   3. a walker who distrusts the vertex it reaches LEAVES the graph, so vertex i keeps
      only (1 - P~[i, i])^beta of its incoming walkers (eqs. 3.3, 3.6). Walkers do not
      steer around distrusted vertices: the penalty is the leak at the vertex itself.
  One iteration (paper order):
      x'_i   = (1 - P~_ii)^beta * (G x)_i,  then x' /= sum(x')      [G = Google matrix]
      T_ij   = (alpha S_ji x_j + M (1-alpha)/n x_j) / (alpha sum_k S_ki x_k + (1-alpha)/n)
               (share of the walkers now in i that came from j; with M < 1 walkers who
                zapped arrive with no opinions -- M = memory after zapping, Sec. 3.4)
      P~     = T P
      P      = P~, then P_ij = 1 for (i, j) in L-, and P_ii = 0     (eq. 3.5)
  initial x = 1/n, P = P~ = A- (the negative adjacency matrix); stop when max|x' - x| < eps.

  Parameter names: the paper's degree of conviction is beta; in our proposal it is
  gamma (beta is PacSum's threshold there), so the argument here is `gamma`.

  Verified against the paper (tests): Tables 1 and 2 (Figure 2 graph, alpha = 0.9) to
  the paper's 2 decimals; no negative edges, or conviction 0 -> exactly PageRank.

  Our additions (not in the paper), each optional:
   * weighted positive edges: S_ji = W+_ji / weighted out-degree of j (paper: 0/1 links)
   * distrust="weighted": rule 2 adopts |W_ij| / max|W| instead of 1, combined as
     P_ij = 1 - (1 - P~_ij)(1 - D_ij); "binary" (default) is the paper's rule
   * dangling vertices (no positive out-link) zap uniformly (paper assumes none, Rem. 2.1)
   * convergence also waits for diag(P~) to settle: on a symmetric graph x can stand
     still on the first step while distrust is still spreading, and the paper's
     x-only test would stop there with the negative links never applied.

  Memory M = 1 (the paper's default) drives every vertex with a negative in-link
  towards x = 0 (paper Table 1: node 4 gets 0 for M = 1) and converges slowly,
  because walkers never forget. On sentence graphs that turns any weak negative edge
  into a hard veto. Tune M over {0, 0.5, 0.9}; the paper allows M in [0, 1] (Sec. 3.4).

Cost: one iteration is an (n x n) @ (n x n) product, O(n^3). Fine for sentences
(n ~ 30 CNN/DM, ~200-300 arXiv); clause graphs (~4x n) are ~64x slower per iteration.
"""

from __future__ import annotations

import numpy as np


def _row_stochastic(A: np.ndarray) -> np.ndarray:
    """Row-normalise; all-zero (dangling) rows become uniform."""
    n = A.shape[0]
    s = A.sum(axis=1, keepdims=True)
    out = np.where(s > 0, A / np.where(s > 0, s, 1.0), 1.0 / n)
    return out


def split_signed(W: np.ndarray, distrust: str = "binary") -> tuple[np.ndarray, np.ndarray]:
    """Return (positive weights, distrust matrix in [0, 1]) with zero diagonals."""
    W = np.array(W, dtype=float)
    np.fill_diagonal(W, 0.0)
    pos = np.clip(W, 0.0, None)
    neg = np.clip(-W, 0.0, None)
    if distrust == "binary":
        D = (neg > 0).astype(float)
    elif distrust == "weighted":
        m = neg.max()
        D = neg / m if m > 0 else neg
    else:
        raise ValueError(distrust)
    return pos, D


def pagerank(W: np.ndarray, alpha: float = 0.85, eps: float = 1e-6, max_iter: int = 100,
             return_iters: bool = False):
    """Weighted PageRank on the positive part of W; scores sum to 1."""
    pos, _ = split_signed(W)
    n = pos.shape[0]
    if n == 0:
        return (np.zeros(0), 0) if return_iters else np.zeros(0)
    G = alpha * _row_stochastic(pos) + (1 - alpha) / n
    x = np.full(n, 1.0 / n)
    for it in range(1, max_iter + 1):
        x_new = x @ G
        done = np.abs(x_new - x).max() < eps
        x = x_new
        if done:
            break
    return (x, it) if return_iters else x


def pagetrust(W: np.ndarray, alpha: float = 0.85, gamma: float = 1.0, memory: float = 1.0,
              eps: float = 1e-6, max_iter: int = 100, distrust: str = "binary",
              return_state: bool = False):
    """PageTrust scores (sum to 1) for signed graph W -- paper Algorithm 4.1.

    gamma   : degree of conviction (paper's beta), >= 0; np.inf = one distrusting
              walker is enough to empty the vertex.
    memory  : M in [0, 1], probability a walker keeps its opinions after zapping.
    distrust: "binary" (paper) or "weighted" (see module docstring).
    return_state: also return (P, iterations).
    """
    pos, D = split_signed(W, distrust)
    n = pos.shape[0]
    if n == 0:
        z = np.zeros(0)
        return (z, np.zeros((0, 0)), 0) if return_state else z
    S = _row_stochastic(pos)                 # S[j, i] = A+_ji / d_j
    G = alpha * S + (1.0 - alpha) / n         # x @ G == paper's G x
    neg = D > 0
    x = np.full(n, 1.0 / n)
    P = D.copy()                              # P(0) = P~(0) = A-
    Pt = D.copy()
    it = 0
    for it in range(1, max_iter + 1):
        # 1) scores: walkers who distrust the vertex they reach leave (eq. 3.6)
        stay = np.diag(Pt).clip(0.0, 1.0)
        if np.isinf(gamma):
            keep = (stay <= 1e-12).astype(float)
        else:
            keep = np.power(1.0 - stay, gamma)
        y = keep * (x @ G)
        x_new = y / y.sum() if y.sum() > 0 else np.full(n, 1.0 / n)
        # 2) opinions travel with the walkers (eq. 3.4), T built from x(t)
        inflow = alpha * (x @ S) + (1.0 - alpha) / n * x.sum()
        T = (alpha * S.T + memory * (1.0 - alpha) / n) * x[None, :] / inflow[:, None]
        Pt_new = T @ P
        # 3) adopt the vertex's negative links, forget distrust of the vertex itself (eq. 3.5)
        P = Pt_new.copy()
        if distrust == "binary":
            P[neg] = 1.0
        else:
            P = 1.0 - (1.0 - P) * (1.0 - D)
        np.fill_diagonal(P, 0.0)
        done = max(np.abs(x_new - x).max(), np.abs(np.diag(Pt_new) - np.diag(Pt)).max()) < eps
        x, Pt = x_new, Pt_new
        if done:
            break
    return (x, P, it) if return_state else x


# ------------------------------------------------------------------ graph builders
def threshold_edges(sim: np.ndarray, beta: float) -> np.ndarray:
    """PacSum thresholding: subtract min + beta * (max - min) (over off-diagonal entries)
    and keep only the positive remainder. Returns a symmetric, non-negative matrix."""
    sim = np.array(sim, dtype=float)
    n = sim.shape[0]
    if n < 2:
        return np.zeros_like(sim)
    off = ~np.eye(n, dtype=bool)
    lo, hi = sim[off].min(), sim[off].max()
    e = sim - (lo + beta * (hi - lo))
    e[~off] = 0.0
    return np.clip(e, 0.0, None)


def pacsum_signed_graph(edges: np.ndarray, lambda_fwd: float, lambda_bwd: float) -> np.ndarray:
    """PacSum's position-directed graph as a signed W for PageTrust.

    PacSum scores vertex i as  lambda_fwd * sum_{j>i} e_ij + lambda_bwd * sum_{j<i} e_ij
    (pass PacSum's lambda1/lambda2 with whatever sign convention its code uses, so that
    this formula is exactly its centrality). As a graph: an edge j -> i carries the
    weight j contributes to i, so  W.sum(axis=0) == PacSum's degree centrality.
    Negative coefficients give negative (distrust) edges -- the signed graph PageTrust
    needs, available before the BERT Directioner exists.
    """
    e = np.array(edges, dtype=float)
    n = e.shape[0]
    j, i = np.indices((n, n))           # W[j, i]: from j to i
    W = np.where(j > i, lambda_fwd * e.T, np.where(j < i, lambda_bwd * e.T, 0.0))
    return W


def degree_centrality(W: np.ndarray) -> np.ndarray:
    """Incoming signed edge sum (PacSum's ranker / the DegreeCentrality baseline)."""
    W = np.array(W, dtype=float)
    np.fill_diagonal(W, 0.0)
    return W.sum(axis=0)
