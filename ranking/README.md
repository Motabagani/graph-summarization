# Hashim's part: Preprocessor, PageTrust ranker, Sentence selector

| File | What it does |
|---|---|
| `preprocess.py` | Raw text, or a pre-split list for arXiv/PubMed, becomes sentences or clauses. Each unit keeps a raw copy (for BERT) and a gensim-style cleaned copy (for tf-idf). |
| `porter.py` | Porter stemmer reused from HW4, so stemming needs no nltk. |
| `pagetrust.py` | Vectorized PageRank, PageTrust and degree centrality, plus PacSum thresholding and a PacSum signed-graph builder. |
| `selector.py` | Turns vertex scores into a summary using the baselines' per-dataset budgets, and combines clause scores into sentence scores. |
| `pipeline.py` | Glue code, using a stand-in graph (tf-idf weights with PacSum position direction) until the BERT Grapher and Directioner exist. |
| `run.py` | Runs one ranker over a JSONL dataset sample and prints mean ROUGE. |
| `rouge_lite.py` | ROUGE-1 / -2 / -Lsum F1 with stemming. It matches `rouge-score` to within 0.05. Use it for local comparisons only. |
| `tools/sweep.py` | Runs Lead, PacSum degree, PageRank and the PageTrust grid on every sample, and writes a CSV to `results/`. |
| `tools/fetch_samples.py`, `tools/convert_raw.py` | Build the 100-document samples in `data_sample/`, from the Hugging Face API or from files you downloaded by hand. |
| `results/` | Sample sweep results and their summary. These are for debugging and aren't reportable. |
| `tests/test_all.py` | 22 tests, including the paper's Figure 2 example. From the repo root, run `.venv/bin/python -m unittest discover -s ranking/tests -v`. |

## Interface with Alfardil (Directioner → Ranker)
The Directioner outputs one `numpy` array `W` of shape (n, n) per document:
- `W[i, j] > 0`: positive edge i → j. The value is the weight, for example similarity × NSP confidence.
- `W[i, j] < 0`: negative edge, meaning NSP rejected both orders.
- `0`: no edge. The diagonal is ignored.

Pass it to `pagetrust(W, alpha, gamma, memory)`.

## Things to sync with Finley's baseline code before running on HPC
1. **Stop list.** We use gensim's `STOPWORDS` when gensim is installed. Use the same list as the TextRank baseline.
2. **Splitter.** Use NLTK punkt on HPC. The regex splitter is only a local fallback.
3. **Word-budget overflow rule** (`include` / `truncate` / `skip`), and the arXiv budget, which is 200 or 220 depending on which note you read.
4. **PacSum λ sign convention.** Fixed: λ_fwd = 1 on edges to later sentences, and λ_bwd ≤ 0 on edges to earlier ones. The first check on HPC should be that `degree_centrality(pacsum_signed_graph(...))` reproduces Finley's PacSum tf-idf score of 38.67 on CNN/DM.
5. **ROUGE-L.** `rouge_lite` reports ROUGE-Lsum. Confirm Finley uses the same on every dataset.

## PageTrust vs. the paper
`pagetrust()` follows Algorithm 4.1 of de Kerchove & Van Dooren (2008). Walkers who distrust the vertex they reach leave the graph, so the score update is x_i ← (1 − P̃_ii)^γ (Gx)_i, followed by renormalizing. Blacklists travel with the walkers (P̃ = T P), and P_ij is reset to 1 for negative links and P_ii to 0.

It's checked against the paper's Figure 2 graph:
- PageRank at α = 0.9 matches the paper exactly (.18 / .18 / .27 / .18 / .18).
- The PageTrust tables (Tables 1 and 2, α = 0.9) match in ranking order and in the zeros for node 4, and every value is within 0.05.
- An exact match isn't possible: the paper's own α = 0.5 PageRank column isn't what standard PageRank gives at 0.5. It matches α ≈ 0.8.

Additions that aren't in the paper:
- weighted positive edges
- an optional weighted distrust (`distrust="weighted"`; the default `"binary"` is the paper's rule)
- uniform zapping from dangling vertices
- a convergence check that also waits for the blacklists to settle

## Open items
- **M = 1 empties distrusted sentences.** The paper shows the same thing (Table 1: node 4 → 0). On our 40-sentence stand-in graph, distrusted sentences keep 76–93% of their PageRank score at M = 0, 25–72% at M = 0.9, and fall to 0% at M = 1, which takes about 2,000 iterations. Tune M over {0, 0.5, 0.9}; the paper allows any M in [0, 1].
- Tuning grid: β ∈ {0, .3, .6}, γ ∈ {0, 1, 2, ∞}, M ∈ {0, .5, .9}, α = 0.85, on about 1,000 dev documents.
- Timing with M = 0: about 0.03 s for a 250-vertex graph and about 1 s for 1,000 vertices (clause graphs on arXiv), roughly 45 iterations.
