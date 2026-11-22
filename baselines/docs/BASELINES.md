# Baselines in detail

Every baseline scores the sentences of one document; the shared pipeline (`baselines/summarize.py`)
then builds the summary and computes ROUGE. Settings shared by all of them:

| Setting | Value |
|---|---|
| Sentence splitting | NLTK (`punkt_tab`) on raw text; WikiHow also splits at line breaks; arXiv/PubMed come pre-split |
| Summary length | CNN/DM 3 sentences · PubMed 203 words · arXiv 220 words · BillSum 184 words · WikiHow 55 words · TL;DR-17 1 sentence |
| Word budgets | Highest-scoring sentences are added until the budget is reached, then cut at exactly N words |
| Scoring | `rouge-score` F1, **no stemming**; ROUGE-L is summary-level (`rougeLsum`) |
| Tuning (LexPageRank, PacSum) | First 1,000 development documents, ROUGE-1 F1: validation split (CNN/DM, PubMed, arXiv), train split (BillSum), first 1,000 documents of the file (WikiHow, TL;DR-17; evaluation starts after them) |

Folder names below are the run folders under `results/all/<dataset>/` (main) or
`results/sensitivity/<dataset>/` (variants).

## Oracle (`oracle`) — upper bound

Greedy search (Nallapati et al. 2017) that uses the reference summary: repeatedly adds the sentence
that most increases the F1 of one ROUGE metric, stopping when no sentence improves it; output in
document order, cut to the dataset's summary length.

| Dataset | Maximizes | Why |
|---|---|---|
| CNN/DM | ROUGE-1 F1, 3 sentences | PacSum doesn't state its Oracle's metric; this was closest to its published numbers |
| PubMed, arXiv | ROUGE-2 F1 | HipoRank labels its Oracle "ROUGE-2, F1" |
| BillSum, WikiHow, TL;DR-17 | ROUGE-1 + ROUGE-2 F1 | No published Oracle; standard combination |

## Lead (`lead`)

First k sentences (CNN/DM 3, TL;DR 1) or first N words (word-budget datasets). No graph, no tuning.

## TextRank

| | Unnormalized (`textrank`) | Normalized (`textrank_cosine`) | Mihalcea 2004 (`textrank_mihalcea`) |
|---|---|---|---|
| Type | Aligned to PacSum | Variant | Paper-faithful |
| Preprocessing | gensim default filters (lowercase; strip tags, punctuation, whitespace, digits, stopwords; drop words < 3 chars; Porter stemming), as in PacSum's code | same | lowercase word tokens only |
| Edge weight | Σ tfᵢ(w)·tfⱼ(w)·idf(w)², idf = log(N − df + 0.5) − log(df + 0.5) over the article's sentences | cosine of the same tf·idf vectors | shared words ÷ (log\|Sᵢ\| + log\|Sⱼ\|); pairs of one-word sentences get no edge |
| Graph | sentences; weighted; undirected; edge if similarity > 0; no self-links | same | same |
| Ranking | weighted PageRank (Mihalcea Eq. 10), damping 0.85, start 1.0, tolerance 1e-4, ≤ 200 iterations | same | same |

## DegreeCentrality

Same graphs as TextRank (unnormalized → `degree_centrality`, normalized → `degree_centrality_cosine`);
each sentence is scored by the sum of its edge weights (PacSum's Eq. 1) instead of PageRank.

## LexPageRank (Erkan & Radev 2004)

| | Corpus idf (`lexpagerank`) | Per-article idf (`lexpagerank_docidf`) |
|---|---|---|
| Type | Paper-faithful | Variant |
| Preprocessing | lowercase word tokens; no stemming or stopword removal (idf down-weights common words) | same |
| Edge weight | cosine of tf·idf vectors, idf = log(N / n) over the first 20,000 training documents (unseen words: log N) | idf over the article's own sentences |
| Graph | edge only if cosine > threshold; unweighted; undirected; every sentence linked to itself | same |
| Ranking | PageRank, damping 0.85, start 1.0, tolerance 1e-4 | same |
| Tuned threshold ∈ {0.1, 0.2, 0.3} | CNN/DM 0.3, PubMed 0.2, arXiv 0.1, BillSum 0.2, WikiHow 0.1, TL;DR 0.1 | CNN/DM 0.3, all others 0.1 |

Verified against the paper's worked example (Figure 1 matrix → Figure 3 scores within 0.0006).
MEAD's position feature, reranker and length cutoff are not used. `--continuous` gives the weighted,
no-threshold variant; `--impl sumy` runs the off-the-shelf sumy LexRank for comparison.

## PacSum (Zheng & Lapata 2019), tf-idf version

| | Unnormalized (`pacsum_tfidf`) | Normalized (`pacsum_cosine`) |
|---|---|---|
| Type | Aligned to PacSum (reimplemented from the released code) | Variant |
| Preprocessing | gensim default filters | same |
| Edge weight | Σ tfᵢ(w)·tfⱼ(w)·idf(w)² (as in PacSum's code); self-similarity 1 | cosine of the same vectors |
| Graph | sentences; weighted; edges below t = min(E) + β·(max(E) − min(E)) removed and t subtracted from the rest | same |
| Direction / ranking | score(i) = λ1·(−Σ similarity to earlier sentences) + λ2·(Σ similarity to later sentences); ties random (fixed seed); documents with ≤ k sentences returned whole | same |
| Tuned (β, λ1, λ2), grid β, λ1 ∈ {0, 0.1, …, 1}, λ2 = 1 − λ1 | CNN/DM (0, 0.7, 0.3), PubMed (0, 0.7, 0.3), arXiv (0, 0.9, 0.1), BillSum (0, 0, 1), WikiHow (0.1, 0.7, 0.3), TL;DR (0, 0.1, 0.9) | CNN/DM (0, 0.6, 0.4), PubMed (0, 0.4, 0.6), arXiv (0, 0.6, 0.4), BillSum (0, 0.2, 0.8), WikiHow (0, 0.1, 0.9), TL;DR (0, 0, 1) |

Even λ1 = 0 keeps a lead bias (early sentences have more later neighbours); fully undirected scoring
would need λ1 = −λ2, outside the grid. The BERT version (`--rep bert`) is implemented but untested and
needs PacSum's fine-tuned checkpoint and a GPU.

## Alignment with published results

`python baselines/compare_published.py` rebuilds this table from your runs (`results/alignment.md`).

| Baseline | Compared with | Mean abs. difference |
|---|---|---|
| PacSum (unnormalized) | PacSum Table 2 (CNN/DM) | 0.41 |
| TextRank (unnormalized) | PacSum Table 2 (CNN/DM) | 0.55 |
| DegreeCentrality (unnormalized) | PacSum Table 2 (CNN/DM) | 0.55 |
| Lead | PacSum Table 2; HipoRank Tables 3–4 | 0.67 |
| LexPageRank (corpus idf) | HipoRank Tables 3–4 (LexRank, from Cohan et al. 2018) | 1.90 |
| Oracle | PacSum Table 2; HipoRank Tables 3–4 | 2.85 |

Known causes: the CNN/DM Lead gap (−1.9 ROUGE-1) comes from the dataset release (raw Kaggle text split
with NLTK vs PacSum's preprocessed files); the +1.3 ROUGE-2 offset shared by TextRank and
DegreeCentrality points to a measurement difference; LexPageRank's PubMed gap matches sumy's, while on
arXiv sumy also exceeds the published LexRank by 3.2, so that published number appears low; our Oracle
maximizes the scored metric directly, giving a stronger upper bound (ROUGE-2 +6.5 / +8.1 on
PubMed / arXiv).
