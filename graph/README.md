# GraphSum: BERT sentence graphs ranked with PageTrust

This code summarizes a document by picking its most important sentences. It turns the document into a graph of sentences and ranks them:

1. Sentences that say similar things get linked. Similarity comes from a frozen BERT model (used as is, not fine-tuned), and weak links are dropped with PacSum's threshold β.
2. BERT's next-sentence prediction (NSP) sets the direction of each link. The sentence that builds on another one points to it. If NSP thinks neither order makes sense, the two sentences get a negative link instead.
3. PageTrust, a version of PageRank that can use negative links, scores every sentence. The highest-scoring sentences form the summary.

This code is the `graph/` folder of the team project [Motabagani/graph-summarization](https://github.com/Motabagani/graph-summarization). The team's `ranking/` folder has the preprocessor, PageTrust and the sentence selector, and this code builds the graph they run on. Run the commands below from inside `graph/`.

## Setup and running

```bash
pip install -r requirements.txt && python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt_tab')"
python evaluate.py --datasets cnn_dm --limit-dev 200 --limit-test 1000   # or ./run_eval.sh for everything
```

Each run tunes every system on dev documents, scores it on test documents and writes three files to `results/` (or to `--out`). `results.md` has the score tables. `results.json` has the same numbers plus the dev score of every setting tried, and both record where in the document the Oracle's sentences fall. `per_doc.jsonl` lists the sentences each system picked in each document, with its ROUGE score.

`run_eval.sh` ends by running `scripts/merge_tables.py`, which turns the run folders into the paper's main-results and ablation tables. `scripts/analysis.py` does the error analysis: it dumps the documents where our system loses most to a simpler one, draws example graphs and plots where the chosen sentences sit in the document.

To check that everything works without downloading data or a real model:

```bash
python tests/make_tiny_bert.py /tmp/tinybert
python evaluate.py --synthetic --model /tmp/tinybert --limit-dev 20 --limit-test 30 --clauses
```

## Files

| File | What it does |
|---|---|
| `bert_graph.py` | Builds the graph: sentences or clauses as nodes, BERT similarity, the β threshold, NSP direction and negative links |
| `splitting.py` | Splits documents and reference summaries into sentences and clauses, the same way the team's `ranking/` code does |
| `rankers.py` | PageTrust (de Kerchove & Van Dooren 2008, Algorithm 4.1), PageRank and degree. Next to the team's `ranking/` folder it calls `ranking/pagetrust.py`; on its own it uses a copy that gives the same scores |
| `baselines.py` | Lead, Random, Oracle, Degree (TF-IDF), TextRank, LexPageRank and PacSum (TF-IDF), on stemmed text |
| `data.py` | Loads CNN/DM, arXiv, PubMed, tldr-17, WikiHow and BillSum, with an 80/10/10 split for datasets that have no dev/test split |
| `pipeline.py` | Puts every system behind one interface; `Summarizer` summarizes a single document |
| `evaluate.py` | Tunes on dev, then reports test ROUGE-1/2/L F1, ablations and per-document logs |

## Working with the team repo

The graph for each document is a matrix `W` in the team's format. `W[i, j] > 0` is a link from sentence i to sentence j, `W[i, j] < 0` is a negative link (NSP rejected both orders), 0 means no link, and the diagonal is ignored. Pass it to `pagetrust(W, alpha, gamma, memory)` in `ranking/pagetrust.py`. To build the graph on the team preprocessor's sentences or clauses, use `vertices_from_units(doc.units, doc.unit_sentence)`.

When `ranking/` sits next to this folder (or `$TEAM_RANKING` points to it), PageRank and PageTrust come from `ranking/pagetrust.py`, so the whole team pipeline uses one implementation. The copy in `rankers.py` gives the same scores to within 1e-15, but two sentences with exactly equal scores can then come out in either order, so only one is used at a time. Set `GRAPHSUM_OWN_RANKERS=1` to use the copy anyway. Each run's `results.md` says which one it used.

The defaults match `ranking/`. NLTK punkt splits sentences, arXiv and PubMed are split at their `" ."` tokens, and reference summaries are split like `ref_lines` in `ranking/run.py`. Clauses come from the team's rule-based splitter, and a sentence's score is the highest score among its clauses. Summary lengths and tie-breaking are the same too, and `cnndm` works as a name for `cnn_dm`. To use this repo's original setup (spaCy splitting and the average clause score), add `--splitter spacy --clause-splitter spacy --aggregate mean`.

`tests/test_team_compat.py` checks the rankers, summary lengths, sentence selection, splitting and clause scoring against the team's code. It looks for `ranking/` in `../ranking` or in `$TEAM_RANKING`.

Preview runs made before this alignment (`results/` in [alfardil/GraphSum_NLP](https://github.com/alfardil/GraphSum_NLP)) are out of date. They used spaCy splitting and an older PageTrust stopping rule that gave different results at γ = ∞, so re-run before comparing with the team's numbers.

## Settings

Link direction: if NSP finds "A then B" more plausible than "B then A", then B builds on A, so the link goes from B to A. A pair gets negative links both ways when NSP rejects both orders (its isNext log-odds are below 0 both ways). A link's confidence is tanh(|Δ log-odds| / 2). The code uses log-odds because NSP probabilities sit close to 1 for almost any two sentences from the same document.

α = 0.85, ε = 1e-6 and the 100-iteration cap are fixed. These are tuned on dev:

- β, the similarity threshold: {0, 0.3, 0.6}
- γ, how much distrust lowers a sentence's score: {0, 1, 2, ∞}
- M, how much a PageTrust walker remembers after a random jump: {0, 0.5, 0.9, 1}, the proposal's {0, 1} plus the team's {0, 0.5, 0.9}
- λ, the extra weight on links between clauses of the same sentence: {0, 0.5, 1}
- PacSum's λ1: {−2, −1, −0.5, 0}, with λ2 = 1

Summary lengths are the team's fixed budgets, shared with the baselines (`TEAM_BUDGETS` in `evaluate.py`): 3 sentences for CNN/DM, 1 sentence for tldr, and 203, 220, 184 and 55 words for PubMed, arXiv, BillSum and WikiHow. For word budgets, sentences are added best first until the budget is reached, and `--overflow` decides what happens to the sentence that crosses it (default `include`). `--length median` uses the median number of sentences in the dev reference summaries instead (always the case with `--synthetic`), and `--n K` forces K sentences.

Scores come from Google's `rouge-score` (with stemming, ROUGE-Lsum). They are close to, but not the same as, the Perl ROUGE-1.5.5 numbers that PacSum and HipoRank report.

Each results table is followed by graph diagnostics. A high "all-tied docs" share means NSP left too few positive links, so every sentence scored the same and the system fell back to picking the first sentences (Lead).

## Cost

NSP runs BERT twice for each sentence pair that passes the threshold. β = 0 keeps every pair, so a 200-sentence arXiv paper needs about 40,000 BERT passes. A GPU handles CNN/DM and tldr. For long documents, add `--max-sentences` and `--betas 0.3 0.6`, as `run_eval.sh` does. On a CPU, only small samples are practical.
