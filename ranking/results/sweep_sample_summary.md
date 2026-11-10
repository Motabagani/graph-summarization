# Sample sweep, 100 docs per dataset

**Debugging only.** The best settings were picked on these same test documents, so the "best" rows are optimistic. Differences under about 1 ROUGE point are within noise on 100 docs. Do not report these numbers.

Setup:
- Data: Hugging Face samples (`data_sample/`), split with NLTK. All 6 are spread across their split (WikiHow and TL;DR across the last part of their data, which has no test split). BillSum, TL;DR and WikiHow came from hand-downloaded files via `tools/convert_raw.py`; `data_raw/wikihowAll.csv` is the full WikiHow file. arXiv and PubMed were cut at their tokenized " ." sentence ends.
- Scoring: `rouge_lite`, with R-L = ROUGE-Lsum. It matches `rouge-score` to within 0.05.
- Full grid: `results/sweep_sample.csv`. Run it with `python3 tools/sweep.py`.

Scores are R-1 / R-2 / R-L. "Finley" is his full-test score for the same baseline, for scale.

| Dataset | Lead | Finley Lead | PacSum, best setting | Finley PacSum tf-idf | PageRank, best | PageTrust, best |
|---|---|---|---|---|---|---|
| CNN/DM | 40.9 / 17.4 / 37.0 | 38.7 / 17.0 / 35.3 | 41.4 / 17.9 / 37.4 | 38.7 / 16.9 / 35.2 | 40.6 / 16.9 / 36.7 | 41.4 / 17.4 / 37.0 |
| PubMed | 36.5 / 11.6 / 33.4 | 35.2 / 12.1 / 31.8 | 42.6 / 16.3 / 38.4 | 36.4 / 12.8 / 32.1 | 39.9 / 14.5 / 36.2 | 40.2 / 14.7 / 36.5 |
| arXiv | 36.9 / 10.8 / 33.2 | 33.7 / 8.9 / 29.8 | 41.5 / 15.0 / 37.0 | 34.3 / 9.4 / 30.1 | 38.4 / 12.2 / 34.5 | 38.2 / 11.9 / 34.1 |
| BillSum | 40.3 / 17.6 / 36.2 | 38.9 / 17.9 / 31.1 | 43.4 / 20.4 / 39.1 | 39.4 / 17.3 / 29.8 | 43.6 / 20.3 / 39.0 | 43.7 / 20.2 / 39.2 |
| WikiHow | 24.4 / 5.5 / 22.3 | 22.6 / 5.0 / 20.9 | 26.7 / 6.7 / 24.3 | 23.5 / 5.3 / 21.7 | 26.2 / 6.4 / 24.0 | 26.1 / 6.4 / 23.8 |
| TL;DR | 10.9 / 2.0 / 8.8 | 12.3 / 2.3 / 9.9 | 13.2 / 2.5 / 10.5 | 14.8 / 2.5 / 11.2 | 12.2 / 2.0 / 9.8 | 12.2 / 2.0 / 9.8 |

Change in average ROUGE for PageTrust against PageRank on the same graph, averaged over all datasets and graphs:

| γ | M = 0 | M = 0.5 | M = 0.9 |
|---|---|---|---|
| 1 | 0.0 | −0.2 | −0.8 |
| 2 | −0.3 | −0.7 | −1.2 |
| ∞ | −1.2 | −1.1 | −1.1 |

## What it says
1. **PageTrust doesn't help on the stand-in graph.** At best (γ = 1, M = 0) it ties PageRank. Stronger conviction or more memory hurts. On this graph the negative edges come from PacSum's position rule: "this sentence resembles an earlier one". That marks a repeat, not a disagreement, so penalizing it removes good sentences. Whether PageTrust helps depends on Alfardil's NSP-based negative edges meaning something different.
2. **PacSum's degree centrality beats PageRank on 4 of 6 datasets.** That's the same finding as the PacSum paper. Its best settings are β = 0 and λ_bwd = −1 or −2, close to PacSum's own tuned values.
3. **Our numbers run higher than Finley's:**
   - Partly that's the 100-doc samples, the Hugging Face data versions, and picking the best setting on test data.
   - On BillSum and WikiHow our R-L is 5–7 points above his for Lead. That looks like he used plain ROUGE-L rather than ROUGE-Lsum for those datasets. On CNN/DM, his 35.3 only fits Lsum. Ask Finley which ROUGE-L he used.
