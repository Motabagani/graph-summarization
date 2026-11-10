# Graph-based summarization: learned directionality and PageTrust

This is the team repo for our NLP project. We modify PacSum (Zheng & Lapata, 2019) in two places:
1. **Learned edge direction:** BERT's next-sentence-prediction head decides edge direction, replacing PacSum's sentence-position rule.
2. **A new ranker:** PageTrust (de Kerchove & Van Dooren, 2008) replaces degree centrality, so the graph's negative edges are used instead of ignored.

## Layout
Each person has their own folder, so nobody's files clash.

| Folder | Owner | What's in it |
|---|---|---|
| `ranking/` | Hashim Motabagani | Preprocessor, PageTrust ranker, sentence selector, local sweep tools. See [ranking/README.md](ranking/README.md). |
| `baselines/` | Finley Torrens | Baselines (Lead, Oracle, TextRank, DegreeCentrality, LexPageRank, PacSum) and evaluation. *To be added.* |
| `graph/` | Alfardil Alam | BERT Grapher and NSP Directioner. *To be added.* |

## Interface between folders
The Directioner hands the ranker one numpy array `W` of shape (n, n) per document:
- `W[i, j] > 0`: positive edge from sentence i to sentence j, with that weight.
- `W[i, j] < 0`: negative edge, meaning BERT rejected both sentence orders.
- `0`: no edge. The diagonal is ignored.

The ranker is called as `ranking/pagetrust.py: pagetrust(W, alpha, gamma, memory)`.

## Setup
```bash
python3 -m venv .venv
.venv/bin/pip install -r ranking/requirements.txt
.venv/bin/python -c "import nltk; nltk.download('punkt_tab')"
.venv/bin/python -m unittest discover -s ranking/tests -v
```

The datasets aren't in the repo. The full ones are on the HPC, in the team scratch folder (ask Finley for the path). To build the 100-document local samples, run `ranking/tools/fetch_samples.py`, or `ranking/tools/convert_raw.py` on files you downloaded by hand.
