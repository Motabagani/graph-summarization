# Team pipeline: Preprocessor → Grapher → Ranker

One pipeline from raw dataset files to ROUGE scores, built from the team's three folders. It
doesn't copy or change anyone's method; it calls them:

| Stage | Program | Uses | Runs on |
|---|---|---|---|
| 1. Preprocessor | `preprocessor.py` | `baselines/` (reading files, choosing documents, sentence splitting, references) + Hashim's `ranking/preprocess.py` (clauses, cleaned tokens) | CPU |
| 2. Grapher | `grapher.py` | Alfardil's `graph/bert_graph.py` (frozen BERT similarity, NSP in both orders) | GPU |
| 3. Ranker | `ranker.py` | Alfardil's `build_graph` (threshold, direction, negative edges) + Hashim's `ranking/pagetrust.py` and `selector.py` + the baselines' summary and ROUGE rules | CPU |

Everything is evaluated exactly like the baselines: same test documents, same development
documents for tuning, same summary lengths (cut at N words), same ROUGE (`rouge-score`, no
stemming). The results land in `baselines/results/ours/`, so they appear as extra rows in
`baselines/results/summary.md`.

## How a document flows through

**1. Preprocessor.** Every dataset comes out in one format, whatever its source (CSV, Parquet,
JSON lines; text as a string or a list; references with `<S>` tags, newlines or plain text):

```json
{"id": "test-17", "dataset": "cnn_dailymail", "part": "test",
 "sentences":     ["The council approved the budget.", "Critics say it ignores roads, which are failing.", "..."],
 "units":         ["The council approved the budget.", "Critics say it ignores roads,", "which are failing.", "..."],
 "unit_sentence": [0, 1, 1, ...],
 "unit_tokens":   [["council", "approv", "budget"], ["critic", "ignor", "road"], ...],
 "reference":     ["The council passed its budget.", "..."]}
```
`units` are the graph's vertices: the sentences themselves, or their clauses with `--unit clause`
(Hashim's rule-based splitter; `unit_sentence` remembers which sentence each clause came from).

**2. Grapher.** For each document, frozen BERT embeds every unit (mean pooling) and scores every
pair: `sim[i, j]` = dot product of the two vectors. For each pair from different sentences that
passes PacSum's threshold at `--beta-min`, the NSP head scores both orders, "j follows i" and
"i follows j", as log-odds. Both matrices are saved to disk. This is the expensive part (two BERT
passes per pair), so it runs once, on a GPU.

**3. Ranker.** For each setting being tried, it rebuilds the graph from the saved scores with
Alfardil's `build_graph`, without rerunning BERT:
- pairs below the threshold `min + β(max − min)` get no edge; kept pairs are weighted by their similarity, rescaled to [0, 1];
- if "i then j" is the more plausible order, j builds on i, so the edge goes **j → i** (the later unit endorses the one it refers back to), weighted by similarity × NSP confidence;
- if NSP rejects both orders, the pair gets a **negative** edge both ways;
- clauses of the same sentence are linked to each other (plus λ), never negatively.

Then Hashim's `pagetrust(W, α, γ, M)` scores the units: a PageRank random walk in which walkers
pick up distrust from negative edges and leave vertices they distrust (γ = how strongly, M = how
much they remember after a random jump). Clause scores become sentence scores (max over clauses),
and the summary is built like every baseline's: best sentences first until the length budget,
cut at exactly N words.

Settings (β, γ, M, λ) are chosen on the 1,000 development documents by ROUGE-1, then the best
setting runs once on the test documents.

## Quick pilot (no tuning, no baseline re-runs)

For a rough idea of how the pipeline does on every dataset:

```bash
source baselines/config.sh
bash pipeline/slurm/submit_pilot.sh            # first 300 test documents of each dataset
cat work/pilot/pilot_summary.md
```

One GPU job per dataset runs the documents through all three stages with fixed settings
(β = 0.3, γ = 1, M = 0, sentence units), then five ranker variants on the same graphs:
NSP + PageTrust (the method), without negative edges, NSP + PageRank, PacSum-style position
direction, and no direction. `compare_pilot.py` then puts them next to every baseline **on the same
documents**, reading the baselines' per-document ROUGE from their saved `predictions.jsonl`, so no
baseline is re-run. The scores are a sample of 300 documents with untuned settings: good for
direction, not for the paper.

## Running it

From the repository root, on the HPC, after `source baselines/config.sh`:

```bash
bash pipeline/slurm/submit.sh cnn_dailymail                 # preprocess -> grapher (GPU) -> ranker, chained
SHARDS=8 bash pipeline/slurm/submit.sh pubmed               # test graphs split over 8 GPU jobs
TEST_LIMIT=10000 bash pipeline/slurm/submit.sh tldr17       # sample a huge test set (see "Cost")
UNIT=clause RANKER_ARGS="--lams 0 0.5 1" bash pipeline/slurm/submit.sh cnn_dailymail
GPU_SBATCH="--partition=<gpu partition>" bash pipeline/slurm/submit.sh arxiv
python baselines/baselines/collect_results.py baselines/results      # table including "Ours" rows
```

Or stage by stage (any machine; the grapher wants a GPU):

```bash
cd pipeline
python preprocessor.py --dataset pubmed --part dev  --data-root $DATA_ROOT --out ../work/pubmed/dev.jsonl
python preprocessor.py --dataset pubmed --part test --data-root $DATA_ROOT --out ../work/pubmed/test.jsonl
python grapher.py ../work/pubmed/dev.jsonl  --out ../work/pubmed/dev.bert  --beta-min 0.3
python grapher.py ../work/pubmed/test.jsonl --out ../work/pubmed/test.bert --beta-min 0.3
python ranker.py --dataset pubmed \
    --dev-docs ../work/pubmed/dev.jsonl --dev-graphs ../work/pubmed/dev.bert \
    --test-docs ../work/pubmed/test.jsonl --test-graphs ../work/pubmed/test.bert \
    --out ../baselines/results/ours/pubmed/nsp_pagetrust
```

Ablations are ranker options on the same saved graphs (no new BERT run): `--ranker pagerank`,
`--ranker degree`, `--direction position` (PacSum-style, tunes λ1), `--direction none`,
`--no-negative`. `--encoder tfidf` in the grapher gives Hashim's tf-idf stand-in graph on CPU
(use with `--direction position` or `none`).

**Sanity check:** `python ranker.py --dataset <d> --system lead --test-docs ... --out ...` must give
exactly the baselines' Lead scores; `tests/test_pipeline.py` checks this, plus that graphs rebuilt
from the saved scores equal `bert_graph.py`'s own (`cd pipeline && python -m unittest discover -s tests -v`).

## Outputs

`baselines/results/ours/<dataset>/<name>/`: `tuning.json` (every setting tried with its dev
ROUGE-1), `metrics.json` (test ROUGE-1/2/L/Lsum, chosen settings, graph statistics: mean units,
share of negative edges, share of documents where every unit scored the same) and
`predictions.jsonl` (one summary per test document).

## Cost

NSP needs two BERT passes per kept pair. With β-min = 0, a 30-sentence CNN/DM article needs
~900 passes and a 200-sentence arXiv paper ~40,000. `--beta-min 0.3` roughly halves that; the ranker
can then only use β ≥ 0.3. TL;DR-17 has 3.8 M test posts, which isn't feasible; use `TEST_LIMIT`
and run the baselines on the same documents (`--offset 1000 --limit N`) for a fair comparison.
