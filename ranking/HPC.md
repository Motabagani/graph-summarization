# Running the ranking code on the HPC

This runs Hashim's part of the project (PacSum degree centrality, PageRank and PageTrust on the
stand-in tf-idf graph) on the full datasets. Everything below is copy-paste. Run it from the
**repo root** (`graph-summarization/`).

Requirements: Python **3.9 or newer**, and a Slurm cluster. Nothing needs a GPU.

---

## 1. One-time setup (on a login node; it needs internet)

```bash
git clone https://github.com/Motabagani/graph-summarization.git
cd graph-summarization
module avail python              # pick a python >= 3.9 if the default is older, then:
module load python/<version>     # (skip if `python3 --version` already says >= 3.9)
bash ranking/hpc/setup.sh
```

`setup.sh` does the following:
- builds `.venv/`
- installs `ranking/requirements.txt` plus gensim
- downloads the NLTK sentence splitter into `.nltk_data/`
- runs the 22 unit tests

It should end with:

```
OK
setup OK  splitter = nltk  stop_list = gensim
```

If it says `stop_list = hw4-closed-class`, gensim didn't install. Everything still runs, but the
stop words won't match the TextRank baseline. Tell Hashim if that happens.

For an interactive shell instead of batch jobs, run this first:

```bash
export NLTK_DATA=$PWD/.nltk_data
source .venv/bin/activate
```

## 2. Turn the datasets into JSONL

The scripts read one JSONL per dataset, `{"id", "article", "summary"}` per line, with the
datasets named `cnndm pubmed arxiv billsum wikihow tldr`. The column names differ between
copies of the data, so **peek first**, then convert:

```bash
source .venv/bin/activate
python ranking/tools/prepare_data.py --peek /path/to/cnn_dailymail/test.parquet
```

Peek prints the format, the columns and the first row. Pass the right columns to the
conversion. Here is CNN/DM test as an example:

```bash
python ranking/tools/prepare_data.py cnndm /path/to/cnn_dailymail/test.parquet \
    --article-col article --summary-col highlights --out $SCRATCH/pt_data/test
```

Make two folders:

| Folder | What goes in | Used for |
|---|---|---|
| `$SCRATCH/pt_data/dev` | about 1,000 docs per dataset from the **validation** split (`--n 1000`) | picking settings (step 4) |
| `$SCRATCH/pt_data/test` | the full **test** split | the numbers we report (step 5) |

The usual column names (always confirm with `--peek`):

| Dataset | Article column | Summary column | Notes |
|---|---|---|---|
| cnndm | `article` | `highlights` | |
| pubmed, arxiv | `article` (HF copy) or `article_text` (original) | `abstract` or `abstract_text` | either layout works; `<S>` tags are removed automatically |
| billsum | `text` | `summary` | |
| wikihow | `text` | `headline` | there's only one split: use the **same 80/10/10 split as the baselines**, e.g. `--start-frac 0.8 --end-frac 0.9` for dev and `--start-frac 0.9` for test |
| tldr | `content` | `summary` | single split, so do the same as for wikihow |

**Use the same dev and test documents as the baselines.** If the baselines used a different
WikiHow or TL;DR split, match theirs.

## 3. Sanity check: reproduce PacSum on CNN/DM

```bash
sbatch ranking/hpc/run.sbatch ranking/run.py $SCRATCH/pt_data/test/cnndm.jsonl \
    --dataset cnndm --ranker degree --beta 0.3 --lambda-bwd -0.5 --rouge official
```

The log goes to `ranking/hpc/logs/pt-run_<jobid>.out`. Compare **R-1 with the PacSum tf-idf
baseline (38.67)**. If it's more than about 1 point off, stop and tell Hashim. Something
differs (the splitter, the stop words, β or λ, or ROUGE-L vs ROUGE-Lsum), and the PageTrust
numbers wouldn't be comparable either. The first line of every log shows the splitter, the
stop list and the ROUGE setting.

## 4. Tune on dev: sweep all settings

```bash
sbatch --export=ALL,DATA=$SCRATCH/pt_data/dev,TAG=dev ranking/hpc/sweep.sbatch
```

- This submits 6 array tasks, one per dataset. Each task tries 73 settings: Lead, PacSum
  degree over β×λ, and PageTrust over β×λ×γ×M (γ = 0 is PageRank).
- Results go to `ranking/results/sweep_dev_<dataset>.csv`, with one row per setting and the
  columns R1, R2 and RL.
- Monitor the jobs with `squeue --me`. To cancel them: `scancel <jobid>`.
- Time, measured on a laptop: 100 CNN/DM docs took about 6 s on 4 cores, but 100 arXiv docs
  took about 35 min on 8 cores. Long documents make big graphs, and the PageTrust settings with
  M = 0.9 converge slowly. Expect roughly 3 h for 1,000 arXiv docs on 16 CPUs. The job allows
  24 h and prints progress every 500 docs.
- Keep dev at about 1,000 docs per dataset. If the dev files are bigger, add `LIMIT=1000` to the
  `--export` list.
- `--rouge lite` (the default) is fast and within 0.05 of the official ROUGE. Add `ROUGE=official`
  to the `--export` list if you want the official ROUGE.

**Don't report numbers from this sweep.** It picks the best of 73 settings, so its numbers are
optimistic.

## 5. Final numbers on test

For each dataset, take the best row from the dev CSVs: one for PacSum degree and one for
PageTrust (and the γ = 0 row for PageRank). Run them on test with the official ROUGE, for
example:

```bash
sbatch ranking/hpc/run.sbatch ranking/run.py $SCRATCH/pt_data/test/arxiv.jsonl \
    --dataset arxiv --ranker pagetrust --beta 0.3 --lambda-bwd -0.5 --gamma 1 --memory 0 --rouge official
sbatch ranking/hpc/run.sbatch ranking/run.py $SCRATCH/pt_data/test/arxiv.jsonl \
    --dataset arxiv --ranker degree --beta 0.3 --lambda-bwd -0.5 --rouge official
```

Rankers: `lead`, `degree` (PacSum), `pagerank` and `pagetrust`. Every option is listed under
`python ranking/run.py -h`.

## 6. Send back
- `ranking/results/sweep_dev_*.csv`
- the test logs from `ranking/hpc/logs/`
- which dev/test split you used for WikiHow, TL;DR and BillSum

Commit them on a branch or drop them in the team folder. Don't commit the data: `.gitignore`
already excludes it.

## If something breaks

| Symptom | Fix |
|---|---|
| `TypeError: 'type' object is not subscriptable` | Python is older than 3.9. Load a newer module, then delete `.venv` and run setup again. |
| `Resource punkt_tab not found` | `NLTK_DATA` isn't set. The sbatch scripts set it; in a shell, run `export NLTK_DATA=$PWD/.nltk_data`. |
| `sbatch: error: ... account` | Uncomment the `#SBATCH --account=` line at the top of the `.sbatch` files and add your account. |
| `<dataset>: ... not found, skipped` | The JSONL isn't named `<dataset>.jsonl` inside `DATA`. |
| A job runs out of time | Raise `--time` in `sweep.sbatch`, or add `LIMIT=500` to `--export`. |
| `n_skipped` is high | Those documents had fewer than 2 sentences after splitting. For arXiv/PubMed, check the article column holds the text and not the IDs. |
