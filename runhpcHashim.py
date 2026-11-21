#!/usr/bin/env python3
"""Run ALL of Hashim's ranking experiments on the HPC with one command, ending in an Excel file.

    1. Edit the CONFIG block below: where each dataset file is and its column names
       (`python ranking/tools/prepare_data.py --peek <file>` prints the columns).
    2. On a LOGIN node, from the repo root:
           python3 runhpcHashim.py
    3. Wait for the jobs to finish (`squeue --me`). The result is
           ranking/results/hashim_results.xlsx

What it does:
    setup   .venv + packages + NLTK splitter + unit tests          (login node, first time only)
    data    converts every configured dataset to JSONL: dev (DEV_DOCS docs) and full test
    sweep   Slurm array, one job per dataset: 73 settings of Lead / PacSum degree / PageRank /
            PageTrust on dev
    report  Slurm job that starts when the sweep ends: picks each ranker's best dev setting,
            runs it on test with official ROUGE, writes the Excel file

Other modes:
    python3 runhpcHashim.py report          only the last step (e.g. re-run after a failure)
    python3 runhpcHashim.py local --limit 5 whole pipeline on the local 100-doc samples, no Slurm
"""

# ================================================================== CONFIG (edit me)
# Two-split datasets: give the validation file as "dev" and the test file as "test".
# One-split datasets (WikiHow, TL;DR): give "file" plus the row fractions for dev and test.
# Use the SAME dev/test documents as the baselines. Leave a path as "EDIT_ME" to skip a dataset.
DATA = {
    "cnndm":   {"dev": "EDIT_ME", "test": "EDIT_ME", "article": "article", "summary": "highlights"},
    "pubmed":  {"dev": "EDIT_ME", "test": "EDIT_ME", "article": "article", "summary": "abstract"},
    "arxiv":   {"dev": "EDIT_ME", "test": "EDIT_ME", "article": "article", "summary": "abstract"},
    "billsum": {"dev": "EDIT_ME", "test": "EDIT_ME", "article": "text",    "summary": "summary"},
    "wikihow": {"file": "EDIT_ME", "dev_frac": (0.8, 0.9), "test_frac": (0.9, 1.0),
                "article": "text", "summary": "headline"},
    "tldr":    {"file": "EDIT_ME", "dev_frac": (0.8, 0.9), "test_frac": (0.9, 1.0),
                "article": "content", "summary": "summary",
                "test_docs": 10000},   # TL;DR's 10% test part is ~380k posts: match the baselines' count
}
WORK_DIR = "$SCRATCH/pt_data"   # converted JSONL goes here ($SCRATCH or ~ is expanded)
DEV_DOCS = 1000                 # dev docs per dataset for tuning (spread over the dev split)
TEST_DOCS = None                # None = the whole test split (a dataset's "test_docs" key overrides)
ACCOUNT = ""                    # Slurm account, if the cluster requires one (e.g. "pr_123_general")
PARTITION = ""                  # Slurm partition, if needed
CPUS = 16
# ================================================================== end of CONFIG

import argparse
import csv
import datetime
import glob
import json
import os
import shlex
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
RANKING = os.path.join(ROOT, "ranking")
RESULTS = os.path.join(RANKING, "results")
LOGS = os.path.join(RANKING, "hpc", "logs")
VENV_PY = os.path.join(ROOT, ".venv", "bin", "python")
NAMES = ["cnndm", "pubmed", "arxiv", "billsum", "wikihow", "tldr"]
PRETTY = {"cnndm": "CNN/DM", "pubmed": "PubMed", "arxiv": "arXiv", "billsum": "BillSum",
          "wikihow": "WikiHow", "tldr": "TL;DR"}
RANKERS = ["Lead", "PacSum degree", "PageRank", "PageTrust"]


def work_dir():
    d = WORK_DIR if os.environ.get("SCRATCH") else WORK_DIR.replace("$SCRATCH", "~")
    return os.path.expanduser(os.path.expandvars(d))


def say(msg):
    print(f"\n=== {msg}", flush=True)


# ------------------------------------------------------------------ setup
def ensure_venv():
    """Run setup.sh once, then continue inside .venv (needs numpy, nltk, openpyxl ...)."""
    if os.path.realpath(sys.prefix) == os.path.realpath(os.path.join(ROOT, ".venv")):
        return
    if not os.path.exists(VENV_PY):
        say("setup: building .venv (first run only)")
        subprocess.run(["bash", os.path.join(RANKING, "hpc", "setup.sh")], check=True,
                       env={**os.environ, "PYTHON": sys.executable})
    os.execv(VENV_PY, [VENV_PY] + sys.argv)


def env_ready():
    os.environ.setdefault("NLTK_DATA", os.path.join(ROOT, ".nltk_data"))
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    sys.path.insert(0, RANKING)
    sys.path.insert(0, os.path.join(RANKING, "tools"))


# ------------------------------------------------------------------ data
def prepare():
    from prepare_data import convert, read_rows
    out = work_dir()
    done = []
    for name in NAMES:
        c = DATA[name]
        files = [c.get("dev"), c.get("test")] if "file" not in c else [c["file"]]
        if any(not f or f == "EDIT_ME" for f in files):
            print(f"{name}: no path in CONFIG, skipped")
            continue
        dev_out, test_out = os.path.join(out, "dev"), os.path.join(out, "test")
        if os.path.exists(os.path.join(dev_out, f"{name}.jsonl")) and os.path.exists(os.path.join(test_out, f"{name}.jsonl")):
            print(f"{name}: already converted in {out}, reused (delete it to redo)")
            done.append(name)
            continue
        if "file" in c:
            rows, fmt = read_rows(c["file"])
            convert(name, rows, c["article"], c["summary"], dev_out, DEV_DOCS, *c["dev_frac"], fmt=fmt)
            convert(name, rows, c["article"], c["summary"], test_out, c.get("test_docs", TEST_DOCS),
                    *c["test_frac"], fmt=fmt)
        else:
            convert(name, c["dev"], c["article"], c["summary"], dev_out, DEV_DOCS)
            convert(name, c["test"], c["article"], c["summary"], test_out, c.get("test_docs", TEST_DOCS))
        done.append(name)
    if not done:
        sys.exit("\nNo datasets configured: fill in the paths in the CONFIG block at the top of "
                 "runhpcHashim.py (find the columns with: python ranking/tools/prepare_data.py --peek <file>)")
    return done


# ------------------------------------------------------------------ slurm
def sbatch(args):
    extra = (["--account", ACCOUNT] if ACCOUNT else []) + (["--partition", PARTITION] if PARTITION else [])
    cmd = ["sbatch", "--parsable"] + extra + args
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"sbatch failed:\n  {' '.join(cmd)}\n{r.stderr}\n"
                 "If it is about the account/partition, set ACCOUNT / PARTITION in the CONFIG block.")
    return r.stdout.strip().split(";")[0]


def submit(names):
    os.makedirs(LOGS, exist_ok=True)
    ids = ",".join(str(NAMES.index(n)) for n in names)
    sweep = sbatch([f"--array={ids}", f"--cpus-per-task={CPUS}",
                    f"--export=ALL,DATA={os.path.join(work_dir(), 'dev')},TAG=dev",
                    "ranking/hpc/sweep.sbatch"])
    report = sbatch([f"--dependency=afterany:{sweep}", "--job-name=pt-report", f"--cpus-per-task={CPUS}",
                     "--mem=16G", "--time=24:00:00", "--output=ranking/hpc/logs/%x_%j.out",
                     f"--wrap=cd {shlex.quote(ROOT)} && {shlex.quote(VENV_PY)} runhpcHashim.py report"])
    say(f"submitted: sweep array {sweep} ({', '.join(names)}), report {report} (starts after the sweep)")
    print(f"  watch:   squeue --me\n  logs:    ranking/hpc/logs/\n"
          f"  result:  ranking/results/hashim_results.xlsx (when job {report} finishes)\n"
          f"  cancel:  scancel {sweep} {report}")


# ------------------------------------------------------------------ report
def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def best_settings(dev_rows):
    """Per ranker: the dev row with the highest mean of R-1 / R-2 / R-L."""
    groups = {"Lead": [], "PacSum degree": [], "PageRank": [], "PageTrust": []}
    for r in dev_rows:
        if r["ranker"] == "lead":
            groups["Lead"].append(r)
        elif r["ranker"] == "degree":
            groups["PacSum degree"].append(r)
        elif num(r["gamma"]) == 0:
            groups["PageRank"].append(r)
        else:
            groups["PageTrust"].append(r)
    score = lambda r: (float(r["R1"]) + float(r["R2"]) + float(r["RL"])) / 3
    return {k: max(v, key=score) for k, v in groups.items() if v}


def report(dev_dir, test_dir, tag="dev", out_name="hashim_results.xlsx", limit=None):
    from run import evaluate, n_workers
    from preprocess import SPLITTER, STOP_LIST
    workers = n_workers()
    detail, sweep_rows = [], []
    for name in NAMES:
        csv_path = os.path.join(RESULTS, f"sweep_{tag}_{name}.csv")      # one file per dataset (Slurm)
        combined = os.path.join(RESULTS, f"sweep_{tag}.csv")             # or one file for all (local)
        test_path = os.path.join(test_dir, f"{name}.jsonl")
        if os.path.exists(csv_path):
            dev = list(csv.DictReader(open(csv_path)))
        elif os.path.exists(combined):
            dev = [r for r in csv.DictReader(open(combined)) if r["dataset"] == name]
        else:
            dev = []
        if not dev:
            continue
        sweep_rows += dev
        if not os.path.exists(test_path):
            print(f"{name}: no test file {test_path}, skipped")
            continue
        with open(test_path, encoding="utf-8") as f:
            rows = [json.loads(l) for l, _ in zip(f, range(limit or 10**9))]
        for label, b in best_settings(dev).items():
            ranker = {"Lead": "lead", "PacSum degree": "degree"}.get(label, "pagetrust")
            beta, lb = num(b["beta"]) or 0.0, num(b["lambda_bwd"]) or 0.0
            kw = {"gamma": num(b["gamma"]), "memory": num(b["memory"])} if ranker == "pagetrust" else {}
            r1, r2, rl = evaluate(rows, name, ranker, beta, 1.0, lb, scorer="official", workers=workers, **kw)
            dev_score = (float(b["R1"]) + float(b["R2"]) + float(b["RL"])) / 3
            detail.append({"dataset": name, "ranker": label, "n_test_docs": len(rows),
                           "R1": r1, "R2": r2, "RL": rl, "beta": b["beta"], "lambda_bwd": b["lambda_bwd"],
                           "gamma": b["gamma"], "memory": b["memory"],
                           "n_dev_docs": int(b["n_docs"]), "dev_mean_R": dev_score})
            print(f"{name:8s} {label:14s} test R-1 {r1:5.2f} R-2 {r2:5.2f} R-L {rl:5.2f}   "
                  f"(beta={b['beta']} lambda_bwd={b['lambda_bwd']} gamma={b['gamma']} M={b['memory']})", flush=True)
    if not detail:
        sys.exit(f"nothing to report: no ranking/results/sweep_{tag}_<dataset>.csv with a matching test file")
    info = {"created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), "git commit": git_commit(),
            "sentence splitter": SPLITTER, "stop list": STOP_LIST, "test ROUGE": "rouge-score (official), F1, stemmed",
            "ROUGE-L": "rougeLsum (summary-level, one sentence per line)", "dev ROUGE": "rouge_lite (within 0.05 of official)",
            "graph": "stand-in: tf-idf cosine, PacSum threshold beta, position direction (lambda_fwd = 1)",
            "setting choice": "per ranker, the dev setting with the highest mean of R-1/R-2/R-L",
            "alpha (damping)": "0.85", "dev data": dev_dir, "test data": test_dir}
    path = write_excel(detail, sweep_rows, info, os.path.join(RESULTS, out_name))
    with open(path.replace(".xlsx", ".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(detail[0]))
        w.writeheader()
        w.writerows(detail)
    say(f"wrote {path} (and a .csv copy)")


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True).stdout.strip() or "?"
    except OSError:
        return "?"


def write_excel(detail, sweep_rows, info, path):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    HEAD = PatternFill("solid", fgColor="1F3864")
    SUB = PatternFill("solid", fgColor="D9E1F2")
    BEST = PatternFill("solid", fgColor="C6EFCE")
    white, bold = Font(bold=True, color="FFFFFF"), Font(bold=True)
    thin = Side(style="thin", color="BFBFBF")
    box = Border(left=thin, right=thin, top=thin, bottom=thin)
    center = Alignment(horizontal="center", vertical="center", wrap_text=True)

    def widths(ws, w):
        for i, x in enumerate(w, 1):
            ws.column_dimensions[get_column_letter(i)].width = x

    wb = Workbook()

    # --- Summary: datasets x rankers, R-1 / R-2 / R-L, best per metric in green
    ws = wb.active
    ws.title = "Summary"
    ws["A1"] = "Test-set ROUGE F1 (official rouge-score); settings tuned on dev. Green = best per dataset and metric."
    ws["A1"].font = Font(italic=True, color="595959")
    ws.cell(row=2, column=1, value="Dataset")
    ws.merge_cells(start_row=2, start_column=1, end_row=3, end_column=1)
    for k, rk in enumerate(RANKERS):
        c = 2 + 3 * k
        ws.cell(row=2, column=c, value=rk)
        ws.merge_cells(start_row=2, start_column=c, end_row=2, end_column=c + 2)
        for j, m in enumerate(("R-1", "R-2", "R-L")):
            ws.cell(row=3, column=c + j, value=m)
    dc = 2 + 3 * len(RANKERS)
    ws.cell(row=2, column=dc, value="PageTrust − PacSum")
    ws.merge_cells(start_row=2, start_column=dc, end_row=2, end_column=dc + 2)
    for j, m in enumerate(("R-1", "R-2", "R-L")):
        ws.cell(row=3, column=dc + j, value=m)
    for row in ws.iter_rows(min_row=2, max_row=3, max_col=dc + 2):
        for cell in row:
            cell.fill, cell.font, cell.alignment, cell.border = (HEAD if cell.row == 2 else SUB), \
                (white if cell.row == 2 else bold), center, box

    by = {(d["dataset"], d["ranker"]): d for d in detail}
    datasets = [n for n in NAMES if any(d["dataset"] == n for d in detail)]
    r = 4
    for name in datasets:
        ws.cell(row=r, column=1, value=PRETTY[name]).font = bold
        for j, m in enumerate(("R1", "R2", "RL")):
            vals = {rk: by[(name, rk)][m] for rk in RANKERS if (name, rk) in by}
            top = max(vals.values())
            for k, rk in enumerate(RANKERS):
                if rk in vals:
                    cell = ws.cell(row=r, column=2 + 3 * k + j, value=round(vals[rk], 2))
                    if abs(vals[rk] - top) < 1e-9:
                        cell.fill, cell.font = BEST, bold
            if "PageTrust" in vals and "PacSum degree" in vals:
                diff = ws.cell(row=r, column=dc + j, value=round(vals["PageTrust"] - vals["PacSum degree"], 2))
                diff.font = Font(color="006100" if diff.value > 0 else "9C0006" if diff.value < 0 else "000000")
        r += 1
    if len(datasets) > 1:                                          # average row
        ws.cell(row=r, column=1, value="Average").font = bold
        for col in range(2, dc + 3):
            vals = [ws.cell(row=i, column=col).value for i in range(4, r)]
            vals = [v for v in vals if isinstance(v, (int, float))]
            if vals:
                ws.cell(row=r, column=col, value=round(sum(vals) / len(vals), 2)).font = bold
        r += 1
    for row in ws.iter_rows(min_row=4, max_row=r - 1, max_col=dc + 2):
        for cell in row:
            cell.border, cell.number_format = box, "0.00"
            if cell.column > 1:
                cell.alignment = Alignment(horizontal="center")
    widths(ws, [12] + [8] * (dc + 1))
    ws.freeze_panes = "B4"

    # --- Test detail: one row per dataset x ranker with the chosen settings
    ws = wb.create_sheet("Test detail")
    cols = [("Dataset", 11), ("Ranker", 15), ("Test docs", 10), ("R-1", 8), ("R-2", 8), ("R-L", 8),
            ("β (threshold)", 12), ("λ_bwd", 8), ("γ (conviction)", 13), ("M (memory)", 11),
            ("Dev docs", 9), ("Dev mean ROUGE", 14)]
    keys = ["dataset", "ranker", "n_test_docs", "R1", "R2", "RL", "beta", "lambda_bwd", "gamma", "memory",
            "n_dev_docs", "dev_mean_R"]
    for i, (h, _) in enumerate(cols, 1):
        c = ws.cell(row=1, column=i, value=h)
        c.fill, c.font, c.alignment, c.border = HEAD, white, center, box
    for i, d in enumerate(detail, 2):
        for j, k in enumerate(keys, 1):
            v = d[k]
            if k == "dataset":
                v = PRETTY[v]
            elif k in ("beta", "lambda_bwd", "gamma", "memory"):
                v = "–" if v == "" else ("∞" if num(v) == float("inf") else num(v))
            c = ws.cell(row=i, column=j, value=v)
            c.border = box
            if k in ("R1", "R2", "RL", "dev_mean_R"):
                c.number_format = "0.00"
    widths(ws, [w for _, w in cols])
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(cols))}{len(detail) + 1}"

    # --- Dev sweep: every setting tried
    ws = wb.create_sheet("Dev sweep (all)")
    if sweep_rows:
        hdr = list(sweep_rows[0])
        for j, h in enumerate(hdr, 1):
            c = ws.cell(row=1, column=j, value=h)
            c.fill, c.font, c.alignment = HEAD, white, center
        for i, row in enumerate(sweep_rows, 2):
            for j, h in enumerate(hdr, 1):
                v = row[h]
                ws.cell(row=i, column=j, value=num(v) if num(v) is not None else (v or None))
        widths(ws, [10, 11] + [10] * (len(hdr) - 2))
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = f"A1:{get_column_letter(len(hdr))}{len(sweep_rows) + 1}"

    # --- Run info
    ws = wb.create_sheet("Run info")
    for i, (k, v) in enumerate(info.items(), 1):
        ws.cell(row=i, column=1, value=k).font = bold
        ws.cell(row=i, column=2, value=v)
    widths(ws, [20, 90])

    os.makedirs(os.path.dirname(path), exist_ok=True)
    wb.save(path)
    return path


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", nargs="?", default="all", choices=["all", "report", "local"])
    ap.add_argument("--limit", type=int, help="local mode: docs per dataset (default 10)")
    a = ap.parse_args()
    ensure_venv()
    env_ready()

    if a.mode == "report":
        report(os.path.join(work_dir(), "dev"), os.path.join(work_dir(), "test"))
    elif a.mode == "local":                    # smoke test on the 100-doc samples, no Slurm
        sample = os.path.join(RANKING, "data_sample")
        if not glob.glob(os.path.join(sample, "*.jsonl")):
            sys.exit("no local samples in ranking/data_sample (build them with ranking/tools/fetch_samples.py)")
        lim = a.limit or 10
        say(f"local smoke test: dev = test = first {lim} docs of ranking/data_sample (numbers are NOT meaningful)")
        subprocess.run([sys.executable, os.path.join(RANKING, "tools", "sweep.py"), "--data", sample,
                        "--limit", str(lim), "--tag", "local"], check=True)
        report(sample, sample, tag="local", out_name="hashim_results_LOCAL_TEST.xlsx", limit=lim)
    else:
        if subprocess.run(["which", "sbatch"], capture_output=True).returncode:
            sys.exit("sbatch not found: run this on the HPC login node (or use `local` mode to test)")
        say("data: converting datasets to JSONL")
        names = prepare()
        say("slurm: submitting jobs")
        submit(names)


if __name__ == "__main__":
    main()
