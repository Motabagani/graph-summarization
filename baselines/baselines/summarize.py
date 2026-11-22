#!/usr/bin/env python3
"""
Shared pipeline for the extractive baselines (a library; run the baselines themselves):

    python lead3.py    --dataset cnn_dailymail ...     # Lead-k
    python textrank.py --dataset cnn_dailymail ...     # TextRank (PageRank centrality)
    python DegreeCentrality.py --dataset cnn_dailymail ...   # Degree centrality

Each baseline goes raw documents -> summaries -> ROUGE, but everything that must
be identical across methods for a fair comparison lives here: dataset loading,
sentence splitting, how a summary is assembled from sentence scores, reference
handling, ROUGE, parallel execution and the output files.

A baseline module provides
    Config (dataclass), add_args(parser), prepare(cfg, args),
    score_sentences(sentences, cfg) -> (scores, info)
optionally
    NEEDS_REFERENCE = True      -> score_sentences(sentences, cfg, reference) (e.g. Oracle)
    tune_grid(cfg) -> [dict]    -> enables --tune (grid search on a validation split)
A score of -inf means "never select this sentence".
and ends with
    if __name__ == "__main__":
        summarize.cli(sys.modules[__name__], "<name>")
"""

import argparse
import dataclasses
import glob
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, fields
from multiprocessing import Pool

import numpy as np


# =============================================================================
# Summary configuration (shared by every method)
# =============================================================================
@dataclass
class SummaryConfig:
    splitter: str = "nltk"            # nltk | regex
    newline_boundaries: bool = False  # treat line breaks as sentence boundaries
    word_limit: int = 100             # take top sentences until this many words (0 = off)
    num_sentences: int = 0            # >0: take exactly this many sentences instead
    truncate: bool = True             # cut the summary at word_limit words
    order: str = "score"              # score | document (original order)


# =============================================================================
# Sentences
# =============================================================================
_WS_RE = re.compile(r"\s+")
_REGEX_SPLIT = re.compile(r"(?<=[.!?])[\"'”’)\]]*\s+(?=[\"'“‘(\[]*[A-Z0-9])")
_cache = {}


def ensure_nltk(resources):
    """resources: list of (nltk.data path, package name). Downloads what is missing."""
    import nltk
    for path, pkg in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"Downloading NLTK resource '{pkg}' ...", file=sys.stderr)
            nltk.download(pkg, quiet=True)


def split_sentences(text, cfg):
    if not text:
        return []
    blocks = text.splitlines() if cfg.newline_boundaries else [text]
    out = []
    for block in blocks:
        block = _WS_RE.sub(" ", block).strip()
        if not block:
            continue
        if cfg.splitter == "nltk":
            if "sent_tokenize" not in _cache:
                from nltk.tokenize import sent_tokenize
                _cache["sent_tokenize"] = sent_tokenize
            out.extend(s.strip() for s in _cache["sent_tokenize"](block))
        else:
            out.extend(s.strip() for s in _REGEX_SPLIT.split(block))
    return [s for s in out if s]


def prepare_sentences(doc, cfg):
    """doc: raw text, or a list of pre-split sentences (arXiv/PubMed)."""
    if isinstance(doc, (list, tuple, np.ndarray)):
        sentences = [str(s) for s in doc]
    else:
        sentences = split_sentences(doc, cfg)
    return [_WS_RE.sub(" ", s).strip() for s in sentences if s and str(s).strip()]


def build_summary(sentences, scores, cfg):
    """Pick sentences by descending score. Returns (summary_text, chosen_indices, summary_sentences)."""
    scores = np.asarray(scores, dtype=float)
    ranking = np.argsort(-scores, kind="stable")       # ties -> earlier first
    chosen, words = [], 0
    for i in ranking:
        if not np.isfinite(scores[i]):                 # -inf = excluded by the method
            continue
        if cfg.num_sentences > 0:
            if len(chosen) >= cfg.num_sentences:
                break
            chosen.append(int(i))
            continue
        chosen.append(int(i))
        words += len(sentences[i].split())
        if cfg.word_limit and words >= cfg.word_limit:
            break

    if cfg.order == "document":
        chosen.sort()
    sents = [sentences[i] for i in chosen]
    if cfg.num_sentences == 0 and cfg.truncate and cfg.word_limit:
        cut, left = [], cfg.word_limit                  # keep sentence boundaries for ROUGE-Lsum
        for s in sents:
            if left <= 0:
                break
            w = s.split()
            cut.append(" ".join(w[:left]))
            left -= len(w)
        sents = cut
    return " ".join(sents), chosen, sents


def summarize_doc(doc, score_fn, method_cfg, summary_cfg, reference=None):
    """Run one method on one document and return a result dict.
    `reference` (list of sentences) is passed only to methods that need it (Oracle)."""
    sentences = prepare_sentences(doc, summary_cfg)
    if not sentences:
        return {"summary": "", "summary_sentences": [], "selected": [], "scores": [],
                "iterations": 0, "converged": True, "num_sentences": 0}
    if reference is None:
        scores, info = score_fn(sentences, method_cfg)
    else:
        scores, info = score_fn(sentences, method_cfg, reference)
    text, chosen, sents = build_summary(sentences, scores, summary_cfg)
    return {"summary": text, "summary_sentences": sents, "selected": chosen,
            "scores": [float(x) if np.isfinite(x) else None for x in scores],
            "num_sentences": len(sentences),
            "iterations": info.get("iterations", 0), "converged": info.get("converged", True)}


# =============================================================================
# Datasets
# =============================================================================
# Patterns are relative to --data-root; {split} is replaced by the split name.
PRESETS = {
    "cnn_dailymail": dict(patterns=["cnn_dailymail/**/{split}.csv"], fmt="csv",
                          text="article", ref="highlights", id="id",
                          split_alias={"val": "validation"}),
    "arxiv": dict(patterns=["long_summarization/**/arxiv-dataset/{split}.txt",
                            "long_summarization/arxiv-dataset_hf/**/{split}*.txt"],
                  fmt="jsonl", text="article_text", ref="abstract_text", id="article_id",
                  alt_text="article", alt_ref="abstract",
                  split_alias={"validation": "val"}),
    "pubmed": dict(patterns=["long_summarization/**/pubmed-dataset/{split}.txt",
                             "long_summarization/pubmed-dataset_hf/**/{split}*.txt"],
                   fmt="jsonl", text="article_text", ref="abstract_text", id="article_id",
                   alt_text="article", alt_ref="abstract",
                   split_alias={"validation": "val"}),
    "tldr17": dict(patterns=["tldr17/**/corpus-webis-tldr-17.json"], fmt="jsonl",
                   text="content", ref="summary", id="id", no_split=True),
    "billsum": dict(patterns=["billsum/**/{split}-*.parquet", "billsum/**/{split}.parquet"],
                    fmt="parquet", text="text", ref="summary", id="title"),
    "wikihow": dict(patterns=["wikihow/manual/wikihowAll.csv", "wikihow/**/wikihowAll.csv"],
                    fmt="csv", text="text", ref="headline", id="title", no_split=True),
}

_TAG_RE = re.compile(r"</?S>")


def reference_sentences(ref):
    """Reference split into sentences (needed for ROUGE-Lsum)."""
    if isinstance(ref, (list, tuple, np.ndarray)):          # arXiv/PubMed: already a list
        parts = [str(r) for r in ref]
    elif "\n" in str(ref):                                 # CNN/DM highlights, WikiHow headlines
        parts = str(ref).splitlines()
    else:
        parts = [str(ref)]
    parts = [_WS_RE.sub(" ", _TAG_RE.sub(" ", p)).strip(" ,") for p in parts]
    return [p for p in parts if p]


def resolve_input(args, preset):
    if args.input:
        return sorted(glob.glob(args.input, recursive=True)) or [args.input]
    split = preset.get("split_alias", {}).get(args.split, args.split)
    files = []
    for pat in preset["patterns"]:
        files = sorted(glob.glob(os.path.join(args.data_root, pat.format(split=split)), recursive=True))
        if files:
            break
    if not files:
        sys.exit(f"No files found for dataset '{args.dataset}' split '{args.split}' under "
                 f"{args.data_root}. Tried: {preset['patterns']}. Use --input to point at the file.")
    return files


def locate(args, split, explicit_input=None):
    """Resolve files/format/field names for `split` (or an explicit file/glob)."""
    preset = PRESETS.get(args.dataset, {})
    if explicit_input:
        files = sorted(glob.glob(explicit_input, recursive=True)) or [explicit_input]
    elif preset:
        files = resolve_input(argparse.Namespace(input=None, split=split, data_root=args.data_root,
                                                 dataset=args.dataset), preset)
    else:
        sys.exit("--input is required with --dataset custom")
    fmt = args.format or preset.get("fmt") or os.path.splitext(files[0])[1].lstrip(".")
    fmt = {"json": "jsonl", "txt": "jsonl", "h5": "pacsum_h5", "h5df": "pacsum_h5"}.get(fmt, fmt)
    return dict(files=files, fmt=fmt,
                text_f=args.text_field or preset.get("text", "text" if fmt != "pacsum_h5" else "article"),
                ref_f=args.ref_field or preset.get("ref", "summary" if fmt != "pacsum_h5" else "abstract"),
                id_f=args.id_field or preset.get("id", "id"),
                alt_text=preset.get("alt_text"), alt_ref=preset.get("alt_ref"),
                no_split=preset.get("no_split", False))


def open_records(loc):
    return iter_records(loc["files"], loc["fmt"], loc["text_f"], loc["ref_f"], loc["id_f"],
                        loc["alt_text"], loc["alt_ref"])


def iter_records(files, fmt, text_f, ref_f, id_f, alt_text=None, alt_ref=None):
    """Yield (id, text, reference) without loading huge JSONL files into memory."""
    n = 0
    if fmt == "pacsum_h5":
        # PacSum's released data: HDF5 'dataset' of JSON strings with
        # 'article' (list of sentences) and 'abstract' (list of sentences).
        import h5py
        for path in files:
            with h5py.File(path, "r") as f:
                for j_str in f["dataset"]:
                    r = json.loads(j_str)
                    yield (str(n), r[text_f if text_f in r else "article"],
                           r[ref_f if ref_f in r else "abstract"])
                    n += 1
        return
    if fmt == "jsonl":
        for path in files:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    text = r.get(text_f, r.get(alt_text) if alt_text else None)
                    ref = r.get(ref_f, r.get(alt_ref) if alt_ref else None)
                    if text is None or ref is None:
                        continue
                    yield (str(r.get(id_f, n)), text, ref)
                    n += 1
    else:
        import pandas as pd
        for path in files:
            df = pd.read_csv(path) if fmt == "csv" else pd.read_parquet(path)
            missing = {text_f, ref_f} - set(df.columns)
            if missing:
                sys.exit(f"{path} lacks columns {missing}; has {list(df.columns)}")
            df = df.dropna(subset=[text_f, ref_f])
            ids = df[id_f].astype(str) if id_f in df.columns else df.index.astype(str)
            for i, t, r in zip(ids, df[text_f], df[ref_f]):
                yield (i, t, r)


def _limited(it, offset, limit):
    for i, x in enumerate(it):
        if i < offset:
            continue
        if limit and i >= offset + limit:
            break
        yield x


# =============================================================================
# Command line
# =============================================================================
def build_parser(description=None, default_out=None):
    p = argparse.ArgumentParser(description=description,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("data / run")
    g.add_argument("--dataset", choices=list(PRESETS) + ["custom"], default="custom")
    g.add_argument("--data-root", default=os.environ.get("DATA_ROOT", "."))
    g.add_argument("--split", default="test")
    g.add_argument("--input", help="explicit file/glob (overrides the preset location)")
    g.add_argument("--format", choices=["csv", "jsonl", "parquet", "txt", "pacsum_h5"])
    g.add_argument("--text-field")
    g.add_argument("--ref-field")
    g.add_argument("--id-field")
    g.add_argument("--limit", type=int, default=0, help="max documents (0 = all)")
    g.add_argument("--offset", type=int, default=0)
    g.add_argument("--out", default=default_out, help="output directory (default: results/<method>)")
    g.add_argument("--workers", type=int,
                   default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)))
    g.add_argument("--chunksize", type=int, default=16)
    g.add_argument("--no-eval", action="store_true", help="skip ROUGE")
    g.add_argument("--rouge-stemmer", action=argparse.BooleanOptionalAction, default=True,
                   help="Porter-stem before ROUGE (pyrouge's default config does not stem)")

    u = p.add_argument_group("tuning (methods with hyperparameters: LexPageRank, PacSum)")
    u.add_argument("--tune", action="store_true",
                   help="grid-search the method's hyperparameters on a validation split first")
    u.add_argument("--tune-split", default="validation")
    u.add_argument("--tune-input", help="explicit validation file/glob (overrides --tune-split)")
    u.add_argument("--tune-docs", type=int, default=1000, help="validation documents used (PacSum: 1000)")
    u.add_argument("--tune-metric", default="rouge1", choices=["rouge1", "rouge2", "rougeL", "rougeLsum"],
                   help="F1 metric to maximise (PacSum: rouge1)")

    d = SummaryConfig()
    s = p.add_argument_group("summary construction (shared by all methods)")
    s.add_argument("--splitter", choices=["nltk", "regex"], default=d.splitter)
    s.add_argument("--newline-boundaries", action=argparse.BooleanOptionalAction, default=d.newline_boundaries)
    s.add_argument("--word-limit", type=int, default=d.word_limit)
    s.add_argument("--num-sentences", type=int, default=d.num_sentences)
    s.add_argument("--truncate", action=argparse.BooleanOptionalAction, default=d.truncate)
    s.add_argument("--order", choices=["score", "document"], default=d.order)
    return p


def config_from_args(cls, args):
    # fields without a CLI flag keep their defaults
    return cls(**{f.name: getattr(args, f.name) for f in fields(cls) if hasattr(args, f.name)})


# =============================================================================
# Parallel driver
# =============================================================================
_W = {}


def _init_worker(score_fn, method_cfg, summary_cfg, with_rouge, rouge_stemmer, needs_reference=False):
    _W.update(score_fn=score_fn, method_cfg=method_cfg, summary_cfg=summary_cfg,
              scorer=None, lsum=None, needs_reference=needs_reference)
    if with_rouge:
        from rouge_score import rouge_scorer
        _W["scorer"] = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"],
                                                use_stemmer=rouge_stemmer)
        # rougeLsum = summary-level LCS over newline-separated sentences, the way
        # ROUGE-1.5.5 / pyrouge computes ROUGE-L (what PacSum and HipoRank report).
        _W["lsum"] = rouge_scorer.RougeScorer(["rougeLsum"], use_stemmer=rouge_stemmer)


def _work(rec):
    doc_id, text, ref = rec
    ref_sents = reference_sentences(ref)
    res = summarize_doc(text, _W["score_fn"], _W["method_cfg"], _W["summary_cfg"],
                        reference=ref_sents if _W["needs_reference"] else None)
    ref = " ".join(ref_sents)
    out = {"id": doc_id, "summary": res["summary"], "reference": ref,
           "selected": res["selected"], "num_sentences": res["num_sentences"],
           "iterations": res["iterations"], "converged": res["converged"]}
    if _W["scorer"] is not None and res["summary"] and ref:
        s = dict(_W["scorer"].score(ref, res["summary"]))
        s.update(_W["lsum"].score("\n".join(ref_sents), "\n".join(res["summary_sentences"])))
        out["rouge"] = {k: [v.precision, v.recall, v.fmeasure] for k, v in s.items()}
    return out


def run(args, method_name, score_fn, method_cfg, summary_cfg, needs_reference=False, extra=None):
    """Load the dataset, summarise every document in parallel, score, write outputs."""
    if summary_cfg.splitter == "nltk":
        ensure_nltk([("tokenizers/punkt_tab", "punkt_tab")])

    loc = locate(args, args.split, args.input)
    files = loc["files"]
    if loc["no_split"]:
        print(f"[{args.dataset}] has no official splits; using the whole file "
              f"(--offset/--limit select a subset).", file=sys.stderr)
    print(f"[{method_name}] files: {files}", file=sys.stderr)
    records = _limited(open_records(loc), args.offset, args.limit)

    os.makedirs(args.out, exist_ok=True)
    with_rouge = not args.no_eval
    sums, iters = {}, []
    n = n_conv = 0
    t0 = time.time()
    init = (score_fn, method_cfg, summary_cfg, with_rouge, args.rouge_stemmer, needs_reference)
    with open(os.path.join(args.out, "predictions.jsonl"), "w", encoding="utf-8") as fout, \
            Pool(args.workers, initializer=_init_worker, initargs=init) as pool:
        for res in pool.imap(_work, records, chunksize=args.chunksize):
            fout.write(json.dumps(res, ensure_ascii=False) + "\n")
            n += 1
            n_conv += res["converged"]
            iters.append(res["iterations"])
            for k, v in res.get("rouge", {}).items():
                acc = sums.setdefault(k, [0.0, 0.0, 0.0, 0])
                for j in range(3):
                    acc[j] += v[j]
                acc[3] += 1
            if n % 1000 == 0:
                print(f"  {n} docs  {n / (time.time() - t0):.1f} docs/s", file=sys.stderr, flush=True)

    elapsed = time.time() - t0
    metrics = {
        "method": method_name,
        "dataset": args.dataset, "split": args.split, "files": files,
        "documents": n, "seconds": round(elapsed, 2),
        "docs_per_sec": round(n / elapsed, 2) if elapsed else None,
        "converged_fraction": round(n_conv / n, 4) if n else None,
        "mean_iterations": round(float(np.mean(iters)), 2) if iters else None,
        "method_config": asdict(method_cfg),
        "summary_config": asdict(summary_cfg),
        "rouge_stemmer": args.rouge_stemmer,
        **(extra or {}),
        "rouge": {k: {"precision": v[0] / v[3], "recall": v[1] / v[3], "f1": v[2] / v[3], "n": v[3]}
                  for k, v in sums.items()},
    }
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[{method_name}] done: {n} documents in {elapsed:.1f}s -> {args.out}/")
    for k, v in metrics["rouge"].items():
        print(f"  {k:9s}  P={v['precision']:.4f}  R={v['recall']:.4f}  F1={v['f1']:.4f}")
    return metrics


# =============================================================================
# Hyperparameter tuning (grid search on a validation split)
# =============================================================================
_T = {}


def _init_tuner(score_fn, cfgs, summary_cfg, needs_reference, metric, stemmer):
    from rouge_score import rouge_scorer
    _T.update(score_fn=score_fn, cfgs=cfgs, summary_cfg=summary_cfg, needs_reference=needs_reference,
              metric=metric, scorer=rouge_scorer.RougeScorer([metric], use_stemmer=stemmer))


def _tune_work(rec):
    """F1 of every grid configuration on one validation document."""
    _, text, ref = rec
    out = np.zeros(len(_T["cfgs"]))
    ref_sents = reference_sentences(ref)
    sentences = prepare_sentences(text, _T["summary_cfg"])
    if not sentences or not ref_sents:
        return out
    lsum = _T["metric"] == "rougeLsum"
    ref_text = ("\n" if lsum else " ").join(ref_sents)
    for k, cfg in enumerate(_T["cfgs"]):
        if _T["needs_reference"]:
            scores, _ = _T["score_fn"](sentences, cfg, ref_sents)
        else:
            scores, _ = _T["score_fn"](sentences, cfg)
        _, _, sents = build_summary(sentences, scores, _T["summary_cfg"])
        if sents:
            out[k] = _T["scorer"].score(ref_text, ("\n" if lsum else " ").join(sents))[_T["metric"]].fmeasure
    return out


def tune(args, method_name, module, base_cfg, summary_cfg, needs_reference=False):
    """Evaluate every configuration in module.tune_grid(cfg) on the first --tune-docs
    validation documents; return the config with the best mean F1 (first one on ties)."""
    grid = module.tune_grid(base_cfg)
    cfgs = [dataclasses.replace(base_cfg, **g) for g in grid]
    loc = locate(args, args.tune_split, args.tune_input)
    print(f"[{method_name}] tuning {len(cfgs)} configurations on {args.tune_docs} docs of "
          f"{loc['files']} ({args.tune_metric} F1)", file=sys.stderr)

    total, n, t0 = np.zeros(len(cfgs)), 0, time.time()
    init = (module.score_sentences, cfgs, summary_cfg, needs_reference, args.tune_metric, args.rouge_stemmer)
    with Pool(args.workers, initializer=_init_tuner, initargs=init) as pool:
        for res in pool.imap(_tune_work, _limited(open_records(loc), 0, args.tune_docs),
                             chunksize=max(1, args.chunksize // 4)):
            total += res
            n += 1
            if n % 100 == 0:
                print(f"  tuning: {n} docs  {n / (time.time() - t0):.1f} docs/s", file=sys.stderr, flush=True)
    means = total / max(n, 1)
    best = int(np.argmax(means))

    os.makedirs(args.out, exist_ok=True)
    ranked = sorted(range(len(cfgs)), key=lambda k: -means[k])
    with open(os.path.join(args.out, "tuning.json"), "w") as f:
        json.dump({"metric": args.tune_metric, "documents": n, "files": loc["files"],
                   "best": grid[best], "best_f1": means[best],
                   "results": [{"params": grid[k], "f1": means[k]} for k in ranked]}, f, indent=2)
    print(f"[{method_name}] tuning done ({n} docs, {time.time() - t0:.0f}s). Top configurations:")
    for k in ranked[:5]:
        print(f"    {args.tune_metric} F1={means[k]:.4f}  {grid[k]}")
    print(f"[{method_name}] using {grid[best]}")
    return cfgs[best], {"tuned": {"metric": args.tune_metric, "documents": n,
                                  "best": grid[best], "best_f1": means[best]}}


# =============================================================================
# Command line for a baseline module
# =============================================================================
def cli(module, method_name, defaults=None, argv=None):
    """Parse shared + method flags, then run the full pipeline for `module`."""
    p = build_parser(module.__doc__, default_out=os.path.join("results", method_name))
    module.add_args(p)
    if defaults:
        p.set_defaults(**defaults)
    args = p.parse_args(argv)
    method_cfg = config_from_args(module.Config, args)
    summary_cfg = config_from_args(SummaryConfig, args)
    needs_reference = getattr(module, "NEEDS_REFERENCE", False)
    if summary_cfg.splitter == "nltk":
        ensure_nltk([("tokenizers/punkt_tab", "punkt_tab")])
    module.prepare(method_cfg, args)      # fail fast / fetch resources the method needs

    extra = None
    if args.tune:
        if not hasattr(module, "tune_grid"):
            sys.exit(f"{method_name} has no hyperparameters to tune")
        method_cfg, extra = tune(args, method_name, module, method_cfg, summary_cfg, needs_reference)
    return run(args, method_name, module.score_sentences, method_cfg, summary_cfg,
               needs_reference=needs_reference, extra=extra)
