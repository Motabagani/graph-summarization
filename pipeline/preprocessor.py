#!/usr/bin/env python3
"""
Stage 1 -- Preprocessor: raw dataset files -> one standard JSONL format.

The six datasets arrive in different shapes (CSV, Parquet, JSON lines; text as a string or as a
list of sentences; references with <S> tags, newlines or plain text). This stage reads any of them
and writes every document the same way:

    {"id": "...", "dataset": "pubmed", "part": "dev" | "test",
     "sentences":     ["...", ...],    # what a summary is built from (and what the baselines use)
     "units":         ["...", ...],    # graph vertices: the sentences, or their clauses (--unit clause)
     "unit_sentence": [0, 0, 1, ...],  # sentence index of each unit
     "unit_tokens":   [["stem", ...], ...],   # cleaned tokens per unit (for tf-idf graphs)
     "reference":     ["...", ...]}    # reference summary, one sentence per item

Which code does what:
    reading files, choosing documents, sentence splitting, reference cleanup
        -> baselines/baselines/summarize.py, so the units are exactly the baselines' sentences
    clauses and cleaned tokens
        -> ranking/preprocess.py (Hashim's Preprocessor: rule-based clauses, gensim-style tokens)

Which documents (the same ones the baselines were tuned and evaluated on):
    --part dev   the first 1,000 documents of the tuning split (validation; BillSum: train;
                 WikiHow / TL;DR-17: the first 1,000 documents of the file)
    --part test  the test split (WikiHow / TL;DR-17: every document after the first 1,000)

    python preprocessor.py --dataset pubmed --part dev  --data-root $DATA_ROOT --out work/pubmed/dev.jsonl
    python preprocessor.py --dataset pubmed --part test --data-root $DATA_ROOT --out work/pubmed/test.jsonl
    python preprocessor.py --dataset arxiv --part dev --unit clause ...        # clause vertices
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import pipeline_common as pc
import summarize
from preprocess import STOP_LIST, preprocess as hashim_preprocess   # ranking/preprocess.py

_A = {}


def _init(dataset, unit, tokens):
    _A.update(cfg=pc.summary_config(dataset), unit=unit, tokens=tokens)


def _convert(rec):
    doc_id, text, ref = rec
    sentences = summarize.prepare_sentences(text, _A["cfg"])          # = the baselines' sentences
    reference = summarize.reference_sentences(ref)
    if not sentences or not reference:
        return None
    d = hashim_preprocess(sentences, unit=_A["unit"])                  # list in -> kept as sentences
    assert d.sentences == sentences, "ranking/preprocess.py changed the sentence list"
    out = {"id": str(doc_id), "sentences": sentences, "units": d.units,
           "unit_sentence": d.unit_sentence, "reference": reference}
    if _A["tokens"]:
        out["unit_tokens"] = d.unit_tokens
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=list(pc.DATASETS))
    p.add_argument("--part", required=True, choices=["dev", "test"])
    p.add_argument("--data-root", default=os.environ.get("DATA_ROOT", "."))
    p.add_argument("--input", help="explicit file/glob instead of the dataset's usual location")
    p.add_argument("--text-field"), p.add_argument("--ref-field")
    p.add_argument("--unit", choices=["sentence", "clause"], default="sentence")
    p.add_argument("--limit", type=int, help="documents to keep (dev default: 1,000; test default: all)")
    p.add_argument("--offset", type=int, default=0, help="extra documents to skip (on top of the test offset)")
    p.add_argument("--no-tokens", action="store_true", help="omit unit_tokens (only tf-idf graphs need them)")
    p.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)))
    p.add_argument("--out", required=True, help="output .jsonl")
    a = p.parse_args(argv)

    summarize.ensure_nltk([("tokenizers/punkt_tab", "punkt_tab")])
    split, base_offset = pc.DATASETS[a.dataset][a.part]
    limit = a.limit if a.limit is not None else (pc.DEV_DOCS if a.part == "dev" else 0)
    loc = summarize.locate(argparse.Namespace(dataset=a.dataset, data_root=a.data_root, format=None,
                                              text_field=a.text_field, ref_field=a.ref_field, id_field=None),
                           split, a.input)
    records = summarize._limited(summarize.open_records(loc), base_offset + a.offset, limit)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    n = skipped = 0
    t0 = time.time()
    with open(a.out, "w", encoding="utf-8") as f, \
            Pool(a.workers, initializer=_init, initargs=(a.dataset, a.unit, not a.no_tokens)) as pool:
        for doc in pool.imap(_convert, records, chunksize=64):
            if doc is None:
                skipped += 1
                continue
            doc.update(dataset=a.dataset, part=a.part)
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            n += 1
    meta = {"dataset": a.dataset, "part": a.part, "split": split, "offset": base_offset + a.offset,
            "limit": limit, "files": loc["files"], "unit": a.unit, "documents": n,
            "skipped_empty": skipped, "stop_list": STOP_LIST, "sentence_splitter": "nltk (baselines)",
            "summary_config": vars(pc.summary_config(a.dataset)) if hasattr(pc.summary_config(a.dataset), "__dict__") else None}
    with open(a.out + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[preprocessor] {a.dataset}/{a.part}: {n} documents ({skipped} empty skipped) "
          f"in {time.time() - t0:.0f}s -> {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
