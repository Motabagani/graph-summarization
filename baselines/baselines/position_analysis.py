#!/usr/bin/env python3
"""
Position analysis: where does summary-worthy material sit inside documents?

For each dataset, on a development sample (never the test set), it records the
relative position (sentence index / number of sentences, 0 = first, 1 = last) of

  1. ORACLE sentences  - the sentences a greedy ROUGE oracle selects against the
     reference summary (same Oracle settings as the main results). This uses the
     references: it measures where *reference-like* content is.
  2. CENTRAL sentences - the k sentences most similar to the rest of their own document
     (mean tf-idf cosine to every other sentence, idf over the document's sentences),
     with k = the number of Oracle sentences for that document. Reference-free: it
     measures where the document's *own* core content is.

Outputs (in --out):
  position_summary.md   per-dataset table: share in first/last 10%, first 3 sentences,
                        mean relative position, histogram over 10 position bins
  positions.json        all numbers, including the per-bin centrality profile
  positions.png         heatmaps of both histograms (datasets x position bins)

    python position_analysis.py --data-root ../datasets --out ../results/position_analysis
"""

import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

import summarize
import oracle
import lexpagerank

BINS = 10
# development split, docs to use, oracle settings (same as the main results)
DATASETS = {
    "cnn_dailymail": dict(split="validation", objective="rouge1",  max_sent=3),
    "pubmed":        dict(split="validation", objective="rouge2",  max_sent=30),
    "arxiv":         dict(split="validation", objective="rouge2",  max_sent=30),
    "billsum":       dict(split="train",      objective="rouge12", max_sent=30),   # no validation split
    "wikihow":       dict(split="all",        objective="rouge12", max_sent=30, max_docs=1000),  # tuning portion
    "tldr17":        dict(split="all",        objective="rouge12", max_sent=1,  max_docs=1000),  # tuning portion
}

_G = {}


def _init(ds, cfg):
    _G["ds"] = ds
    _G["summary_cfg"] = summarize.SummaryConfig(newline_boundaries=(ds == "wikihow"))
    _G["oracle_cfg"] = oracle.Config(search="greedy", objective=cfg["objective"],
                                     max_sentences=cfg["max_sent"], rouge_stemmer=False)
    _G["sim_cfg"] = lexpagerank.Config(idf_source="document")


def _doc(rec):
    """Relative positions of oracle and central sentences in one document."""
    _, text, ref = rec
    sents = summarize.prepare_sentences(text, _G["summary_cfg"])
    refs = summarize.reference_sentences(ref)
    n = len(sents)
    if n < 2 or not refs:
        return None
    scores, _ = oracle.score_sentences(sents, _G["oracle_cfg"], refs)
    chosen = [i for i in range(n) if np.isfinite(scores[i])]
    if not chosen:
        return None
    S = lexpagerank.cosine_matrix(sents, _G["sim_cfg"])
    centrality = S.sum(axis=1) / (n - 1)                          # mean cosine to the other sentences
    central = list(np.argsort(-centrality, kind="stable")[:len(chosen)])
    rel = (np.arange(n) + 0.5) / n                                 # position of each sentence's centre
    bins = np.minimum((rel * BINS).astype(int), BINS - 1)
    mean_c = centrality.mean()
    profile = np.full(BINS, np.nan)                                # centrality per bin, relative to doc mean
    if mean_c > 0:
        for b in range(BINS):
            if (bins == b).any():
                profile[b] = centrality[bins == b].mean() / mean_c
    return dict(n=n,
                oracle_rel=[float(rel[i]) for i in chosen], oracle_idx=[int(i) for i in chosen],
                central_rel=[float(rel[i]) for i in central], central_idx=[int(i) for i in central],
                profile=profile.tolist())


def _stats(rels, idxs, ns):
    rels = np.asarray(rels)
    hist = np.bincount(np.minimum((rels * BINS).astype(int), BINS - 1), minlength=BINS) / max(len(rels), 1)
    return {"selected": int(len(rels)),
            "mean_rel_position": round(float(rels.mean()), 3) if len(rels) else None,
            "share_first_10pct": round(float((rels < 0.1).mean()), 3) if len(rels) else None,
            "share_last_10pct": round(float((rels >= 0.9).mean()), 3) if len(rels) else None,
            "share_first_3_sentences": round(float((np.asarray(idxs) < 3).mean()), 3) if len(idxs) else None,
            "histogram": [round(float(h), 3) for h in hist]}


def analyse(ds, cfg, args):
    a = argparse.Namespace(dataset=ds, data_root=args.data_root, format=None,
                           text_field=None, ref_field=None, id_field=None)
    loc = summarize.locate(a, cfg["split"])
    n_docs = min(args.docs, cfg.get("max_docs", args.docs))
    o_rel, o_idx, c_rel, c_idx, ns, profiles = [], [], [], [], [], []
    with Pool(args.workers, initializer=_init, initargs=(ds, cfg)) as pool:
        for r in pool.imap(_doc, summarize._limited(summarize.open_records(loc), 0, n_docs), chunksize=8):
            if r is None:
                continue
            ns.append(r["n"]); profiles.append(r["profile"])
            o_rel += r["oracle_rel"]; o_idx += r["oracle_idx"]
            c_rel += r["central_rel"]; c_idx += r["central_idx"]
    prof = np.nanmean(np.array(profiles, dtype=float), axis=0) if profiles else np.full(BINS, np.nan)
    return {"dataset": ds, "split": cfg["split"], "files": loc["files"], "documents": len(ns),
            "mean_sentences_per_doc": round(float(np.mean(ns)), 1) if ns else None,
            "oracle_settings": {k: cfg[k] for k in ("objective", "max_sent")},
            "oracle": _stats(o_rel, o_idx, ns),
            "central": _stats(c_rel, c_idx, ns),
            "centrality_profile": [round(float(x), 3) for x in prof]}


def write_outputs(results, out):
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "positions.json"), "w") as f:
        json.dump(results, f, indent=2)

    md = ["# Where summary material sits in documents (development samples)\n",
          "Relative position: 0 = first sentence, 1 = last. With no position effect each 10% bin "
          "would hold ~10% of selections.\n",
          "\n## Summary\n",
          "| Dataset | Docs | Sent./doc | Oracle: first 10% | Oracle: last 10% | Oracle: first 3 sent. | "
          "Oracle mean pos. | Central: first 10% | Central: last 10% | Central mean pos. |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    pct = lambda x: "–" if x is None else f"{100 * x:.1f}%"
    for r in results:
        o, c = r["oracle"], r["central"]
        md.append(f"| {r['dataset']} | {r['documents']} | {r['mean_sentences_per_doc']} | {pct(o['share_first_10pct'])} | "
                  f"{pct(o['share_last_10pct'])} | {pct(o['share_first_3_sentences'])} | {o['mean_rel_position']} | "
                  f"{pct(c['share_first_10pct'])} | {pct(c['share_last_10pct'])} | {c['mean_rel_position']} |")
    for key, title in (("oracle", "Oracle sentences (reference-based)"),
                       ("central", "Most central sentences (reference-free)")):
        md += [f"\n## {title}: share per 10% position bin\n",
               "| Dataset | " + " | ".join(f"{10*b}-{10*b+10}%" for b in range(BINS)) + " |",
               "|---|" + "---|" * BINS]
        for r in results:
            md.append(f"| {r['dataset']} | " + " | ".join(pct(h) for h in r[key]["histogram"]) + " |")
    md += ["\n## Centrality profile: mean centrality per bin / document mean (1.00 = average)\n",
           "| Dataset | " + " | ".join(f"{10*b}-{10*b+10}%" for b in range(BINS)) + " |",
           "|---|" + "---|" * BINS]
    for r in results:
        md.append(f"| {r['dataset']} | " + " | ".join("–" if np.isnan(x) else f"{x:.2f}"
                                                         for x in r["centrality_profile"]) + " |")
    with open(os.path.join(out, "position_summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = [r["dataset"] for r in results]
        fig, axes = plt.subplots(1, 2, figsize=(14, 0.6 * len(names) + 2), sharey=True)
        for ax, key, title in zip(axes, ("oracle", "central"),
                                  ("Oracle sentences (reference-based)", "Most central sentences (reference-free)")):
            M = np.array([r[key]["histogram"] for r in results]) * 100
            im = ax.imshow(M, cmap="Blues", aspect="auto", vmin=0, vmax=max(40, M.max()))
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=8,
                            color="white" if M[i, j] > 25 else "black")
            ax.set_xticks(range(BINS)); ax.set_xticklabels([f"{10*b}%" for b in range(BINS)], fontsize=8)
            ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
            ax.set_xlabel("relative position in document (bin start)"); ax.set_title(title)
        fig.colorbar(im, ax=axes, label="% of selected sentences")
        fig.savefig(os.path.join(out, "positions.png"), dpi=200, bbox_inches="tight")
        print(f"\nSaved {out}/positions.png")
    except ImportError:
        print("\n(matplotlib not installed: skipped positions.png)")
    print(f"Saved {out}/position_summary.md and positions.json")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default="../datasets")
    p.add_argument("--out", default="../results/position_analysis")
    p.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    p.add_argument("--docs", type=int, default=5000, help="development documents per dataset")
    p.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)))
    args = p.parse_args()
    summarize.ensure_nltk([("tokenizers/punkt_tab", "punkt_tab")])
    results = []
    for ds in args.datasets:
        print(f"[position] {ds} ...", file=sys.stderr, flush=True)
        results.append(analyse(ds, DATASETS[ds], args))
        write_outputs(results, args.out)          # save after each dataset
    print("[position] done", file=sys.stderr)


if __name__ == "__main__":
    main()
