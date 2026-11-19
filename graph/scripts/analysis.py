"""Error analysis from the proposal (step 5). Everything goes to results/analysis/.

    python scripts/analysis.py inspect   results/small/cnn_dm results/small/arxiv ... --device mps
    python scripts/analysis.py graphs    results/small/cnn_dm --docs 5 --device mps
    python scripts/analysis.py positions results/small/*

inspect    (a) For each dataset, the 20 DEV docs where full_pipeline most underperforms its
               strongest simpler rival (Lead or an ablation, whichever has the best dev mean),
               at the parameters tuned on dev. Dumps the document (numbered sentences, as
               the systems saw it), both summaries, the reference and ROUGE, with a blank
               "error category" field to fill in. Re-runs BERT on dev (dev selections are
               not stored by evaluate.py).
graphs     (b) The PageTrust graph for a few short test docs: positive / negative edges with
               direction, PageTrust scores, our picks and Oracle's picks. PNGs.
positions  (c) Where Oracle's sentences sit by document decile, per dataset (from
               results.json), next to Lead's and ours (from per_doc.jsonl). PNG + CSV.

Each run folder must hold results.json and RUN_INFO.txt (written by scripts/run_logged.sh);
the exact evaluate.py command in RUN_INFO.txt is parsed so the same samples are rebuilt.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shlex
import sys
import textwrap
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "analysis"

DS_LABEL = {"cnn_dm": "CNN/DM", "arxiv": "arXiv", "pubmed": "PubMed", "tldr": "tldr-17",
            "wikihow": "WikiHow", "billsum": "BillSum"}
DS_ORDER = ["cnn_dm", "tldr", "wikihow", "arxiv", "pubmed", "billsum"]
RIVALS = ["lead", "abl_position", "abl_positive_pagerank", "abl_degree"]

# Colors: reference palette of the dataviz skill (light mode). Polarity (positive vs negative
# edges) uses the diverging blue/red pair; identity (systems) uses categorical slots 1-3.
INK, INK2, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#8a8984", "#fcfcfb", "#e4e3df"
POS, NEG = "#2a78d6", "#e34948"
SERIES = {"oracle": "#2a78d6", "lead": "#eb6834", "full_pipeline": "#1baf7a"}
OURS_RING, ORACLE_MARK = "#eb6834", "#6250d6"


# --------------------------------------------------------------------------- run folders

def run_config(run: Path) -> dict:
    info = (run / "RUN_INFO.txt").read_text()
    cmd = shlex.split(re.search(r"^command: (.*)$", info, re.M).group(1))
    def opt(name, cast=str, default=None, many=False):
        if name not in cmd:
            return default
        k = cmd.index(name)
        if not many:
            return cast(cmd[k + 1])
        vals = []
        for t in cmd[k + 1:]:
            if t.startswith("--"):
                break
            vals.append(cast(t))
        return vals
    res = json.loads((run / "results.json").read_text())
    return {
        "results": res,
        "datasets": list(res),
        "limit_dev": opt("--limit-dev", int, 200), "limit_test": opt("--limit-test", int, 1000),
        "max_sentences": opt("--max-sentences", int), "betas": opt("--betas", float, many=True),
        "seed": opt("--seed", int, 0), "model": opt("--model", str, "bert-base-uncased"),
        "nsp_max_length": opt("--nsp-max-length", int, 256), "wikihow_csv": opt("--wikihow-csv"),
        "clauses": "--clauses" in cmd, "overflow": opt("--overflow", str, "include"),
        # runs from before splitting.py have no "splitting" entry: they used spaCy throughout
        "splitting": next(iter(res.values())).get(
            "splitting", {"splitter": "spacy", "clause_splitter": "spacy", "aggregate": "mean"}),
    }


def budget_of(res: dict, overflow: str):
    from pipeline import Budget
    m = re.match(r"(\d+) (sentences|words)", res.get("budget", f"{res['n']} sentences"))
    return Budget(m.group(2), int(m.group(1)), overflow)


def load_docs(cfg, ds, split, nlp, encoder):
    from data import load_split
    from pipeline import prepare
    limit = cfg["limit_dev"] if split == "dev" else cfg["limit_test"]
    sp = cfg["splitting"]
    exs = load_split(ds, split, limit, nlp, cfg["seed"], cfg["wikihow_csv"], sp["splitter"])
    docs = [prepare(e, nlp, encoder, cfg["max_sentences"], sp["splitter"], sp["clause_splitter"])
            for e in exs]
    return [d for d in docs if d.sentences]


_ENC = {}


def encoder_for(cfg, device):
    from bert_graph import FrozenBert
    key = (cfg["model"], cfg["nsp_max_length"], device)
    if key not in _ENC:
        _ENC[key] = FrozenBert(cfg["model"], device=device, batch_size=32, max_length=cfg["nsp_max_length"])
    return _ENC[key]


def fix_params(p: dict) -> dict:
    return {k: (math.inf if v in ("inf", "Infinity") else v) for k, v in p.items()}


# --------------------------------------------------------------------------- (a) inspection

def cmd_inspect(args):
    import spacy
    from evaluate import RougeCache
    from pipeline import build_systems, summary_text
    nlp = spacy.load("en_core_web_sm")
    out = OUT / "inspection"
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    for run in map(Path, args.runs):
        cfg = run_config(run)
        enc = encoder_for(cfg, args.device)
        systems = build_systems(clauses=cfg["clauses"], seed=cfg["seed"], betas=cfg["betas"],
                                aggregate=cfg["splitting"]["aggregate"])
        for ds in cfg["datasets"]:
            res = cfg["results"][ds]
            budget = budget_of(res, cfg["overflow"])
            print(f"[{ds}] rebuilding {cfg['limit_dev']} dev docs ({budget})", file=sys.stderr)
            dev = load_docs(cfg, ds, "dev", nlp, enc)
            rouge = RougeCache(budget)
            names = ["full_pipeline"] + [r for r in RIVALS if r in res["systems"]]
            sel, sc = {}, {}
            for name in names:
                params = fix_params(res["systems"][name]["params"])
                sel[name] = [systems[name].select(d, budget, **params) for d in dev]
                sc[name] = np.array([rouge(d, s) for d, s in zip(dev, sel[name])])
            means = {k: float(v.mean()) for k, v in sc.items()}
            rival = max((r for r in names if r != "full_pipeline"), key=lambda r: means[r])
            gap = sc[rival].mean(1) - sc["full_pipeline"].mean(1)     # > 0: rival better on this doc
            worst = np.argsort(-gap, kind="stable")[: args.top]
            oracle_p = fix_params(res["systems"]["oracle"]["params"]) if "oracle" in res["systems"] else {}
            rows, md = [], [f"# Targeted inspection: {DS_LABEL.get(ds, ds)} (dev)\n",
                            f"Strongest simpler rival on dev: **{rival}** (dev mean ROUGE "
                            f"{means[rival]:.2f} vs ours {means['full_pipeline']:.2f}). "
                            f"Ours tuned: {res['systems']['full_pipeline']['params']}; rival tuned: "
                            f"{res['systems'][rival]['params']}. Budget: {budget}. "
                            f"Ours is worse than the rival on {int((gap > 0).sum())}/{len(dev)} dev docs; "
                            f"listed: the {len(worst)} largest gaps.\n",
                            "Sentence markers: **O** ours, **R** rival, **★** Oracle.\n"]
            for rank, k in enumerate(worst, 1):
                d = dev[k]
                orc = systems["oracle"].select(d, budget, **oracle_p)
                ours_s, riv_s = sel["full_pipeline"][k], sel[rival][k]
                rec = {"dataset": ds, "rank": rank, "doc": d.id, "n_sentences": len(d.sentences),
                       "gap": round(float(gap[k]), 2), "rival": rival,
                       "ours_selected": ours_s, "rival_selected": riv_s, "oracle_selected": orc,
                       "ours_rouge": sc["full_pipeline"][k].round(2).tolist(),
                       "rival_rouge": sc[rival][k].round(2).tolist(),
                       "ours_summary": summary_text(d.sentences, ours_s, budget),
                       "rival_summary": summary_text(d.sentences, riv_s, budget),
                       "reference": d.ex.reference, "sentences": d.sentences, "error_category": ""}
                rows.append(rec)
                md += [f"\n---\n\n## {rank}. `{d.id}` — gap {gap[k]:.2f} ROUGE "
                       f"({len(d.sentences)} sentences)\n",
                       "| | R-1 | R-2 | R-L |", "|---|---:|---:|---:|",
                       "| ours | " + " | ".join(f"{v:.2f}" for v in sc["full_pipeline"][k]) + " |",
                       f"| {rival} | " + " | ".join(f"{v:.2f}" for v in sc[rival][k]) + " |",
                       "\n**Error category:** ______\n",
                       "**Reference**\n", *[f"> {r}" for r in d.ex.reference],
                       f"\n**Ours** (sentences {ours_s})\n", *[f"> {d.sentences[i]}" for i in ours_s],
                       f"\n**{rival}** (sentences {riv_s})\n", *[f"> {d.sentences[i]}" for i in riv_s],
                       "\n<details><summary>Document</summary>\n"]
                for i, s in enumerate(d.sentences):
                    marks = "".join(m for m, on in (("O", i in ours_s), ("R", i in riv_s), ("★", i in orc)) if on)
                    md.append(f"{i:>3} {marks:<4} {s}  ")
                md.append("\n</details>")
            (out / f"{ds}.md").write_text("\n".join(md) + "\n")
            with open(out / f"{ds}.jsonl", "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            with open(out / f"{ds}_labels.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["rank", "doc", "gap", "ours_rouge_mean", "rival", "rival_rouge_mean", "error_category", "notes"])
                for r in rows:
                    w.writerow([r["rank"], r["doc"], r["gap"], round(float(np.mean(r["ours_rouge"])), 2), rival,
                                round(float(np.mean(r["rival_rouge"])), 2), "", ""])
            summary.append({"dataset": ds, "rival": rival, "dev_mean_ours": round(means["full_pipeline"], 2),
                            "dev_mean_rival": round(means[rival], 2),
                            "docs_ours_worse": int((gap > 0).sum()), "dev_docs": len(dev)})
            print(f"[{ds}] wrote {out / (ds + '.md')}  (rival {rival})", file=sys.stderr)
            for d in dev:
                d.drop_bert_cache()
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


# --------------------------------------------------------------------------- (b) graphs

def cmd_graphs(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx
    import spacy
    from matplotlib.lines import Line2D
    from bert_graph import GraphConfig, sentence_scores
    from pipeline import ALPHA, EPS, MAX_ITER, build_systems
    from rankers import pagetrust

    run = Path(args.runs[0])
    cfg = run_config(run)
    ds = cfg["datasets"][0]
    res = cfg["results"][ds]
    budget = budget_of(res, cfg["overflow"])
    nlp = spacy.load("en_core_web_sm")
    enc = encoder_for(cfg, args.device)
    systems = build_systems(seed=cfg["seed"], betas=cfg["betas"])
    p = fix_params(res["systems"]["full_pipeline"]["params"])
    test = load_docs(cfg, ds, "test", nlp, enc)
    short = sorted([d for d in test if args.min_sents <= len(d.sentences) <= args.max_sents],
                   key=lambda d: (len(d.sentences), d.id))[: args.docs]
    out = OUT / "graphs"
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": INK,
                         "axes.edgecolor": GRID, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
    index = []
    for d in short:
        G = d.graph(False, GraphConfig(beta=p["beta"]))
        W = G.W
        x = pagetrust(W, alpha=ALPHA, gamma=p["gamma"], memory=p["memory"], eps=EPS, max_iter=MAX_ITER)
        ss = sentence_scores(x, G.vertices, cfg["splitting"]["aggregate"])
        ours = systems["full_pipeline"].select(d, budget, **p)
        orc = systems["oracle"].select(d, budget)
        n = len(d.sentences)
        g = nx.DiGraph()
        g.add_nodes_from(range(n))
        pos_e = [(i, j, W[i, j]) for i in range(n) for j in range(n) if i != j and W[i, j] > 0]
        neg_e = [(i, j) for i in range(n) for j in range(i + 1, n) if W[i, j] < 0]  # symmetric: draw once
        g.add_weighted_edges_from(pos_e)
        # document order clockwise from 12 o'clock
        lay = {i: (math.sin(2 * math.pi * i / n), math.cos(2 * math.pi * i / n)) for i in range(n)}

        fig = plt.figure(figsize=(13, 6.6))
        ax = fig.add_axes([0.01, 0.06, 0.47, 0.84])
        tx = fig.add_axes([0.50, 0.06, 0.49, 0.84])
        ax.set_axis_off(); tx.set_axis_off()
        wmax = max([w for *_, w in pos_e], default=1.0) or 1.0
        for i, j in neg_e:   # negative: undirected (both ways), dashed red
            ax.plot(*zip(lay[i], lay[j]), color=NEG, lw=1.0, ls=(0, (4, 3)), alpha=0.75, zorder=1)
        if pos_e:
            nx.draw_networkx_edges(g, lay, ax=ax, edgelist=[(i, j) for i, j, _ in pos_e],
                                   width=[0.4 + 2.2 * w / wmax for *_, w in pos_e],
                                   edge_color=POS, alpha=0.55, arrows=True, arrowstyle="-|>",
                                   arrowsize=9, connectionstyle="arc3,rad=0.12", node_size=900)
        smax = ss.max() if ss.max() > 0 else 1.0
        sizes = [250 + 1400 * s / smax for s in ss]
        nx.draw_networkx_nodes(g, lay, ax=ax, node_size=sizes, node_color="#f0efec",
                               edgecolors=[OURS_RING if i in ours else MUTED for i in range(n)],
                               linewidths=[3.0 if i in ours else 0.8 for i in range(n)])
        nx.draw_networkx_labels(g, lay, ax=ax, labels={i: str(i) for i in range(n)}, font_size=8, font_color=INK)
        for i in orc:
            xx, yy = lay[i]
            ax.scatter([xx * 1.17], [yy * 1.17], marker="*", s=130, color=ORACLE_MARK, zorder=5,
                       edgecolors=SURFACE, linewidths=0.8)
        ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3); ax.set_aspect("equal")
        handles = [Line2D([], [], color=POS, lw=2, label="positive edge i → j (i endorses j)"),
                   Line2D([], [], color=NEG, lw=1.2, ls=(0, (4, 3)), label="negative edge (NSP rejects both orders)"),
                   Line2D([], [], marker="o", ls="", markerfacecolor="#f0efec", markeredgecolor=OURS_RING,
                          markeredgewidth=2.5, markersize=10, label="our pick (ring)"),
                   Line2D([], [], marker="*", ls="", color=ORACLE_MARK, markersize=11, label="Oracle pick"),
                   Line2D([], [], marker="o", ls="", markerfacecolor="#f0efec", markeredgecolor=MUTED,
                          markersize=7, label="node size = PageTrust score")]
        ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, -0.08), fontsize=7.5,
                  frameon=False, ncol=2)
        fig.suptitle(f"{DS_LABEL.get(ds, ds)} test doc {d.id.split('-')[-1]}: PageTrust graph "
                     f"(β={p['beta']}, γ={'∞' if math.isinf(p['gamma']) else p['gamma']}, M={p['memory']})",
                     x=0.01, ha="left", fontsize=11, fontweight="bold")
        fig.text(0.01, 0.915, f"{n} sentences · {len(pos_e)} positive, {len(neg_e)} negative edges · "
                 f"ours {ours}, Oracle {orc}, Lead {list(range(min(n, budget.size)))}" if budget.kind == "sentences"
                 else f"{n} sentences · {len(pos_e)} positive, {len(neg_e)} negative edges · ours {ours}, Oracle {orc}",
                 color=INK2, fontsize=8.5)
        y = 1.0
        for i, s in enumerate(d.sentences):
            mark = ("O" if i in ours else " ") + ("★" if i in orc else " ")
            wrapped = textwrap.wrap(s, 92)[:2]
            if len(textwrap.wrap(s, 92)) > 2:
                wrapped[-1] = wrapped[-1][:88] + " …"
            color = INK if (i in ours or i in orc) else INK2
            tx.text(0.0, y, f"{i:>2} {mark}  {ss[i]:.3f}", fontsize=7.5, family="DejaVu Sans Mono",
                    color=color, va="top")
            tx.text(0.2, y, "\n".join(wrapped), fontsize=7.5, color=color, va="top")
            y -= 0.022 + 0.0255 * len(wrapped)
        tx.text(0.0, 1.035, " #  pick  score    sentence (O ours, ★ Oracle)", fontsize=7.5,
                family="DejaVu Sans Mono", color=INK2, va="bottom")
        ref = " ".join(d.ex.reference)
        tx.text(0.0, max(y - 0.02, -0.02), "Reference: " + "\n".join(textwrap.wrap(ref, 110)[:4]),
                fontsize=7.5, color=INK2, va="top", style="italic")
        path = out / f"{ds}_{d.id.split('-')[-1]}.png"
        fig.savefig(path, dpi=170)
        plt.close(fig)
        index.append({"doc": d.id, "png": str(path.relative_to(ROOT)), "n": n, "ours": ours, "oracle": orc,
                      "positive_edges": len(pos_e), "negative_edges": len(neg_e),
                      "scores": [round(float(v), 4) for v in ss]})
        print(f"wrote {path}", file=sys.stderr)
    (out / "index.json").write_text(json.dumps(index, indent=1))


# --------------------------------------------------------------------------- (c) positions

def cmd_positions(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = {}
    for run in map(Path, args.runs):
        if not (run / "results.json").exists():
            continue
        res = json.loads((run / "results.json").read_text())
        per = [json.loads(l) for l in open(run / "per_doc.jsonl")] if (run / "per_doc.jsonl").exists() else []
        for ds, r in res.items():
            if ds not in DS_LABEL or "oracle_positions" not in r:
                continue
            rows = {"oracle": r["oracle_positions"]["deciles"]}
            for sysname in ("lead", "full_pipeline"):
                rel = [i / max(p["n_sentences"] - 1, 1) for p in per
                       if p["dataset"] == ds and p["system"] == sysname for i in p["selected"]]
                if rel:
                    h, _ = np.histogram(rel, bins=10, range=(0, 1))
                    rows[sysname] = (h / h.sum()).round(3).tolist()
            data[ds] = rows
    if not data:
        raise SystemExit("no results with oracle_positions")
    out = OUT
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "positions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "system"] + [f"decile_{k + 1}" for k in range(10)])
        for ds, rows in data.items():
            for s, v in rows.items():
                w.writerow([ds, s] + v)
    dss = [d for d in DS_ORDER if d in data]
    cols = min(3, len(dss))
    rws = math.ceil(len(dss) / cols)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": INK,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.edgecolor": GRID, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
    fig, axes = plt.subplots(rws, cols, figsize=(max(4.2 * cols, 6.4), 3.0 * rws + 1.1), squeeze=False, sharey=True)
    label = {"oracle": "Oracle", "lead": "Lead", "full_pipeline": "Ours"}
    order = ["oracle", "full_pipeline", "lead"]
    bw = 0.27
    for k, ds in enumerate(dss):
        ax = axes[k // cols][k % cols]
        present = [s for s in order if s in data[ds]]
        for m, s in enumerate(present):
            xs = np.arange(10) + (m - (len(present) - 1) / 2) * (bw + 0.02)
            ax.bar(xs, data[ds][s], width=bw, color=SERIES[s], label=label[s], zorder=2,
                   edgecolor=SURFACE, linewidth=1.0)
        ax.set_title(DS_LABEL[ds], loc="left", fontsize=10, fontweight="bold", color=INK)
        ax.set_xticks(range(10), [f"{10 * i}" for i in range(10)], fontsize=7.5)
        ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(length=0)
        if k % cols == 0:
            ax.set_ylabel("share of picked sentences")
        if k // cols == rws - 1:
            ax.set_xlabel("relative position in document (%, decile start)")
    for k in range(len(dss), rws * cols):
        axes[k // cols][k % cols].set_axis_off()
    h, l = axes[0][0].get_legend_handles_labels()
    top = 1 - 0.75 / fig.get_figheight()            # room for a title row and a legend row
    fig.suptitle("Where selected sentences sit in the document (test)", x=0.01, y=0.995, ha="left",
                 va="top", fontsize=11, fontweight="bold")
    fig.legend(h, l, loc="upper left", ncol=3, frameon=False, fontsize=9,
               bbox_to_anchor=(0.0, 1 - 0.32 / fig.get_figheight()))
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(out / "positions.png", dpi=170)
    print(f"wrote {out / 'positions.png'} and {out / 'positions.csv'}")
    print(json.dumps(data, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["inspect", "graphs", "positions"])
    ap.add_argument("runs", nargs="+", help="run folders (results.json + RUN_INFO.txt)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--top", type=int, default=20, help="inspect: docs per dataset")
    ap.add_argument("--docs", type=int, default=5, help="graphs: how many docs")
    ap.add_argument("--min-sents", type=int, default=7, help="graphs: shortest doc to draw")
    ap.add_argument("--max-sents", type=int, default=14, help="graphs: longest doc to draw")
    args = ap.parse_args()
    {"inspect": cmd_inspect, "graphs": cmd_graphs, "positions": cmd_positions}[args.what](args)


if __name__ == "__main__":
    main()
