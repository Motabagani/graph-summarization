"""End-to-end checks for the team pipeline. No GPU, real data or BERT download needed.

    cd pipeline && python -m unittest discover -s tests -v

1. preprocessor -> grapher (fake encoder) -> ranker runs on a small synthetic CNN/DM-style
   and PubMed-style dataset, tuning on dev and writing metrics on test.
2. Graphs rebuilt from the grapher's cache equal graph/bert_graph.py's live graphs.
3. The pipeline's Lead (--system lead) gives exactly the baselines' Lead ROUGE.
"""
import json
import os
import random
import subprocess
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PIPE = os.path.dirname(HERE)
sys.path.insert(0, PIPE)

import pipeline_common as pc  # noqa: E402

WORDS = "budget school council river energy policy patient trial drug model data result court law vote".split()


def _sent(rng):
    return " ".join(rng.choice(WORDS) for _ in range(rng.randint(6, 16))).capitalize() + "."


def make_data(root):
    import pandas as pd
    rng = random.Random(0)
    os.makedirs(f"{root}/cnn_dailymail/cnn_dailymail")
    for sp in ("validation", "test"):
        pd.DataFrame([{"id": f"{sp}{i}", "article": " ".join(_sent(rng) for _ in range(12)),
                       "highlights": "\n".join(_sent(rng) for _ in range(3))} for i in range(12)]
                     ).to_csv(f"{root}/cnn_dailymail/cnn_dailymail/{sp}.csv", index=False)
    os.makedirs(f"{root}/long_summarization/pubmed-dataset")
    for sp in ("val", "test"):
        with open(f"{root}/long_summarization/pubmed-dataset/{sp}.txt", "w") as f:
            for i in range(12):
                f.write(json.dumps({"article_id": f"{sp}{i}", "article_text": [_sent(rng) for _ in range(25)],
                                    "abstract_text": [f"<S> {_sent(rng)} </S>" for _ in range(4)]}) + "\n")


def run(script, *args):
    r = subprocess.run([sys.executable, os.path.join(PIPE, script), *args], cwd=PIPE,
                       capture_output=True, text=True, timeout=600)
    if r.returncode:
        raise AssertionError(f"{script} failed:\n{r.stderr[-3000:]}")
    return r.stdout


class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.data, cls.work = f"{cls.tmp}/data", f"{cls.tmp}/work"
        make_data(cls.data)
        for ds in ("cnn_dailymail", "pubmed"):
            for part in ("dev", "test"):
                run("preprocessor.py", "--dataset", ds, "--part", part, "--data-root", cls.data,
                    "--out", f"{cls.work}/{ds}/{part}.jsonl", "--workers", "1")
                run("grapher.py", f"{cls.work}/{ds}/{part}.jsonl", "--out", f"{cls.work}/{ds}/{part}.g",
                    "--encoder", "fake", "--beta-min", "0.3", "--docs-per-file", "5")

    def test_standard_format(self):
        for ds in ("cnn_dailymail", "pubmed"):
            for d in pc.read_docs(f"{self.work}/{ds}/test.jsonl"):
                self.assertEqual(len(d["units"]), len(d["unit_sentence"]))
                self.assertTrue(d["sentences"] and d["reference"])
                self.assertFalse(any("<S>" in r for r in d["reference"]))

    def test_end_to_end(self):
        out = f"{self.work}/pubmed/ours"
        run("ranker.py", "--dataset", "pubmed", "--dev-docs", f"{self.work}/pubmed/dev.jsonl",
            "--dev-graphs", f"{self.work}/pubmed/dev.g", "--test-docs", f"{self.work}/pubmed/test.jsonl",
            "--test-graphs", f"{self.work}/pubmed/test.g", "--gammas", "0", "1", "--memories", "0",
            "--out", out, "--workers", "1")
        m = json.load(open(f"{out}/metrics.json"))
        self.assertEqual(m["documents"], 12)
        self.assertGreater(m["rouge"]["rouge1"]["f1"], 0)
        self.assertTrue(os.path.exists(f"{out}/tuning.json"))

    def test_cache_matches_live_graph(self):
        import grapher
        import ranker
        bg = pc.import_bert_graph()
        store, enc = ranker.GraphStore(f"{self.work}/pubmed/test.g"), grapher.FakeEncoder()
        for d in list(pc.read_docs(f"{self.work}/pubmed/test.jsonl"))[:4]:
            verts = bg.vertices_from_units(d["units"], d["unit_sentence"])
            for beta in (0.3, 0.6):
                cfg = bg.GraphConfig(beta=beta)
                live = bg.build_graph(bg.DocumentScores(verts, enc), cfg).W
                cached = bg.build_graph(ranker.CachedScores(verts, *store.get(d["id"])), cfg).W
                self.assertLess(np.abs(live - cached).max(), 1e-5)

    def test_lead_matches_baselines(self):
        for ds, length in (("cnn_dailymail", ["--num-sentences", "3"]),
                           ("pubmed", ["--num-sentences", "0", "--word-limit", "203"])):
            ours = f"{self.work}/{ds}/lead"
            run("ranker.py", "--dataset", ds, "--system", "lead", "--test-docs", f"{self.work}/{ds}/test.jsonl",
                "--out", ours, "--workers", "1")
            base = f"{self.work}/{ds}/baseline_lead"
            r = subprocess.run([sys.executable, os.path.join(pc.BASELINES_DIR, "lead3.py"), "--dataset", ds,
                                "--split", "test", "--data-root", self.data, "--no-rouge-stemmer",
                                "--workers", "1", "--out", base, *length], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            a, b = (json.load(open(f"{p}/metrics.json"))["rouge"] for p in (ours, base))
            for k in ("rouge1", "rouge2", "rougeLsum"):
                self.assertAlmostEqual(a[k]["f1"], b[k]["f1"], places=10, msg=f"{ds} {k}")


if __name__ == "__main__":
    unittest.main()
