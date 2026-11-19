import sys, tempfile, numpy as np, pandas as pd, spacy, datasets
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))))
import data
nlp = spacy.load("en_core_web_sm")
# pooled splits reproduce the proposal's sizes
for total, want in [(23455, (18764, 2345, 2346)), (3848330, (3078664, 384833, 384833))]:
    p = data._pooled_indices(total, 0)
    assert (len(p["train"]), len(p["dev"]), len(p["test"])) == want, (total, {k: len(v) for k, v in p.items()})
    assert len(np.unique(np.concatenate(list(p.values())))) == total
# references
assert data._split_reference("Line one .\nLine two", "newline", nlp) == ["Line one .", "Line two"]
assert len(data._split_reference("<S> First sent. </S> <S> Second one here. </S>", "spacy", nlp)) == 2
# HF path with a stubbed loader (official splits and pooled)
fake = lambda n, tf, rf: datasets.Dataset.from_dict({tf: [f"Doc {i}. It has two sentences." for i in range(n)],
                                                     rf: [f"Summary {i}." for i in range(n)]})
data._hf_load = lambda spec, split: fake(50, spec.text, spec.ref)
cnn = data.load_split("cnn_dm", "test", 10, nlp)
bill_dev, bill_test = data.load_split("billsum", "dev", None, nlp), data.load_split("billsum", "test", None, nlp)
assert len(cnn) == 10 and cnn[0].reference and len(bill_dev) == 15 and len(bill_test) == 15   # 3 pooled splits x 50
assert not {e.id.split("-")[-1] for e in bill_dev} & {e.id.split("-")[-1] for e in bill_test}
# WikiHow CSV path
with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
    pd.DataFrame({"headline": ["\nStep one,\nStep two,"] * 40, "title": ["t"] * 40,
                  "text": ["Do a thing. Then another."] * 40}).to_csv(f.name, index=False)
wh = data.load_split("wikihow", "test", 3, nlp, wikihow_csv=f.name, splitter="spacy")
assert len(wh) == 3 and wh[0].reference == ["Step one", "Step two"], wh[0].reference
wh = data.load_split("wikihow", "test", 3, nlp, wikihow_csv=f.name)            # team: lines kept as-is
assert wh[0].reference == ["Step one,", "Step two,"] and wh[0].dataset == "wikihow", wh[0].reference
assert cnn[0].dataset == "cnn_dm"
print("data tests pass")
