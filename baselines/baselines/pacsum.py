#!/usr/bin/env python3
"""
PacSum (Zheng & Lapata, 2019, "Sentence Centrality Revisited for Unsupervised
Summarization"), reimplemented from the authors' code (github.com/mswellhao/PacSum,
code/extractor.py) so results match theirs step by step.

Per document:
  1. Similarity matrix E (diagonal = 1, as in their code)
       --rep tfidf : gensim preprocessing, idf over the document's sentences,
                     E_ij = sum_w tf_i(w) tf_j(w) idf(w)^2          (--sim dot, PacSum's code)
                     or its length-normalised cosine version          (--sim cosine, our variant)
       --rep bert  : E_ij = <v_i, v_j>, v = pooled [CLS] output of their fine-tuned
                     BERT for each sentence ([CLS] s [SEP], max 60 word pieces)
  2. Threshold (Eq. 5-6):  t = min(E) + beta * (max(E) - min(E));  keep E_ij - t if > 0
  3. Directed centrality (Eq. 2), as implemented in their _select_tops:
       centrality(s_i) = lambda1 * ( - sum_{j<i} e_ij ) + lambda2 * sum_{j>i} e_ij
     (their code negates the "similarity to previous sentences" term, so lambda1 >= 0
      here corresponds to the paper's negative lambda1)
  4. Top sentences by centrality, ties broken at random (their code shuffles);
     documents with <= k sentences are returned whole.

Hyperparameters (their run.py defaults: beta=0, lambda1=0, lambda2=1) are tuned like
the paper: --tune searches beta in {0,.1,..,1} x lambda1 in {0,.1,..,1}, lambda2 =
1 - lambda1, on the first 1,000 validation documents, maximising ROUGE-1 F.

    python pacsum.py --dataset cnn_dailymail --split test --data-root ../datasets \
        --num-sentences 3 --no-rouge-stemmer --rep tfidf --tune

BERT needs their fine-tuned model (README: drive.google.com/file/d/1wbMlLmnbD_0j7Qs8YY8cSCh935WKKdsP),
torch and transformers, ideally a GPU, and --workers 1:
    python pacsum.py ... --rep bert --bert-model path/to/pacsum_model.bin --workers 1 --tune

Loading, sentence splitting, summary assembly and ROUGE come from summarize.py.
"""

import sys
from dataclasses import dataclass

import numpy as np

import summarize
import textrank


@dataclass
class Config:
    rep: str = "tfidf"               # tfidf | bert
    sim: str = "dot"                 # tfidf only: dot (PacSum's code) | cosine (length-normalised variant)
    beta: float = 0.0                # their run.py defaults
    lambda1: float = 0.0
    lambda2: float = 1.0
    extract_num: int = 3             # set from --num-sentences by prepare()
    seed: int = 0                    # tie-breaking (their code shuffles before sorting)
    bert_model: str = ""             # PacSum's fine-tuned BertModel state dict
    bert_config: str = "bert-base-uncased"   # their default: uncased_L-12_H-768_A-12
    bert_vocab: str = "bert-base-uncased"
    max_len: int = 60                # their Dataset._max_len
    device: str = "auto"             # auto | cuda | cpu
    batch_size: int = 64


# =============================================================================
# Similarity matrices
# =============================================================================
_CACHE = {"key": None, "E": None}       # last document's matrix (reused across the tuning grid)


def _tfidf_matrix(sentences, sim="dot"):
    toks = textrank.tokenize_sentences(sentences, textrank.GraphConfig(preprocess="gensim"))
    n = len(toks)
    # their _calculate_idf_scores / _idf_modified_dot, vectorised (or its cosine-normalised form)
    E = textrank._tfidf_similarity(toks, textrank.GraphConfig(
        similarity="tfidf_cosine" if sim == "cosine" else "tfidf_dot", min_tokens=0))
    # textrank's version clips negatives (impossible here: idf is squared) and zeroes the diagonal;
    # PacSum sets the diagonal to 1
    np.fill_diagonal(E, 1.0)
    return E if n else np.zeros((0, 0))


_BERT = {}


def _bert_encoder(cfg):
    if "model" not in _BERT:
        import torch
        from transformers import BertConfig, BertModel, BertTokenizer
        device = cfg.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        config = (BertConfig.from_json_file(cfg.bert_config) if cfg.bert_config.endswith(".json")
                  else BertConfig.from_pretrained(cfg.bert_config))
        model = BertModel(config, add_pooling_layer=True)
        state = torch.load(cfg.bert_model, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        # their checkpoint uses the original pytorch_pretrained_bert names (LayerNorm.gamma/beta)
        renamed = {}
        for k, v in state.items():
            k = k[len("bert."):] if k.startswith("bert.") else k
            k = k.replace("LayerNorm.gamma", "LayerNorm.weight").replace("LayerNorm.beta", "LayerNorm.bias")
            renamed[k] = v
        missing, unexpected = model.load_state_dict(renamed, strict=False)
        missing = [m for m in missing if not m.endswith("position_ids")]
        if missing or unexpected:
            sys.exit(f"PacSum BERT checkpoint does not match BertModel.\n missing: {missing[:10]}\n"
                     f" unexpected: {unexpected[:10]}")
        if cfg.bert_vocab.endswith(".txt"):
            tok = BertTokenizer(cfg.bert_vocab, do_lower_case=True)
        else:
            tok = BertTokenizer.from_pretrained(cfg.bert_vocab, do_lower_case=True)
        model.to(device).eval()
        _BERT.update(model=model, tok=tok, device=device, torch=torch)
    return _BERT


def _bert_matrix(sentences, cfg):
    b = _bert_encoder(cfg)
    torch, tok, model = b["torch"], b["tok"], b["model"]
    ids = []
    for s in sentences:
        pieces = tok.tokenize(s)[: cfg.max_len - 2]
        ids.append(tok.convert_tokens_to_ids(["[CLS]"] + pieces + ["[SEP]"]))
    vecs = []
    with torch.no_grad():
        for i in range(0, len(ids), cfg.batch_size):
            batch = ids[i:i + cfg.batch_size]
            width = max(len(x) for x in batch)
            x = torch.zeros(len(batch), width, dtype=torch.long)
            m = torch.zeros(len(batch), width, dtype=torch.long)
            for r, seq in enumerate(batch):
                x[r, :len(seq)] = torch.tensor(seq)
                m[r, :len(seq)] = 1
            out = model(input_ids=x.to(b["device"]), attention_mask=m.to(b["device"]),
                        token_type_ids=torch.zeros_like(x).to(b["device"]))
            vecs.append(out.pooler_output.float().cpu().numpy())
    V = np.concatenate(vecs) if vecs else np.zeros((0, 1))
    E = V @ V.T                 # their logits: <pooled(s_i), pooled(s_j)> (same encoder at test time)
    np.fill_diagonal(E, 1.0)    # their pairdown(): "1 for self score"
    return E


def similarity(sentences, cfg):
    key = (cfg.rep, cfg.sim, tuple(sentences))
    if _CACHE["key"] != key:
        _CACHE["key"] = key
        _CACHE["E"] = _bert_matrix(sentences, cfg) if cfg.rep == "bert" else _tfidf_matrix(sentences, cfg.sim)
    return _CACHE["E"]


# =============================================================================
# Directed, position-augmented centrality (their _select_tops / _compute_scores)
# =============================================================================
def score_sentences(sentences, cfg):
    n = len(sentences)
    if n <= cfg.extract_num:                                # their extract_summary(): whole article
        return np.arange(n, 0, -1, dtype=float), {"iterations": 0, "converged": True}
    E = similarity(sentences, cfg)
    t = E.min() + cfg.beta * (E.max() - E.min())
    U = np.triu(E - t, k=1)
    U[U <= 0] = 0
    forward = U.sum(axis=0)            # sum over earlier sentences j < i
    backward = U.sum(axis=1)           # sum over later sentences j > i
    central = cfg.lambda1 * (-forward) + cfg.lambda2 * backward

    # their code: random.shuffle, then a stable sort by score -> random tie-breaking
    order = np.random.default_rng(cfg.seed).permutation(n)
    ranked = order[np.argsort(-central[order], kind="stable")]
    scores = np.empty(n)
    scores[ranked] = np.arange(n, 0, -1)
    return scores, {"iterations": 0, "converged": True}


def tune_grid(cfg):
    """Their _tune_extractor grid: 11 x 11 = 121 configurations."""
    return [{"beta": k / 10, "lambda1": i / 10, "lambda2": 1 - i / 10}
            for k in range(11) for i in range(11)]


# =============================================================================
# Command-line hooks
# =============================================================================
def add_args(parser):
    d = Config()
    g = parser.add_argument_group("PacSum")
    g.add_argument("--rep", choices=["tfidf", "bert"], default=d.rep)
    g.add_argument("--sim", choices=["dot", "cosine"], default=d.sim,
                   help="tf-idf similarity: dot (PacSum's code, default) or length-normalised cosine")
    g.add_argument("--beta", type=float, default=d.beta)
    g.add_argument("--lambda1", type=float, default=d.lambda1)
    g.add_argument("--lambda2", type=float, default=d.lambda2)
    g.add_argument("--seed", type=int, default=d.seed)
    g.add_argument("--bert-model", default=d.bert_model, help="PacSum fine-tuned BERT checkpoint (--rep bert)")
    g.add_argument("--bert-config", default=d.bert_config, help="config .json or HF model name")
    g.add_argument("--bert-vocab", default=d.bert_vocab, help="vocab.txt or HF model name")
    g.add_argument("--max-len", type=int, default=d.max_len)
    g.add_argument("--device", choices=["auto", "cuda", "cpu"], default=d.device)
    g.add_argument("--batch-size", type=int, default=d.batch_size)


def prepare(cfg, args=None):
    if args is not None:
        # Fixed-k extraction (PacSum's setup): documents with <= k sentences are returned whole.
        # With a word budget instead (--num-sentences 0 --word-limit N, as HipoRank ran PacSum on
        # arXiv/PubMed) that rule does not apply, so it is switched off.
        cfg.extract_num = max(args.num_sentences, 0)
    if cfg.rep == "tfidf":
        textrank._lazy("gensim")
    else:
        if not cfg.bert_model:
            sys.exit("--rep bert needs --bert-model (PacSum's fine-tuned checkpoint)")
        try:
            import torch, transformers  # noqa: F401
        except ImportError:
            sys.exit("--rep bert needs torch and transformers: pip install torch transformers")
        if args is not None and args.workers > 1:
            print("[PacSum] note: --rep bert loads BERT in every worker; use --workers 1 (GPU) "
                  "unless you have memory to spare.", file=sys.stderr)


if __name__ == "__main__":
    summarize.cli(sys.modules[__name__], "pacsum")
