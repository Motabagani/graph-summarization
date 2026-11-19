"""Build a tiny random BERT (with NSP head) whose vocab covers the synthetic data."""
import re, sys, os, torch
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))))
from transformers import BertConfig, BertForNextSentencePrediction, BertTokenizerFast
from data import synthetic
d = sys.argv[1] if len(sys.argv) > 1 else "/tmp/tinybert"
os.makedirs(d, exist_ok=True)
text = " ".join(e.text + " " + " ".join(e.reference) for e in synthetic(50) + synthetic(50, 1))
words = sorted(set(re.findall(r"[a-z]+|[^\sa-z]", text.lower())))
open(f"{d}/vocab.txt", "w").write("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + words))
BertTokenizerFast(vocab_file=f"{d}/vocab.txt").save_pretrained(d)
torch.manual_seed(0)
BertForNextSentencePrediction(BertConfig(vocab_size=5 + len(words), hidden_size=32, num_hidden_layers=2,
                              num_attention_heads=2, intermediate_size=64)).save_pretrained(d)
print("tiny bert at", d)
