import sys
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))))
import numpy as np
import baselines as B
from pipeline import Budget, apply_budget, summary_text

S = ["a b c d e", "f g", "h i j", "k l m n o p q", "r"]            # 5, 2, 3, 7, 1 words
rank = [3, 0, 1, 2, 4]
assert apply_budget(rank, S, Budget("sentences", 2)) == [0, 3]
# include: keep adding until >= 10 words; the crossing sentence stays whole (7 + 5 = 12)
assert apply_budget(rank, S, Budget("words", 10, "include")) == [0, 3]
assert apply_budget(rank, S, Budget("words", 13, "include")) == [0, 1, 3]
# skip: 7, then 5 would exceed 10 -> skip, 2 fits (9), 3 would exceed, 1 fits (10)
assert apply_budget(rank, S, Budget("words", 10, "skip")) == [1, 3, 4]
# truncate: same selection as include, text cut at exactly 10 words
sel = apply_budget(rank, S, Budget("words", 10, "truncate"))
assert sum(len(l.split()) for l in summary_text(S, sel, Budget("words", 10, "truncate")).split("\n")) == 10
assert Budget("words", 18).covers(S) and not Budget("words", 17).covers(S) and Budget("sentences", 5).covers(S)
# baselines: full ranking with n=None, document-order top-n otherwise
sc = np.array([0.1, 0.9, 0.5, 0.9, 0.0])
assert B.top(sc, None) == [1, 3, 2, 0, 4] and B.top(sc, 2) == [1, 3]
# oracle under a word budget stops once the budget is used, and "skip" never overshoots
sents = ["the cat sat on the mat today", "dogs bark loudly", "the cat sat", "on the mat", "birds fly south"]
ref = ["the cat sat on the mat"]
assert B.oracle(sents, ref, n=1) == [0]
inc = B.oracle(sents, ref, max_words=4)
assert sum(len(sents[i].split()) for i in inc[:-1]) < 4 <= sum(len(sents[i].split()) for i in inc)
sk = B.oracle(sents, ref, max_words=4, overflow="skip")
assert sum(len(sents[i].split()) for i in sk) <= 4 and sk, sk
print("budget tests pass:", inc, sk)
