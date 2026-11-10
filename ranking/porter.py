"""Porter (1980) stemmer, copied unchanged from HW4 (tfidf_ir.py) so the pipeline has no nltk dependency for stemming."""

class PorterStemmer:
    """Standard Porter (1980) stemming algorithm."""

    def _cons(self, w, i):
        c = w[i]
        if c in "aeiou":
            return False
        if c == "y":
            return i == 0 or not self._cons(w, i - 1)
        return True

    def _m(self, stem):
        # number of VC sequences
        n, i, L = 0, 0, len(stem)
        while i < L and self._cons(stem, i):
            i += 1
        while i < L:
            while i < L and not self._cons(stem, i):
                i += 1
            if i >= L:
                break
            while i < L and self._cons(stem, i):
                i += 1
            n += 1
        return n

    def _has_vowel(self, stem):
        return any(not self._cons(stem, i) for i in range(len(stem)))

    def _double_c(self, w):
        return len(w) >= 2 and w[-1] == w[-2] and self._cons(w, len(w) - 1)

    def _cvc(self, w):
        if len(w) < 3:
            return False
        return (self._cons(w, len(w) - 3) and not self._cons(w, len(w) - 2)
                and self._cons(w, len(w) - 1) and w[-1] not in "wxy")

    def _replace(self, w, rules, cond):
        for suf, rep in rules:
            if w.endswith(suf):
                stem = w[: len(w) - len(suf)]
                return stem + rep if cond(stem) else w
        return w

    def stem(self, w):
        if len(w) <= 2:
            return w
        # step 1a
        if w.endswith("sses"):
            w = w[:-2]
        elif w.endswith("ies"):
            w = w[:-2]
        elif w.endswith("ss"):
            pass
        elif w.endswith("s"):
            w = w[:-1]
        # step 1b
        flag = False
        if w.endswith("eed"):
            if self._m(w[:-3]) > 0:
                w = w[:-1]
        elif w.endswith("ed") and self._has_vowel(w[:-2]):
            w, flag = w[:-2], True
        elif w.endswith("ing") and self._has_vowel(w[:-3]):
            w, flag = w[:-3], True
        if flag:
            if w.endswith(("at", "bl", "iz")):
                w += "e"
            elif self._double_c(w) and w[-1] not in "lsz":
                w = w[:-1]
            elif self._m(w) == 1 and self._cvc(w):
                w += "e"
        # step 1c
        if w.endswith("y") and self._has_vowel(w[:-1]):
            w = w[:-1] + "i"
        # step 2
        w = self._replace(w, [
            ("ational", "ate"), ("tional", "tion"), ("enci", "ence"), ("anci", "ance"),
            ("izer", "ize"), ("abli", "able"), ("alli", "al"), ("entli", "ent"),
            ("eli", "e"), ("ousli", "ous"), ("ization", "ize"), ("ation", "ate"),
            ("ator", "ate"), ("alism", "al"), ("iveness", "ive"), ("fulness", "ful"),
            ("ousness", "ous"), ("aliti", "al"), ("iviti", "ive"), ("biliti", "ble"),
        ], lambda s: self._m(s) > 0)
        # step 3
        w = self._replace(w, [
            ("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"),
            ("ical", "ic"), ("ful", ""), ("ness", ""),
        ], lambda s: self._m(s) > 0)
        # step 4
        for suf in ("al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement",
                    "ment", "ent", "ion", "ou", "ism", "ate", "iti", "ous", "ive", "ize"):
            if w.endswith(suf):
                stem = w[: len(w) - len(suf)]
                if self._m(stem) > 1 and (suf != "ion" or stem.endswith(("s", "t"))):
                    w = stem
                break
        # step 5a
        if w.endswith("e"):
            stem = w[:-1]
            m = self._m(stem)
            if m > 1 or (m == 1 and not self._cvc(stem)):
                w = stem
        # step 5b
        if self._m(w) > 1 and self._double_c(w) and w.endswith("l"):
            w = w[:-1]
        return w


