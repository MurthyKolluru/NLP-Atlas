from __future__ import annotations
import re
from typing import List
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

_stop = set(stopwords.words("english"))
_lem = WordNetLemmatizer()

# Minimal demo synonym list (extend as needed)
_SYNONYMS = {
    "pc": "computer",
    "computers": "computer",
    "hockeys": "hockey",
    "meds": "medicine",
    "physician": "doctor",
}

def _normalize_token(tok: str) -> str:
    tok = tok.lower()
    tok = re.sub(r"[^a-z0-9]+", "", tok)
    return tok

def preprocess(texts: List[str]) -> List[str]:
    out = []
    for txt in texts:
        txt = re.sub(r"\s+", " ", txt).strip()
        tokens = [t for t in word_tokenize(txt)]
        normed = []
        for t in tokens:
            t = _normalize_token(t)
            if not t or t in _stop or t.isdigit():
                continue
            t = _SYNONYMS.get(t, t)
            # Try verb, noun, adj, adv lemmatization
            for pos in ("v", "n", "a", "r"):
                t = _lem.lemmatize(t, pos=pos)
            if t and t not in _stop:
                normed.append(t)
        out.append(" ".join(normed))
    return out
