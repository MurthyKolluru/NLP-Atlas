
import re
from typing import List, Iterable
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# -- Contraction map (compact but effective) ----------------------------------
_CONTRACTIONS = {
    "n't": " not", "'re": " are", "'s": " is", "'ll": " will",
    "'d": " would", "'ve": " have", "'m": " am"
}

# -- Domain extras common in 20NG dumps (even after header removal) -----------
STOPWORDS_EXTRA = {
    "ax","lines","subject","organization","organisations","article","writes","wrote",
    "edu","com","ca","uk","gmt","max","etc","thanks","reply","email","phone",
    "use","used","using","like","just"
}
STOPWORDS = set(ENGLISH_STOP_WORDS) | STOPWORDS_EXTRA

_PUNCT = re.compile(r"[^\w\s]")           # drop punctuation
_MULTI_WS = re.compile(r"\s+")
_DIGIT_WS = re.compile(r"\s{2,}")

def _expand_contractions(text: str) -> str:
    # Replace common contractions; simple pass is enough for our case
    for k, v in _CONTRACTIONS.items():
        text = text.replace(k, v)
    return text

def tokenize_clean(text: str) -> List[str]:
    """
    Normalize -> expand contractions -> strip punctuation -> split -> filter.
    - lowercased
    - contractions expanded (don't -> do not)
    - punctuation removed
    - tokens len>=2
    - english stopwords + domain extras removed
    """
    if not isinstance(text, str):
        return []
    t = text.lower()
    t = _expand_contractions(t)
    t = _PUNCT.sub(" ", t)
    t = _MULTI_WS.sub(" ", t).strip()
    # split and filter
    toks = [w for w in t.split() if len(w) >= 2 and w not in STOPWORDS]
    return toks

def clean_for_vectorizer(text: str) -> str:
    """Return a single space-joined string (good for TF-IDF / embeddings)."""
    return " ".join(tokenize_clean(text))

def clean_corpus_to_tokens(docs: Iterable[str]) -> List[List[str]]:
    return [tokenize_clean(d) for d in docs]

def clean_corpus_to_strings(docs: Iterable[str]) -> List[str]:
    return [" ".join(tokenize_clean(d)) for d in docs]
