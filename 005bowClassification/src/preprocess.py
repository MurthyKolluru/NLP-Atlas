import re, string
from typing import List, Dict
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure NLTK resources (first run will download)
try:
    _ = stopwords.words("english")
except:
    nltk.download("stopwords")
try:
    _ = nltk.corpus.wordnet.ensure_loaded()
except:
    nltk.download("wordnet")
    nltk.download("omw-1.4")

NEGATION_SET = {"no","nor","not","n't","never","none","nobody","nothing","neither","nowhere"}

def clean_text(text:str, lowercase=True, remove_urls=True, remove_html=True, remove_punct=True):
    if lowercase:
        text = text.lower()
    if remove_urls:
        text = re.sub(r"http\S+|www\.\S+", " ", text)
    if remove_html:
        text = re.sub(r"<.*?>", " ", text)
    if remove_punct:
        text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text:str) -> List[str]:
    # keep alphabetic tokens
    return re.findall(r"[a-zA-Z]+", text)

def remove_stopwords(tokens:List[str], keep_negations=True):
    sw = set(stopwords.words("english"))
    if keep_negations:
        sw = sw - NEGATION_SET
    return [t for t in tokens if t not in sw]

def replace_synonyms(tokens:List[str], synmap:Dict[str,str]):
    if not synmap:
        return tokens
    return [synmap.get(t, t) for t in tokens]

def lemmatize(tokens:List[str]):
    lemm = WordNetLemmatizer()
    return [lemm.lemmatize(t) for t in tokens]

def preprocess_pipeline(text:str, cfg) -> str:
    t = clean_text(
        text,
        lowercase=cfg["preprocess"]["lowercase"],
        remove_urls=cfg["preprocess"]["remove_urls"],
        remove_html=cfg["preprocess"]["remove_html"],
        remove_punct=cfg["preprocess"]["remove_punct"]
    )
    toks = tokenize(t)
    toks = remove_stopwords(toks, keep_negations=cfg["preprocess"]["keep_negations"])
    toks = replace_synonyms(toks, cfg["preprocess"].get("synonym_map", {}))
    if cfg["preprocess"]["lemmatize"]:
        toks = lemmatize(toks)
    return " ".join(toks)

# -------- Picklable callable for scikit pipeline --------
class Preprocessor:
    def __init__(self, cfg):
        self.cfg = cfg
    def __call__(self, txt:str) -> str:
        return preprocess_pipeline(txt, self.cfg)
