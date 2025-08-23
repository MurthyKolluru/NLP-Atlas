import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell("""# 00 — Explore: Data, Preprocessing, and TF-IDF Features

This notebook shows:
- Peek at the dataset & class distribution
- Run preprocessing on a sample
- Build TF-IDF features
- Show **top chi² terms per class** (most indicative features)
- (Optional) Evaluate a trained model"""))

cells.append(nbf.v4.new_code_cell("""# Setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
import joblib

from utils import load_config, load_raw, load_target_names
from preprocess import Preprocessor

cfg = load_config("configs/config.yaml")
df = load_raw("data/raw/20news.jsonl")
target_names = load_target_names("data/raw/target_names.json")
print(df.shape, target_names)"""))

cells.append(nbf.v4.new_code_cell("""# Class distribution
df['target_name'] = df['target'].map(lambda i: target_names[i])
ax = df['target_name'].value_counts().plot(kind='bar', rot=45)
_ = plt.title('Class Distribution')
plt.tight_layout()
plt.show()"""))

cells.append(nbf.v4.new_code_cell("""# Preprocessing demo
pp = Preprocessor(cfg)
ex = df.iloc[0]['text']
print("RAW:\\n", ex[:600], "...\\n")
print("\\nPREPROCESSED:\\n", pp(ex)[:600], "...")"""))

cells.append(nbf.v4.new_code_cell("""# Build TF-IDF features
feats = cfg['features']
vectorizer = TfidfVectorizer(
    preprocessor=Preprocessor(cfg),
    tokenizer=str.split,
    ngram_range=tuple(feats['ngram_range']),
    max_df=feats['max_df'],
    min_df=feats['min_df'],
    max_features=feats.get('max_features')
)
X = vectorizer.fit_transform(df['text'])
y = df['target'].values
X.shape"""))

cells.append(nbf.v4.new_code_cell("""# Top chi² features per class
def top_chi2_per_class(X, y, feature_names, class_index, topk=20):
    mask = (y == class_index).astype(int)
    chi2_scores, _ = chi2(X, mask)
    idx = np.argsort(chi2_scores)[-topk:][::-1]
    return [(feature_names[i], chi2_scores[i]) for i in idx]

feature_names = np.array(vectorizer.get_feature_names_out())
for i, name in enumerate(target_names):
    top = top_chi2_per_class(X, y, feature_names, i, topk=20)
    print(f"\\nTop terms for class: {name}")
    for term, score in top:
        print(f"{term:25s} {score:.3f}")"""))

cells.append(nbf.v4.new_code_cell("""# (Optional) Evaluate trained model
from sklearn.metrics import accuracy_score, classification_report
try:
    pipe = joblib.load("models/clf.joblib")
    X_tr, X_te, y_tr, y_te = train_test_split(df['text'].tolist(), y,
                                              test_size=cfg['train']['test_size'],
                                              random_state=cfg['train']['random_state'],
                                              stratify=y)
    preds = pipe.predict(X_te)
    print("Accuracy:", round(accuracy_score(y_te, preds), 4))
    print(classification_report(y_te, preds, target_names=target_names))
except Exception as e:
    print("Train the model first with python src/train.py. Error:", e)"""))

nb["cells"] = cells
nbf.write(nb, "notebooks/00_explore.ipynb")
print("Wrote notebooks/00_explore.ipynb")
