from __future__ import annotations
from pathlib import Path
import json
import joblib
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

from src.utils import load_config, ensure_dirs, top_terms_from_coef
from src.preprocess import preprocess

def build_pipeline(cfg):
    tfidf = TfidfVectorizer(
        max_features=cfg["pipeline"]["tfidf"]["max_features"],
        ngram_range=tuple(cfg["pipeline"]["tfidf"]["ngram_range"]),
        min_df=cfg["pipeline"]["tfidf"]["min_df"],
        max_df=cfg["pipeline"]["tfidf"]["max_df"]
    )
    svd = TruncatedSVD(
        n_components=cfg["pipeline"]["svd"]["n_components"],
        random_state=cfg["pipeline"]["svd"]["random_state"]
    )
    norm = Normalizer(copy=False)
    clf = LogisticRegression(**cfg["pipeline"]["classifier"]["params"])
    return Pipeline([("tfidf", tfidf), ("svd", svd), ("norm", norm), ("clf", clf)])

def main():
    cfg = load_config()
    models_dir, _ = ensure_dirs()

    cats = cfg["dataset"]["categories"]
    remove = tuple(cfg["dataset"]["remove"])
    data_home = cfg["dataset"]["data_home"]

    train = fetch_20newsgroups(subset="train", categories=cats, remove=remove, data_home=data_home)
    test  = fetch_20newsgroups(subset="test",  categories=cats, remove=remove, data_home=data_home)

    X_train = preprocess(train.data)
    X_test  = preprocess(test.data)
    y_train, y_test = train.target, test.target
    labels = train.target_names

    pipe = build_pipeline(cfg)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")

    # Save model
    model_path = models_dir / "svd_lr_pipeline.joblib"
    joblib.dump(pipe, model_path)
    print(f"Saved: {model_path}")

    # Save class names for fast prediction
    labels_path = models_dir / "target_names.json"
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False)
    print(f"Saved: {labels_path}")

    # Report
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=labels, digits=4))

    # Top terms per class (approximate back-projection)
    tfidf_step = pipe.named_steps["tfidf"]
    svd_step   = pipe.named_steps["svd"]
    clf_step   = pipe.named_steps["clf"]
    tops = top_terms_from_coef(tfidf_step, svd_step, clf_step, k=15)
    for c_idx, pairs in tops:
        print(f"\nTop terms for class '{labels[c_idx]}':")
        for term, score in pairs:
            print(f"  {term:20s} {score:+.4f}")

    # Persist top terms to reports/top_terms.txt
    _, figures_dir = ensure_dirs()
    out_txt = figures_dir.parent / "top_terms.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        for c_idx, pairs in tops:
            f.write(f"Top terms for class '{labels[c_idx]}':\n")
            for term, score in pairs:
                f.write(f"  {term:20s} {score:+.4f}\n")
            f.write("\n")
    print(f"Saved top terms: {out_txt}")

if __name__ == "__main__":
    main()
