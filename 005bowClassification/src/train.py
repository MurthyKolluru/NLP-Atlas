from pathlib import Path
import argparse
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier

from utils import load_config, load_raw, load_target_names
from preprocess import Preprocessor  # changed import

def make_vectorizer(cfg):
    pp = Preprocessor(cfg)  # picklable callable
    feats = cfg["features"]
    return TfidfVectorizer(
        preprocessor=pp,
        tokenizer=str.split,
        ngram_range=tuple(feats["ngram_range"]),
        max_df=feats["max_df"],
        min_df=feats["min_df"],
        max_features=feats.get("max_features")
    )

def make_selector(cfg):
    fs = cfg.get("feature_selection", {})
    if fs.get("use_chi2", False):
        return SelectKBest(score_func=chi2, k=fs.get("k_best", 5000))
    return "passthrough"

def make_model(cfg):
    m = cfg["model"]
    if m["name"].lower() == "randomforest":
        return RandomForestClassifier(
            n_estimators=m.get("n_estimators", 300),
            max_depth=m.get("max_depth", None),
            max_features=m.get("max_features", "sqrt"),
            n_jobs=m.get("n_jobs", -1),
            random_state=m.get("random_state", 42)
        )
    raise ValueError("Unsupported model in config")

def top_features_from_forest(selector, vectorizer, rf, topn=30):
    feat_names = np.array(vectorizer.get_feature_names_out())
    if hasattr(selector, "get_support"):
        support_idx = selector.get_support(indices=True)
        feat_names = feat_names[support_idx]
    importances = rf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:topn]
    return list(zip(feat_names[top_idx], importances[top_idx]))

def main(cfg_path):
    cfg = load_config(cfg_path)
    df = load_raw()
    target_names = load_target_names()
    X, y = df["text"].tolist(), df["target"].values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y,
        test_size=cfg["train"]["test_size"],
        random_state=cfg["train"]["random_state"],
        stratify=y
    )

    vectorizer = make_vectorizer(cfg)
    selector = make_selector(cfg)
    clf = make_model(cfg)

    pipe = Pipeline([
        ("tfidf", vectorizer),
        ("chi2", selector),
        ("clf", clf)
    ])

    pipe.fit(X_tr, y_tr)
    preds = pipe.predict(X_te)

    print(f"Accuracy: {accuracy_score(y_te, preds):.4f}")
    print(classification_report(y_te, preds, target_names=target_names))

    # Top global TF-IDF features by RF importance
    try:
        rf = pipe.named_steps["clf"]
        sel = pipe.named_steps["chi2"]
        vect = pipe.named_steps["tfidf"]
        top = top_features_from_forest(sel, vect, rf, topn=30)
        print("\nTop features by RandomForest importance:")
        for w, imp in top:
            print(f"{w:25s} {imp:.6f}")
    except Exception as e:
        print(f"(Could not compute top features: {e})")

    Path("models").mkdir(exist_ok=True, parents=True)
    joblib.dump(pipe, "models/clf.joblib")
    print("Saved model → models/clf.joblib")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    args = p.parse_args()
    main(args.config)
