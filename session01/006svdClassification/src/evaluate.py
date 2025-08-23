from __future__ import annotations
from pathlib import Path
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import fetch_20newsgroups

from src.utils import load_config, ensure_dirs, save_confusion_matrix
from src.preprocess import preprocess

def main():
    cfg = load_config()
    _, figures_dir = ensure_dirs()

    cats = cfg["dataset"]["categories"]
    remove = tuple(cfg["dataset"]["remove"])
    data_home = cfg["dataset"]["data_home"]

    test  = fetch_20newsgroups(subset="test", categories=cats, remove=remove, data_home=data_home)
    X_test = preprocess(test.data)
    y_test = test.target
    labels = test.target_names

    model_path = (Path(__file__).resolve().parents[1] / cfg["paths"]["models_dir"] / "svd_lr_pipeline.joblib")
    pipe = joblib.load(model_path)

    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=labels, digits=4))

    cm = confusion_matrix(y_test, y_pred)
    out_png = figures_dir / "confusion_matrix.png"
    save_confusion_matrix(cm, labels, out_png, title="SVD-LogReg Confusion Matrix")
    print(f"Saved confusion matrix: {out_png}")

if __name__ == "__main__":
    main()
