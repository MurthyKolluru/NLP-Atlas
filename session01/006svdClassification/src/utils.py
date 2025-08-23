from __future__ import annotations
import os
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

ROOT = Path(__file__).resolve().parents[1]  # project root

def load_config():
    cfg_path = ROOT / "configs" / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dirs():
    cfg = load_config()
    models_dir = ROOT / cfg["paths"]["models_dir"]
    figures_dir = ROOT / cfg["paths"]["figures_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    return models_dir, figures_dir

def save_confusion_matrix(cm, labels, out_path: Path, title="Confusion matrix"):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def top_terms_from_coef(tfidf, svd, clf, k=15):
    """
    Approximate per-class term importances by mapping classifier weights from SVD space
    back to TF-IDF term space: term_weights ≈ Vt.T @ w_class
    """
    import numpy as np
    feature_names = np.array(tfidf.get_feature_names_out())
    Vt = svd.components_                    # [n_components, n_terms]
    W = clf.coef_                           # [n_classes, n_components] or [1, n_components]
    term_scores = W @ Vt                    # [n_classes, n_terms]
    results = []
    for c_idx, row in enumerate(term_scores):
        top_idx = np.argsort(row)[::-1][:k]
        results.append((c_idx, list(zip(feature_names[top_idx].tolist(),
                                        row[top_idx].round(4).tolist()))))
    return results
