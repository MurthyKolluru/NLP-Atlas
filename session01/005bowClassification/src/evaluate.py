import argparse, joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from utils import load_config, load_raw, load_target_names

def main(cfg_path):
    cfg = load_config(cfg_path)
    df = load_raw()
    target_names = load_target_names()
    X, y = df["text"].tolist(), df["target"].values

    _, X_te, _, y_te = train_test_split(
        X, y,
        test_size=cfg["train"]["test_size"],
        random_state=cfg["train"]["random_state"],
        stratify=y
    )

    pipe = joblib.load("models/clf.joblib")
    y_pred = pipe.predict(X_te)

    cm = confusion_matrix(y_te, y_pred, labels=range(len(target_names)))
    disp = ConfusionMatrixDisplay(cm, display_labels=target_names)
    disp.plot(xticks_rotation=45)
    plt.tight_layout()

    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    out_path = "reports/figures/confusion_matrix.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved → {out_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    args = p.parse_args()
    main(args.config)
