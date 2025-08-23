from __future__ import annotations
import sys, json
from pathlib import Path
import joblib
from src.utils import load_config
from src.preprocess import preprocess

def main():
    if len(sys.argv) < 2:
        print('Usage: python -m src.predict "text 1" ["text 2" ...]')
        sys.exit(1)

    cfg = load_config()
    project_root = Path(__file__).resolve().parents[1]
    model_path = project_root / cfg["paths"]["models_dir"] / "svd_lr_pipeline.joblib"
    labels_path = project_root / cfg["paths"]["models_dir"] / "target_names.json"

    # Load model
    pipe = joblib.load(model_path)

    # Prefer labels saved during training, else fall back to config categories
    target_names = None
    if labels_path.exists():
        try:
            with open(labels_path, "r", encoding="utf-8") as f:
                target_names = json.load(f)
        except Exception:
            target_names = None
    if target_names is None:
        target_names = cfg.get("dataset", {}).get("categories", None)

    inputs = sys.argv[1:]
    preds = pipe.predict(preprocess(inputs))

    for text, idx in zip(inputs, preds):
        try:
            idx_int = int(idx)  # robust for numpy scalars
            if target_names and 0 <= idx_int < len(target_names):
                print(f"Predicted class: {idx_int} -> {target_names[idx_int]}")
            else:
                print(f"Predicted class index: {idx_int}")
        except Exception:
            print(f"Predicted class index: {idx}")
if __name__ == "__main__":
    main()
