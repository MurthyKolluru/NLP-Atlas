# Bag-of-Words Text Classification (TF-IDF → χ² → RandomForest)

A minimal, production-ish pipeline for classic document classification using **TF-IDF** features, **chi-square feature selection**, and a **RandomForest** classifier. Designed for teaching: clear preprocessing, configurable features, and simple scripts.

---

## Project Structure

bowClassification/
├─ data/
│ ├─ raw/ # downloaded/original texts (created by download script)
│ └─ processed/ # optional cache
├─ models/ # saved model (created after training)
├─ reports/
│ └─ figures/ # confusion matrix, etc.
├─ src/
│ ├─ download_data.py # fetch 20 Newsgroups subset
│ ├─ preprocess.py # clean → tokenize → stopwords → synonyms → lemmatize
│ ├─ train.py # TF-IDF → χ² (SelectKBest) → RandomForest
│ ├─ evaluate.py # confusion matrix image
│ ├─ predict.py # quick CLI inference
│ └─ utils.py # config + IO helpers
├─ configs/
│ └─ config.yaml # all knobs in one place
├─ notebooks/
│ └─ 00_explore.ipynb # optional: EDA & feature exploration
├─ requirements.txt
└─ README.md


---

## Setup (Windows / PowerShell)

```powershell
# from bowClassification/
py -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

Data

We use a clean subset (4 classes) of 20 Newsgroups shipped with scikit-learn.

python src\download_data.py --config configs\config.yaml
# outputs data\raw\20news.jsonl and data\raw\target_names.json

Train (TF-IDF → χ² → RandomForest)
python src\train.py --config configs\config.yaml
# prints accuracy, per-class precision/recall/F1
# also prints "Top features by RandomForest importance"
# saves the model to models\clf.joblib

Why RandomForest for text?
Students like trees. We cap TF-IDF features (max_features) and apply chi² selection (k_best) to keep it fast and accurate on sparse, high-dimensional text.

Evaluate (Confusion Matrix)
python src\evaluate.py --config configs\config.yaml
# writes reports\figures\confusion_matrix.png

Predict (Try your own text)
python src\predict.py --text "The shuttle reached low earth orbit and NASA confirmed the satellite deployment."

Teaching Notes

Preprocessing order: clean → tokenize → stopwords → synonym replacement → lemmatize → TF-IDF.

Keep negations (e.g., “not”) — improves polarity/topic signals.

Feature control: tune features.max_features, features.ngram_range, and feature_selection.k_best.

Swap the model in configs/config.yaml (e.g., LinearSVC or LogisticRegression) to compare vs. RandomForest (optional).

Troubleshooting

If downloads feel “stuck” in cloud-synced folders (OneDrive), keep the project on local disk.

If joblib.dump fails: we use a picklable Preprocessor class (no lambdas) — ensure src\preprocess.py is the provided version.

If plots don’t show in notebook, run cells individually and check the kernel is .venv.


---

## Step 4 — Save the file
Press **Ctrl+S**.

---

## Step 5 — (Optional) Preview the Markdown
Press **Ctrl+Shift+V** in VS Code to open the Markdown preview and check formatting.

---
