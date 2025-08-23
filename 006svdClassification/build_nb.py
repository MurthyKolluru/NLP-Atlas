import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
    "# Explore SVD Classification\n\nQuick sanity checks: load pipeline, show config, inspect SVD + top terms."
))

cells.append(nbf.v4.new_code_cell("""from pathlib import Path
import json, joblib
from src.utils import load_config
cfg = load_config()
root = Path.cwd()
model_path = root / cfg['paths']['models_dir'] / 'svd_lr_pipeline.joblib'
labels_path = root / cfg['paths']['models_dir'] / 'target_names.json'
pipe = joblib.load(model_path)
target_names = json.load(open(labels_path, 'r', encoding='utf-8'))
target_names
"""))

cells.append(nbf.v4.new_code_cell("""import matplotlib.pyplot as plt
import numpy as np
svd = pipe.named_steps['svd']
explained = svd.explained_variance_ratio_
plt.figure(figsize=(6,3))
plt.plot(np.cumsum(explained))
plt.xlabel('Components')
plt.ylabel('Cumulative explained variance')
plt.title('SVD Components — Cumulative Variance')
plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""from src.preprocess import preprocess
samples = [
    "OpenGL shaders for 3D graphics",
    "The goalie stops the puck in overtime",
    "New therapy shows effect in clinical trials",
    "Policy debate continues in national politics"
]
preds = pipe.predict(preprocess(samples))
list(zip(samples, [target_names[i] for i in preds]))
"""))

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {
    "display_name": "Python (.venv) svdClassification",
    "language": "python",
    "name": "svdClassification"
}
nb["metadata"]["language_info"] = {"name": "python"}

Path("notebooks").mkdir(parents=True, exist_ok=True)
outp = Path("notebooks/00_explore_svd.ipynb")
with outp.open("w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Wrote {outp}")
