from pathlib import Path
import yaml, json, pandas as pd

def load_config(path="configs/config.yaml"):
    return yaml.safe_load(open(path, "r", encoding="utf-8"))

def load_raw(path="data/raw/20news.jsonl"):
    return pd.read_json(path, lines=True)

def load_target_names(path="data/raw/target_names.json"):
    return json.loads(Path(path).read_text(encoding="utf-8"))
