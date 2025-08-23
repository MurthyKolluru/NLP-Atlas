from __future__ import annotations
from sklearn.datasets import fetch_20newsgroups
from src.utils import load_config

def main():
    cfg = load_config()
    cats = cfg["dataset"]["categories"]
    remove = tuple(cfg["dataset"]["remove"])
    data_home = cfg["dataset"]["data_home"]
    for subset in ("train", "test"):
        ds = fetch_20newsgroups(subset=subset, categories=cats, remove=remove,
                                data_home=data_home, download_if_missing=True)
        print(f"{subset}: {len(ds.data)} docs | classes={len(ds.target_names)}")
    print("OK: 20NG cached.")

if __name__ == "__main__":
    main()
