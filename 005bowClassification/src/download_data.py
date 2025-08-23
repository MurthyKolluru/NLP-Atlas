from sklearn.datasets import fetch_20newsgroups
import pandas as pd
from pathlib import Path
import yaml, argparse, json

def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))
    ds = cfg["dataset"]

    # Use a cache OUTSIDE the project to avoid any path/permission locks
    data_home = Path(r"C:\Users\murth\Desktop\NLP\cache\sk_cache")
    data_home.mkdir(parents=True, exist_ok=True)

    data = fetch_20newsgroups(
        subset="all",
        categories=ds.get("categories"),
        remove=tuple(ds.get("remove", [])),
        shuffle=True,
        random_state=42,
        data_home=str(data_home)
    )

    df = pd.DataFrame({"text": data.data, "target": data.target})
    target_names = data.target_names

    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)

    df.to_json(out_dir / "20news.jsonl", orient="records", lines=True, force_ascii=False)
    (out_dir / "target_names.json").write_text(json.dumps(target_names, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved {len(df)} docs to {out_dir/'20news.jsonl'} with {len(target_names)} classes.")
    print(f"sklearn cache: {data_home.resolve()}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    args = p.parse_args()
    main(args.config)
