"""
prepare_dataset.py
------------------
Purpose:
    - Load AG News dataset (train split) using Hugging Face `datasets`
    - Randomly sample 30,000 docs
    - Normalize/clean fields and write to JSONL for Elasticsearch bulk indexing

Why JSONL:
    - Elasticsearch _bulk API expects action/metadata and source per line (NDJSON)
    - We emit ONLY the document JSON here; bulk loader will add the action lines

Notes:
    - If you already have a CSV, you can pass --csv path/to/file.csv and we will read that instead.
    - Categories in AG News are integers {0..3}; we map them to strings: ["World","Sports","Business","Sci/Tech"].
    - AG News lacks real timestamps; we optionally synthesize a pseudo date if needed later.

Usage:
    python ingest/prepare_dataset.py --out data/processed/agnews_30k.jsonl --size 30000
    # Optional (CSV fallback):
    python ingest/prepare_dataset.py --csv data/raw/ag_news.csv --out data/processed/agnews_30k.jsonl --size 30000
"""

# =========================
# IMPORTS
# =========================
from __future__ import annotations

import argparse                     # CLI args parsing
import json                         # JSON serialization (we'll write compact lines)
import random                       # sampling reproducibly
from pathlib import Path            # safe filesystem paths
from typing import Optional         # typing hints

from bs4 import BeautifulSoup       # simple HTML strip
from datasets import load_dataset   # AG News loader
from datetime import datetime       # (optional) synth timestamp
from dateutil.relativedelta import relativedelta  # (optional) date math
from tqdm import tqdm               # progress bars

# =========================
# CONSTANTS
# =========================
CAT_MAP = {                         # AG News label → human-readable category
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech",
}

DEFAULT_SIZE = 30000                # how many documents to sample
RNG_SEED = 42                       # deterministic sampling for repeatability


def html_strip(text: str) -> str:
    """
    Remove any HTML tags/entities safely.
    - BeautifulSoup is robust for messy inputs.
    - Also normalizes whitespace (strip + single spaces).
    """
    if not text:
        return ""
    cleaned = BeautifulSoup(text, "html.parser").get_text(separator=" ", strip=True)
    # Collapse excessive internal whitespace
    return " ".join(cleaned.split())


def normalize_record(i: int, title: str, desc: str, label: int) -> dict:
    """
    Create a normalized document:
      - id: stable string id ("agnews-{index}")
      - title/body: cleaned text
      - category: mapped string
      - url: empty for now (kept for display/extensibility)
      - published_at: OPTIONAL synthesized date (commented out by default)
    """
    title = html_strip(title)
    body = html_strip(desc)

    # Skip records that are effectively empty (defensive)
    if not title and not body:
        return {}

    doc = {
        "id": f"agnews-{i}",
        "title": title,
        "body": body,
        "category": CAT_MAP.get(label, "Unknown"),
        "url": "",
        # "published_at": synth_timestamp(i),  # uncomment if you want dates
    }
    return doc


def synth_timestamp(i: int) -> str:
    """
    OPTIONAL: Create a pseudo timestamp spaced by minutes from a base date.
    - Useful if you plan to add a recency boost later.
    """
    base = datetime(2021, 1, 1, 0, 0, 0)
    ts = base + relativedelta(minutes=i)
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def load_ag_news_as_rows() -> list[tuple[int, str, str, int]]:
    """
    Use Hugging Face datasets to load AG News (train split).
    Returns a list of tuples: (row_index, title, description, label)

    Why tuple list?
      - Simple and serializable, easy to sample and iterate.
    """
    ds = load_dataset("ag_news", split="train")  # ~120K rows
    rows: list[tuple[int, str, str, int]] = []
    for idx, item in enumerate(ds):
        # Dataset fields are: {"text": "...", "label": int}
        # text is typically "Title: ...\nDescription: ..."
        text = item.get("text") or ""
        label = int(item.get("label", -1))

        # Heuristic split: AG News often has title on first line, rest is body
        parts = text.split("\n", 1)
        title = parts[0].replace("Title: ", "").strip()
        desc = parts[1].replace("Description: ", "").strip() if len(parts) > 1 else ""

        rows.append((idx, title, desc, label))
    return rows


def load_csv_as_rows(csv_path: Path) -> list[tuple[int, str, str, int]]:
    """
    CSV fallback loader.
    Expects columns: id (optional), title, description/body, label (0..3 or string)
    This path is provided for flexibility if you already have a CSV dump.
    """
    import pandas as pd  # local import to keep global deps minimal

    df = pd.read_csv(csv_path)
    # Try to be flexible with column names
    title_col = next((c for c in df.columns if c.lower() in ("title", "headline")), None)
    body_col  = next((c for c in df.columns if c.lower() in ("description", "body", "text")), None)
    label_col = next((c for c in df.columns if c.lower() in ("label", "category")), None)

    if not title_col or not body_col or not label_col:
        raise ValueError("CSV must contain columns: title/headline, description/body/text, label/category")

    rows: list[tuple[int, str, str, int]] = []
    for i, r in df.iterrows():
        title = str(r[title_col]) if pd.notna(r[title_col]) else ""
        desc  = str(r[body_col])  if pd.notna(r[body_col])  else ""
        lbl   = r[label_col]
        # Map label string to AG categories if needed
        if isinstance(lbl, str):
            inv = {v.lower(): k for k, v in CAT_MAP.items()}
            lbl = inv.get(lbl.lower(), -1)
        else:
            lbl = int(lbl) if lbl == lbl else -1  # NaN guard

        rows.append((i, title, desc, lbl))
    return rows


def write_jsonl(rows: list[tuple[int, str, str, int]], out_path: Path, size: int) -> int:
    """
    Normalize, sample, and write as newline-delimited JSON.
    Returns the number of records written.
    """
    random.seed(RNG_SEED)
    if size < len(rows):
        rows = random.sample(rows, size)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for idx, title, desc, label in tqdm(rows, desc="Writing JSONL", unit="doc"):
            doc = normalize_record(idx, title, desc, label)
            if not doc:
                continue
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            written += 1
    return written


def main():
    # -------------------------
    # CLI: allow CSV fallback and custom output/size
    # -------------------------
    ap = argparse.ArgumentParser(description="Prepare AG News subset as JSONL for Elasticsearch bulk load")
    ap.add_argument("--csv", type=str, default="", help="Optional path to CSV (if not using datasets)")
    ap.add_argument("--out", type=str, default="data/processed/agnews_30k.jsonl", help="Output JSONL path")
    ap.add_argument("--size", type=int, default=DEFAULT_SIZE, help="Sample size (default: 30000)")
    args = ap.parse_args()

    out_path = Path(args.out)

    # -------------------------
    # Load rows either from CSV or datasets
    # -------------------------
    if args.csv:
        rows = load_csv_as_rows(Path(args.csv))
    else:
        rows = load_ag_news_as_rows()

    # -------------------------
    # Write JSONL
    # -------------------------
    n = write_jsonl(rows, out_path, size=args.size)
    print(f"✅ Wrote {n} documents to {out_path}")


if __name__ == "__main__":
    main()
