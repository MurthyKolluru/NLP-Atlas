# ingest/bulk_load.py
"""
Bulk loader for Elasticsearch (_bulk API) with:
- Chunking (default 2000 docs per request)
- Retries on 429/5xx with exponential backoff
- Progress bars + basic error logging
- Post-load count verification

Usage:
  python ingest/bulk_load.py --index agnews_v1 --input data/processed/agnews_30k.jsonl --batch 2000

Notes:
- Assumes Elasticsearch is reachable at http://localhost:9200 with security disabled (dev).
- Input file is plain JSONL: one doc JSON per line (no action lines). We add the "index" action line.
- Document IDs are expected inside each JSON as "id"; if absent, we auto-generate.
"""

from __future__ import annotations

import argparse              # to parse CLI flags cleanly
import json                  # serialize/deserialize lines
import math                  # for ceil calculations on batches
import time                  # for sleep during backoff
from pathlib import Path     # path-safe file handling
from typing import List      # type hints for clarity

import requests              # HTTP client for ES
from tqdm import tqdm        # nice progress bars


ES_URL = "http://localhost:9200"   # dev default endpoint


def iter_jsonl(path: Path):
    """
    Stream JSON objects line-by-line from a JSONL file.
    - Yields dicts.
    - Skips blank lines defensively.
    """
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def make_bulk_payload(index: str, docs: List[dict]) -> str:
    """
    Build NDJSON string for the _bulk API:
      { "index": {"_index": "index", "_id": "ID"} }
      { ...document body... }
    - Requires an "_id"; we use doc["id"] if present else synthesize one.
    """
    lines = []
    for i, d in enumerate(docs):
        _id = d.get("id") or f"{index}-{int(time.time()*1000)}-{i}"
        action = {"index": {"_index": index, "_id": _id}}
        lines.append(json.dumps(action, ensure_ascii=False))
        lines.append(json.dumps(d, ensure_ascii=False))
    return "\n".join(lines) + "\n"  # bulk requires trailing newline


def bulk_send(index: str, docs: List[dict], max_retries: int = 5) -> tuple[int, int]:
    """
    Send one bulk chunk with retries.
    Returns: (success_count, error_count) for items in this chunk.
    Retries on 429/5xx with exponential backoff: 1s, 2s, 4s, ...
    """
    url = f"{ES_URL}/_bulk"
    payload = make_bulk_payload(index, docs)
    headers = {"Content-Type": "application/x-ndjson"}

    delay = 1.0
    for attempt in range(max_retries):
        resp = requests.post(url, data=payload.encode("utf-8"), headers=headers, timeout=60)
        status = resp.status_code

        # Retry on 429 (too many requests) or 5xx (server hiccups)
        if status in (429,) or 500 <= status <= 599:
            time.sleep(delay)
            delay = min(delay * 2, 16)  # cap backoff
            continue

        # Parse response, count item-level errors
        try:
            data = resp.json()
        except Exception:
            # If ES returns non-JSON, treat whole chunk as failure
            return (0, len(docs))

        if not data.get("errors"):
            return (len(docs), 0)

        # Some docs failed at item level; count precisely
        ok, bad = 0, 0
        for item in data.get("items", []):
            res = item.get("index", {})
            if 200 <= res.get("status", 500) < 300:
                ok += 1
            else:
                bad += 1
        return (ok, bad)

    # If we exhausted retries without a non-429/5xx status, count none as loaded
    return (0, len(docs))


def count_index(index: str) -> int:
    """
    Ask ES for the current document count.
    """
    url = f"{ES_URL}/{index}/_count"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return int(r.json().get("count", 0))


def main():
    # ---------- CLI ----------
    ap = argparse.ArgumentParser(description="Bulk-load JSONL into Elasticsearch")
    ap.add_argument("--index", required=True, help="Target index name (e.g., agnews_v1)")
    ap.add_argument("--input", required=True, help="Path to JSONL with docs")
    ap.add_argument("--batch", type=int, default=2000, help="Bulk chunk size (default: 2000)")
    args = ap.parse_args()

    index = args.index
    in_path = Path(args.input)
    batch = max(1, args.batch)

    assert in_path.exists(), f"Input file not found: {in_path}"

    # ---------- Read total lines to size the progress bar (cheap scan) ----------
    total_lines = sum(1 for _ in in_path.open("r", encoding="utf-8"))
    total_batches = math.ceil(total_lines / batch)

    print(f"Target index      : {index}")
    print(f"Input file        : {in_path}  ({total_lines:,} lines)")
    print(f"Batch size        : {batch}")
    print(f"Estimated batches : {total_batches}")

    # ---------- Bulk loop ----------
    ok_total, bad_total = 0, 0
    docs_buf: list[dict] = []
    pbar = tqdm(total=total_lines, unit="doc", desc="Bulk indexing", leave=True)

    for doc in iter_jsonl(in_path):
        docs_buf.append(doc)
        if len(docs_buf) >= batch:
            ok, bad = bulk_send(index, docs_buf)
            ok_total += ok
            bad_total += bad
            pbar.update(len(docs_buf))
            docs_buf.clear()

    # Flush remaining docs (last partial chunk)
    if docs_buf:
        ok, bad = bulk_send(index, docs_buf)
        ok_total += ok
        bad_total += bad
        pbar.update(len(docs_buf))
        docs_buf.clear()

    pbar.close()

    # ---------- Post-check ----------
    es_count = count_index(index)
    print(f"✅ Bulk summary: ok={ok_total:,}  errors={bad_total:,}")
    print(f"✅ ES _count   : {es_count:,} docs in index '{index}'")

    # Basic acceptance hint (optional)
    if bad_total == 0 and es_count >= ok_total:
        print("🎯 Looks good: all chunks acknowledged. Proceed to API layer.")
    else:
        print("⚠️ Some items failed or counts mismatch. Check Docker logs and retry bulk on failed docs if needed.")


if __name__ == "__main__":
    main()
