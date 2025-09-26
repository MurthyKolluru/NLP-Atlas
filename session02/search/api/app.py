# api/app.py
"""
FastAPI thin layer over Elasticsearch (BM25) for teaching/demo purposes.

Endpoints
---------
GET /search
    - q: query string (required)
    - cat: optional category filter (keyword match)
    - from: offset (default 0)
    - size: page size (default 10)
    - Returns: hits with _id, _score, title, snippet (highlight if available)

GET /doc/{doc_id}
    - Returns the full stored document (title, body, category, url)

GET /explain/{doc_id}
    - q: query string (required)
    - Returns Elasticsearch _explain output (teaching visibility into BM25)

GET /suggest
    - prefix: the text a user is typing (required)
    - size: number of suggestions (default 5)
    - Uses the "title.autocomplete" subfield (edge-ngrams) for lightweight suggest

Notes
-----
- This is intentionally compact and "production-ish", not production-hardened:
  timeouts, retries, and input validation are sensible but minimal.
- CORS is enabled for localhost so Streamlit can call these endpoints later.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# -----------------------------
# Config (env or defaults)
# -----------------------------
ES_URL = os.getenv("ES_URL", "http://localhost:9200")   # single-node dev ES
INDEX  = os.getenv("ES_INDEX", "agnews_v1")             # our index name
HTTP_TIMEOUT = 10                                       # seconds for ES calls

app = FastAPI(title="Mid-Sized Search API", version="0.1.0")

# Allow local UIs (Streamlit/React) to call the API during dev.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Helpers
# -----------------------------
def es_post(path: str, body: Dict[str, Any]) -> Dict[str, Any]:
    """POST helper with small timeout and basic error mapping."""
    url = f"{ES_URL}{path}"
    try:
        r = requests.post(url, json=body, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Elasticsearch error: {e}") from e


def es_get(path: str) -> Dict[str, Any]:
    """GET helper with small timeout and basic error mapping."""
    url = f"{ES_URL}{path}"
    try:
        r = requests.get(url, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Elasticsearch error: {e}") from e


# -----------------------------
# Routes
# -----------------------------
@app.get("/search")
def search(
    q: str = Query(..., description="User query text"),
    cat: Optional[str] = Query(None, description="Optional category filter (keyword)"),
    from_: int = Query(0, alias="from", ge=0),
    size: int = Query(10, ge=1, le=50),
):
    """
    Execute a BM25 search over title^2 + body with optional category filter.
    - Highlights 'body' to produce a readable snippet.
    - Uses multi_match to combine fields and give the title a boost.
    - Applies a keyword filter on 'category' if provided.
    """
    must: List[Dict[str, Any]] = [
        {
            "multi_match": {
                "query": q,
                "fields": ["title^2", "body"],
                "type": "best_fields",
                "operator": "and",         # slightly stricter matching
            }
        }
    ]
    filter_clause: List[Dict[str, Any]] = []
    if cat:
        filter_clause.append({"term": {"category": cat}})

    body = {
        "from": from_,
        "size": size,
        "query": {
            "bool": {
                "must": must,
                "filter": filter_clause
            }
        },
        "highlight": {
            "pre_tags": ["<mark>"],
            "post_tags": ["</mark>"],
            "fields": {
                "body": {"fragment_size": 150, "number_of_fragments": 1}
            }
        },
        "_source": ["title", "body", "category", "url"],  # keep response small
    }

    res = es_post(f"/{INDEX}/_search", body)

    hits_out: List[Dict[str, Any]] = []
    for h in res.get("hits", {}).get("hits", []):
        src = h.get("_source", {})
        # Prefer highlight snippet if present; else fall back to start of body.
        snippet = None
        hl = h.get("highlight", {})
        if "body" in hl and hl["body"]:
            snippet = hl["body"][0]
        else:
            b = (src.get("body") or "")[:200]
            snippet = b + ("..." if len(b) == 200 else "")

        hits_out.append({
            "id": h.get("_id"),
            "score": h.get("_score"),
            "title": src.get("title"),
            "snippet": snippet,
            "category": src.get("category"),
            "url": src.get("url"),
        })

    return {
        "total": res.get("hits", {}).get("total", {}).get("value", 0),
        "took_ms": res.get("took", 0),
        "hits": hits_out,
    }


@app.get("/doc/{doc_id}")
def get_doc(doc_id: str):
    """
    Fetch the full stored document by its _id.
    """
    res = es_get(f"/{INDEX}/_doc/{doc_id}")
    if not res.get("found", False):
        raise HTTPException(status_code=404, detail="Document not found")
    out = res.get("_source", {})
    out["id"] = res.get("_id")
    return out


@app.get("/explain/{doc_id}")
def explain(doc_id: str, q: str = Query(..., description="Query to explain against")):
    """
    Return Elasticsearch _explain for a given document and query.
    - Very useful for teaching BM25 contributions and clause matches.
    """
    body = {
        "query": {
            "multi_match": {
                "query": q,
                "fields": ["title^2", "body"],
                "type": "best_fields",
                "operator": "and",
            }
        }
    }
    return es_post(f"/{INDEX}/_explain/{doc_id}", body)


@app.get("/suggest")
def suggest(prefix: str = Query(..., min_length=2), size: int = Query(5, ge=1, le=20)):
    """
    Lightweight prefix suggest using the 'title.autocomplete' subfield (edge-ngrams).
    - We query with match_phrase_prefix to leverage the subfield.
    """
    body = {
        "size": size,
        "_source": ["title"],
        "query": {
            "match_phrase_prefix": {
                "title.autocomplete": {
                    "query": prefix
                }
            }
        }
    }
    res = es_post(f"/{INDEX}/_search", body)
    suggestions = [h.get("_source", {}).get("title") for h in res.get("hits", {}).get("hits", [])]
    # Deduplicate while preserving order (tiny lists)
    dedup: List[str] = []
    seen = set()
    for s in suggestions:
        if s and s not in seen:
            seen.add(s); dedup.append(s)
    return {"prefix": prefix, "suggestions": dedup[:size]}
