# ops/smoke_tests.py
"""
Python smoke tests (cross-platform), mirrors the PowerShell scripts:
- Simple search
- Filtered search
- Suggest
- Explain (uses the first hit)
Run:
  python ops/smoke_tests.py
"""
from __future__ import annotations
import requests, sys

API = "http://127.0.0.1:8000"

def call(path, **params):
    r = requests.get(API + path, params=params, timeout=10)
    r.raise_for_status()
    return r.json()

def main():
    print(">>> Simple search")
    r = call("/search", q="stock market", size=3)
    print({k: r[k] for k in ("took_ms","total")})
    hits = r["hits"]
    for h in hits:
        print(h["id"], round(h["score"],2), h["title"][:60])

    print("\n>>> Category filter = Business")
    r = call("/search", q="stock market", cat="Business", size=3)
    print({k: r[k] for k in ("took_ms","total")})
    for h in r["hits"]:
        print(h["category"], "—", h["title"][:60])

    print("\n>>> Suggest 'mic'")
    s = call("/suggest", prefix="mic", size=5)
    print(s.get("suggestions", [])[:5])

    if hits:
        doc_id = hits[0]["id"]
        print(f"\n>>> Explain first hit: {doc_id}")
        e = call(f"/explain/{doc_id}", q="stock market")
        print("explained:", "matched" if e.get("matched") else "not matched")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Smoke failed:", e)
        sys.exit(1)
