# eval/run_eval.py
"""
Tiny evaluation harness:
- Reads qrels TSV with columns: query <TAB> doc_id <TAB> gain
- Calls the FastAPI /search endpoint for each query
- Aligns top-K results to qrels to compute P@K, AP, nDCG@K, MAP
Usage:
    python eval/run_eval.py --qrels data/qrels/qrels.tsv --k 10 --size 50
"""
from __future__ import annotations
import argparse, requests
from collections import defaultdict
from typing import Dict, List, Tuple
from metrics import precision_at_k, average_precision, ndcg_at_k, mean_average_precision

API = "http://127.0.0.1:8000"

def read_qrels(path: str) -> Tuple[List[str], Dict[str, Dict[str, float]]]:
    """
    Returns:
      queries: list of unique query strings in order of appearance
      rels: mapping query -> {doc_id -> gain}
    """
    queries, rels = [], defaultdict(dict)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            q, doc_id, gain = line.split("\t")
            gain = float(gain)
            if q not in rels:
                queries.append(q)
            rels[q][doc_id] = gain
    return queries, rels

def search_api(q: str, size: int) -> List[str]:
    """Return ranked list of doc_ids from API."""
    r = requests.get(f"{API}/search", params={"q": q, "size": size}, timeout=15)
    r.raise_for_status()
    hits = r.json().get("hits", [])
    return [h["id"] for h in hits if "id" in h]

def gains_from_ranked(ranked_ids: List[str], rels_for_q: Dict[str, float]) -> List[float]:
    """Map ranked ids to numeric gains (0 if not relevant)."""
    return [rels_for_q.get(docid, 0.0) for docid in ranked_ids]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qrels", required=True, help="TSV: query \\t doc_id \\t gain")
    ap.add_argument("--size", type=int, default=50, help="API result depth per query")
    ap.add_argument("--k", type=int, default=10, help="Cutoff K for P@K and nDCG@K")
    args = ap.parse_args()

    queries, rels = read_qrels(args.qrels)
    ap_list, ndcgs, ps = [], [], []

    print(f"Evaluating {len(queries)} queries at K={args.k} (depth={args.size})\n")
    for q in queries:
        ranked = search_api(q, args.size)
        gains = gains_from_ranked(ranked, rels[q])
        ap = average_precision(gains)
        nd = ndcg_at_k(gains, args.k)
        p = precision_at_k(gains, args.k)

        ap_list.append(ap); ndcgs.append(nd); ps.append(p)
        print(f"Q: {q}\n  P@{args.k}={p:.3f}  AP={ap:.3f}  nDCG@{args.k}={nd:.3f}")

    print("\n==== Averages ====")
    print(f"MAP={mean_average_precision([gains_from_ranked(search_api(q, args.size), rels[q]) for q in queries]):.3f}")
    print(f"mean P@{args.k}={sum(ps)/len(ps):.3f}")
    print(f"mean nDCG@{args.k}={sum(ndcgs)/len(ndcgs):.3f}")

if __name__ == "__main__":
    main()
