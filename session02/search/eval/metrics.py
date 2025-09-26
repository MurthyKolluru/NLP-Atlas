# eval/metrics.py
"""
Classic IR metrics with small, clear implementations:
- precision_at_k
- average_precision (AP)
- mean_average_precision (MAP)
- ndcg_at_k
All functions accept a "gain" (relevance) list aligned to the ranked list.
"""
from __future__ import annotations
from math import log2
from typing import List

def precision_at_k(gains: List[float], k: int) -> float:
    k = max(1, min(k, len(gains)))
    rel = sum(1 for g in gains[:k] if g > 0)
    return rel / k

def average_precision(gains: List[float]) -> float:
    """AP = mean of precision@k at each rank k where gain > 0."""
    num_rel, ap = 0, 0.0
    for i, g in enumerate(gains, start=1):
        if g > 0:
            num_rel += 1
            ap += num_rel / i
    return ap / num_rel if num_rel else 0.0

def mean_average_precision(list_of_gain_lists: List[List[float]]) -> float:
    if not list_of_gain_lists:
        return 0.0
    return sum(average_precision(gs) for gs in list_of_gain_lists) / len(list_of_gain_lists)

def dcg_at_k(gains: List[float], k: int) -> float:
    s = 0.0
    for i, g in enumerate(gains[:k], start=1):
        denom = log2(i + 1) if i > 1 else 1.0
        s += (2**g - 1) / denom
    return s

def ndcg_at_k(gains: List[float], k: int) -> float:
    k = max(1, min(k, len(gains)))
    best = sorted(gains, reverse=True)
    idcg = dcg_at_k(best, k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(gains, k) / idcg
