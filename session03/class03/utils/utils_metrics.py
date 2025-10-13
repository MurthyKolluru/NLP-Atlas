import numpy as np
import torch
import torch.nn.functional as F

def safe_mean(x):
    x = np.array(x, dtype=float)
    return float(x.mean()) if x.size else float("nan")

def kl_divergence(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    """KL(P||Q) over last dim (token) for batches of sequences."""
    p_logprob = F.log_softmax(p_logits, dim=-1)
    q_logprob = F.log_softmax(q_logits, dim=-1)
    p_prob = p_logprob.exp()
    kl = (p_prob * (p_logprob - q_logprob)).sum(dim=-1)
    return kl

def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    logprob = F.log_softmax(logits, dim=-1)
    prob = logprob.exp()
    ent = -(prob * logprob).sum(dim=-1)
    return ent

def win_rate(scores_new: np.ndarray, scores_ref: np.ndarray) -> float:
    """Fraction of pairs where new > ref."""
    return float((scores_new > scores_ref).mean())
