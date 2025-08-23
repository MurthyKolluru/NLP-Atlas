#!/usr/bin/env python
"""
Explore SVD embeddings from saved artifacts.

Usage examples (from project root):
    python lecture1/SVD/svd_explore.py --rank 5 --save
    python lecture1/SVD/svd_explore.py --interactive

Options:
    --rank R           Rank r (1..min(docs,terms)) to use for embeddings
    --save             Save embeddings CSVs to lecture1/SVD/
    --top-dim K        Show top 10 terms by |dimension K|
    --interactive      Enter an interactive loop to try multiple ranks/dims
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# ---- Paths ----
ROOT = Path(__file__).resolve().parents[2] if (Path(__file__).parts[-3].lower() == "lecture1") else Path(".")
SVD_DIR = ROOT / "lecture1" / "SVD"
TDM_PATH = ROOT / "lecture1" / "tdm" / "tdm.csv"

def load_tdm(tdm_path: Path) -> pd.DataFrame:
    if not tdm_path.exists():
        raise FileNotFoundError(f"Missing TDM: {tdm_path}")
    try:
        df = pd.read_csv(tdm_path, index_col=0)
    except Exception:
        df = pd.read_csv(tdm_path)
    return df

def load_svd(dirpath: Path):
    U_path, S_path, VT_path = dirpath / "U.npy", dirpath / "S.npy", dirpath / "VT.npy"
    for p in [U_path, S_path, VT_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing SVD artifact: {p}")
    U = np.load(U_path)
    S = np.load(S_path)
    VT = np.load(VT_path)
    return U, S, VT

def compute_embeddings(U, S, VT, r):
    """Given full U,S,VT and rank r, return (D, W)."""
    Ur  = U[:, :r]         # (docs x r)
    Sr  = S[:r]            # (r,)
    VTr = VT[:r, :]        # (r x terms)
    sqrtS = np.diag(np.sqrt(Sr))
    D = Ur @ sqrtS               # (docs x r)
    W = (VTr.T) @ sqrtS          # (terms x r)
    return D, W

def save_embeddings(D, W, doc_ids, terms, out_dir: Path, r: int):
    dcols = [f"dim_{i+1}" for i in range(D.shape[1])]
    D_df = pd.DataFrame(D, index=doc_ids, columns=dcols)
    W_df = pd.DataFrame(W, index=terms,  columns=dcols)
    D_path = out_dir / f"doc_embeddings_r{r}.csv"
    W_path = out_dir / f"word_embeddings_r{r}.csv"
    D_df.to_csv(D_path)
    W_df.to_csv(W_path)
    print(f"[Saved] {D_path.name}, {W_path.name} -> {out_dir}")

def show_heads(D, W, doc_ids, terms, r, head_n=8):
    dcols = [f"dim_{i+1}" for i in range(r)]
    D_df = pd.DataFrame(D, index=doc_ids, columns=dcols)
    W_df = pd.DataFrame(W, index=terms,  columns=dcols)

    print(f"\nRank r = {r}")
    print(f"Document embeddings D shape: {D.shape}  (docs x r)")
    print(f"Word embeddings     W shape: {W.shape}  (terms x r)")

    with pd.option_context("display.width", 120, "display.max_columns", 12):
        print("\n=== Document Embeddings (head) ===")
        print(D_df.head(head_n))
        print("\n=== Word Embeddings (head) ===")
        print(W_df.head(head_n))

def top_terms_on_dim(W, terms, dim_k: int, top_n: int = 10):
    vec = W[:, dim_k-1]  # 1-based to 0-based
    idx = np.argsort(-np.abs(vec))[:top_n]
    rows = [(terms[i], float(vec[i])) for i in idx]
    df = pd.DataFrame(rows, columns=["term", f"score_dim_{dim_k}"])
    print(f"\nTop {top_n} terms by |dim_{dim_k}| (signed scores):")
    print(df)

def main():
    parser = argparse.ArgumentParser(description="Explore SVD embeddings.")
    parser.add_argument("--rank", type=int, default=None, help="Rank r to use")
    parser.add_argument("--save", action="store_true", help="Save embeddings CSVs")
    parser.add_argument("--top-dim", type=int, default=None, help="Show top 10 terms by |dimension K|")
    parser.add_argument("--interactive", action="store_true", help="Interactive loop to try multiple ranks/dims")
    args = parser.parse_args()

    # Load artifacts
    tdm_df = load_tdm(TDM_PATH)
    U, S, VT = load_svd(SVD_DIR)

    terms = list(tdm_df.columns)
    doc_ids = list(tdm_df.index)
    m_docs, n_terms = tdm_df.shape
    rmax = min(m_docs, n_terms)

    if args.interactive:
        print("[INFO] Interactive mode. Press Ctrl+C to exit.\n")
        while True:
            try:
                r = input(f"Enter rank r (1..{rmax}): ").strip()
                if not r:
                    continue
                r = int(r)
                if r < 1 or r > rmax:
                    print(f"  -> r must be in 1..{rmax}")
                    continue

                D, W = compute_embeddings(U, S, VT, r)
                show_heads(D, W, doc_ids, terms, r)

                cmd = input("\nCommands: [s]ave, [t]op-dim, [c]ontinue, [q]uit: ").strip().lower()
                if cmd == "s":
                    save_embeddings(D, W, doc_ids, terms, SVD_DIR, r)
                elif cmd == "t":
                    k = input(f"  Enter dimension k (1..{r}): ").strip()
                    try:
                        k = int(k)
                        if 1 <= k <= r:
                            top_terms_on_dim(W, terms, k)
                        else:
                            print("  -> Invalid k.")
                    except Exception:
                        print("  -> Invalid input.")
                elif cmd == "q":
                    break
                # else continue
            except KeyboardInterrupt:
                print("\n[INFO] Exiting interactive mode.")
                break
        return

    # Non-interactive path
    if args.rank is None:
        print("[ERROR] Please specify --rank R or use --interactive.")
        return

    r = int(args.rank)
    if r < 1 or r > rmax:
        raise ValueError(f"Rank r must be in 1..{rmax}")

    D, W = compute_embeddings(U, S, VT, r)
    show_heads(D, W, doc_ids, terms, r)

    if args.save:
        save_embeddings(D, W, doc_ids, terms, SVD_DIR, r)

    if args.top_dim is not None:
        k = int(args.top_dim)
        if 1 <= k <= r:
            top_terms_on_dim(W, terms, k)
        else:
            print(f"[WARN] --top-dim must be in 1..{r}")

if __name__ == "__main__":
    main()
