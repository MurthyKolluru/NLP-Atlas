#!/usr/bin/env python
"""
Compute SVD for the Term–Document Matrix.

Orientation:
    A is (docs x terms). A = U Σ V^T
Embeddings (used by svd_explore.py):
    - Document embeddings: D = U_r @ sqrt(Σ_r)     -> shape (docs, r)
    - Word embeddings:     W = (V_r^T).T @ sqrt(Σ_r) -> shape (terms, r)
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd

# ---- Paths ----
ROOT = Path(__file__).resolve().parents[2] if (Path(__file__).parts[-3].lower() == "lecture1") else Path(".")
LECTURE_DIR = ROOT / "lecture1"
TDM_PATH = LECTURE_DIR / "tdm" / "tdm.csv"
OUT_DIR = LECTURE_DIR / "SVD"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_tdm(tdm_path: Path) -> pd.DataFrame:
    if not tdm_path.exists():
        raise FileNotFoundError(f"Could not find TDM at: {tdm_path}")
    # Try with index_col=0 (common when saved from pandas), fall back to default
    try:
        df = pd.read_csv(tdm_path, index_col=0)
    except Exception:
        df = pd.read_csv(tdm_path)
    return df

def main():
    print(f"[INFO] Reading TDM from: {TDM_PATH}")
    tdm_df = load_tdm(TDM_PATH)

    terms = list(tdm_df.columns)
    docs = list(tdm_df.index)
    A = tdm_df.values.astype(float)  # docs x terms

    m, n = A.shape
    print(f"[INFO] A.shape (docs x terms) = {A.shape}")
    print("[INFO] Preview (first 5 rows):")
    with pd.option_context("display.width", 120, "display.max_columns", 20):
        print(tdm_df.head())

    print("[INFO] Computing SVD ... (numpy.linalg.svd, full_matrices=False)")
    U, S, VT = np.linalg.svd(A, full_matrices=False)

    print("\n=== SVD Shapes ===")
    print(f"U.shape  = {U.shape}")
    print(f"S.shape  = {S.shape}  (vector of singular values)")
    print(f"VT.shape = {VT.shape}")
    print("\nFirst singular values:", S[: min(10, len(S))])

    # ---- Save outputs ----
    np.save(OUT_DIR / "U.npy", U)
    np.save(OUT_DIR / "S.npy", S)
    np.save(OUT_DIR / "VT.npy", VT)

    pd.DataFrame({"sigma": S}).to_csv(OUT_DIR / "singular_values.csv", index=False)
    # CSVs are fine for small demos
    pd.DataFrame(U, index=docs, columns=[f"u{i+1}" for i in range(U.shape[1])]).to_csv(OUT_DIR / "U.csv")
    pd.DataFrame(VT, index=[f"sv{i+1}" for i in range(VT.shape[0])], columns=terms).to_csv(OUT_DIR / "VT.csv")

    with open(OUT_DIR / "vocab.txt", "w", encoding="utf-8") as f:
        for t in terms:
            f.write(t + "\n")

    with open(OUT_DIR / "svd_meta.txt", "w", encoding="utf-8") as f:
        f.write(f"A_shape={m},{n}\n")
        f.write(f"rank_upper_bound={min(m,n)}\n")
        f.write("note=Orientation: A is docs x terms; doc embeddings use U, word embeddings use V\n")

    print(f"\n[OK] Saved into: {OUT_DIR}")
    print("     Files: U.npy, S.npy, VT.npy, U.csv, VT.csv, singular_values.csv, vocab.txt, svd_meta.txt")

if __name__ == "__main__":
    main()
