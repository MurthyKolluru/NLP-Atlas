#!/usr/bin/env python
"""
word2vec_download.py

Downloads pre-trained GloVe 100d vectors (no training), converts to Word2Vec format,
loads as KeyedVectors, and saves a compact .kv model for fast loading later.

References:
- GloVe page: https://nlp.stanford.edu/projects/glove/
- GloVe 6B zip: https://nlp.stanford.edu/data/glove.6B.zip
- Gensim KeyedVectors & glove2word2vec:
  https://radimrehurek.com/gensim/models/keyedvectors.html
"""

from pathlib import Path
import sys
import zipfile
import urllib.request
import shutil
from typing import Optional

# gensim imports (ensure installed: pip install gensim)
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
ROOT = Path(".")
W2V_DIR = ROOT / "lecture1" / "word2vec"
W2V_DIR.mkdir(parents=True, exist_ok=True)

GLOVE_ZIP_URL = "https://nlp.stanford.edu/data/glove.6B.zip"
GLOVE_ZIP_PATH = W2V_DIR / "glove.6B.zip"
GLOVE_TXT_NAME = "glove.6B.100d.txt"             # member inside the zip
GLOVE_TXT_PATH = W2V_DIR / GLOVE_TXT_NAME        # extracted target
W2V_TXT_PATH = W2V_DIR / "glove.6B.100d.word2vec.txt"
KV_PATH       = W2V_DIR / "glove.6B.100d.kv"


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------
def download_with_progress(url: str, dest: Path) -> None:
    """Download a URL to dest with a simple progress indicator."""
    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, downloaded * 100 / (total_size or 1))
        end = "\r" if downloaded < total_size else "\n"
        print(f"Downloading: {percent:6.2f}% ({downloaded // (1024*1024)} MB)", end=end)
    print(f"[INFO] Downloading {url} -> {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest, _progress)


def ensure_glove_zip() -> None:
    """Ensure the GloVe zip is present."""
    if GLOVE_ZIP_PATH.exists():
        print(f"[OK] Zip already present: {GLOVE_ZIP_PATH}")
        return
    download_with_progress(GLOVE_ZIP_URL, GLOVE_ZIP_PATH)
    print(f"[OK] Downloaded: {GLOVE_ZIP_PATH}")


def extract_glove_100d() -> None:
    """Extract only glove.6B.100d.txt from the zip into W2V_DIR."""
    if GLOVE_TXT_PATH.exists():
        print(f"[OK] Found extracted file: {GLOVE_TXT_PATH}")
        return
    print(f"[INFO] Extracting {GLOVE_TXT_NAME} from {GLOVE_ZIP_PATH}")
    with zipfile.ZipFile(GLOVE_ZIP_PATH, "r") as zf:
        # Safety: ensure member exists
        members = zf.namelist()
        if GLOVE_TXT_NAME not in members:
            raise FileNotFoundError(
                f"{GLOVE_TXT_NAME} not found in {GLOVE_ZIP_PATH}. "
                f"Available members: {members[:5]}..."
            )
        # Extract only the needed file
        with zf.open(GLOVE_TXT_NAME) as src, open(GLOVE_TXT_PATH, "wb") as dst:
            shutil.copyfileobj(src, dst)
    print(f"[OK] Extracted: {GLOVE_TXT_PATH}")


def convert_glove_to_word2vec() -> None:
    """Convert GloVe txt to Word2Vec txt format if needed."""
    if W2V_TXT_PATH.exists():
        print(f"[OK] Word2Vec-format file already present: {W2V_TXT_PATH}")
        return
    print(f"[INFO] Converting to word2vec format: {GLOVE_TXT_PATH.name} -> {W2V_TXT_PATH.name}")
    glove2word2vec(str(GLOVE_TXT_PATH), str(W2V_TXT_PATH))
    print(f"[OK] Converted: {W2V_TXT_PATH}")


def build_and_save_kv() -> KeyedVectors:
    """Load the word2vec text and save as a compact keyed-vectors .kv file."""
    if KV_PATH.exists():
        print(f"[OK] KeyedVectors already saved: {KV_PATH}")
        # Still load to print summary
        wv = KeyedVectors.load(str(KV_PATH), mmap="r")
        return wv
    print(f"[INFO] Loading word2vec text to KeyedVectors (this can take a minute) ...")
    wv = KeyedVectors.load_word2vec_format(str(W2V_TXT_PATH), binary=False)
    print(f"[INFO] Saving KeyedVectors to: {KV_PATH}")
    wv.save(str(KV_PATH))
    return wv


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main() -> int:
    print(f"[INFO] Target directory: {W2V_DIR.resolve()}")
    try:
        ensure_glove_zip()
        extract_glove_100d()
        convert_glove_to_word2vec()
        wv = build_and_save_kv()
    except Exception as e:
        print(f"[ERROR] {e}")
        return 1

    # Summary
    vocab_size = len(wv.index_to_key)
    print("\n========== SUMMARY ==========")
    print(f"Vectors file (KV): {KV_PATH}")
    print(f"GloVe source zip : {GLOVE_ZIP_PATH}")
    print(f"GloVe 100d txt   : {GLOVE_TXT_PATH}")
    print(f"Word2Vec txt     : {W2V_TXT_PATH}")
    print(f"Vocabulary size  : {vocab_size}")
    print("Sample tokens    :", ", ".join(wv.index_to_key[:10]))
    print("================================\n")
    print("[DONE] You can now load the vectors in code, e.g.:")
    print(f"    from gensim.models import KeyedVectors")
    print(f"    wv = KeyedVectors.load(r'{KV_PATH}')")
    return 0


if __name__ == "__main__":
    sys.exit(main())
