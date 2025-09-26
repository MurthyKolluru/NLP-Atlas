# ui/app_streamlit.py
"""
Streamlit UI for the Mid-Sized Search demo.
- Query box, category filter, page size
- Shows ranked results with score + snippet
- Per-result "Explain" and "Open Doc" expanders
- Talks to the FastAPI server at http://127.0.0.1:8000
"""
from __future__ import annotations
import os, requests, math
import streamlit as st

API = os.getenv("SEARCH_API", "http://127.0.0.1:8000")
CATS = ["", "World", "Sports", "Business", "Sci/Tech"]   # "" means no filter

st.set_page_config(page_title="Mid-Sized Search", layout="wide")
st.title("🔎 Mid-Sized Search (BM25)")

# -------- Controls (left) --------
with st.sidebar:
    st.header("Controls")
    q = st.text_input("Query", value="stock market", help="Type a query phrase")
    cat = st.selectbox("Category filter", CATS, index=0)
    size = st.slider("Results per page", 5, 30, 10, step=5)
    page = st.number_input("Page (0-based)", min_value=0, value=0, step=1)
    run = st.button("Run Search")

# -------- Helper calls --------
def api_get(path: str, params: dict | None = None):
    try:
        r = requests.get(API + path, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        st.error(f"API error: {e}")
        return None

# -------- Search trigger --------
if run:
    params = {"q": q, "from": page * size, "size": size}
    if cat:
        params["cat"] = cat
    res = api_get("/search", params)
    if res is None:
        st.stop()

    total = res.get("total", 0)
    took = res.get("took_ms", 0)
    hits = res.get("hits", [])

    st.markdown(f"**Results:** {total:,} • **took:** {took} ms • **page:** {page}")
    pages = math.ceil(total / size) if size else 0
    if pages:
        st.progress(min((page + 1) / pages, 1.0))

    # ---- Results list ----
    for h in hits:
        with st.container(border=True):
            cols = st.columns([0.75, 0.25])
            with cols[0]:
                st.markdown(f"**{h.get('title','(no title)')}**  \n"
                            f"_{h.get('category','')}_ • score={h.get('score',0):.2f}")
                st.markdown(h.get("snippet") or "")
            with cols[1]:
                with st.expander("🔍 Explain"):
                    params = {"q": q}
                    doc_id = h.get("id")
                    if doc_id:
                        exp = api_get(f"/explain/{doc_id}", params)
                        if exp:
                            st.json(exp)
                with st.expander("📄 Full doc"):
                    doc_id = h.get("id")
                    if doc_id:
                        doc = api_get(f"/doc/{doc_id}")
                        if doc:
                            st.json(doc)

    # ---- Autocomplete demo ----
    st.subheader("📝 Autocomplete")
    pref = st.text_input("Prefix", value=q.split(" ")[0][:4] if q else "mic")
    if st.button("Suggest"):
        sres = api_get("/suggest", {"prefix": pref, "size": 8})
        if sres:
            st.write(sres.get("suggestions", []))
else:
    st.info("Use the sidebar to run a search. Pro tip: try filters, then click Explain on a result.")
