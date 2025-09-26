# Project Structure & How Each Piece Works

search/
├─ data/
│ ├─ raw/ # (optional) original CSV/JSONL dumps
│ ├─ processed/ # cleaned JSONL for ES bulk load (agnews_30k.jsonl)
│ └─ qrels/ # tiny hand-made relevance sets for eval (TSV)
├─ es/
│ ├─ mappings/
│ │ └─ agnews_v1.json # index settings + mappings (english analyzer + autocomplete)
│ ├─ pipelines/ # (optional later) ingest pipelines (HTML strip, lowercase)
│ └─ scripts/ # (optional) bulk templates, helpers
├─ ingest/
│ ├─ prepare_dataset.py # loads AG News → cleans → writes data/processed/agnews_30k.jsonl
│ └─ bulk_load.py # sends JSONL to ES via _bulk (chunked, retries, backoff)
├─ api/
│ └─ app.py # FastAPI: /search /suggest /doc/{id} /explain (talks to ES)
├─ ui/
│ └─ app_streamlit.py # Streamlit teaching UI (query, filters, explain)
├─ eval/
│ ├─ metrics.py # P@k, AP, MAP, nDCG@k (clean, student-friendly)
│ └─ run_eval.py # reads qrels → queries API → prints per-query + averages
├─ ops/
│ ├─ run_es.ps1 # start/stop/status for the es-dev container
│ └─ smoke_tests.py # cross-platform mini smoke (search/filter/suggest/explain)
├─ .gitignore # ignores venv, caches, data/, etc.
├─ requirements.txt # pinned package versions
├─ README.md # one-page runbook (ES → ingest → API → UI → eval)
├─ scripts2run.txt # Pow­erShell one-liners for live demos
└─ docs/
└─ STRUCTURE.md # (this file)


## Data flow (end-to-end)
1. **prepare_dataset.py** → writes **`data/processed/agnews_30k.jsonl`**  
2. **bulk_load.py** → streams that JSONL into **Elasticsearch** (`agnews_v1`)  
3. **api/app.py** → queries ES (BM25 over `title^2 + body`) and exposes endpoints  
4. **ui/app_streamlit.py** → calls the API to visualize results/explainability  
5. **eval/run_eval.py** → calls the API for a small **qrels** file to measure nDCG/MAP

## Key files and knobs

### `es/mappings/agnews_v1.json`
- **Shards/replicas**: 1 / 0 (single-node dev)  
- **Analyzers**: built-in `english` for `title` & `body`; custom edge-ngram via `autocomplete_analyzer` bound to `title.autocomplete`.  
- **Fields**:  
  - `title`: `text` + `autocomplete` + `keyword`  
  - `body`: `text`  
  - `category`: `keyword` (for filters)  
  - `url`, `published_at` reserved for later

> Change analyzers/fields here and **recreate** the index before re-ingest.

### `ingest/prepare_dataset.py`
- Loads **AG News** (`datasets`), samples **30,000**, strips HTML, normalizes whitespace.  
- Maps labels `{0..3}` → `{World, Sports, Business, Sci/Tech}`.  
- Writes one JSON per line with keys: `id`, `title`, `body`, `category`, `url`.

> Knobs: `--size`, RNG seed, optional `synth_timestamp()` if you want recency boosting.

### `ingest/bulk_load.py`
- Builds `_bulk` **NDJSON** with action lines + document lines.  
- **Batching** (`--batch`), **retries/backoff** on 429/5xx, and post-load `_count` check.

> Knobs: `--batch` (start at 2000); tweak timeout/backoff if your laptop is busy.

### `api/app.py` (FastAPI)
- `GET /search`: multi_match over `title^2` + `body`, optional `cat` filter, highlight snippets.  
- `GET /suggest`: prefix suggestions using `title.autocomplete`.  
- `GET /doc/{id}`: return the full stored document.  
- `GET /explain/{id}?q=`: ES `_explain` (teaches BM25 scoring).

**Config via env vars**:
- `ES_URL` (default `http://localhost:9200`)  
- `ES_INDEX` (default `agnews_v1`)  

> For quick switches, run:  
> `setx ES_INDEX agnews_v1` (new shell picks it up), or launch with `ES_INDEX=... uvicorn ...` in bash.

### `ui/app_streamlit.py`
- Sidebar: query, category, page size, page number.  
- Shows scores/snippets; per-hit **Explain** + **Full doc** expanders.  
- Autocomplete demo at bottom.

**Config via env var**:
- `SEARCH_API` (default `http://127.0.0.1:8000`)

### `eval/metrics.py` & `eval/run_eval.py`
- **`qrels.tsv` format**: `query<TAB>doc_id<TAB>gain` (use 2 for strong relevance, 1 for weak).  
- Computes **P@K, AP, nDCG@K, MAP**; prints per-query + averages.

> Start with 10–15 queries; target **nDCG@10 ≥ 0.75** (adjust once you see results).

### `ops/run_es.ps1`
- `./ops/run_es.ps1 start|stop|rm|status` to manage the `es-dev` container.  
- Good for classroom resets.

---

## Typical demo run
1. Start ES → Create index → Ingest 30K  
2. Run API (`uvicorn`) + UI (`streamlit`)  
3. Show `/search`, filters, **Explain**, and **Suggest**  
4. (Optional) Run `eval/run_eval.py` on **qrels** to talk metrics

