# Mid-Sized Search (AG News 30K)

## What this is
A production-ish single-node Elasticsearch + FastAPI + Streamlit setup for ~30K docs:
- BM25 over title^2 + body
- Category filters
- Autocomplete via edge n-grams
- Explain endpoint for teaching
- Tiny eval harness (MAP, nDCG)

## Quickstart
1) **Prereqs** (Windows Home ok):
- Docker Desktop (WSL2 backend)
- Python 3.10 (venv)
- VS Code

2) **Setup**
```powershell
cd C:\Users\murth\Desktop\nlpSession02\codeBase\search
& ..\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

3). **Start Elastic Search**

docker pull docker.elastic.co/elasticsearch/elasticsearch:8.14.0
docker run --name es-dev -p 9200:9200 `
  -e "discovery.type=single-node" `
  -e "xpack.security.enabled=false" `
  -d docker.elastic.co/elasticsearch/elasticsearch:8.14.0

Invoke-RestMethod http://localhost:9200/

4) Create index (mapping)
Invoke-RestMethod -Method DELETE http://localhost:9200/agnews_v1 -ErrorAction SilentlyContinue
Invoke-RestMethod -Method PUT -ContentType "application/json" `
  -InFile es\mappings\agnews_v1.json `
  http://localhost:9200/agnews_v1

5) Prepare dataset
python ingest/prepare_dataset.py --out data/processed/agnews_30k.jsonl --size 30000

6) Bulk load into ES
python ingest/bulk_load.py --index agnews_v1 --input data/processed/agnews_30k.jsonl --batch 2000

Check:
Invoke-RestMethod http://localhost:9200/agnews_v1/_count

7) Run API & UI
Terminal 1 (API):
uvicorn api.app:app --reload --port 8000
Terminal 2 (UI):

cd C:\Users\murth\Desktop\nlpSession02\codeBase\search
& ..\.venv\Scripts\Activate.ps1
streamlit run ui/app_streamlit.py --server.port 8501

Open: http://127.0.0.1:8501

8) Evaluation harness
Make a small data/qrels/qrels.tsv like:

stocks	agnews-123	2
stocks	agnews-456	1
world cup	agnews-789	2

Run:

python eval/run_eval.py --qrels data/qrels/qrels.tsv --k 10 --size 50

Troubleshooting
ES not responding: docker logs -f es-dev or ./ops/run_es.ps1 status
CORS issues: API enables localhost origins (127.0.0.1:8501) by default
Slow queries: warmup once; p95 goals are for simple queries


