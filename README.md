Minimal RAG v0 for FAQ (FastAPI + FAISS)

Setup
- python3 -m venv .venv && source .venv/bin/activate
- pip install --upgrade pip
- pip install -r requirements.txt

Ingest
- python ingest.py --input harmony.jsonl --out rag_store
  - Prints: "Indexed N Q/A pairs."

Serve
- export OPENAI_BASE_URL="http://localhost:8000/v1"
- export OPENAI_MODEL="gpt-oss-20b"
- # export OPENAI_API_KEY=...
- uvicorn serve:app --reload --host 0.0.0.0 --port 11435

Test
- curl -s http://localhost:11435/ask -H 'Content-Type: application/json' \
  -d '{"query":"Как подключить SMS-уведомления?","k":3}' | jq

Notes
- Embeddings: BAAI/bge-m3 (normalized, IP similarity)
- Index: FAISS IndexFlatIP, one vector per "Q: ...\nA: ..." pair
- Lang detection: RU/UZ heuristic (regex), langdetect fallback
- Guardrails: Answers strictly from context; say insufficient if not found; no PII; concise
