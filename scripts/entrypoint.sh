#!/usr/bin/env sh
set -e

# Optional: show configured upstream for visibility
echo "OPENAI_BASE_URL=${OPENAI_BASE_URL}"
echo "OPENAI_MODEL=${OPENAI_MODEL}"

# If FAISS index is missing, build it from harmony.jsonl
if [ ! -f "/app/rag_store/index.faiss" ] || [ ! -f "/app/rag_store/meta.jsonl" ]; then
  echo "[entrypoint] RAG store not found; running ingest..."
  mkdir -p /app/rag_store
  python /app/ingest.py --input /app/harmony.jsonl --out /app/rag_store || {
    echo "[entrypoint] Ingest failed" >&2
    exit 1
  }
else
  echo "[entrypoint] RAG store found; skipping ingest."
fi

exec uvicorn serve:app --host ${UVICORN_HOST:-0.0.0.0} --port ${UVICORN_PORT:-8000}

