import argparse
import json
import os
import re
from typing import List, Dict, Any

import faiss  # type: ignore
import numpy as np
from sentence_transformers import SentenceTransformer
import httpx




def detect_lang_heuristic(text: str) -> str:
    # Simple regex-based detection for ingest step
    if re.search(r"[а-яё]", text.lower()):
        return "ru"
    if re.search(r"[ўқғҳ]", text.lower()):
        return "uz_cyr"
    return "uz_lat_or_other"




def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms


def build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings.astype(np.float32))
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest harmony.jsonl into FAISS RAG store")
    parser.add_argument("--input", required=True, help="Path to harmony.jsonl")
    parser.add_argument("--out", required=True, help="Output directory (e.g., rag_store)")
    parser.add_argument("--backend", default=os.getenv("EMBEDDING_BACKEND", "local"), choices=["local", "api"], help="Embedding backend: local or api")
    parser.add_argument("--model", default=os.getenv("EMBEDDING_MODEL", "nvidia/bge-m3"), help="Embedding model id")
    parser.add_argument("--batch", type=int, default=int(os.getenv("EMBEDDING_BATCH", "64")), help="Embedding batch size")
    args = parser.parse_args()

    input_path = args.input
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    backend = args.backend.lower()
    model_id = args.model
    batch_size = max(1, int(args.batch))
    openai_base = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1").rstrip("/")
    openai_key = os.getenv("OPENAI_API_KEY") or os.getenv("NVIDIA_API_KEY")

    # Load local model if needed
    model: Optional[SentenceTransformer] = None
    if backend == "local":
        model = SentenceTransformer(model_id)

    texts: List[str] = []
    metas: List[Dict[str, Any]] = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            messages = row.get("messages", [])
            if not isinstance(messages, list):
                continue

            question = None
            answer = None
            for m in messages:
                role = m.get("role")
                content = m.get("content")
                if role == "user" and question is None and isinstance(content, str):
                    question = content.strip()
                elif role == "assistant" and answer is None and isinstance(content, str):
                    answer = content.strip()
                if question is not None and answer is not None:
                    break

            if not question or not answer:
                # skip rows missing either question or answer
                continue

            meta = row.get("meta", {}) or {}
            lang = meta.get("lang")
            if not lang:
                # infer by regex
                lang = detect_lang_heuristic(question + "\n" + answer)
            source = meta.get("source")

            pair_text = f"Q: {question}\nA: {answer}"
            texts.append(pair_text)
            metas.append({
                "question": question,
                "answer": answer,
                "lang": lang,
                "source": source,
            })

    if not texts:
        print("Indexed 0 Q/A pairs.")
        return

    # Encode and normalize
    if backend == "local":
        embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False)  # type: ignore[union-attr]
        if isinstance(embeddings, list):
            embeddings = np.array(embeddings)
        embeddings = embeddings.astype(np.float32)
        embeddings = l2_normalize(embeddings)
    else:
        headers = {"Content-Type": "application/json"}
        if openai_key:
            headers["Authorization"] = f"Bearer {openai_key}"
        all_vecs: List[List[float]] = []
        with httpx.Client(timeout=60.0) as client:
            for i in range(0, len(texts), batch_size):
                chunk = texts[i:i+batch_size]
                base_body: Dict[str, Any] = {"model": model_id, "input": chunk}
                alt_body: Dict[str, Any] = {"model": model_id, "input": chunk}
                if "api.nvidia.com" in openai_base:
                    for b in (base_body, alt_body):
                        b["encoding_format"] = "float"
                        b["truncate"] = "NONE"
                    alt_body["input_type"] = "passage"
                url = f"{openai_base}/embeddings"
                resp = client.post(url, headers=headers, json=base_body)
                if resp.status_code >= 400:
                    resp2 = client.post(url, headers=headers, json=alt_body)
                    resp2.raise_for_status()
                    data = resp2.json()
                else:
                    data = resp.json()
                try:
                    for item in data["data"]:
                        all_vecs.append(item["embedding"])
                except Exception:
                    raise RuntimeError("Invalid embeddings response from API")
        embeddings = np.asarray(all_vecs, dtype=np.float32)
        embeddings = l2_normalize(embeddings)

    # Build FAISS index (IP similarity on normalized vectors == cosine similarity)
    index = build_index(embeddings)

    # Persist
    index_path = os.path.join(out_dir, "index.faiss")
    meta_path = os.path.join(out_dir, "meta.jsonl")
    faiss.write_index(index, index_path)
    with open(meta_path, "w", encoding="utf-8") as mf:
        for m in metas:
            mf.write(json.dumps(m, ensure_ascii=False) + "\n")

    print(f"Indexed {len(texts)} Q/A pairs.")


if __name__ == "__main__":
    main()
