import argparse
import json
import os
import re
from typing import List, Dict, Any

import faiss  # type: ignore
import numpy as np
from sentence_transformers import SentenceTransformer



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
    args = parser.parse_args()

    input_path = args.input
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    # Load model once
    model = SentenceTransformer("BAAI/bge-m3")

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
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=False)
    if isinstance(embeddings, list):
        embeddings = np.array(embeddings)
    embeddings = embeddings.astype(np.float32)
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

