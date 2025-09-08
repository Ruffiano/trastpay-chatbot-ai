import os
import json
import re
import uuid
from typing import List, Optional, Dict, Any, Literal

import faiss  # type: ignore
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, MetaData, Table, Column, String, Text, Integer, DateTime, func, select, delete, insert
import anyio
from sentence_transformers import SentenceTransformer
import httpx
from fastapi.middleware.cors import CORSMiddleware


# -------- Config ---------
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-oss-20b")

STORE_DIR = "rag_store"
INDEX_PATH = os.path.join(STORE_DIR, "index.faiss")
META_PATH = os.path.join(STORE_DIR, "meta.jsonl")

# DB config (MySQL)
DB_HOST = os.getenv("DB_HOST", "db")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_NAME = os.getenv("DB_NAME", "rag")
DB_USER = os.getenv("DB_USER", "rag")
DB_PASSWORD = os.getenv("DB_PASSWORD", "ragpw")
DB_URL = os.getenv("DB_URL", f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")


# -------- Globals --------
# OpenAPI/Swagger metadata
TAGS_METADATA = [
    {"name": "RAG", "description": "Retrieval-augmented answering over a FAISS store."},
    {"name": "Health", "description": "Health and readiness probes."},
    {"name": "Chat", "description": "Multi-turn chat sessions with lightweight history."},
    {"name": "Export", "description": "Download sessions and messages as CSV/JSONL."},
]

app = FastAPI(
    title="RAG FAQ API",
    description=(
        "Simple retrieval-augmented generation (RAG) API backed by a FAISS index "
        "built from harmony.jsonl. Use /ask to query."
    ),
    version="0.1.0",
    openapi_tags=TAGS_METADATA,
)
# Enable permissive CORS so a simple static page can call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"]
)
_embed_model: Optional[SentenceTransformer] = None
_faiss_index: Optional[faiss.Index] = None
_metas: List[Dict[str, Any]] = []
# Database: sessions and messages
engine = create_engine(DB_URL, pool_pre_ping=True, pool_recycle=1800, future=True)
metadata = MetaData()

sessions_table = Table(
    "sessions",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("created_at", DateTime, server_default=func.now(), nullable=False),
)

messages_table = Table(
    "messages",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("session_id", String(64), nullable=False, index=True),
    Column("role", String(16), nullable=False),
    Column("content", Text, nullable=False),
    Column("created_at", DateTime, server_default=func.now(), nullable=False, index=True),
)


def load_store() -> None:
    global _embed_model, _faiss_index, _metas
    if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
        raise RuntimeError("RAG store not found. Run ingest.py first.")

    _faiss_index = faiss.read_index(INDEX_PATH)
    _metas = []
    with open(META_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                _metas.append(json.loads(line))
            except json.JSONDecodeError:
                continue


# --- DB helpers (run in thread to avoid blocking async loop) ---
def _db_create_tables() -> None:
    metadata.create_all(engine)


def _db_create_session() -> str:
    sid = uuid.uuid4().hex
    with engine.begin() as conn:
        conn.execute(insert(sessions_table).values(id=sid))
    return sid


def _db_add_message(session_id: str, role: str, content: str) -> None:
    with engine.begin() as conn:
        conn.execute(insert(messages_table).values(session_id=session_id, role=role, content=content))


def _db_get_history(session_id: str, limit: int = 50) -> List[Dict[str, str]]:
    with engine.begin() as conn:
        rows = conn.execute(
            select(messages_table.c.role, messages_table.c.content)
            .where(messages_table.c.session_id == session_id)
            .order_by(messages_table.c.id.asc())
            .limit(limit)
        ).all()
    return [{"role": r.role, "content": r.content} for r in rows]


def _db_clear_session(session_id: str) -> None:
    with engine.begin() as conn:
        conn.execute(delete(messages_table).where(messages_table.c.session_id == session_id))
        conn.execute(delete(sessions_table).where(sessions_table.c.id == session_id))


def detect_query_language(text: str) -> str:
    # Heuristic first
    if re.search(r"[а-яё]", text.lower()):
        return "ru"
    if re.search(r"[ўқғҳ]", text.lower()):
        return "uz"
    # Fallback to langdetect if available
    try:
        from langdetect import detect

        lang = detect(text)
        if lang.startswith("ru"):
            return "ru"
        if lang.startswith("uz"):
            return "uz"
    except Exception:
        pass
    # Default to Uzbek policy if unsure
    return "uz"


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms


def get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        # Lazy-load the embedding model to avoid slow API startup
        _embed_model = SentenceTransformer("BAAI/bge-m3")
    return _embed_model


def embed_query(text: str) -> np.ndarray:
    model = get_embed_model()
    vec = model.encode([text], show_progress_bar=False)
    if isinstance(vec, list):
        vec = np.array(vec)
    vec = vec.astype(np.float32)
    vec = l2_normalize(vec)
    return vec


class AskRequest(BaseModel):
    query: str
    k: int = 3
    lang: Optional[str] = None

    # Example payloads for Swagger UI
    model_config = {
        "json_schema_extra": {
            "examples": [
                {"query": "Can I cancel a transfer?", "k": 3},
                {"query": "Могу ли я отменить перевод?", "k": 3, "lang": "ru"},
            ]
        }
    }


class RetrievedItem(BaseModel):
    score: float
    lang: Optional[str] = None
    source: Optional[str] = None
    question: Optional[str] = None


class AskResponse(BaseModel):
    answer: str
    retrieved: List[RetrievedItem]
    suggestions: List[str] = []


class ChatStartResponse(BaseModel):
    session_id: str


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatSendRequest(BaseModel):
    session_id: Optional[str] = Field(default=None, description="Existing chat session. If omitted, a session is created and returned in the response.")
    message: str = Field(description="User message for this turn.")
    k: int = Field(default=3, ge=1, le=10, description="Top-k contexts (1–10).")
    lang: Optional[str] = Field(default=None, description="Override language detection (e.g., 'ru' or 'uz').")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"message": "How to cancel a transfer?", "k": 3},
                {"session_id": "<existing>", "message": "Tell me more", "k": 3},
            ]
        }
    }


class ChatHistoryResponse(BaseModel):
    session_id: str
    history: List[ChatMessage]


class ChatSendResponse(BaseModel):
    session_id: str
    answer: str
    retrieved: List[RetrievedItem]
    suggestions: List[str]
    history: List[ChatMessage]


@app.on_event("startup")
def _on_startup() -> None:
    # Ensure FAISS store and DB tables are ready
    load_store()
    # Create tables if they don't exist
    _db_create_tables()


@app.post(
    "/ask",
    tags=["RAG"],
    summary="Ask a question with RAG",
    response_model=AskResponse,
)
async def ask(req: AskRequest):
    if _faiss_index is None or not _metas:
        raise HTTPException(status_code=500, detail="RAG store not loaded")

    query_lang = req.lang or detect_query_language(req.query)
    is_ru = (query_lang == "ru")

    # Build policy and prompt scaffold
    if is_ru:
        policy = (
            "Отвечай строго на основе контекста. Если ответа нет в контексте — скажи, что информации недостаточно.\n"
            "Не запрашивай личные данные. Отвечай кратко и по делу."
        )
    else:
        policy = (
            "Faqdan tashqarida fakt qo‘shma. Kontekstdan topilmasa — ma’lumot yetarli emas de.\n"
            "Shaxsiy ma’lumot so‘rama. Qisqa va aniq javob ber."
        )

    # FAISS search
    k = max(1, int(req.k))
    k = min(k, len(_metas))
    qvec = embed_query(req.query)
    D, I = _faiss_index.search(qvec, k)  # type: ignore
    scores = D[0].tolist() if len(D) else []
    idxs = I[0].tolist() if len(I) else []

    retrieved_items = []
    contexts: List[str] = []
    for rank, (sid, score) in enumerate(zip(idxs, scores), start=1):
        if sid < 0 or sid >= len(_metas):
            continue
        meta = _metas[sid]
        # Context: only answers per spec
        ans = meta.get("answer", "")
        contexts.append(f"[CTX#{rank}] {ans}")
        retrieved_items.append({
            "score": float(score),
            "lang": meta.get("lang"),
            "source": meta.get("source"),
            "question": meta.get("question"),
        })

    context_block = "\n\n".join(contexts) if contexts else ""

    prompt = (
        f"{policy}\n\n"
        f"[QUERY]\n{req.query}\n\n"
        f"[CONTEXT]\n{context_block}\n\n"
        f"[ANSWER IN LANGUAGE OF QUERY]"
    )

    headers = {"Content-Type": "application/json"}
    if OPENAI_API_KEY:
        headers["Authorization"] = f"Bearer {OPENAI_API_KEY}"

    body = {
        "model": OPENAI_MODEL,
        "temperature": 0.1,
        "messages": [
            {"role": "user", "content": prompt}
        ],
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{OPENAI_BASE_URL.rstrip('/')}/chat/completions",
                headers=headers,
                json=body,
            )
            if resp.status_code == 422:
                return JSONResponse(status_code=422, content={"detail": "Invalid request to chat endpoint"})
            if resp.status_code >= 400:
                return JSONResponse(status_code=500, content={"detail": f"Upstream error: {resp.status_code}"})
            data = resp.json()
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": f"Upstream exception: {str(e)}"})

    try:
        answer = data["choices"][0]["message"]["content"].strip()
    except Exception:
        answer = ""

    # Ensure response matches the schema
    typed_items = [
        RetrievedItem(
            score=it.get("score", 0.0),
            lang=it.get("lang"),
            source=it.get("source"),
            question=it.get("question"),
        )
        for it in retrieved_items[:k]
    ]
    # Build simple suggestions (questions) based on retrieved items or language defaults
    suggestions: List[str] = []
    for it in retrieved_items:
        q = (it.get("question") or "").strip()
        if q and q not in suggestions:
            suggestions.append(q)
        if len(suggestions) >= 3:
            break

    if not suggestions:
        if is_ru:
            suggestions = [
                "Как пополнить карту?",
                "Как отменить перевод?",
                "Как изменить лимит?",
            ]
        else:
            suggestions = [
                "Kartani qanday to‘ldirish mumkin?",
                "Pul o‘tkazmasini qanday bekor qilaman?",
                "Limitni qanday o‘zgartiraman?",
            ]

    return AskResponse(answer=answer or "", retrieved=typed_items, suggestions=suggestions)


@app.get("/health", tags=["Health"], summary="Health check")
def health() -> Dict[str, str]:
    return {"status": "ok"}


# ------------- Chat Endpoints -------------

@app.post(
    "/chat/start",
    tags=["Chat"],
    summary="Start a new chat session",
    description="Creates and returns a new session_id. You can also omit session_id in POST /chat/send to auto-create one.",
    response_model=ChatStartResponse,
)
def chat_start() -> ChatStartResponse:
    sid = _db_create_session()
    return ChatStartResponse(session_id=sid)


@app.post(
    "/chat/send",
    tags=["Chat"],
    summary="Send a message (creates session if missing)",
    description="Sends a user message, performs RAG on the latest turn, and returns the assistant reply. If session_id is not provided, a new session is created and included in the response.",
    response_model=ChatSendResponse,
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "new-session": {
                            "summary": "First message without session_id",
                            "value": {"message": "How can I cancel a transfer?", "k": 3},
                        },
                        "existing-session": {
                            "summary": "Continue an existing session",
                            "value": {"session_id": "abc123", "message": "Tell me more", "k": 3},
                        },
                    }
                }
            }
        },
        "responses": {
            "200": {
                "content": {
                    "application/json": {
                        "examples": {
                            "ok": {
                                "summary": "Assistant response (session auto-created)",
                                "value": {
                                    "session_id": "f1e2d3c4b5a6",
                                    "answer": "You can cancel a transfer within 30 minutes...",
                                    "retrieved": [
                                        {"score": 0.87, "lang": "ru", "source": "faq.md", "question": "Как отменить перевод?"}
                                    ],
                                    "suggestions": [
                                        "Как пополнить карту?",
                                        "Как изменить лимит?",
                                        "Как проверить статус перевода?"
                                    ],
                                    "history": [
                                        {"role": "user", "content": "How can I cancel a transfer?"},
                                        {"role": "assistant", "content": "You can cancel a transfer within 30 minutes..."}
                                    ]
                                },
                            }
                        }
                    }
                }
            }
        }
    },
)
async def chat_send(req: ChatSendRequest) -> ChatSendResponse:
    if _faiss_index is None or not _metas:
        raise HTTPException(status_code=500, detail="RAG store not loaded")

    # Create or retrieve session
    sid = req.session_id or await anyio.to_thread.run_sync(_db_create_session)
    history = await anyio.to_thread.run_sync(_db_get_history, sid, 100)

    # Append user message to history
    user_text = req.message.strip()
    if not user_text:
        raise HTTPException(status_code=422, detail="Empty message")
    await anyio.to_thread.run_sync(_db_add_message, sid, "user", user_text)

    # Language & policy
    query_lang = req.lang or detect_query_language(user_text)
    is_ru = (query_lang == "ru")
    if is_ru:
        policy = (
            "Отвечай строго на основе контекста. Если ответа нет в контексте — скажи, что информации недостаточно.\n"
            "Не запрашивай личные данные. Отвечай кратко и по делу."
        )
    else:
        policy = (
            "Faqdan tashqarida fakt qo‘shma. Kontekstdan topilmasa — ma’lumot yetarli emas de.\n"
            "Shaxsiy ma’lumot so‘rama. Qisqa va aniq javob ber."
        )

    # Retrieve context based on the latest user message
    k = max(1, int(req.k))
    k = min(k, len(_metas))
    qvec = embed_query(user_text)
    D, I = _faiss_index.search(qvec, k)  # type: ignore
    scores = D[0].tolist() if len(D) else []
    idxs = I[0].tolist() if len(I) else []

    retrieved_items = []
    contexts: List[str] = []
    for rank, (sid_, score) in enumerate(zip(idxs, scores), start=1):
        if sid_ < 0 or sid_ >= len(_metas):
            continue
        meta = _metas[sid_]
        ans = meta.get("answer", "")
        contexts.append(f"[CTX#{rank}] {ans}")
        retrieved_items.append({
            "score": float(score),
            "lang": meta.get("lang"),
            "source": meta.get("source"),
            "question": meta.get("question"),
        })

    context_block = "\n\n".join(contexts) if contexts else ""

    # Build chat message list: system policy, prior turns, then current turn with context scaffold
    messages = []
    messages.append({"role": "system", "content": policy})
    for m in history:
        # include prior turns as-is
        if m["role"] in ("user", "assistant"):
            messages.append(m)
    # Current turn with context
    messages.append({
        "role": "user",
        "content": (
            f"[QUERY]\n{user_text}\n\n" +
            f"[CONTEXT]\n{context_block}\n\n" +
            f"[ANSWER IN LANGUAGE OF QUERY]"
        ),
    })

    headers = {"Content-Type": "application/json"}
    if OPENAI_API_KEY:
        headers["Authorization"] = f"Bearer {OPENAI_API_KEY}"

    body = {
        "model": OPENAI_MODEL,
        "temperature": 0.1,
        "messages": messages,
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{OPENAI_BASE_URL.rstrip('/')}/chat/completions",
                headers=headers,
                json=body,
            )
            if resp.status_code == 422:
                return JSONResponse(status_code=422, content={"detail": "Invalid request to chat endpoint"})
            if resp.status_code >= 400:
                return JSONResponse(status_code=500, content={"detail": f"Upstream error: {resp.status_code}"})
            data = resp.json()
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": f"Upstream exception: {str(e)}"})

    try:
        answer = data["choices"][0]["message"]["content"].strip()
    except Exception:
        answer = ""

    # Append assistant message to history
    await anyio.to_thread.run_sync(_db_add_message, sid, "assistant", answer)

    # Suggestions (reuse logic)
    suggestions: List[str] = []
    for it in retrieved_items:
        q = (it.get("question") or "").strip()
        if q and q not in suggestions:
            suggestions.append(q)
        if len(suggestions) >= 3:
            break
    if not suggestions:
        suggestions = (
            ["Как пополнить карту?", "Как отменить перевод?", "Как изменить лимит?"] if is_ru else
            ["Kartani qanday to‘ldirish mumkin?", "Pul o‘tkazmasini qanday bekor qilaman?", "Limitni qanday o‘zgartiraman?"]
        )

    typed_items = [
        RetrievedItem(
            score=it.get("score", 0.0),
            lang=it.get("lang"),
            source=it.get("source"),
            question=it.get("question"),
        ) for it in retrieved_items[:k]
    ]

    return ChatSendResponse(
        session_id=sid,
        answer=answer or "",
        retrieved=typed_items,
        suggestions=suggestions,
        history=[ChatMessage(**m) for m in (history + [{"role": "assistant", "content": answer}])[-20:]],
    )


@app.get("/chat/{session_id}/history", tags=["Chat"], summary="Get chat history", response_model=ChatHistoryResponse)
def chat_history(session_id: str) -> ChatHistoryResponse:
    hist = _db_get_history(session_id, 200)
    return ChatHistoryResponse(session_id=session_id, history=[ChatMessage(**m) for m in hist[-50:]])


@app.delete("/chat/{session_id}", tags=["Chat"], summary="Reset chat session")
def chat_reset(session_id: str) -> Dict[str, str]:
    _db_clear_session(session_id)
    return {"status": "cleared", "session_id": session_id}


# ------------- Export Endpoints -------------

@app.get("/export/sessions.csv", tags=["Export"], summary="Export sessions as CSV")
def export_sessions_csv() -> StreamingResponse:
    import csv
    import io

    with engine.begin() as conn:
        rows = conn.execute(select(sessions_table.c.id, sessions_table.c.created_at).order_by(sessions_table.c.created_at.asc())).all()

    def generate():
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["id", "created_at"])
        for r in rows:
            writer.writerow([r.id, r.created_at.isoformat() if r.created_at else ""])
        yield buf.getvalue()

    return StreamingResponse(
        generate(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=sessions.csv"},
    )


@app.get("/export/messages.csv", tags=["Export"], summary="Export messages as CSV")
def export_messages_csv(
    session_id: Optional[str] = Query(None, description="Filter by session_id"),
    limit: int = Query(100000, ge=1, le=1_000_000, description="Max rows to export"),
) -> StreamingResponse:
    import csv
    import io

    stmt = select(
        messages_table.c.id,
        messages_table.c.session_id,
        messages_table.c.role,
        messages_table.c.content,
        messages_table.c.created_at,
    ).order_by(messages_table.c.id.asc()).limit(limit)
    if session_id:
        stmt = stmt.where(messages_table.c.session_id == session_id)

    with engine.begin() as conn:
        rows = conn.execute(stmt).all()

    def generate():
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["id", "session_id", "role", "content", "created_at"])
        for r in rows:
            writer.writerow([
                r.id,
                r.session_id,
                r.role,
                r.content.replace("\r", " ").replace("\n", " ") if isinstance(r.content, str) else r.content,
                r.created_at.isoformat() if r.created_at else "",
            ])
        yield buf.getvalue()

    return StreamingResponse(
        generate(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=messages.csv"},
    )


@app.get("/export/messages.jsonl", tags=["Export"], summary="Export messages as JSONL")
def export_messages_jsonl(
    session_id: Optional[str] = Query(None, description="Filter by session_id"),
    limit: int = Query(100000, ge=1, le=1_000_000, description="Max rows to export"),
) -> StreamingResponse:
    import io

    stmt = select(
        messages_table.c.id,
        messages_table.c.session_id,
        messages_table.c.role,
        messages_table.c.content,
        messages_table.c.created_at,
    ).order_by(messages_table.c.id.asc()).limit(limit)
    if session_id:
        stmt = stmt.where(messages_table.c.session_id == session_id)

    def generate():
        with engine.begin() as conn:
            for r in conn.execute(stmt):
                obj = {
                    "id": r.id,
                    "session_id": r.session_id,
                    "role": r.role,
                    "content": r.content,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
                yield (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={"Content-Disposition": "attachment; filename=messages.jsonl"},
    )
