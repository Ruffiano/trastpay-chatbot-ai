import os
import json
import re
import uuid
from typing import List, Optional, Dict, Any, Literal, Generic, TypeVar, Union

import faiss  # type: ignore
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, MetaData, Table, Column, String, Text, Integer, DateTime, func, select, delete, insert, text
import anyio
from sentence_transformers import SentenceTransformer
import httpx
from fastapi.middleware.cors import CORSMiddleware


# -------- Envelope Models ---------
T = TypeVar('T')

class PaginationMeta(BaseModel):
    page: int
    page_size: int
    total: int

class MetaInfo(BaseModel):
    pagination: Optional[PaginationMeta] = None

class ErrorInfo(BaseModel):
    code: str
    message: str
    details: Optional[Union[Dict[str, Any], List[Any]]] = None

class SuccessEnvelope(BaseModel, Generic[T]):
    data: T
    error: None = None
    meta: Optional[MetaInfo] = None
    trace_id: str

class ErrorEnvelope(BaseModel):
    data: None = None
    error: ErrorInfo
    meta: None = None
    trace_id: str

# -------- Config ---------
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("NVIDIA_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-oss-20b")

STORE_DIR = "rag_store"
INDEX_PATH = os.path.join(STORE_DIR, "index.faiss")
META_PATH = os.path.join(STORE_DIR, "meta.jsonl")

# Embeddings config
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "local").lower()  # 'local' or 'api'
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nvidia/bge-m3")

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
    Column("user_id", String(64), nullable=False, index=True),
    Column("created_at", DateTime, server_default=func.now(), nullable=False),
    Column("status", String(16), nullable=False, default="active"),
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
    # Add status and user_id columns if they don't exist (for existing databases)
    try:
        with engine.begin() as conn:
            # Check if status column exists first
            result = conn.execute(text("SHOW COLUMNS FROM sessions LIKE 'status'")).fetchone()
            if not result:
                conn.execute(text("ALTER TABLE sessions ADD COLUMN status VARCHAR(16) DEFAULT 'active'"))
            
            # Check if user_id column exists
            result = conn.execute(text("SHOW COLUMNS FROM sessions LIKE 'user_id'")).fetchone()
            if not result:
                conn.execute(text("ALTER TABLE sessions ADD COLUMN user_id VARCHAR(64) NOT NULL DEFAULT 'anonymous'"))
                conn.execute(text("ALTER TABLE sessions ADD INDEX idx_user_id (user_id)"))
    except Exception:
        # Column already exists or other error, ignore
        pass


def _db_create_session(user_id: str = "anonymous") -> str:
    sid = uuid.uuid4().hex
    with engine.begin() as conn:
        try:
            # Try with all columns first
            conn.execute(insert(sessions_table).values(id=sid, user_id=user_id, status="active"))
        except Exception:
            # Fallback for databases without new columns - use raw SQL
            conn.execute(text(f"INSERT INTO sessions (id, user_id) VALUES ('{sid}', '{user_id}')"))
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


def _db_close_session(session_id: str) -> None:
    with engine.begin() as conn:
        try:
            # Try to update status column
            conn.execute(
                sessions_table.update()
                .where(sessions_table.c.id == session_id)
                .values(status="closed")
            )
        except Exception:
            # Fallback: delete the session if status column doesn't exist - use raw SQL
            conn.execute(text(f"DELETE FROM sessions WHERE id = '{session_id}'"))


def _db_get_session_status(session_id: str) -> Optional[str]:
    with engine.begin() as conn:
        try:
            # Try to get status column
            result = conn.execute(
                select(sessions_table.c.status)
                .where(sessions_table.c.id == session_id)
            ).first()
            return result.status if result else None
        except Exception:
            # Fallback: check if session exists (assume active if it does) - use raw SQL
            result = conn.execute(text(f"SELECT id FROM sessions WHERE id = '{session_id}'")).first()
            return "active" if result else None


def _db_get_sessions_by_user(user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    with engine.begin() as conn:
        try:
            # Try to get sessions with all columns
            result = conn.execute(
                select(sessions_table.c.id, sessions_table.c.user_id, sessions_table.c.created_at, sessions_table.c.status)
                .where(sessions_table.c.user_id == user_id)
                .order_by(sessions_table.c.created_at.desc())
                .limit(limit)
            ).all()
            return [
                {
                    "id": r.id,
                    "user_id": r.user_id,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                    "status": r.status
                }
                for r in result
            ]
        except Exception:
            # Fallback: use raw SQL
            result = conn.execute(
                text(f"SELECT id, user_id, created_at, status FROM sessions WHERE user_id = '{user_id}' ORDER BY created_at DESC LIMIT {limit}")
            ).all()
            return [
                {
                    "id": r[0],
                    "user_id": r[1],
                    "created_at": r[2].isoformat() if r[2] else None,
                    "status": r[3] if len(r) > 3 else "active"
                }
                for r in result
            ]


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


def get_greeting_message(lang: str) -> str:
    """Get greeting message based on language code."""
    greetings = {
        "uz": "Salom! Men Trastpay yordamchingizman. Bugun sizga qanday yordam bera olishim mumkin?",
        "ru": "Здравствуйте! Я ваш помощник Trastpay. Чем могу помочь вам сегодня?",
        "en": "Hello! I'm your Trastpay assistant. How can I assist you today?"
    }
    return greetings.get(lang, greetings["uz"])  # Default to Uzbek


# -------- Response Wrappers ---------
def create_success_response(data: Any, meta: Optional[MetaInfo] = None) -> JSONResponse:
    """Create a standardized success response."""
    trace_id = str(uuid.uuid4())
    envelope = SuccessEnvelope(data=data, meta=meta, trace_id=trace_id)
    return JSONResponse(content=envelope.model_dump(), status_code=200)

def create_error_response(
    code: str, 
    message: str, 
    details: Optional[Union[Dict[str, Any], List[Any]]] = None,
    status_code: int = 500
) -> JSONResponse:
    """Create a standardized error response."""
    trace_id = str(uuid.uuid4())
    error_info = ErrorInfo(code=code, message=message, details=details)
    envelope = ErrorEnvelope(error=error_info, trace_id=trace_id)
    return JSONResponse(content=envelope.model_dump(), status_code=status_code)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms


def get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        # Lazy-load the embedding model to avoid slow API startup
        _embed_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embed_model


async def embed_query(text: str) -> np.ndarray:
    # API backend (OpenAI-compatible /embeddings)
    if EMBEDDING_BACKEND == "api":
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if OPENAI_API_KEY:
            headers["Authorization"] = f"Bearer {OPENAI_API_KEY}"
        base_body: Dict[str, Any] = {"model": EMBEDDING_MODEL, "input": [text]}
        alt_body: Dict[str, Any] = {"model": EMBEDDING_MODEL, "input": text}
        if "api.nvidia.com" in OPENAI_BASE_URL:
            # NVIDIA specifics: top-level encoding_format and truncate
            for b in (base_body, alt_body):
                b["encoding_format"] = "float"
                b["truncate"] = "NONE"
            alt_body["input_type"] = "query"
        url = f"{OPENAI_BASE_URL.rstrip('/')}/embeddings"
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                # Attempt 1
                resp = await client.post(url, headers=headers, json=base_body)
                if resp.status_code >= 400:
                    # Attempt 2 with alternative body shape
                    resp2 = await client.post(url, headers=headers, json=alt_body)
                    resp2.raise_for_status()
                    data = resp2.json()
                else:
                    data = resp.json()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Embedding upstream error: {e}")

        try:
            emb = data["data"][0]["embedding"]
        except Exception:
            raise HTTPException(status_code=500, detail="Invalid embeddings response")
        vec = np.asarray([emb], dtype=np.float32)
        return l2_normalize(vec)

    # Local backend (SentenceTransformers)
    model = get_embed_model()
    vec = await anyio.to_thread.run_sync(lambda: model.encode([text], show_progress_bar=False))
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


class ChatStartRequest(BaseModel):
    lang: str = Field(default="uz", description="Language code: uz, ru, or en")
    user_id: str = Field(description="User identifier")


class ChatStartResponse(BaseModel):
    session_id: str
    greeting: str


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
    status: str
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
)
async def ask(req: AskRequest):
    if _faiss_index is None or not _metas:
        return create_error_response(
            code="RAG_STORE_NOT_LOADED",
            message="RAG store not loaded",
            status_code=500
        )

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
    qvec = await embed_query(req.query)
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
                return create_error_response(
                    code="INVALID_REQUEST",
                    message="Invalid request to chat endpoint",
                    status_code=422
                )
            if resp.status_code >= 400:
                return create_error_response(
                    code="UPSTREAM_ERROR",
                    message=f"Upstream error: {resp.status_code}",
                    status_code=500
                )
            data = resp.json()
    except Exception as e:
        return create_error_response(
            code="UPSTREAM_EXCEPTION",
            message=f"Upstream exception: {str(e)}",
            details={"exception_type": type(e).__name__},
            status_code=500
        )

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

    response_data = AskResponse(answer=answer or "", retrieved=typed_items, suggestions=suggestions)
    return create_success_response(response_data.model_dump())


@app.get("/health", tags=["Health"], summary="Health check")
def health():
    return create_success_response({"status": "ok"})


# ------------- Chat Endpoints -------------

@app.post(
    "/chat/start",
    tags=["Chat"],
    summary="Start a new chat session",
    description="Creates and returns a new session_id with a greeting message in the specified language. You can also omit session_id in POST /chat/send to auto-create one.",
)
def chat_start(req: ChatStartRequest):
    sid = _db_create_session(req.user_id)
    greeting = get_greeting_message(req.lang)
    # Save greeting message to database
    _db_add_message(sid, "assistant", greeting)
    response_data = ChatStartResponse(session_id=sid, greeting=greeting)
    return create_success_response(response_data.model_dump())


@app.post(
    "/chat/send",
    tags=["Chat"],
    summary="Send a message (creates session if missing)",
    description="Sends a user message, performs RAG on the latest turn, and returns the assistant reply. If session_id is not provided, a new session is created and included in the response.",
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
async def chat_send(req: ChatSendRequest):
    if _faiss_index is None or not _metas:
        return create_error_response(
            code="RAG_STORE_NOT_LOADED",
            message="RAG store not loaded",
            status_code=500
        )

    # Create or retrieve session
    sid = req.session_id or await anyio.to_thread.run_sync(_db_create_session)
    
    # Check if session exists and is active
    if req.session_id:
        session_status = await anyio.to_thread.run_sync(_db_get_session_status, sid)
        if not session_status:
            return create_error_response(
                code="SESSION_NOT_FOUND",
                message="Session not found",
                status_code=404
            )
        if session_status == "closed":
            return create_error_response(
                code="SESSION_CLOSED",
                message="Session is closed. Please start a new chat.",
                status_code=400
            )
    
    # Append user message to history
    user_text = req.message.strip()
    if not user_text:
        return create_error_response(
            code="EMPTY_MESSAGE",
            message="Empty message",
            status_code=422
        )
    # Add user message to database immediately
    await anyio.to_thread.run_sync(_db_add_message, sid, "user", user_text)
    
    # Get updated history including the user message
    history = await anyio.to_thread.run_sync(_db_get_history, sid, 100)

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
    qvec = await embed_query(user_text)
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
                return create_error_response(
                    code="INVALID_REQUEST",
                    message="Invalid request to chat endpoint",
                    status_code=422
                )
            if resp.status_code >= 400:
                return create_error_response(
                    code="UPSTREAM_ERROR",
                    message=f"Upstream error: {resp.status_code}",
                    status_code=500
                )
            data = resp.json()
    except Exception as e:
        return create_error_response(
            code="UPSTREAM_EXCEPTION",
            message=f"Upstream exception: {str(e)}",
            details={"exception_type": type(e).__name__},
            status_code=500
        )

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

    response_data = ChatSendResponse(
        session_id=sid,
        answer=answer or "",
        retrieved=typed_items,
        suggestions=suggestions,
        history=[ChatMessage(**m) for m in (history + [{"role": "assistant", "content": answer}])[-20:]],
    )
    return create_success_response(response_data.model_dump())


@app.get("/chat/{session_id}/history", tags=["Chat"], summary="Get chat history")
def chat_history(session_id: str):
    hist = _db_get_history(session_id, 200)
    session_status = _db_get_session_status(session_id)
    if not session_status:
        return create_error_response(
            code="SESSION_NOT_FOUND",
            message="Session not found",
            status_code=404
        )
    
    response_data = ChatHistoryResponse(
        session_id=session_id, 
        status=session_status,
        history=[ChatMessage(**m) for m in hist[-50:]]
    )
    return create_success_response(response_data.model_dump())


@app.post("/chat/{session_id}/close", tags=["Chat"], summary="Close chat session")
def chat_close(session_id: str):
    session_status = _db_get_session_status(session_id)
    if not session_status:
        return create_error_response(
            code="SESSION_NOT_FOUND",
            message="Session not found",
            status_code=404
        )
    if session_status == "closed":
        return create_error_response(
            code="SESSION_ALREADY_CLOSED",
            message="Session already closed",
            status_code=400
        )
    
    _db_close_session(session_id)
    return create_success_response({"status": "closed", "session_id": session_id})


@app.delete("/chat/{session_id}", tags=["Chat"], summary="Reset chat session")
def chat_reset(session_id: str):
    _db_clear_session(session_id)
    return create_success_response({"status": "cleared", "session_id": session_id})


@app.get("/chat/sessions", tags=["Chat"], summary="Get sessions by user_id")
def get_sessions(
    user_id: str = Query(description="User identifier"),
    limit: int = Query(default=50, ge=1, le=100, description="Maximum number of sessions to return")
):
    sessions = _db_get_sessions_by_user(user_id, limit)
    return create_success_response(sessions)


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
