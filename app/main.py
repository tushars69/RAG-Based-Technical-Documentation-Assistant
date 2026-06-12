"""
FastAPI application — exposes the RAG pipeline via HTTP.

Endpoints:
  POST /session         — Create a new chat session
  POST /query           — Ask a question (with optional session_id)
  POST /ingest          — Add new documents from URLs
  POST /ingest/file     — Upload a .md / .txt / .html file
  GET  /documents       — List all indexed documents
  POST /feedback        — Submit thumbs up/down on an answer
  GET  /session/{id}    — Get chat history for a session
  DELETE /session/{id}  — Clear a session's history
"""

import uuid
from datetime import datetime
from typing import Optional, List
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

from app.graph import rag_graph
from app.vector_store import vector_store

load_dotenv()

# ── App Setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="RAG Documentation Assistant",
    description="Self-corrective RAG pipeline with LangGraph + conversation memory.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── In-memory stores ───────────────────────────────────────────────────────────
feedback_store: List[dict] = []

# session_id → list of LangChain messages
# In production this would be Redis or a DB
session_store: dict = {}


# ── Schemas ────────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    max_retries: int = 2
    session_id: Optional[str] = None   # if None, stateless single-turn query

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[dict]
    retry_count: int
    query_id: str
    hallucination_score: str
    web_search_used: bool
    session_id: Optional[str] = None   # echoed back so client can reuse it

class IngestURLRequest(BaseModel):
    urls: List[str]
    title: Optional[str] = None

class FeedbackRequest(BaseModel):
    query_id: str
    rating: int
    comment: Optional[str] = None

# ── Text Splitter ──────────────────────────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=80,
    separators=["\n\n", "\n", ". ", " ", ""]
)


# ── Helpers ────────────────────────────────────────────────────────────────────
def fetch_url_as_markdown(url: str) -> str:
    response = requests.get(url, timeout=15, headers={"User-Agent": "RAG-Bot/1.0"})
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["nav", "footer", "script", "style", "header"]):
        tag.decompose()
    main = soup.find("main") or soup.find("article") or soup.find("body")
    return markdownify(str(main), heading_style="ATX")


def chunk_and_store(text: str, source: str, title: str = "") -> int:
    chunks = splitter.split_text(text)
    documents = [
        Document(
            page_content=chunk,
            metadata={
                "source": source,
                "title": title or source,
                "chunk_index": i,
                "ingested_at": datetime.utcnow().isoformat()
            }
        )
        for i, chunk in enumerate(chunks)
    ]
    return vector_store.add_documents(documents)


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "RAG Documentation Assistant",
        "docs": "/docs",
        "chunks_indexed": vector_store.get_chunk_count(),
        "active_sessions": len(session_store)
    }


@app.post("/session")
def create_session():
    """
    Create a new chat session.
    Returns a session_id to pass in subsequent /query calls.
    """
    session_id = str(uuid.uuid4())
    session_store[session_id] = []
    print(f"[Session] Created: {session_id}")
    return {"session_id": session_id}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    Ask a question. Optionally pass a session_id for multi-turn conversation.

    Without session_id → stateless, no memory between calls
    With session_id    → full conversation history sent to LLM
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if vector_store.get_chunk_count() == 0:
        raise HTTPException(
            status_code=503,
            detail="No documents indexed yet. POST to /ingest first."
        )

    # Load chat history if session_id provided
    session_id = request.session_id
    chat_history = []

    if session_id:
        if session_id not in session_store:
            # Auto-create session if ID not found
            session_store[session_id] = []
        chat_history = session_store[session_id]
        print(f"[Session] {session_id} | history length: {len(chat_history)}")

    initial_state = {
        "question": request.question,
        "chat_history": chat_history,
        "rewritten_query": "",
        "documents": [],
        "relevant_documents": [],
        "all_irrelevant": False,
        "answer": "",
        "hallucination_score": "grounded",
        "hallucination_feedback": "",
        "web_search_used": False,
        "web_search_results": [],
        "retry_count": 0,
        "max_retries": request.max_retries
    }

    final_state = rag_graph.invoke(initial_state)
    answer = final_state["answer"]

    # Save this Q&A turn to session history
    if session_id and session_id in session_store:
        session_store[session_id].append(HumanMessage(content=request.question))
        session_store[session_id].append(AIMessage(content=answer))
        print(f"[Session] {session_id} | saved turn, "
              f"history now: {len(session_store[session_id])} messages")

    # Build sources
    sources = []
    seen_sources = set()
    for doc in final_state.get("relevant_documents", []):
        src = doc.metadata.get("source", "unknown")
        if src not in seen_sources:
            seen_sources.add(src)
            sources.append({
                "source": src,
                "title": doc.metadata.get("title", src),
                "similarity_score": doc.metadata.get("similarity_score"),
                "from_web": doc.metadata.get("from_web", False)
            })

    return QueryResponse(
        question=request.question,
        answer=answer,
        sources=sources,
        retry_count=final_state["retry_count"],
        query_id=str(uuid.uuid4()),
        hallucination_score=final_state.get("hallucination_score", "grounded"),
        web_search_used=final_state.get("web_search_used", False),
        session_id=session_id
    )


@app.get("/session/{session_id}")
def get_session_history(session_id: str):
    """Get the full chat history for a session."""
    if session_id not in session_store:
        raise HTTPException(status_code=404, detail="Session not found")

    history = session_store[session_id]
    formatted = []
    for msg in history:
        formatted.append({
            "role": "user" if isinstance(msg, HumanMessage) else "assistant",
            "content": msg.content
        })

    return {
        "session_id": session_id,
        "message_count": len(formatted),
        "history": formatted
    }


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    """Clear a session's chat history."""
    if session_id not in session_store:
        raise HTTPException(status_code=404, detail="Session not found")

    session_store[session_id] = []
    return {"status": "cleared", "session_id": session_id}


@app.post("/ingest")
def ingest_urls(request: IngestURLRequest):
    results = []
    for url in request.urls:
        try:
            print(f"[Ingest] Fetching {url}")
            text = fetch_url_as_markdown(url)
            count = chunk_and_store(text=text, source=url, title=request.title or url)
            results.append({"url": url, "status": "success", "chunks_added": count})
        except Exception as e:
            results.append({"url": url, "status": "error", "error": str(e)})

    return {
        "results": results,
        "total_chunks_in_db": vector_store.get_chunk_count()
    }


@app.post("/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    allowed = {".md", ".txt", ".html"}
    suffix = Path(file.filename).suffix.lower()

    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {allowed}"
        )

    content = await file.read()
    text = content.decode("utf-8", errors="ignore")

    if suffix == ".html":
        text = markdownify(text)

    count = chunk_and_store(text=text, source=file.filename, title=file.filename)
    return {
        "filename": file.filename,
        "chunks_added": count,
        "total_chunks_in_db": vector_store.get_chunk_count()
    }


@app.get("/documents")
def list_documents():
    docs = vector_store.list_documents()
    return {
        "documents": docs,
        "total_documents": len(docs),
        "total_chunks": vector_store.get_chunk_count()
    }


@app.post("/feedback")
def submit_feedback(request: FeedbackRequest):
    if request.rating not in (1, -1):
        raise HTTPException(status_code=400, detail="Rating must be 1 or -1")

    record = {
        "query_id": request.query_id,
        "rating": request.rating,
        "comment": request.comment,
        "timestamp": datetime.utcnow().isoformat()
    }
    feedback_store.append(record)
    return {"status": "recorded", "feedback_id": str(uuid.uuid4())}


@app.get("/feedback/summary")
def feedback_summary():
    if not feedback_store:
        return {"total": 0, "thumbs_up": 0, "thumbs_down": 0}

    ups = sum(1 for f in feedback_store if f["rating"] == 1)
    return {
        "total": len(feedback_store),
        "thumbs_up": ups,
        "thumbs_down": len(feedback_store) - ups,
        "approval_rate": round(ups / len(feedback_store), 2)
    }
