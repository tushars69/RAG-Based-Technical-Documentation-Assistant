"""
FastAPI application — exposes the RAG pipeline via HTTP.

Endpoints:
  POST /query       — Ask a question, get answer with citations
  POST /ingest      — Add new documents from URLs
  POST /ingest/file — Upload a .md / .txt / .html file
  GET  /documents   — List all indexed documents
  POST /feedback    — Submit thumbs up/down on an answer
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
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from dotenv import load_dotenv

from app.graph import rag_graph
from app.vector_store import vector_store

load_dotenv()

# ── App Setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="RAG Documentation Assistant",
    description="Self-corrective RAG pipeline with LangGraph.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

feedback_store: List[dict] = []

# ── Schemas ────────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    max_retries: int = 2

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[dict]
    retry_count: int
    query_id: str

class IngestURLRequest(BaseModel):
    urls: List[str]
    title: Optional[str] = None

class FeedbackRequest(BaseModel):
    query_id: str
    rating: int        # 1 = thumbs up, -1 = thumbs down
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
        "chunks_indexed": vector_store.get_chunk_count()
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if vector_store.get_chunk_count() == 0:
        raise HTTPException(
            status_code=503,
            detail="No documents indexed yet. POST to /ingest first."
        )

    initial_state = {
        "question": request.question,
        "rewritten_query": "",
        "documents": [],
        "relevant_documents": [],
        "all_irrelevant": False,
        "answer": "",
        "retry_count": 0,
        "max_retries": request.max_retries
    }

    final_state = rag_graph.invoke(initial_state)

    sources = []
    seen_sources = set()
    for doc in final_state.get("relevant_documents", []):
        src = doc.metadata.get("source", "unknown")
        if src not in seen_sources:
            seen_sources.add(src)
            sources.append({
                "source": src,
                "title": doc.metadata.get("title", src),
                "similarity_score": doc.metadata.get("similarity_score")
            })

    return QueryResponse(
        question=request.question,
        answer=final_state["answer"],
        sources=sources,
        retry_count=final_state["retry_count"],
        query_id=str(uuid.uuid4())
    )


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
