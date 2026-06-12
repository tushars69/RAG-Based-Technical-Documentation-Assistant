"""
ingest.py — One-time script to pre-load your document corpus.

Run this BEFORE starting the FastAPI server:
    python ingest.py

Ingests FastAPI documentation pages from the official docs.
You can swap these URLs for any technical docs you prefer.
"""

import sys
from app.main import chunk_and_store, fetch_url_as_markdown

# ── Your document corpus ──────────────────────────────────────────────────────
# These are FastAPI official docs pages — well-structured, good for demo
CORPUS = [
    {
        "url": "https://fastapi.tiangolo.com/tutorial/first-steps/",
        "title": "FastAPI - First Steps"
    },
    {
        "url": "https://fastapi.tiangolo.com/tutorial/path-params/",
        "title": "FastAPI - Path Parameters"
    },
    {
        "url": "https://fastapi.tiangolo.com/tutorial/body/",
        "title": "FastAPI - Request Body"
    },
    {
        "url": "https://fastapi.tiangolo.com/tutorial/security/oauth2-jwt/",
        "title": "FastAPI - OAuth2 with JWT"
    },
    {
        "url": "https://fastapi.tiangolo.com/tutorial/handling-errors/",
        "title": "FastAPI - Handling Errors"
    },
    {
        "url": "https://docs.langchain.com/oss/python/langgraph/agentic-rag",
        "title": "Agentic RAG Tutorial"
    },
    {
        "url": "https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_adaptive_rag/",
        "title": "Adaptive RAG Tutorial"
    },
    {

        "url": "https://raw.githubusercontent.com/langchain-ai/langgraph/main/examples/rag/langgraph_crag.ipynb",
        "title": "Corrective RAG (CRAG) Notebook"
    },
    {
        "url": "https://blog.langchain.com/agentic-rag-with-langgraph/",
        "title": "Self-RAG Blog Post"
    },
    {
        "url": "https://langchain-opentutorial.gitbook.io/langchain-opentutorial/17-langgraph/02-structures/06-langgraph-agentic-rag",
        "title": "Community - Adaptive RAG Walkthrough"
    },
    {
        "url": "https://blog.futuresmart.ai/langgraph-rag-agent-tutorial-basics-to-advanced-multi-agent-ai-chatbot",
        "title": "FutureSmart AI - LangGraph RAG Agent"
    }
]


def main():
    print("=" * 60)
    print("RAG Assistant — Document Ingestion")
    print("=" * 60)

    total_chunks = 0
    errors = []

    for item in CORPUS:
        url = item["url"]
        title = item["title"]

        print(f"\n→ Fetching: {title}")
        print(f"  URL: {url}")

        try:
            text = fetch_url_as_markdown(url)
            count = chunk_and_store(text=text, source=url, title=title)
            total_chunks += count
            print(f"  ✓ {count} chunks indexed")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            errors.append({"url": url, "error": str(e)})

    print("\n" + "=" * 60)
    print(f"Ingestion complete: {total_chunks} chunks indexed")
    if errors:
        print(f"Errors ({len(errors)}):")
        for e in errors:
            print(f"  - {e['url']}: {e['error']}")
    print("=" * 60)
    print("\nYou can now start the server with:")
    print("  uvicorn app.main:app --reload")


if __name__ == "__main__":
    main()
    