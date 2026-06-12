"""
State schema for the RAG LangGraph workflow.

Every node receives this object and returns a dict of fields to update.
Think of it as the shared memory between all pipeline stages.
"""

from typing import TypedDict, List
from langchain_core.documents import Document


class RAGState(TypedDict):
    # --- Input ---
    question: str                        # Original user question (never modified)

    # --- Query Analysis ---
    rewritten_query: str                 # Expanded/clarified version for better retrieval

    # --- Retrieval ---
    documents: List[Document]            # Raw chunks returned from vector store

    # --- Grading ---
    relevant_documents: List[Document]   # Filtered subset that passed grading
    all_irrelevant: bool                 # True if zero chunks passed grading

    # --- Generation ---
    answer: str                          # Final generated answer with citations

    # --- Control Flow ---
    retry_count: int                     # How many times we've rewritten + re-retrieved
    max_retries: int                     # Ceiling to prevent infinite loops (default: 2)
    