#  RAG-Based Technical Documentation Assistant

An intelligent, self-correcting Retrieval-Augmented Generation (RAG) system built with **FastAPI**, **LangGraph**, and **ChromaDB**. This project serves as a technical documentation assistant capable of answering complex queries, routing around irrelevant data, fact-checking its own answers, and falling back to real-time web searches when local knowledge fails.

Built for the **Express Analytics AI/ML Engineer Intern** assignment.


## What It Does

A user asks a natural language question. The system:


Rewrites the query for better retrieval (expanding abbreviations, resolving pronouns from chat history)
Retrieves the most semantically similar document chunks from ChromaDB
Grades each chunk — irrelevant chunks are filtered out before generation
Retries with a different query if no relevant chunks were found (up to 2 times)
Falls back to web search via Tavily if the vector store has nothing useful
Generates a grounded answer with inline source citations
Checks for hallucination — if the answer makes claims not in the context, it auto-regenerates with a stricter prompt
Returns the final answer with sources, hallucination score, and a flag indicating whether web search was used


##  Features

### Core Capabilities
* **LangGraph Workflow:** A stateful, cyclical graph that intelligently routes queries based on continuous self-evaluation.
* **Self-Corrective Routing:** An LLM-powered grader strictly evaluates retrieved documents. If all documents are irrelevant, the system rewrites the query and retries.
* **FastAPI Backend:** Fully featured REST API serving endpoints for querying, document ingestion, and chat history management.
* **Local Vector Database:** ChromaDB implementation using localized ONNX embeddings for fast, private document retrieval.

### Advanced Pipeline Additions
* **Hallucination Checker:** A post-generation verification node cross-references the LLM's answer with the source documents. If a hallucination is detected, it triggers a strict, mandatory-citation regeneration.
* **Tavily Web Search Fallback:** If the local document corpus fails to answer the question after multiple retries, the system seamlessly falls back to a web search.
* **Contextual Memory (Entity Resolution):** The query analyzer aggressively resolves pronouns and abstract references from previous conversational turns to maintain precise technical context.
* **Streamlit Interactive UI:** A custom frontend featuring real-time 
* **One-click Chat Log Exports:** It allows users to download their entire session history as a formatted `.txt` file.


## Project Structure

RAG-Based-Technical-Documentation-Assistant/
├── app/
│   ├── __init__.py
│   ├── state.py         # RAGState TypedDict — shared memory between nodes
│   ├── vector_store.py  # ChromaDB wrapper with embeddings
│   ├── nodes.py         # All pipeline nodes + routing functions
│   ├── graph.py         # LangGraph StateGraph assembly
│   ├── main.py          # FastAPI app + session management
│   └── frontend.py      # Streamlit UI
├── ingest.py            # One-time corpus ingestion script
├── pyproject.toml       # Poetry dependencies
└── README.md


## Architecture & Workflow

The pipeline utilizes a self-reflective graph architecture to ensure high-accuracy, grounded responses.

```mermaid
graph TD
    %% Styling
    classDef primary fill:#005f73,stroke:#00e5ff,stroke-width:2px,color:#fff;
    classDef secondary fill:#0a0908,stroke:#00e5ff,stroke-width:1px,color:#fff;
    classDef decision fill:#94d2bd,stroke:#005f73,stroke-width:2px,color:#000;
    classDef terminal fill:#e9d8a6,stroke:#ee9b00,stroke-width:2px,color:#000;

    %% Nodes
    START((Start)) --> QA[Query Analysis]:::primary
    QA --> RET[(ChromaDB Retrieval)]:::secondary
    RET --> GR{Document Grader}:::decision
    
    %% Routing logic from Grader
    GR -- "Relevant Docs Found" --> GEN[Generation]:::primary
    GR -- "All Irrelevant (Retry < Max)" --> QA
    GR -- "All Irrelevant (Retries Exhausted)" --> WEB[Tavily Web Search]:::secondary
    
    %% Web Search Routing
    WEB --> GEN
    
    %% Hallucination logic
    GEN --> HC{Hallucination Check}:::decision
    HC -- "Grounded" --> END((End: Output)):::terminal
    HC -- "Hallucinated" --> FIX[Strict Regeneration]:::primary
    FIX --> END