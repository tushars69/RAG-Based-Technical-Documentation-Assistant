# ⚡ RAG-Based Technical Documentation Assistant

An intelligent, self-correcting Retrieval-Augmented Generation (RAG) system built with **FastAPI**, **LangGraph**, and **ChromaDB**. This project serves as a technical documentation assistant capable of answering complex queries, routing around irrelevant data, fact-checking its own answers, and falling back to real-time web searches when local knowledge fails.

[cite_start]Built for the **Express Analytics AI/ML Engineer Intern** assignment[cite: 3, 4].

---

## 🚀 Features

### Core Requirements Implemented
* [cite_start]**LangGraph Workflow:** A stateful, cyclical graph that handles query analysis, retrieval, document grading, and generation[cite: 14].
* **Self-Corrective Routing:** An LLM-powered grader strictly evaluates retrieved documents. [cite_start]If all documents are irrelevant, the system rewrites the query and retries before falling back[cite: 37, 40].
* [cite_start]**FastAPI Backend:** Fully featured REST API serving endpoints for querying, document ingestion, listing documents, and feedback[cite: 63, 64].
* [cite_start]**Local Vector Database:** ChromaDB implementation using localized ONNX embeddings for fast, private document retrieval[cite: 35, 57].

### Bonus Features Implemented
* **Hallucination Checker:** A post-generation verification node that cross-references the LLM's answer with the source documents. [cite_start]If a hallucination is detected, it strictly regenerates a grounded response[cite: 68].
* [cite_start]**Tavily Web Search Fallback:** If the local document corpus fails to answer the question after multiple retries, the system seamlessly falls back to a web search to find the answer[cite: 69].
* [cite_start]**Conversation Memory:** The API accepts a `session_id`, allowing the LLM to understand context and resolve pronouns in follow-up questions[cite: 70].
* [cite_start]**Streamlit Interactive UI:** A custom, neon-themed frontend that visualizes the pipeline's internal state (retries, web search usage, hallucination flags) and chat history[cite: 70].

---

## 🧠 Architecture & Workflow

The system is built as a highly resilient `StateGraph` in LangGraph.

1. [cite_start]**Query Analysis:** Takes the user's raw query and rewrites it for optimal retrieval, taking chat history into account[cite: 27, 28, 29].
2. **Retrieval:** Queries ChromaDB for the top 5 chunks. [cite_start]Chunks below a `0.45` similarity threshold are aggressively filtered out[cite: 33, 34, 35].
3. **Document Grading:** An LLM acts as a strict grader, evaluating each retrieved chunk. 
   - *If relevant docs are found:* Routes to Generation.
   - [cite_start]*If all docs are irrelevant:* Routes back to Query Analysis to rewrite and retry (max 2 retries)[cite: 36, 39, 40, 41].
4. [cite_start]**Web Search Fallback:** If the retry limit is exhausted, the query is truncated and sent to the Tavily API to fetch real-time web context[cite: 69].
5. [cite_start]**Generation:** Synthesizes an answer with citations based *only* on the relevant context provided[cite: 42, 45].
6. **Hallucination Check:** A secondary LLM pass evaluates the generated answer against the source chunks. [cite_start]If flagged as hallucinated, the system triggers a strict regeneration node before returning the final answer to the user[cite: 68].

---

## 🏗️ Design Decisions & Tradeoffs

**1. [cite_start]Chunking Strategy:** I used `RecursiveCharacterTextSplitter` with a `chunk_size` of 600 and a `chunk_overlap` of 80[cite: 55]. 
* *Reasoning:* Technical documentation often contains code blocks and step-by-step lists. A 600-character chunk is large enough to capture the context of a code snippet without polluting the vector space with overly broad concepts. The 80-character overlap ensures we don't break sentences or logical flows in half.

**2. [cite_start]Embedding Strategy:** I chose ChromaDB's default `ONNXMiniLM_L6_V2` local embedding function instead of an API-based provider like OpenAI[cite: 56, 57]. 
* *Reasoning:* It provides excellent semantic search capabilities for technical text without the latency or cost of network calls, making ingestion and retrieval incredibly fast.

**3. Strict String Matching in Grading:** Initially, the grader checked if `"relevant" in grade`. This caused false positives when the LLM returned `"irrelevant"`. 
* *Fix:* The system now uses exact string matching (`grade == "relevant"`) to ensure mathematically strict gating of context.

**4. LLM Selection:** Powered by Groq (`llama-3.1-8b-instant`). 
* *Reasoning:* Groq's LPU architecture provides blazing-fast inference[cite: 109]. Because this pipeline utilizes multiple LLM calls per query (rewriting, grading, generation, hallucination checking), high-speed inference is critical for an acceptable UX.

---

## ⚙️ Setup Instructions

### Prerequisites
* Python 3.10+
* [Poetry](https://python-poetry.org/) (for dependency management)

### 1. Clone & Install
```bash
git clone <your-repo-url>
cd RAG-Based-Technical-Documentation-Assistant-main
poetry install
