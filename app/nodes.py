"""
The four LangGraph nodes that form the RAG pipeline.

Each node:
  - Receives the full RAGState
  - Does one focused job
  - Returns a dict of only the fields it updates

Node order: query_analysis → retrieval → grading → generation
                               ↑                        |
                               └── (retry if needed) ───┘
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.state import RAGState
from app.vector_store import vector_store

load_dotenv()

# ── LLM Setup ─────────────────────────────────────────────────────────────────
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)


# ── Node 1: Query Analysis ─────────────────────────────────────────────────────
def query_analysis_node(state: RAGState) -> dict:
    print(f"\n[Node 1] Query Analysis | retry={state['retry_count']}")

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at reformulating technical questions for better document retrieval.

Your job:
1. Expand abbreviations and add relevant technical synonyms
2. Make the intent explicit (e.g., "how to" vs "what is" vs "why")
3. Add context that helps find the right documentation section

If this is a retry (retry_count > 0), the previous query failed to find relevant docs.
Try a significantly different phrasing or break it into simpler terms.

Return ONLY the rewritten query. No explanation, no preamble."""),
        ("human", """Original question: {question}
Retry count: {retry_count}
Previous rewritten query: {previous_query}

Rewrite this question for better technical document retrieval:""")
    ])

    chain = prompt | llm | StrOutputParser()
    rewritten = chain.invoke({
        "question": state["question"],
        "retry_count": state["retry_count"],
        "previous_query": state.get("rewritten_query", "none")
    })

    print(f"[Node 1] Rewritten: {rewritten.strip()}")
    return {"rewritten_query": rewritten.strip()}


# ── Node 2: Retrieval ──────────────────────────────────────────────────────────
def retrieval_node(state: RAGState) -> dict:
    print(f"\n[Node 2] Retrieval | query='{state['rewritten_query'][:60]}...'")

    documents = vector_store.similarity_search(query=state["rewritten_query"], k=5)

    print(f"[Node 2] Retrieved {len(documents)} chunks")
    for i, doc in enumerate(documents):
        print(f"  chunk {i+1}: score={doc.metadata.get('similarity_score')} | "
              f"source={doc.metadata.get('source', 'unknown')[:40]}")

    return {"documents": documents}


# ── Node 3: Document Grading ───────────────────────────────────────────────────
def grading_node(state: RAGState) -> dict:
    print(f"\n[Node 3] Grading {len(state['documents'])} chunks...")

    grading_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a relevance grader for a RAG system.

Given a user question and a document chunk, decide if the chunk contains
information that would help answer the question.

Be generous — if the chunk is even partially relevant, grade it as relevant.
Only grade as irrelevant if the chunk is completely off-topic.

Respond with ONLY one word: "relevant" or "irrelevant"."""),
        ("human", """Question: {question}

Document chunk:
{chunk}

Grade (relevant/irrelevant):""")
    ])

    grading_chain = grading_prompt | llm | StrOutputParser()
    relevant_docs = []

    for i, doc in enumerate(state["documents"]):
        grade = grading_chain.invoke({
            "question": state["question"],
            "chunk": doc.page_content[:1000]
        }).strip().lower()

        print(f"  chunk {i+1}: {grade} | {doc.page_content[:60]}...")

        if "relevant" in grade:
            relevant_docs.append(doc)

    all_irrelevant = len(relevant_docs) == 0
    print(f"[Node 3] {len(relevant_docs)}/{len(state['documents'])} relevant | "
          f"all_irrelevant={all_irrelevant}")

    return {"relevant_documents": relevant_docs, "all_irrelevant": all_irrelevant}


# ── Node 4: Generation ─────────────────────────────────────────────────────────
def generation_node(state: RAGState) -> dict:
    print(f"\n[Node 4] Generation | {len(state['relevant_documents'])} context chunks")

    context_parts = []
    for i, doc in enumerate(state["relevant_documents"]):
        source = doc.metadata.get("source", f"Document {i+1}")
        context_parts.append(f"[Source {i+1}: {source}]\n{doc.page_content}")

    context = "\n\n---\n\n".join(context_parts)

    generation_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a technical documentation assistant.

Rules:
1. Answer ONLY using the provided context. Do not use outside knowledge.
2. Be precise and technical — your audience are developers.
3. After each key claim, cite the source like: [Source 1] or [Source 2]
4. If the context partially answers the question, answer what you can and say what's missing.
5. Use markdown formatting: code blocks for code, bullet points for lists.
6. Keep the answer focused and concise."""),
        ("human", """Context from documentation:
{context}

Question: {question}

Answer (with citations):""")
    ])

    chain = generation_prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": state["question"]})

    print(f"[Node 4] Answer generated ({len(answer)} chars)")
    return {"answer": answer}


# ── Fallback Node ──────────────────────────────────────────────────────────────
def fallback_node(state: RAGState) -> dict:
    print(f"\n[Fallback] No relevant docs after {state['retry_count']} retries")
    answer = (
        f"I wasn't able to find relevant information in the documentation "
        f"to answer: **{state['question']}**\n\n"
        f"This question may be outside the scope of the indexed documents. "
        f"Try rephrasing, or check the official documentation directly."
    )
    return {"answer": answer}


# ── Routing Function ───────────────────────────────────────────────────────────
def route_after_grading(state: RAGState) -> str:
    if not state["all_irrelevant"]:
        return "generate"

    if state["retry_count"] < state["max_retries"]:
        print(f"[Router] No relevant docs — retrying "
              f"({state['retry_count'] + 1}/{state['max_retries']})")
        return "rewrite"

    print(f"[Router] Retry limit reached — falling back")
    return "fallback"
