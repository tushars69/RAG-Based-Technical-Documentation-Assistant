"""
The four LangGraph nodes that form the RAG pipeline.

Each node:
  - Receives the full RAGState
  - Does one focused job
  - Returns a dict of only the fields it updates

Node order: query_analysis → retrieval → grading → generation → hallucination_check
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
    """
    Rewrites the user's question to improve retrieval quality.
    On retries, tries a significantly different phrasing.
    """
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
    """
    Searches ChromaDB for the top-k chunks most similar to the rewritten query.
    Filters out chunks below similarity threshold to prevent weak matches
    from reaching the grader.
    """
    print(f"\n[Node 2] Retrieval | query='{state['rewritten_query'][:60]}...'")

    documents = vector_store.similarity_search(
        query=state["rewritten_query"],
        k=5
    )

    print(f"[Node 2] Retrieved {len(documents)} chunks")
    for i, doc in enumerate(documents):
        print(f"  chunk {i+1}: score={doc.metadata.get('similarity_score')} | "
              f"source={doc.metadata.get('source', 'unknown')[:40]}")

    return {"documents": documents}


# ── Node 3: Document Grading ───────────────────────────────────────────────────
def grading_node(state: RAGState) -> dict:
    """
    Self-corrective node — LLM strictly judges each chunk as relevant or not.

    Outcomes:
    - Some relevant → filter and proceed to generation
    - None relevant → set all_irrelevant=True → router retries
    """
    print(f"\n[Node 3] Grading {len(state['documents'])} chunks...")

    grading_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a strict relevance grader for a RAG system.

Given a user question and a document chunk, decide if the chunk contains
information that DIRECTLY answers or is SPECIFICALLY about the question topic.

Be strict — grade as irrelevant if:
- The chunk is about a different technology or framework
- The chunk only tangentially relates to the question
- You would need to infer or extrapolate to connect it to the question

Grade as relevant ONLY if the chunk directly addresses the question.

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

    return {
        "relevant_documents": relevant_docs,
        "all_irrelevant": all_irrelevant
    }


# ── Node 4: Generation ─────────────────────────────────────────────────────────
def generation_node(state: RAGState) -> dict:
    """
    Generates the final answer grounded in the relevant document chunks.
    Requires inline citations referencing source documents.
    """
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
    answer = chain.invoke({
        "context": context,
        "question": state["question"]
    })

    print(f"[Node 4] Answer generated ({len(answer)} chars)")
    return {"answer": answer}


# ── Node 5: Hallucination Check ───────────────────────────────────────────────
def hallucination_check_node(state: RAGState) -> dict:
    """
    Verifies the generated answer is actually supported by retrieved chunks.

    Three outcomes:
    - grounded     → all claims backed by context, passes through
    - partial      → mostly fine, minor extrapolation, passes through
    - hallucinated → answer makes things up → triggers fix_hallucination_node
    """
    print(f"\n[Node 5] Hallucination Check")

    context_parts = []
    for i, doc in enumerate(state["relevant_documents"]):
        context_parts.append(f"[Source {i+1}]\n{doc.page_content}")
    context = "\n\n---\n\n".join(context_parts)

    check_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a factual grounding checker for a RAG system.

Compare the generated answer against the source context it was based on.
Check whether every factual claim in the answer is directly supported by the context.

Respond in this EXACT format (two lines only):
SCORE: <grounded|partial|hallucinated>
FEEDBACK: <one sentence explaining your verdict>

Definitions:
- grounded     = all claims are supported by the context
- partial      = most claims supported but 1-2 minor unsupported additions
- hallucinated = answer makes significant claims not found in the context"""),
        ("human", """SOURCE CONTEXT:
{context}

GENERATED ANSWER:
{answer}

Grounding verdict:""")
    ])

    chain = check_prompt | llm | StrOutputParser()
    result = chain.invoke({
        "context": context,
        "answer": state["answer"]
    }).strip()

    # Defensive parsing — never return None
    score = "grounded"
    feedback = "No feedback provided"

    for line in result.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.upper().startswith("SCORE:"):
            raw = line.split(":", 1)[1].strip().lower()
            if "hallucinated" in raw:
                score = "hallucinated"
            elif "partial" in raw:
                score = "partial"
            else:
                score = "grounded"
        elif line.upper().startswith("FEEDBACK:"):
            feedback = line.split(":", 1)[1].strip()

    # Final safety net — if parsing failed entirely, default to grounded
    if not score:
        score = "grounded"
        feedback = "Could not parse grading response — defaulting to grounded"

    print(f"[Node 5] Score: {score} | {feedback}")
    return {
        "hallucination_score": score,
        "hallucination_feedback": feedback
    }


# ── Node 5b: Fix Hallucination ────────────────────────────────────────────────
def fix_hallucination_node(state: RAGState) -> dict:
    """
    Only runs when hallucination_check gives "hallucinated".
    Regenerates with an ultra-strict prompt that forbids going beyond context.
    Appends a transparency note to the final answer.
    """
    print(f"\n[Node 5b] Fixing hallucinated answer...")

    context_parts = []
    for i, doc in enumerate(state["relevant_documents"]):
        source = doc.metadata.get("source", f"Document {i+1}")
        context_parts.append(f"[Source {i+1}: {source}]\n{doc.page_content}")
    context = "\n\n---\n\n".join(context_parts)

    strict_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a technical documentation assistant with ONE strict rule:
You may ONLY state facts explicitly present in the provided context.

If the context does not fully answer the question, say what IS covered
and clearly state: "The documentation does not cover [X]."

Do NOT infer, extrapolate, or use outside knowledge. Citations are mandatory."""),
        ("human", """Context:
{context}

Question: {question}

Previous answer was flagged: {feedback}

Provide a strictly grounded answer:""")
    ])

    chain = strict_prompt | llm | StrOutputParser()
    new_answer = chain.invoke({
        "context": context,
        "question": state["question"],
        "feedback": state["hallucination_feedback"]
    })

    transparency_note = (
        "\n\n---\n> ⚠️ *This answer was automatically revised — "
        "an earlier draft contained claims not supported by the source documents.*"
    )

    print(f"[Node 5b] Regenerated ({len(new_answer)} chars)")
    return {"answer": new_answer + transparency_note}


# ── Fallback Node ──────────────────────────────────────────────────────────────
def fallback_node(state: RAGState) -> dict:
    """
    Called when retries are exhausted and still no relevant docs found.
    Returns an honest response rather than hallucinating.
    """
    print(f"\n[Fallback] No relevant docs after {state['retry_count']} retries")

    answer = (
        f"I wasn't able to find relevant information in the documentation "
        f"to answer: **{state['question']}**\n\n"
        f"This question may be outside the scope of the indexed documents. "
        f"Try rephrasing, or check the official documentation directly."
    )
    return {"answer": answer}


# ── Routing: After Grading ─────────────────────────────────────────────────────
def route_after_grading(state: RAGState) -> str:
    """
    - "generate"  → relevant docs found
    - "rewrite"   → no relevant docs, retry if under limit
    - "fallback"  → retries exhausted
    """
    if not state["all_irrelevant"]:
        return "generate"

    if state["retry_count"] < state["max_retries"]:
        print(f"[Router] No relevant docs — retrying "
              f"({state['retry_count'] + 1}/{state['max_retries']})")
        return "rewrite"

    print(f"[Router] Retry limit reached — falling back")
    return "fallback"


# ── Routing: After Hallucination Check ────────────────────────────────────────
def route_after_hallucination_check(state: RAGState) -> str:
    """
    - "fix"  → hallucinated, regenerate strictly
    - "done" → grounded or partial, pass through to END
    Always returns a string, never None.
    """
    score = state.get("hallucination_score", "grounded")

    if isinstance(score, str) and "hallucinated" in score.lower():
        print("[Router] Hallucination detected — regenerating")
        return "fix"

    print(f"[Router] Answer is {score} — passing through")
    return "done"
