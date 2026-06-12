"""
The four LangGraph nodes that form the RAG pipeline + bonus nodes.

Node order: query_analysis → retrieval → grading → generation → hallucination_check
                               ↑                        |
                               └── (retry if needed) ───┘
                                           |
                              (retries exhausted → web_search → generation)
                                           |
                                    (web fails → fallback)
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from tavily import TavilyClient

from app.state import RAGState
from app.vector_store import vector_store

load_dotenv()

# ── LLM Setup ─────────────────────────────────────────────────────────────────
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

# ── Tavily Setup ───────────────────────────────────────────────────────────────
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


# ── Node 1: Query Analysis ─────────────────────────────────────────────────────
def query_analysis_node(state: RAGState) -> dict:
    """
    Rewrites the user's question to improve retrieval quality.

    Now chat_history aware — if the user asks "what about its performance?"
    after a question about Django, the rewrite understands the context
    and expands it to "what is the performance of Django ORM?"
    """
    print(f"\n[Node 1] Query Analysis | retry={state['retry_count']}")

    chat_history = state.get("chat_history", [])
    has_history = len(chat_history) > 0

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at reformulating technical questions for better document retrieval.
The domain is strictly Software Engineering, AI, Python, and LangGraph. 
Always treat 'RAG' as 'Retrieval-Augmented Generation'.

Your job:
1. Add relevant technical synonyms if needed.
2. Make the user's intent explicit.
3. If chat history is provided, resolve any pronouns or references.
4. STRICT LENGTH LIMIT: The rewritten query MUST be concise and under 250 characters.

If this is a retry (retry_count > 0), try a different phrasing.
Return ONLY the rewritten query. No explanation, no preamble."""),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", """Original question: {question}
Retry count: {retry_count}
Previous rewritten query: {previous_query}

Rewrite this question for better technical document retrieval:""")
    ])

    chain = prompt | llm | StrOutputParser()
    rewritten = chain.invoke({
        "question": state["question"],
        "retry_count": state["retry_count"],
        "previous_query": state.get("rewritten_query", "none"),
        "chat_history": chat_history if has_history else []
    })

    print(f"[Node 1] Rewritten: {rewritten.strip()}")
    return {"rewritten_query": rewritten.strip()}


# ── Node 2: Retrieval ──────────────────────────────────────────────────────────
def retrieval_node(state: RAGState) -> dict:
    """
    Searches ChromaDB for the top-k chunks most similar to the rewritten query.
    Filters out chunks below similarity threshold.
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

        # FIX: Use exact match to avoid "irrelevant" passing the check
        if grade == "relevant":
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
    Generates the final answer grounded in relevant_documents.

    Now chat_history aware — uses previous Q&A pairs so the LLM
    understands follow-up questions and can refer back to prior answers.
    """
    print(f"\n[Node 4] Generation | {len(state['relevant_documents'])} context chunks"
          f" | web_search={state.get('web_search_used', False)}")

    chat_history = state.get("chat_history", [])

    context_parts = []
    for i, doc in enumerate(state["relevant_documents"]):
        source = doc.metadata.get("source", f"Document {i+1}")
        context_parts.append(f"[Source {i+1}: {source}]\n{doc.page_content}")
    context = "\n\n---\n\n".join(context_parts)

    source_note = (
        "web search results" if state.get("web_search_used") else "documentation"
    )

    generation_prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are a technical assistant answering from {source_note}.

Rules:
1. Answer ONLY using the provided context. Do not use outside knowledge.
2. Be precise and technical — your audience are developers.
3. After each key claim, cite the source like: [Source 1] or [Source 2]
4. If the context partially answers the question, answer what you can and say what's missing.
5. Use markdown formatting: code blocks for code, bullet points for lists.
6. Keep the answer focused and concise.
7. If the user refers to something from earlier in the conversation, use that context."""),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", """Context:
{context}

Question: {question}

Answer (with citations):""")
    ])

    chain = generation_prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "context": context,
        "question": state["question"],
        "chat_history": chat_history
    })

    print(f"[Node 4] Answer generated ({len(answer)} chars)")
    return {"answer": answer}


# ── Node 5: Hallucination Check ───────────────────────────────────────────────
def hallucination_check_node(state: RAGState) -> dict:
    """
    Verifies the generated answer is actually supported by retrieved chunks.
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
    Regenerates with an ultra-strict prompt.
    """
    print(f"\n[Node 5b] Fixing hallucinated answer...")

    context_parts = []
    for i, doc in enumerate(state["relevant_documents"]):
        source = doc.metadata.get("source", f"Document {i+1}")
        context_parts.append(f"[Source {i+1}: {source}]\n{doc.page_content}")
    context = "\n\n---\n\n".join(context_parts)

    strict_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a technical assistant with ONE strict rule:
You may ONLY state facts explicitly present in the provided context.

If the context does not fully answer the question, say what IS covered
and clearly state: "The source does not cover [X]."

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


# ── Node 6: Web Search ────────────────────────────────────────────────────────
def web_search_node(state: RAGState) -> dict:
    """
    Fallback when vector store has no relevant docs after all retries.
    Searches the web using Tavily and converts results into Documents.
    """
    # FIX: Use the rewritten, context-aware query for web search
    print(f"\n[Node 6] Web Search Fallback | query='{state['rewritten_query'][:60]}'")

    try:
        response = tavily.search(
            query=state["rewritten_query"][:390], 
            max_results=4,
            search_depth="basic"
        )

        results = response.get("results", [])

        if not results:
            print("[Node 6] No web results found")
            return {
                "web_search_used": True,
                "web_search_results": [],
                "relevant_documents": []
            }

        web_docs = []
        for r in results:
            doc = Document(
                page_content=r.get("content", ""),
                metadata={
                    "source": r.get("url", "web"),
                    "title": r.get("title", "Web Result"),
                    "similarity_score": r.get("score", 0.0),
                    "from_web": True
                }
            )
            web_docs.append(doc)

        print(f"[Node 6] Found {len(web_docs)} web results")
        for i, doc in enumerate(web_docs):
            print(f"  result {i+1}: {doc.metadata.get('title', '')[:50]}")

        return {
            "web_search_used": True,
            "web_search_results": web_docs,
            "relevant_documents": web_docs
        }

    except Exception as e:
        print(f"[Node 6] Web search error: {e}")
        return {
            "web_search_used": True,
            "web_search_results": [],
            "relevant_documents": []
        }


# ── Fallback Node ──────────────────────────────────────────────────────────────
def fallback_node(state: RAGState) -> dict:
    """
    Last resort — called when both vector store and web search fail.
    """
    print(f"\n[Fallback] No relevant docs after {state['retry_count']} retries"
          f" | web_search_used={state.get('web_search_used', False)}")

    answer = (
        f"I wasn't able to find relevant information to answer: "
        f"**{state['question']}**\n\n"
        f"I searched both the indexed documentation and the web, "
        f"but couldn't find anything relevant. "
        f"Try rephrasing your question or check the official documentation directly."
    )
    return {"answer": answer}


# ── Routing: After Grading ─────────────────────────────────────────────────────
def route_after_grading(state: RAGState) -> str:
    if not state["all_irrelevant"]:
        return "generate"

    if state["retry_count"] < state["max_retries"]:
        print(f"[Router] No relevant docs — retrying "
              f"({state['retry_count'] + 1}/{state['max_retries']})")
        return "rewrite"

    print(f"[Router] Retry limit reached — trying web search")
    return "web_search"


# ── Routing: After Web Search ──────────────────────────────────────────────────
def route_after_web_search(state: RAGState) -> str:
    if state.get("relevant_documents"):
        print("[Router] Web results found — generating answer")
        return "generate"

    print("[Router] Web search empty — falling back")
    return "fallback"


# ── Routing: After Hallucination Check ────────────────────────────────────────
def route_after_hallucination_check(state: RAGState) -> str:
    score = state.get("hallucination_score", "grounded")

    if isinstance(score, str) and "hallucinated" in score.lower():
        print("[Router] Hallucination detected — regenerating")
        return "fix"

    print(f"[Router] Answer is {score} — passing through")
    return "done"
