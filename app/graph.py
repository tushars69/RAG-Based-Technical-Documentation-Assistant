"""
LangGraph StateGraph — wires all nodes into the RAG workflow.

Graph structure:

  START → query_analysis → retrieval → grading ──→ generation → hallucination_check → END
                ↑                          │                             │
                │                          ↓ (all irrelevant+retries)   ↓ (hallucinated)
                └──── increment_retry ─────┘                   fix_hallucination → END
                                           │
                                           ↓ (retries exhausted)
                                        fallback → END
"""

from langgraph.graph import StateGraph, END
from app.state import RAGState
from app.nodes import (
    query_analysis_node,
    retrieval_node,
    grading_node,
    generation_node,
    fallback_node,
    route_after_grading,
    hallucination_check_node,
    fix_hallucination_node,
    route_after_hallucination_check
)


def increment_retry(state: RAGState) -> dict:
    return {"retry_count": state["retry_count"] + 1}


def build_rag_graph():
    graph = StateGraph(RAGState)

    # ── Register all nodes ────────────────────────────────────────
    graph.add_node("query_analysis", query_analysis_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("grading", grading_node)
    graph.add_node("generation", generation_node)
    graph.add_node("hallucination_check", hallucination_check_node)
    graph.add_node("fix_hallucination", fix_hallucination_node)
    graph.add_node("fallback", fallback_node)
    graph.add_node("increment_retry", increment_retry)

    # ── Entry point ───────────────────────────────────────────────
    graph.set_entry_point("query_analysis")

    # ── Linear edges ──────────────────────────────────────────────
    graph.add_edge("query_analysis", "retrieval")
    graph.add_edge("retrieval", "grading")

    # ── Conditional edge after grading ────────────────────────────
    graph.add_conditional_edges(
        "grading",
        route_after_grading,
        {
            "generate": "generation",
            "rewrite": "increment_retry",
            "fallback": "fallback"
        }
    )

    # ── Retry loop ────────────────────────────────────────────────
    graph.add_edge("increment_retry", "query_analysis")

    # ── Generation → hallucination check (not straight to END) ───
    graph.add_edge("generation", "hallucination_check")

    # ── Conditional edge after hallucination check ────────────────
    graph.add_conditional_edges(
        "hallucination_check",
        route_after_hallucination_check,
        {
            "fix": "fix_hallucination",
            "done": END
        }
    )

    # ── Terminal edges ────────────────────────────────────────────
    graph.add_edge("fix_hallucination", END)
    graph.add_edge("fallback", END)

    return graph.compile()


rag_graph = build_rag_graph()
