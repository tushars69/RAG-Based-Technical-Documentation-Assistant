"""
LangGraph StateGraph — wires all nodes into the RAG workflow.

Graph structure:

  START → query_analysis → retrieval → grading ──→ generation → END
                ↑                          │
                │                          ↓ (all irrelevant + retries left)
                └──── increment_retry ─────┘
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
    route_after_grading
)


def increment_retry(state: RAGState) -> dict:
    """Bumps the retry counter before looping back to query_analysis."""
    return {"retry_count": state["retry_count"] + 1}


def build_rag_graph():
    graph = StateGraph(RAGState)

    graph.add_node("query_analysis", query_analysis_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("grading", grading_node)
    graph.add_node("generation", generation_node)
    graph.add_node("fallback", fallback_node)
    graph.add_node("increment_retry", increment_retry)

    graph.set_entry_point("query_analysis")

    graph.add_edge("query_analysis", "retrieval")
    graph.add_edge("retrieval", "grading")

    graph.add_conditional_edges(
        "grading",
        route_after_grading,
        {
            "generate": "generation",
            "rewrite": "increment_retry",
            "fallback": "fallback"
        }
    )

    graph.add_edge("increment_retry", "query_analysis")
    graph.add_edge("generation", END)
    graph.add_edge("fallback", END)

    return graph.compile()


rag_graph = build_rag_graph()
