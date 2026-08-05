from typing import Any

from langgraph.graph import StateGraph, END

from schemas.state import GraphState
from graph.nodes import (
    loader_node,
    splitter_node,
    vectorstore_node,
    retrieve_node,
    grade_node,
    generate_node,
    verify_node,
    clarify_node,
)
from chains.agent import build_react_agent


def route_after_grade(state: GraphState) -> str:
    """Returns 'clarify' if graded_chunks is empty, otherwise 'generate'."""
    if not state.get("graded_chunks"):
        return "clarify"
    return "generate"


def route_after_verify(state: GraphState) -> str:
    """Returns 'clarify' if verification failed entirely, otherwise routes to agent for follow-ups."""
    warning = state.get("verification_warning", "")
    if warning and "FAIL" in warning.upper():
        return "clarify"
    return "agent"


def build_workflow(llm: Any, vectorstore: Any) -> StateGraph:
    """Builds the full LangGraph StateGraph with nodes and conditional edges."""
    workflow = StateGraph(GraphState)

    workflow.add_node("loader", loader_node)
    workflow.add_node("splitter", splitter_node)
    workflow.add_node("vectorstore", vectorstore_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade", grade_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("verify", verify_node)
    workflow.add_node("clarify", clarify_node)

    workflow.add_conditional_edges(
        "grade",
        route_after_grade,
        {
            "clarify": "clarify",
            "generate": "generate",
        },
    )

    workflow.add_conditional_edges(
        "verify",
        route_after_verify,
        {
            "clarify": "clarify",
            "agent": "agent",
        },
    )

    workflow.add_edge("loader", "splitter")
    workflow.add_edge("splitter", "vectorstore")
    workflow.add_edge("vectorstore", "retrieve")
    workflow.add_edge("retrieve", "grade")
    workflow.add_edge("generate", "verify")
    workflow.add_edge("clarify", END)
    workflow.add_edge("agent", END)

    workflow.set_entry_point("loader")

    return workflow.compile()
