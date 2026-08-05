from typing import Any

from langchain.agents import create_react_agent, Tool
from langchain_core.language_models import BaseChatModel
from langchain_community.vectorstores import FAISS

from chains.vectorstore import retrieve_top_k


def search_clause_tool(vectorstore: FAISS) -> Tool:
    """Returns a Tool that queries the FAISS index for a keyword or phrase."""

    def _search_clause(query: str) -> str:
        chunks = retrieve_top_k(vectorstore, query, k=5)
        if not chunks:
            return "No matching clauses found in the lease document."
        results = []
        for idx, chunk in enumerate(chunks, start=1):
            results.append(f"Chunk {idx}:\n{chunk}")
        return "\n\n---\n\n".join(results)

    return Tool(
        name="search_clause",
        description=(
            "Search the lease agreement for a keyword or phrase and return the "
            "matching clause text with citations. Use this tool when the user asks "
            "about a specific provision, term, or condition in the lease."
        ),
        func=_search_clause,
    )


def summarize_fees_tool(vectorstore: FAISS) -> Tool:
    """Returns a Tool that aggregates and summarizes fee-related clauses."""

    def _summarize_fees(_query: str) -> str:
        fee_keywords = ["deposit", "rent", "late fee", "security deposit", "penalty"]
        collected = []
        for keyword in fee_keywords:
            chunks = retrieve_top_k(vectorstore, keyword, k=3)
            for chunk in chunks:
                if chunk not in collected:
                    collected.append(chunk)

        if not collected:
            return (
                "No fee-related clauses (deposit, rent, late fee, security deposit, "
                "penalty) were found in the lease document."
            )

        summary_lines = []
        for idx, chunk in enumerate(collected, start=1):
            summary_lines.append(f"Relevant clause {idx}:\n{chunk}")

        header = (
            "Summary of fee-related clauses found in the lease document "
            "(covering deposits, rent, late fees, and penalties):\n\n"
        )
        return header + "\n\n---\n\n".join(summary_lines)

    return Tool(
        name="summarize_fees",
        description=(
            "Aggregate and summarize all fee-related clauses in the lease agreement, "
            "including deposits, rent, late fees, security deposits, and penalties. "
            "Use this tool when the user asks about fees, costs, deposits, or "
            "penalties in the lease."
        ),
        func=_summarize_fees,
    )


def build_react_agent(llm: BaseChatModel, vectorstore: FAISS) -> Any:
    """Builds a ReAct agent with both tools for follow-up conversational turns."""
    tools = [
        search_clause_tool(vectorstore),
        summarize_fees_tool(vectorstore),
    ]

    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a lease agreement assistant. Use the available tools to "
                "answer the user's questions about the lease document. "
                "Always ground your answers in the clause text returned by the tools. "
                "If the tools do not return enough information, say so clearly.",
            ),
            ("human", "{input}"),
            ("assistant", "{agent_scratchpad}"),
        ]
    )

    agent = create_react_agent(llm, tools, prompt)
    return agent
