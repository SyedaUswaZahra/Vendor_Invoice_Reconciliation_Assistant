from typing import Any, List

from langchain_core.language_models import BaseChatModel

from prompts.verify_prompt import build_verify_prompt


def build_verifier_chain(llm: BaseChatModel) -> Any:
    """Builds the verification chain with the verify prompt."""
    prompt = build_verify_prompt()
    chain = prompt | llm
    return chain


def verify_answer(llm: BaseChatModel, answer: str, clauses: List[str]) -> str:
    """Returns a verification_warning string, empty if all claims verified."""
    chain = build_verifier_chain(llm)
    clauses_text = "\n\n---\n\n".join(clauses)
    response = chain.invoke({"answer": answer, "clauses": clauses_text})
    content = response.content if hasattr(response, "content") else str(response)
    return content
