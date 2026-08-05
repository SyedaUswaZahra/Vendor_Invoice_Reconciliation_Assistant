from typing import Any, List

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel

from prompts.system_prompt import build_generation_prompt


def build_generator_chain(
    llm: BaseChatModel, callbacks: List[BaseCallbackHandler] = None
) -> Any:
    """Builds the generation chain with ChatPromptTemplate and streaming callbacks."""
    prompt = build_generation_prompt()
    chain = prompt | llm
    if callbacks:
        chain = chain.with_config({"callbacks": callbacks})
    return chain


def generate_answer(
    llm: BaseChatModel,
    question: str,
    chunks: List[str],
    callbacks: List[BaseCallbackHandler] = None,
) -> str:
    """Generates an answer string from graded chunks with citations."""
    chain = build_generator_chain(llm, callbacks)
    chunks_text = "\n\n---\n\n".join(chunks)
    response = chain.invoke({"question": question, "chunks": chunks_text})
    return response.content if hasattr(response, "content") else str(response)
