from typing import Any, List

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


class ChunkGrade(BaseModel):
    chunk_index: int = Field(description="Index of the chunk in the input list")
    relevant: bool = Field(description="Whether the chunk is relevant to the question")


def build_grader_chain(llm: BaseChatModel) -> Any:
    """Builds an LLMChain that grades each chunk's relevance."""
    parser = PydanticOutputParser(pydantic_object=ChunkGrade)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a relevance grader. Determine whether the provided document chunk "
                "contains information relevant to answering the user's question. "
                "Answer with a JSON object matching the schema:\n{format_instructions}",
            ),
            (
                "human",
                "Question: {question}\n\nDocument chunk:\n\n{chunk}\n\n"
                "Is this chunk relevant to the question?",
            ),
        ]
    ).partial(format_instructions=parser.get_format_instructions())

    chain = prompt | llm | parser
    return chain


def grade_chunks(llm: BaseChatModel, question: str, chunks: List[str]) -> List[str]:
    """Returns only the relevant chunk texts from the input chunks."""
    chain = build_grader_chain(llm)
    relevant_chunks = []

    for idx, chunk in enumerate(chunks):
        result = chain.invoke({"question": question, "chunk": chunk})
        if result.relevant:
            relevant_chunks.append(chunk)

    return relevant_chunks
