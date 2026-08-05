from langchain_core.language_models import BaseLanguageModel
from langchain_core.runnables import RunnableLambda
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from schemas.resume_schema import RewrittenBullet


class BulletRewriteChain:
    def __init__(self, llm: BaseLanguageModel) -> None:
        self.llm = llm
        self.parser = PydanticOutputParser(pydantic_object=RewrittenBullet)

        self.prompt = PromptTemplate(
            template=(
                "You are an expert resume writer. Rewrite the given resume bullet so that it "
                "embeds the relevant job requirement keywords naturally, while preserving all "
                "facts, numbers, and achievements from the original bullet.\n\n"
                "Do not invent new facts. Do not remove any quantifiable result. "
                "Only rephrase and weave in the requirement keywords where they fit.\n\n"
                "Requirement keyword:\n{requirement}\n\n"
                "Original bullet:\n{bullet}\n\n"
                "Return the result as a JSON object matching this schema:\n"
                "{format_instructions}"
            ),
            input_variables=["bullet", "requirement"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
        )

        self.runnable = (
            self.prompt
            | self.llm
            | self.parser
        )

    def run(self, bullets: list[str], requirement: str) -> list[RewrittenBullet]:
        results = []
        for bullet in bullets:
            result = self.runnable.invoke({"bullet": bullet, "requirement": requirement})
            results.append(result)
        return results
