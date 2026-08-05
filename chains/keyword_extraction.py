from langchain.output_parsers import PydanticOutputParser
from langchain_core.language_models import BaseLanguageModel
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from schemas.requirements_schema import JobRequirementsList


class KeywordExtractionChain:
    def __init__(self, llm: BaseLanguageModel) -> None:
        self.parser = PydanticOutputParser(pydantic_object=JobRequirementsList)

        prompt_template = PromptTemplate(
            template=(
                "You are an expert recruiter. Extract structured job requirements from the "
                "following job description.\n\n"
                "For each requirement, identify:\n"
                "- skill: a specific technical or professional skill mentioned.\n"
                "- qualification: an educational or experiential qualification required.\n"
                "- responsibility: a responsibility or duty the role entails.\n\n"
                "Return the result as a JSON object matching the schema below.\n"
                "{format_instructions}\n\n"
                "Job Description:\n{job_description}"
            ),
            input_variables=["job_description"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
        )

        self.chain = LLMChain(
            llm=llm,
            prompt=prompt_template,
            output_parser=self.parser,
        )

    def run(self, job_description: str) -> JobRequirementsList:
        output = self.chain.invoke({"job_description": job_description})
        if isinstance(output, dict):
            return output["text"]
        return output
