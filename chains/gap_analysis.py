from langchain_core.language_models import BaseLanguageModel
from langchain.chains import LLMClassifierChain
from langchain.prompts import PromptTemplate

from schemas.gap_analysis_schema import GapAnalysisItem, GapAnalysisResult


class GapAnalysisChain:
    def __init__(self, llm: BaseLanguageModel) -> None:
        prompt_template = PromptTemplate(
            template=(
                "Given the following rewritten resume, determine whether each job requirement "
                "is met, partially met, or missing.\n\n"
                "Rewritten Resume:\n{rewritten_resume}\n\n"
                "Job Requirement:\n{query}\n\n"
                "Classify the requirement as one of: met, partially_met, missing."
            ),
            input_variables=["rewritten_resume", "query"],
        )

        self.chain = LLMClassifierChain.from_llm(
            llm=llm,
            prompt=prompt_template,
            labels=["met", "partially_met", "missing"],
        )

    def run(self, requirements: list[str], rewritten_resume: str) -> GapAnalysisResult:
        items = []
        for requirement in requirements:
            result = self.chain.invoke(
                {"rewritten_resume": rewritten_resume, "query": requirement}
            )
            if isinstance(result, dict):
                label = result["label"]
            else:
                label = result

            evidence = self._extract_evidence(requirement, rewritten_resume, label)
            items.append(
                GapAnalysisItem(
                    requirement=requirement,
                    status=label,
                    evidence=evidence,
                )
            )
        return GapAnalysisResult(items=items)

    def _extract_evidence(
        self, requirement: str, rewritten_resume: str, label: str
    ) -> str:
        if label == "missing":
            return "No supporting evidence found in the rewritten resume."
        requirement_lower = requirement.lower()
        resume_lower = rewritten_resume.lower()
        if requirement_lower in resume_lower:
            return f"Found direct match for '{requirement}' in the rewritten resume."
        return f"Related content in the rewritten resume suggests partial alignment with '{requirement}'."
