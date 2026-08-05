from langchain_core.language_models import BaseLanguageModel
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


class CoverLetterChain:
    def __init__(self, llm: BaseLanguageModel) -> None:
        few_shot_examples = """
Example 1:
Top requirements:
- Python
- Machine Learning
- Cloud Deployment
Resume summary: 5 years building ML pipelines in Python and deploying to AWS.
Cover letter:
Dear Hiring Manager,

I am excited to apply for this role. With five years of experience building machine learning pipelines in Python and deploying them to AWS, I bring strong skills in Python, machine learning, and cloud deployment. My background aligns closely with your top requirements, and I am eager to contribute to your team.

Sincerely,
[Your Name]

Example 2:
Top requirements:
- Project Management
- Agile
- Stakeholder Communication
Resume summary: Led cross-functional teams using Agile practices and communicated regularly with stakeholders.
Cover letter:
Dear Hiring Manager,

I am writing to express my interest in this position. My experience leading cross-functional teams with Agile methodologies and maintaining clear stakeholder communication directly addresses your need for project management, Agile, and stakeholder communication expertise. I look forward to the opportunity to bring this experience to your organization.

Sincerely,
[Your Name]
"""

        prompt_template = PromptTemplate(
            template=(
                "You are a professional cover letter writer. Draft a concise, compelling cover "
                "letter that echoes the top three matched requirements from the job description "
                "and highlights how the candidate's resume summary supports them.\n\n"
                "Use the following examples as style guidance:\n"
                f"{few_shot_examples}\n\n"
                "Top matched requirements:\n{top_requirements}\n\n"
                "Resume summary:\n{resume_summary}\n\n"
                "Write the cover letter now."
            ),
            input_variables=["top_requirements", "resume_summary"],
        )

        self.chain = LLMChain(
            llm=llm,
            prompt=prompt_template,
        )

    def run(self, top_requirements: list[str], resume_summary: str) -> str:
        output = self.chain.invoke(
            {
                "top_requirements": "\n".join(top_requirements),
                "resume_summary": resume_summary,
            }
        )
        if isinstance(output, dict):
            return output["text"]
        return output
