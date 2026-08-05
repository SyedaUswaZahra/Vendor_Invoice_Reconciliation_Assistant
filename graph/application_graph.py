from typing import TypedDict

from langchain_core.language_models import BaseLanguageModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from graph.nodes import (
    ApplicationState,
    ingest,
    extract_keywords,
    retrieve_bullets,
    rewrite_bullets,
    reorder_sections,
    draft_cover_letter,
    gap_analysis,
    human_review,
)
from schemas.requirements_schema import JobRequirementsList
from schemas.resume_schema import RewrittenBullet, ResumeSection
from schemas.gap_analysis_schema import GapAnalysisResult


class ApplicationState(TypedDict):
    resume_path: str
    job_description: str
    job_description_is_url: bool
    output_dir: str
    resume_text: str
    job_text: str
    requirements: JobRequirementsList | None
    retrieved_bullets: dict[str, list[str]]
    rewritten_bullets: list[RewrittenBullet]
    reordered_sections: list[ResumeSection]
    cover_letter: str
    gap_analysis: GapAnalysisResult | None
    low_confidence: bool


class ApplicationGraph:
    def __init__(self, llm: BaseLanguageModel, checkpoint: BaseCheckpointSaver) -> None:
        self.llm = llm
        self.checkpoint = checkpoint
        self.graph = None

    def build(self) -> CompiledStateGraph:
        graph = StateGraph(ApplicationState)

        graph.add_node("ingest", ingest)
        graph.add_node("extract_keywords", extract_keywords)
        graph.add_node("retrieve_bullets", retrieve_bullets)
        graph.add_node("rewrite_bullets", rewrite_bullets)
        graph.add_node("reorder_sections", reorder_sections)
        graph.add_node("draft_cover_letter", draft_cover_letter)
        graph.add_node("gap_analysis", gap_analysis)
        graph.add_node("human_review", human_review)

        graph.set_entry_point("ingest")

        graph.add_edge("ingest", "extract_keywords")
        graph.add_edge("extract_keywords", "retrieve_bullets")
        graph.add_edge("retrieve_bullets", "rewrite_bullets")
        graph.add_edge("rewrite_bullets", "reorder_sections")
        graph.add_edge("reorder_sections", "draft_cover_letter")
        graph.add_edge("draft_cover_letter", "gap_analysis")

        graph.add_conditional_edges(
            "gap_analysis",
            lambda state: "human_review" if state.get("low_confidence", False) else END,
            {
                "human_review": "human_review",
                END: END,
            },
        )

        graph.add_edge("human_review", END)

        self.graph = graph.compile(checkpointer=self.checkpoint)
        return self.graph

    def run(
        self,
        resume_path: str,
        job_description: str,
        is_url: bool,
        output_dir: str,
    ) -> ApplicationState:
        if self.graph is None:
            self.build()

        initial_state: ApplicationState = {
            "resume_path": resume_path,
            "job_description": job_description,
            "job_description_is_url": is_url,
            "output_dir": output_dir,
            "resume_text": "",
            "job_text": "",
            "requirements": None,
            "retrieved_bullets": {},
            "rewritten_bullets": [],
            "reordered_sections": [],
            "cover_letter": "",
            "gap_analysis": None,
            "low_confidence": False,
        }

        config = {"configurable": {"thread_id": "resume-application"}}
        result = self.graph.invoke(initial_state, config=config)
        return result
