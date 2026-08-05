from typing import TypedDict
from schemas.quote import QuoteDocument, CategoryMapping


class GraphState(TypedDict):
    parsed_quotes: list[QuoteDocument]
    normalized_line_items: list[dict]
    category_alignments: list[CategoryMapping]
    pending_confirmation: list[dict]
    comparison_results: dict | None
    final_report: str | None
    processed_source_keys: set[str]
    raw_inputs: list[dict]