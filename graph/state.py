from typing import TypedDict, Optional, Any


class AppState(TypedDict):
    chunks: list[str]
    invoice_metadata: Optional[dict]
    line_items: list[dict]
    retrieved_documents: list[dict]
    discrepancies: list[dict]
    is_duplicate: bool
    report: Optional[str]