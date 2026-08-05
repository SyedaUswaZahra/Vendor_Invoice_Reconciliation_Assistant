from matcher.embedding_store import EmbeddingStore
from schemas.line_item import LineItemSchema
from typing import List, Tuple


class MatcherNode:
    """Fuzzy-matches invoice line items against PO line items via FAISS similarity (FR-4)."""

    def __init__(self, po_store: EmbeddingStore) -> None:
        """Accept an EmbeddingStore pre-indexed with PO line items."""
        self.po_store = po_store

    def match(
        self, invoice_items: List[LineItemSchema]
    ) -> List[Tuple[LineItemSchema, LineItemSchema, float]]:
        """For each invoice item, query the PO store and return the top-1 match."""
        matches: List[Tuple[LineItemSchema, LineItemSchema, float]] = []

        for invoice_item in invoice_items:
            description = getattr(invoice_item, "description", None) or ""
            quantity = getattr(invoice_item, "quantity", None) or 0
            query = f"{description} quantity: {quantity}"

            results = self.po_store.search(query=query, top_k=1)
            if not results:
                continue

            # similarity_search_with_score returns (Document, score); the Document
            # stores the serialized PO line item text, not the original schema.
            # We reconstruct a LineItemSchema from the matched document metadata.
            doc, score = results[0]
            matched_po_item = self._doc_to_line_item(doc)
            if matched_po_item is not None:
                matches.append((invoice_item, matched_po_item, score))

        return matches

    @staticmethod
    def _doc_to_line_item(doc) -> LineItemSchema:
        """Reconstruct a LineItemSchema from a matched FAISS Document."""
        metadata = getattr(doc, "metadata", None) or {}
        try:
            return LineItemSchema(**metadata)
        except Exception:
            # Fallback: parse from page_content if metadata is unavailable
            content = getattr(doc, "page_content", "") or ""
            try:
                import json

                parsed = json.loads(content)
                return LineItemSchema(**parsed)
            except Exception:
                return None
