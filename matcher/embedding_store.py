from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from schemas.line_item import LineItemSchema
from typing import List


class EmbeddingStore:
    """Embeds line items with OpenAIEmbeddings and stores them in a FAISS vector store (FR-3)."""

    def __init__(
        self,
        embeddings=None,
        index_path: str = None,
    ) -> None:
        self.embeddings = embeddings or OpenAIEmbeddings()
        self.index_path = index_path
        self._vector_store = None

    def _to_documents(self, line_items: List[LineItemSchema]) -> List[str]:
        """Serialize each line item into a searchable text string (description + quantity)."""
        documents = []
        for item in line_items:
            description = getattr(item, "description", None) or ""
            quantity = getattr(item, "quantity", None) or 0
            documents.append(
                f"{description} quantity: {quantity}"
            )
        return documents

    def index(self, line_items: List[LineItemSchema]) -> None:
        """Embed canonical line items and build a FAISS index."""
        documents = self._to_documents(line_items)
        if not documents:
            self._vector_store = None
            return
        self._vector_store = FAISS.from_texts(
            texts=documents,
            embedding=self.embeddings,
        )

    def search(self, query: str, top_k: int = 1) -> List:
        """Return top-k (line_item, score) matches for a query text."""
        if self._vector_store is None:
            return []
        results = self._vector_store.similarity_search_with_score(
            query=query,
            k=top_k,
        )
        return results
