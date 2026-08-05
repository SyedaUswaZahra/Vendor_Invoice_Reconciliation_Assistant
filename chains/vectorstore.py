from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

SIMILARITY_THRESHOLD = 0.75


def build_vectorstore(chunks: list[str], model_name: str = "all-MiniLM-L6-v2"):
    """Embed chunks with a local embedding model and build a FAISS index.

    Args:
        chunks: List of chunk strings to embed.
        model_name: Name of the HuggingFace embedding model to use.

    Returns:
        A FAISS vectorstore object built from the chunks.
    """
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore


def retrieve_top_k(vectorstore, query: str, k: int = 5) -> list[str]:
    """Retrieve top-k chunks from FAISS index, filtering below SIMILARITY_THRESHOLD.

    Args:
        vectorstore: A FAISS vectorstore object.
        query: The query string to search for.
        k: Number of top results to retrieve.

    Returns:
        A filtered list of chunk text strings whose similarity score is at or
        above SIMILARITY_THRESHOLD.
    """
    results = vectorstore.similarity_search_with_relevance_scores(query, k=k)
    filtered_chunks = [
        chunk for chunk, score in results if score >= SIMILARITY_THRESHOLD
    ]
    return filtered_chunks
