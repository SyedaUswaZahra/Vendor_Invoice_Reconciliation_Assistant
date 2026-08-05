from langchain_text_splitters import ParagraphTextSplitter


CLAUSE_PATTERNS = [
    r"^Section\s+\d+",
    r"^Clause\s+\d+",
    r"^\d+\.",
]


def split_lease_text(raw_text: str) -> list[str]:
    """Split raw lease text into clause-aware chunks.

    Args:
        raw_text: Full lease text as a single string.

    Returns:
        A list of chunk strings.
    """
    splitter = ParagraphTextSplitter(separators=CLAUSE_PATTERNS)
    return splitter.split_text(raw_text)
