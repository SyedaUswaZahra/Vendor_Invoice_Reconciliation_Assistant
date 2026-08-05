from langchain.text_splitter import RecursiveCharacterTextSplitter


class TextSplitter:
    def __init__(self, chunk_size: int = 200, chunk_overlap: int = 20) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split_text(self, text: str) -> list[str]:
        return self.splitter.split_text(text)
