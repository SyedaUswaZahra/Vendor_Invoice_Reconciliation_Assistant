import os

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader


class ResumeLoader:
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path

    def load(self) -> str:
        ext = os.path.splitext(self.file_path)[1].lower()
        if ext == ".pdf":
            loader = PyPDFLoader(self.file_path)
        elif ext == ".docx":
            loader = Docx2txtLoader(self.file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}. Only PDF and DOCX are supported.")

        documents = loader.load()
        return "\n".join(doc.page_content for doc in documents)
