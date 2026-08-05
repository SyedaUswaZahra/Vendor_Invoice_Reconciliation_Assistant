from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import UnstructuredHTMLLoader


class InvoiceLoader:
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.file_extension = Path(file_path).suffix.lower()

    def load(self) -> str:
        if self.file_extension == ".pdf":
            return self._load_pdf()
        if self.file_extension in (".png", ".jpg", ".jpeg", ".tiff"):
            return self._load_scanned_image()
        if self.file_extension in (".html", ".htm"):
            return self._load_html()
        raise ValueError(f"Unsupported file type: {self.file_extension}")

    def _load_pdf(self) -> str:
        loader = PyPDFLoader(self.file_path)
        pages = loader.load()
        return "\n".join(page.page_content for page in pages)

    def _load_scanned_image(self) -> str:
        loader = UnstructuredPDFLoader(self.file_path, mode="elements", ocr_mode="entire_page")
        documents = loader.load()
        return "\n".join(doc.page_content for doc in documents)

    def _load_html(self) -> str:
        loader = UnstructuredHTMLLoader(self.file_path)
        documents = loader.load()
        return "\n".join(doc.page_content for doc in documents)
