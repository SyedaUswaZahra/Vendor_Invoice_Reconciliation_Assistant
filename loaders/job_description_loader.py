from langchain_community.document_loaders import WebBaseLoader


class JobDescriptionLoader:
    def __init__(self, source: str, is_url: bool = False) -> None:
        self.is_url = is_url
        self.source = source
        self._text = None
        self._loader = None
        if self.is_url:
            self._loader = WebBaseLoader(self.source)
        else:
            self._text = self.source

    def load(self) -> str:
        if self.is_url:
            documents = self._loader.load()
            return "\n".join(doc.page_content for doc in documents)
        return self._text
