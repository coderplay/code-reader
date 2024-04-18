import copy
from typing import Iterable, List

from langchain_core.documents import Document
from langchain_text_splitters.character import RecursiveCharacterTextSplitter


class LanguageSplitter(RecursiveCharacterTextSplitter):
    """Attempts to split the code along the syntax implied by its file suffix."""

    def split_documents(self, documents: Iterable[Document]) -> List[Document]:
        """Split documents."""
        new_documents = []
        for doc in documents:
            text = doc.page_content
            metadata = doc.metadata

            index = 0
            previous_chunk_len = 0
            separators = self.get_separators_for_language(metadata["language"]) \
                if "language" in metadata else ["\n\n", "\n", " ", ""]
            for chunk in self._split_text(doc.page_content, separators=separators):
                new_metadata = copy.deepcopy(metadata)
                if self._add_start_index:
                    offset = index + previous_chunk_len - self._chunk_overlap
                    index = text.find(chunk, max(0, offset))
                    new_metadata["start_index"] = index
                    previous_chunk_len = len(chunk)
                new_document = Document(page_content=chunk, metadata=new_metadata)
                new_documents.append(new_document)
        return new_documents

