from abc import ABC, abstractmethod

from app.domain.models import ParsedDocument, Chunk

class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, doc: ParsedDocument) -> list[Chunk]:
        raise NotImplementedError
