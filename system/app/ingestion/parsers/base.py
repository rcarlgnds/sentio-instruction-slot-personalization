from abc import ABC, abstractmethod

from app.domain.models import ParsedDocument


class BaseParser(ABC):
    SUPPORTED_EXTENSIONS: tuple[str, ...] = ()

    def supports(self, file_extension: str) -> bool:
        ext = file_extension.lower().lstrip(".")
        return ext in self.SUPPORTED_EXTENSIONS

    @abstractmethod
    def parse(self, file_bytes: bytes, filename: str) -> ParsedDocument:
        raise NotImplementedError


class ParsingError(Exception):
    pass
