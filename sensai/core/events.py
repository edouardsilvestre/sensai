from dataclasses import dataclass

from sensai.core.rag.models import SearchResult


@dataclass
class SourcesEvent:
    sources: list[SearchResult]


@dataclass
class TokenEvent:
    text: str


@dataclass
class DoneEvent:
    content: str


Event = SourcesEvent | TokenEvent | DoneEvent
