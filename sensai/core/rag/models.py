from dataclasses import dataclass, field


@dataclass(frozen=True)
class Locator:
    kind: str
    number: int
    scope: str = ""


@dataclass
class Section:
    text: str
    locator: Locator | None = None
    separator: str = "\n\n"


@dataclass
class Document:
    kind: str
    sections: list[Section]
    title: str = ""
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class Chunk:
    text: str
    location: str = ""


@dataclass
class SearchResult:
    chunk_id: int
    text: str
    location: str
    source: str
    description: str
    score: float

    @property
    def citation(self) -> str:
        return f"{self.source}, {self.location}" if self.location else self.source


_DESCRIBED_FIELDS = {"from": "From", "to": "To", "cc": "Cc", "date": "Date", "author": "Author", "attachments": "Attachments"}


def describe(title: str, metadata: dict[str, str]) -> str:
    lines = [f"Title: {title}"] if title else []
    lines += [f"{label}: {metadata[key]}" for key, label in _DESCRIBED_FIELDS.items() if metadata.get(key)]
    return "\n".join(lines)
