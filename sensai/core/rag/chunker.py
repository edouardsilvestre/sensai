import re
from dataclasses import dataclass

from sensai.core.rag.models import Chunk, Document, Locator

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_SENTENCE_BREAK = re.compile(r"((?<=[.!?…])\s+|\s*\n\s*)")


@dataclass
class _Unit:
    text: str
    separator: str
    locator: Locator | None

    @property
    def scope(self) -> tuple[str, str] | None:
        return (self.locator.kind, self.locator.scope) if self.locator else None


class Chunker:
    def __init__(self, size: int, overlap: int):
        if size <= 0 or not 0 <= overlap < size:
            raise ValueError("chunk size must be positive and overlap smaller than size")
        self.size = size
        self.overlap = overlap

    def split(self, document: Document) -> list[Chunk]:
        chunks: list[Chunk] = []
        window: list[_Unit] = []
        for unit in self._units(document):
            if window and (unit.scope != window[-1].scope or _length(window + [unit]) > self.size):
                chunks.append(_to_chunk(window))
                window = self._overlap(window) if unit.scope == window[-1].scope else []
                while window and _length(window + [unit]) > self.size:
                    window.pop(0)
            window.append(unit)
        if window:
            chunks.append(_to_chunk(window))
        return chunks

    def _units(self, document: Document):
        for section in document.sections:
            separator = section.separator
            for paragraph in _PARAGRAPH_BREAK.split(section.text):
                parts = _SENTENCE_BREAK.split(paragraph.strip())
                for i in range(0, len(parts), 2):
                    for piece in self._fit(parts[i].strip()):
                        yield _Unit(piece, separator, section.locator)
                        separator = " "
                    if i + 1 < len(parts):
                        separator = "\n" if "\n" in parts[i + 1] else " "
                separator = "\n\n"

    def _fit(self, text: str) -> list[str]:
        if len(text) <= self.size:
            return [text] if text else []
        pieces: list[str] = []
        current = ""
        for word in text.split():
            while len(word) > self.size:
                if current:
                    pieces.append(current)
                    current = ""
                pieces.append(word[: self.size])
                word = word[self.size :]
            if not word:
                continue
            if current and len(current) + 1 + len(word) > self.size:
                pieces.append(current)
                current = word
            else:
                current = f"{current} {word}" if current else word
        if current:
            pieces.append(current)
        return pieces

    def _overlap(self, window: list[_Unit]) -> list[_Unit]:
        tail: list[_Unit] = []
        length = 0
        for unit in reversed(window[1:]):
            length += len(unit.text) + len(unit.separator)
            if length > self.overlap:
                break
            tail.insert(0, unit)
        return tail


def _length(units: list[_Unit]) -> int:
    return sum(len(u.text) for u in units) + sum(len(u.separator) for u in units[1:])


def _to_chunk(units: list[_Unit]) -> Chunk:
    text = units[0].text + "".join(u.separator + u.text for u in units[1:])
    return Chunk(text, format_location([u.locator for u in units if u.locator]))


def format_location(locators: list[Locator]) -> str:
    if not locators:
        return ""
    first = min(locator.number for locator in locators)
    last = max(locator.number for locator in locators)
    span = str(first) if first == last else f"{first}-{last}"
    kind = locators[0].kind
    if kind == "page":
        label = f"p. {span}"
    else:
        label = f"{kind}{'' if first == last else 's'} {span}"
    scope = locators[0].scope
    return f"{scope}, {label}" if scope else label
