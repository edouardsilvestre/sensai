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
        repeated = 0
        for unit in self._units(document):
            if window and (unit.scope != window[-1].scope or _length(window + [unit]) > self.size):
                chunks.append(_to_chunk(window, repeated))
                window = self._overlap(window) if unit.scope == window[-1].scope else []
                while window and _length(window + [unit]) > self.size:
                    window.pop(0)
                repeated = len(window)
            window.append(unit)
        if window:
            chunks.append(_to_chunk(window, repeated))
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


def join_chunks(chunks: list[Chunk]) -> str:
    """The text of consecutive chunks of a document, without the start of each one that repeats the previous one."""
    joined = chunks[0].text
    for chunk in chunks[1:]:
        joined += chunk.text[chunk.overlap:] if chunk.overlap else "\n\n" + chunk.text
    return joined


_LOCATION = re.compile(r"^(?:(?P<scope>.+), )?(?:p\. |(?P<kind>[a-z]+?)s? )(?P<first>\d+)(?:-(?P<last>\d+))?$")


def join_locations(locations: list[str]) -> str:
    """The location of consecutive chunks, like "p. 2-4" for "p. 2", "p. 3" and "p. 3-4"."""
    locators: list[Locator] = []
    for location in locations:
        match = _LOCATION.match(location)
        if not match:
            locators = []
            break
        kind, scope = match["kind"] or "page", match["scope"] or ""
        locators += [Locator(kind, int(match["first"]), scope), Locator(kind, int(match["last"] or match["first"]), scope)]
    if locators and len({(locator.kind, locator.scope) for locator in locators}) == 1:
        return format_location(locators)
    # Another sheet, or a location of another form: each one, once.
    return "; ".join(dict.fromkeys(filter(None, locations)))


def _length(units: list[_Unit]) -> int:
    return sum(len(u.text) for u in units) + sum(len(u.separator) for u in units[1:])


def _to_chunk(units: list[_Unit], repeated: int = 0) -> Chunk:
    text = units[0].text + "".join(u.separator + u.text for u in units[1:])
    overlap = _length(units[:repeated]) if repeated else 0
    return Chunk(text, format_location([u.locator for u in units if u.locator]), overlap)


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
