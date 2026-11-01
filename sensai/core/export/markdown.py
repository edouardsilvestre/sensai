import html
import re
from dataclasses import dataclass

FENCE = "```"
_HEADING = re.compile(r"^(#{1,4})\s+(.*)$")
_BULLET = re.compile(r"^\s*[-*•]\s+(.*)$")
_NUMBERED = re.compile(r"^\s*(\d+)[.)]\s+(.*)$")
_CODE = re.compile(r"(`[^`\n]+`)")
_BOLD = re.compile(r"\*\*([^*\n]+)\*\*")
_ITALIC = re.compile(r"(^|[\s(])\*([^*\s][^*\n]*)\*")


@dataclass
class Block:
    """One block of an answer or of the document, read the way the web page reads it."""

    kind: str
    text: str
    # The level of a heading, the number of a numbered item.
    number: int = 0


def blocks(text: str) -> list[Block]:
    found: list[Block] = []
    paragraph: list[str] = []
    fence: list[str] | None = None

    def close_paragraph() -> None:
        if paragraph:
            found.append(Block("paragraph", "\n".join(paragraph)))
            paragraph.clear()

    for line in text.replace("\r\n", "\n").split("\n"):
        if fence is not None:
            if line.strip().startswith(FENCE):
                found.append(Block("code", "\n".join(fence)))
                fence = None
            else:
                fence.append(line)
            continue
        heading, bullet, numbered = _HEADING.match(line), _BULLET.match(line), _NUMBERED.match(line)
        if line.strip().startswith(FENCE):
            close_paragraph()
            fence = []
        elif heading:
            close_paragraph()
            found.append(Block("heading", heading[2], len(heading[1])))
        elif bullet:
            close_paragraph()
            found.append(Block("bullet", bullet[1]))
        elif numbered:
            close_paragraph()
            found.append(Block("numbered", numbered[2], int(numbered[1])))
        elif line.strip():
            paragraph.append(line)
        else:
            close_paragraph()
    close_paragraph()
    # A block of code that was never closed, in an interrupted answer.
    if fence is not None:
        found.append(Block("code", "\n".join(fence)))
    return found


def _wrap(tags: tuple[str, str], text: str) -> str:
    return f"{tags[0]}{text}{tags[1]}"


def inline(text: str, bold: tuple[str, str], italic: tuple[str, str], code: tuple[str, str]) -> str:
    """
    Escapes the text, then adds the tags. The code is cut out first, and the bold and the italic never contain a `*`:
    the tags are always nested properly, which the parser of the PDF requires.
    """
    pieces = []
    for number, piece in enumerate(_CODE.split(text)):
        escaped = html.escape(piece[1:-1] if number % 2 else piece, quote=False)
        if number % 2:
            pieces.append(_wrap(code, escaped))
            continue
        styled = _BOLD.sub(lambda match: _wrap(bold, match[1]), escaped)
        pieces.append(_ITALIC.sub(lambda match: match[1] + _wrap(italic, match[2]), styled))
    return "".join(pieces)
