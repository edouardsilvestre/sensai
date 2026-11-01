import unicodedata
from io import BytesIO

from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, Preformatted, SimpleDocTemplate, Spacer

from sensai.core.export.markdown import blocks, inline
from sensai.core.export.page import NOTICE

# The fonts reportlab comes with: nothing to install, and every accent, quote and dash a French text needs.
FONT, BOLD, ITALIC, BOLD_ITALIC, CODE = "Vera", "VeraBd", "VeraIt", "VeraBI", "Courier"
for _name in (FONT, BOLD, ITALIC, BOLD_ITALIC):
    pdfmetrics.registerFont(TTFont(_name, f"{_name}.ttf"))
pdfmetrics.registerFontFamily(FONT, normal=FONT, bold=BOLD, italic=ITALIC, boldItalic=BOLD_ITALIC)
_GLYPHS = set.intersection(*(set(pdfmetrics.getFont(name).face.charToGlyph) for name in (FONT, BOLD, ITALIC, BOLD_ITALIC)))

MISSING = "?"
# The characters models write the most that the fonts lack, with what says the same.
_REPLACEMENTS = str.maketrans({
    " ": " ", " ": " ", " ": " ", "‐": "-", "‑": "-", "→": "->", "←": "<-",
})
# Courier takes this many characters on the width of the page.
CODE_WIDTH = 96
INK, MUTED, ACCENT, BUBBLE = HexColor("#2b2a27"), HexColor("#76746c"), HexColor("#c96442"), HexColor("#f0ede4")
MARGIN = 20 * mm
_BODY = ParagraphStyle("body", fontName=FONT, fontSize=10, leading=14.5, textColor=INK, spaceAfter=5)
_TITLE = ParagraphStyle("title", parent=_BODY, fontName=BOLD, fontSize=18, leading=22, spaceAfter=3)
_DETAILS = ParagraphStyle("details", parent=_BODY, fontSize=8.5, leading=12, textColor=MUTED, spaceAfter=8)
_AUTHOR = ParagraphStyle(
    "author", parent=_BODY, fontName=BOLD, fontSize=8.5, leading=11, textColor=ACCENT, spaceBefore=11, spaceAfter=3,
    keepWithNext=1,
)
_ITEM = ParagraphStyle("item", parent=_BODY, leftIndent=14, bulletIndent=3, bulletFontName=FONT, spaceAfter=2.5)
_SOURCE = ParagraphStyle("source", parent=_DETAILS, spaceAfter=1)
_CODE = ParagraphStyle(
    "code", parent=_BODY, fontName=CODE, fontSize=8, leading=10.5, backColor=BUBBLE, borderPadding=4, spaceBefore=4,
    spaceAfter=9,
)
_HEADINGS = {
    level: ParagraphStyle(
        f"heading{level}", parent=_BODY, fontName=BOLD, fontSize=size, leading=size + 4, spaceBefore=8, spaceAfter=3,
        keepWithNext=1,
    )
    for level, size in {1: 15, 2: 13, 3: 11, 4: 10}.items()
}


class _Printable:
    """Replaces what the fonts cannot draw and counts it: a PDF must say what it lost, not leave it to be found out."""

    def __init__(self) -> None:
        self.lost = 0

    def __call__(self, text: str) -> str:
        out = []
        # Composed: macOS writes an accent apart from its letter, and the fonts only have the accented letter.
        for character in unicodedata.normalize("NFC", text).translate(_REPLACEMENTS):
            if character in "\n\t" or ord(character) in _GLYPHS:
                out.append(character)
            elif unicodedata.category(character) not in ("Cc", "Cf"):
                self.lost += 1
                out.append(MISSING)
        return "".join(out)


def _inline(text: str) -> str:
    return inline(text, ("<b>", "</b>"), ("<i>", "</i>"), (f'<font face="{CODE}">', "</font>"))


def _flowables(text: str, printable: _Printable, heading_shift: int) -> list:
    flowables = []
    for block in blocks(printable(text)):
        if block.kind == "heading":
            flowables.append(Paragraph(_inline(block.text), _HEADINGS[min(block.number + heading_shift, 4)]))
        elif block.kind == "paragraph":
            flowables.append(Paragraph("<br/>".join(_inline(line) for line in block.text.split("\n")), _BODY))
        elif block.kind == "bullet":
            flowables.append(Paragraph(_inline(block.text), _ITEM, bulletText="•"))
        elif block.kind == "numbered":
            flowables.append(Paragraph(_inline(block.text), _ITEM, bulletText=f"{block.number}."))
        else:
            flowables.append(Preformatted(block.text, _CODE, maxLineLength=CODE_WIDTH))
    return flowables


def _page_number(canvas, document) -> None:
    canvas.saveState()
    canvas.setFont(FONT, 8)
    canvas.setFillColor(MUTED)
    canvas.drawRightString(A4[0] - MARGIN, 10 * mm, str(canvas.getPageNumber()))
    canvas.restoreState()


def to_pdf(title: str, details: str, parts: list[tuple[str, str, list[tuple[int, str]]]]) -> bytes:
    printable = _Printable()
    story = [Paragraph(_inline(printable(title)), _TITLE), Paragraph(_inline(printable(details)), _DETAILS)]
    for author, text, sources in parts:
        if author:
            story.append(Paragraph(_inline(printable(author)), _AUTHOR))
        story += _flowables(text, printable, 1 if author else 0)
        story += [Paragraph(f"[{number}] {_inline(printable(citation))}", _SOURCE) for number, citation in sources]
    story += [Spacer(0, 6 * mm), Paragraph(_inline(NOTICE), _DETAILS)]
    if printable.lost:
        story.append(Paragraph(
            f"{printable.lost} characters are not in the font of this PDF and are shown as {MISSING}: "
            "the Markdown and JSON exports keep them.", _DETAILS,
        ))
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=MARGIN, bottomMargin=MARGIN,
        title=title, author="Sensai",
    )
    document.build(story, onFirstPage=_page_number, onLaterPages=_page_number)
    return buffer.getvalue()
