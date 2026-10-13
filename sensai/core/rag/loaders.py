import codecs
import csv
import email
import email.policy
import io
import logging
import re
from collections.abc import Callable, Iterable
from datetime import date, datetime
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from pathlib import Path

from openpyxl import load_workbook
from pypdf import PdfReader

from sensai.core.rag.models import Document, Locator, Section

logging.getLogger("pypdf").setLevel(logging.ERROR)


class DocumentLoadError(Exception):
    pass


def load_document(path: Path) -> Document:
    loader = _LOADERS.get(path.suffix.lower())
    if loader is None:
        raise DocumentLoadError(f"unsupported file type '{path.suffix or path.name}'")
    try:
        document = loader(path)
    except DocumentLoadError:
        raise
    except OSError as e:
        raise DocumentLoadError(f"cannot open file: {e.strerror or e}") from e
    except Exception as e:
        raise DocumentLoadError(f"cannot parse file: {e}") from e
    if not any(section.text.strip() for section in document.sections):
        raise DocumentLoadError("no text found")
    return document


def is_supported(path: Path) -> bool:
    return path.suffix.lower() in _LOADERS


def _read_text(path: Path) -> str:
    data = path.read_bytes()
    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return data.decode("utf-16")
    if b"\x00" in data[:8192]:
        raise DocumentLoadError("binary content in a text file")
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            pass
    return data.decode("latin-1")


def _load_text(path: Path) -> Document:
    text = _read_text(path)
    title = ""
    if path.suffix.lower() in (".md", ".markdown"):
        heading = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
        title = heading.group(1).strip() if heading else ""
    return Document("text", [Section(text)], title=title)


def _clean_pdf_text(text: str) -> str:
    text = re.sub(r"(\w)-\n(?=[a-zà-ÿ])", r"\1", text)
    paragraphs = re.split(r"\n\s*\n", text)
    return "\n\n".join(" ".join(paragraph.split()) for paragraph in paragraphs if paragraph.strip())


def _load_pdf(path: Path) -> Document:
    reader = PdfReader(path)
    if reader.is_encrypted and not reader.decrypt(""):
        raise DocumentLoadError("encrypted PDF")
    sections = [
        Section(_clean_pdf_text(page.extract_text() or ""), Locator("page", number))
        for number, page in enumerate(reader.pages, start=1)
    ]
    if not any(section.text for section in sections):
        raise DocumentLoadError("no extractable text (scanned PDF? OCR is not supported)")

    title, metadata = "", {}
    info = reader.metadata
    if info is not None:
        title = (info.title or "").strip()
        if info.author:
            metadata["author"] = info.author.strip()
        try:
            created = info.creation_date
        except Exception:
            created = None
        if created:
            metadata["date"] = created.date().isoformat()
    return Document("pdf", sections, title=title, metadata=metadata)


def _load_email(path: Path) -> Document:
    with path.open("rb") as file:
        message = email.message_from_binary_file(file, policy=email.policy.default)

    metadata = {key: str(message[key]) for key in ("from", "to", "cc") if message[key]}
    if message["date"]:
        try:
            metadata["date"] = parsedate_to_datetime(str(message["date"])).isoformat(timespec="minutes")
        except (TypeError, ValueError):
            metadata["date"] = str(message["date"])
    attachments = [part.get_filename() for part in message.iter_attachments() if part.get_filename()]
    if attachments:
        metadata["attachments"] = ", ".join(attachments)

    body = ""
    part = message.get_body(preferencelist=("plain", "html"))
    if part is not None:
        body = part.get_content()
        if part.get_content_type() == "text/html":
            body = html_to_text(body)[1]
    return Document("email", [Section(body.strip() or "(empty body)")], title=str(message["subject"] or ""), metadata=metadata)


def _cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.date().isoformat() if value.time() == datetime.min.time() else value.isoformat(timespec="minutes")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _table_sections(rows: Iterable[Iterable[object]], scope: str) -> list[Section]:
    header: list[str] | None = None
    sections = []
    for number, row in enumerate(rows, start=1):
        cells = [_cell(value) for value in row]
        if not any(cells):
            continue
        if header is None:
            header = [name or f"column {i}" for i, name in enumerate(cells, start=1)]
            continue
        pairs = [
            f"{header[i] if i < len(header) else f'column {i + 1}'}: {value}" for i, value in enumerate(cells) if value
        ]
        sections.append(Section(" | ".join(pairs), Locator("row", number, scope), separator="\n"))
    return sections


def _load_csv(path: Path) -> Document:
    text = _read_text(path)
    try:
        dialect = csv.Sniffer().sniff(text[:8192], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    return Document("spreadsheet", _table_sections(csv.reader(io.StringIO(text), dialect), scope=""))


def _load_xlsx(path: Path) -> Document:
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        sections = []
        for sheet in workbook.worksheets:
            sections += _table_sections(sheet.iter_rows(values_only=True), scope=sheet.title)
        metadata = {"author": workbook.properties.creator} if workbook.properties.creator else {}
        return Document("spreadsheet", sections, title=workbook.properties.title or "", metadata=metadata)
    finally:
        workbook.close()


_BLOCK_TAGS = {
    "address", "article", "aside", "blockquote", "br", "dd", "div", "dl", "dt", "figcaption", "footer",
    "form", "h1", "h2", "h3", "h4", "h5", "h6", "header", "hr", "li", "main", "nav", "ol", "p", "pre",
    "section", "table", "td", "th", "tr", "ul",
}
_HIDDEN_TAGS = {"script", "style", "noscript", "template"}


class _HTMLText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title = ""
        self._hidden = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in _HIDDEN_TAGS:
            self._hidden += 1
        elif tag == "title":
            self._in_title = True
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in _HIDDEN_TAGS:
            self._hidden = max(0, self._hidden - 1)
        elif tag == "title":
            self._in_title = False
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._hidden:
            self.parts.append(data)


def html_to_text(html: str) -> tuple[str, str]:
    parser = _HTMLText()
    parser.feed(html)
    parser.close()
    lines = [" ".join(line.split()) for line in "".join(parser.parts).split("\n")]
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    return " ".join(parser.title.split()), text


def _load_html(path: Path) -> Document:
    title, text = html_to_text(_read_text(path))
    return Document("html", [Section(text)], title=title)


_LOADERS: dict[str, Callable[[Path], Document]] = {
    ".txt": _load_text,
    ".md": _load_text,
    ".markdown": _load_text,
    ".pdf": _load_pdf,
    ".eml": _load_email,
    ".csv": _load_csv,
    ".tsv": _load_csv,
    ".xlsx": _load_xlsx,
    ".html": _load_html,
    ".htm": _load_html,
}

SUPPORTED_EXTENSIONS = frozenset(_LOADERS)
