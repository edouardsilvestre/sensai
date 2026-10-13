from email.message import EmailMessage

import pytest
from openpyxl import Workbook

from sensai.core.rag.loaders import DocumentLoadError, html_to_text, load_document
from sensai.core.rag.models import Locator


def make_pdf(pages: list[str]) -> bytes:
    page_ids = [4 + 2 * i for i in range(len(pages))]
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{' '.join(f'{p} 0 R' for p in page_ids)}] /Count {len(pages)} >>".encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for page_id, text in zip(page_ids, pages):
        content = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {page_id + 1} 0 R >>".encode()
        )
        objects.append(b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def test_markdown_title_is_its_first_heading(tmp_path):
    path = tmp_path / "notes.md"
    path.write_text("Intro\n# Offshore accounts\nBody text.", encoding="utf-8")
    document = load_document(path)
    assert document.title == "Offshore accounts"
    assert "Body text." in document.sections[0].text


def test_text_in_legacy_encoding_is_decoded(tmp_path):
    path = tmp_path / "note.txt"
    path.write_bytes("Réunion à Genève".encode("cp1252"))
    assert load_document(path).sections[0].text == "Réunion à Genève"


def test_empty_file_is_rejected(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("  \n", encoding="utf-8")
    with pytest.raises(DocumentLoadError, match="no text"):
        load_document(path)


def test_unsupported_type_is_rejected(tmp_path):
    path = tmp_path / "image.png"
    path.write_bytes(b"\x89PNG")
    with pytest.raises(DocumentLoadError, match="unsupported"):
        load_document(path)


def test_pdf_keeps_one_section_per_page(tmp_path):
    path = tmp_path / "report.pdf"
    path.write_bytes(make_pdf(["First page text", "Second page text"]))
    document = load_document(path)
    assert [(section.text, section.locator) for section in document.sections] == [
        ("First page text", Locator("page", 1)),
        ("Second page text", Locator("page", 2)),
    ]


def test_pdf_without_text_is_reported_as_scanned(tmp_path):
    path = tmp_path / "scan.pdf"
    path.write_bytes(make_pdf([""]))
    with pytest.raises(DocumentLoadError, match="OCR"):
        load_document(path)


def test_corrupted_pdf_is_reported(tmp_path):
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"not a pdf at all")
    with pytest.raises(DocumentLoadError, match="cannot parse"):
        load_document(path)


def test_csv_rows_read_as_column_value_pairs(tmp_path):
    path = tmp_path / "transfers.csv"
    path.write_text("date;amount;to\n2023-04-12;250000;Delta Holding\n\n2023-05-02;;Omega\n", encoding="utf-8")
    sections = load_document(path).sections
    assert [(section.text, section.locator) for section in sections] == [
        ("date: 2023-04-12 | amount: 250000 | to: Delta Holding", Locator("row", 2)),
        ("date: 2023-05-02 | to: Omega", Locator("row", 4)),
    ]


def test_xlsx_rows_are_located_by_sheet(tmp_path):
    workbook = Workbook()
    workbook.active.title = "Transfers"
    workbook.active.append(["amount", "to"])
    workbook.active.append([1500.0, "Delta"])
    workbook.create_sheet("Accounts").append(["iban"])
    workbook["Accounts"].append(["CH93 0076"])
    path = tmp_path / "ledger.xlsx"
    workbook.save(path)

    sections = load_document(path).sections
    assert [(section.text, section.locator) for section in sections] == [
        ("amount: 1500 | to: Delta", Locator("row", 2, "Transfers")),
        ("iban: CH93 0076", Locator("row", 2, "Accounts")),
    ]


def test_email_headers_become_metadata(tmp_path):
    message = EmailMessage()
    message["From"] = "Alice Martin <alice@corp.example>"
    message["To"] = "bob@corp.example"
    message["Subject"] = "Transfer to Delta"
    message["Date"] = "Wed, 12 Apr 2023 09:30:00 +0200"
    message.set_content("Please approve the transfer.")
    message.add_attachment(b"%PDF", maintype="application", subtype="pdf", filename="contract.pdf")
    path = tmp_path / "mail.eml"
    path.write_bytes(message.as_bytes())

    document = load_document(path)
    assert document.title == "Transfer to Delta"
    assert document.metadata == {
        "from": "Alice Martin <alice@corp.example>",
        "to": "bob@corp.example",
        "date": "2023-04-12T09:30+02:00",
        "attachments": "contract.pdf",
    }
    assert document.sections[0].text == "Please approve the transfer."


def test_html_only_email_is_converted_to_text(tmp_path):
    message = EmailMessage()
    message["Subject"] = "Hi"
    message.set_content("<p>Meet at <b>noon</b>.</p><script>evil()</script>", subtype="html")
    path = tmp_path / "mail.eml"
    path.write_bytes(message.as_bytes())
    assert load_document(path).sections[0].text == "Meet at noon."


def test_html_to_text_skips_scripts_and_keeps_blocks():
    title, text = html_to_text(
        "<html><head><title> Board  minutes </title><style>p{}</style></head>"
        "<body><h1>Minutes</h1><p>Vote approved.</p><script>x()</script></body></html>"
    )
    assert title == "Board minutes"
    assert text == "Minutes\n\nVote approved."
