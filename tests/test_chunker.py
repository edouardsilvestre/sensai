import pytest

from sensai.core.rag.chunker import Chunker, format_location
from sensai.core.rag.models import Document, Locator, Section


def sentences(count: int) -> str:
    return " ".join(f"Sentence number {i} says something." for i in range(count))


def test_short_document_is_one_chunk():
    chunks = Chunker(200, 40).split(Document("text", [Section("Just one line.")]))
    assert [chunk.text for chunk in chunks] == ["Just one line."]


def test_chunks_respect_size_and_keep_every_sentence():
    text = sentences(40)
    chunks = Chunker(200, 40).split(Document("text", [Section(text)]))
    assert len(chunks) > 1
    assert all(len(chunk.text) <= 200 for chunk in chunks)
    for i in range(40):
        assert any(f"Sentence number {i} says" in chunk.text for chunk in chunks)


def test_consecutive_chunks_overlap():
    chunks = Chunker(200, 80).split(Document("text", [Section(sentences(40))]))
    for previous, current in zip(chunks, chunks[1:]):
        first_sentence = current.text.split(".")[0]
        assert first_sentence in previous.text


def test_no_overlap_when_disabled():
    chunks = Chunker(200, 0).split(Document("text", [Section(sentences(40))]))
    assert "".join(chunk.text.replace(" ", "") for chunk in chunks) == sentences(40).replace(" ", "")


def test_paragraph_breaks_are_kept():
    chunks = Chunker(500, 0).split(Document("text", [Section("First paragraph.\n\nSecond paragraph.")]))
    assert chunks[0].text == "First paragraph.\n\nSecond paragraph."


def test_oversized_word_is_cut():
    chunks = Chunker(100, 10).split(Document("text", [Section("x" * 450)]))
    assert all(len(chunk.text) <= 100 for chunk in chunks)
    assert sum(len(chunk.text) for chunk in chunks) >= 450


def test_chunks_span_pages_with_a_page_range():
    pages = [Section(sentences(3), Locator("page", number)) for number in (1, 2, 3)]
    chunks = Chunker(220, 0).split(Document("pdf", pages))
    assert chunks[0].location == "p. 1-2"
    assert all(chunk.location.startswith("p. ") for chunk in chunks)


def test_rows_of_different_sheets_never_share_a_chunk():
    rows = [Section(f"name: a{i}", Locator("row", i, "Sheet A"), separator="\n") for i in range(2, 5)]
    rows += [Section(f"name: b{i}", Locator("row", i, "Sheet B"), separator="\n") for i in range(2, 5)]
    chunks = Chunker(1000, 100).split(Document("spreadsheet", rows))
    assert [(chunk.location, chunk.text) for chunk in chunks] == [
        ("Sheet A, rows 2-4", "name: a2\nname: a3\nname: a4"),
        ("Sheet B, rows 2-4", "name: b2\nname: b3\nname: b4"),
    ]


@pytest.mark.parametrize(
    ("locators", "expected"),
    [
        ([], ""),
        ([Locator("page", 4)], "p. 4"),
        ([Locator("page", 4), Locator("page", 6)], "p. 4-6"),
        ([Locator("row", 3)], "row 3"),
        ([Locator("row", 3, "Transfers"), Locator("row", 9, "Transfers")], "Transfers, rows 3-9"),
    ],
)
def test_format_location(locators, expected):
    assert format_location(locators) == expected


def test_invalid_settings_are_rejected():
    with pytest.raises(ValueError):
        Chunker(100, 100)
