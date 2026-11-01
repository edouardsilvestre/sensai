import json
import sys
from io import BytesIO

import pytest
from pypdf import PdfReader

from sensai.core.document.document import Document, Edit
from sensai.core.export.markdown import Block, blocks, inline
from sensai.core.export.page import body
from sensai.core.export.report import (
    SUMMARY,
    ExportError,
    Report,
    conversation_report,
    document_report,
    render,
    to_html,
    to_markdown,
)
from sensai.core.memory.conversation import ConversationMemory, ToolUse
from sensai.core.memory.sessions import SessionStore, session_data

HTML_TAGS = (("<b>", "</b>"), ("<i>", "</i>"), ("<c>", "</c>"))


def pdf_text(data: bytes) -> str:
    return "\n".join(page.extract_text() for page in PdfReader(BytesIO(data)).pages)


@pytest.fixture
def memory(config) -> ConversationMemory:
    memory = ConversationMemory(config)
    memory.commit(
        "Qui a validé le virement ?", "**Marc Vidal** [1], pour 12 000 €.", ["bank/statement.pdf, p. 3", "unused.pdf"],
        tools=[ToolUse("search_documents", {"query": "virement"}, "2 excerpts")],
    )
    return memory


def report_of(memory) -> object:
    return conversation_report(memory, "chat-model", session_data("20261101-120000", memory, "chat-model"))


def test_blocks_reads_the_markdown_like_the_page():
    text = "# Title\nFirst\nsecond\n\n- one\n* two\n3. three\n\n```\ncode\n# not a title\n```\nend"
    assert blocks(text) == [
        Block("heading", "Title", 1), Block("paragraph", "First\nsecond"), Block("bullet", "one"), Block("bullet", "two"),
        Block("numbered", "three", 3), Block("code", "code\n# not a title"), Block("paragraph", "end"),
    ]


def test_a_block_of_code_never_closed_is_kept():
    assert blocks("```\ninterrupted") == [Block("code", "interrupted")]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("**bold** and *italic*", "<b>bold</b> and <i>italic</i>"),
        ("a `b**c` d", "a <c>b**c</c> d"),
        ("<b> & co", "&lt;b&gt; &amp; co"),
        ("`<script>`", "<c>&lt;script&gt;</c>"),
        # A bold that straddles a piece of code stays plain: the tags would cross.
        ("**a `b** c`", "**a <c>b** c</c>"),
        ("*a **b** c*", "<i>a <b>b</b> c</i>"),
        ("2 * 3 * 4", "2 * 3 * 4"),
    ],
)
def test_inline_nests_its_tags(text, expected):
    assert inline(text, *HTML_TAGS) == expected


def test_a_numbered_list_keeps_its_numbers():
    assert '<li value="4">four</li>' in body("4. four")


def test_a_list_is_closed_before_what_follows():
    assert body("- one\n\ntext\n\n1. two") == '<ul>\n<li>one</li>\n</ul>\n<p>text</p>\n<ol>\n<li value="1">two</li>\n</ol>'


def test_markdown_holds_the_messages_and_the_sources_cited(memory):
    text = to_markdown(report_of(memory))
    assert text.startswith("# Qui a validé le virement ?\n\n*Model chat-model, ")
    assert "## You\n\nQui a validé le virement ?\n\n## Sensai\n\n**Marc Vidal** [1], pour 12 000 €." in text
    assert "- [1] bank/statement.pdf, p. 3" in text and "unused.pdf" not in text


def test_what_was_condensed_is_shown_as_a_summary(memory):
    memory.summary = "Vidal runs Alvor."
    assert f"## {SUMMARY}\n\nVidal runs Alvor." in to_markdown(report_of(memory))


def test_the_branch_is_named_when_there_are_several(memory):
    assert "branch" not in report_of(memory).details
    memory.fork(1)
    memory.commit("Other question", "Other answer")
    report = report_of(memory)
    assert "branch 2 of 2" in report.details and "Marc Vidal" not in to_markdown(report)


def test_the_json_holds_everything_and_can_be_resumed(memory, config):
    memory.fork(1)
    memory.commit("Other question", "Other answer")
    data = json.loads(render(report_of(memory), ".json"))
    assert data["turns"][0]["user"] == "Other question"
    assert data["other_branches"][0]["turns"][0]["tools"][0]["name"] == "search_documents"

    config.sessions_dir.mkdir(parents=True)
    (config.sessions_dir / "imported.json").write_bytes(render(report_of(memory), ".json"))
    resumed = ConversationMemory(config)
    SessionStore(config.sessions_dir).load("imported", resumed)
    assert resumed.turns[0].assistant == "Other answer" and len(resumed.branches()) == 2


def test_the_markdown_of_the_document_is_written_as_it_is():
    document = Document()
    document.apply([Edit("append", "Summary", "Marc Vidal approved it.")])
    assert render(document_report(document), ".md").decode("utf-8") == document.markdown()


def test_the_json_of_the_document_has_its_sections():
    document = Document()
    document.apply([Edit("append", "Timeline", "4 April: transfer.")])
    data = json.loads(render(document_report(document), ".json"))
    assert data["title"] == "Investigation"
    assert {"name": "Timeline", "body": "4 April: transfer."} in data["sections"]


def test_the_page_stands_alone_and_runs_nothing(memory):
    memory.commit('<script>alert(1)</script> <img src=x onerror=alert(2)>', "[click](javascript:alert(3))")
    page = to_html(report_of(memory))
    assert "<script" not in page and "<img" not in page and "<a " not in page
    assert "&lt;script&gt;" in page
    assert "src=" not in page.replace("&lt;img src=x", "") and "http" not in page and "url(" not in page


def test_the_title_is_escaped_on_the_page(config):
    memory = ConversationMemory(config)
    memory.commit("</title><script>alert(1)</script>", "No.")
    assert "<script" not in to_html(report_of(memory))


def test_the_pdf_holds_the_accents_and_the_sources(memory):
    data = render(report_of(memory), ".pdf")
    assert data.startswith(b"%PDF")
    text = pdf_text(data)
    assert "Qui a validé le virement ?" in text and "12 000 €" in text and "[1] bank/statement.pdf, p. 3" in text
    assert "Marc Vidal" in text and "**" not in text and "not in the font" not in text
    assert PdfReader(BytesIO(data)).metadata.title == "Qui a validé le virement ?"


def test_the_pdf_says_what_its_fonts_cannot_draw(config):
    memory = ConversationMemory(config)
    memory.commit("Who?", "Иван paid 12 000 € → Orion")
    text = pdf_text(render(report_of(memory), ".pdf"))
    assert "???? paid 12 000 € -> Orion" in text
    assert "4 characters are not in the font of this PDF" in text


@pytest.mark.parametrize(
    "answer",
    [
        "<b>not closed & <para> **bold `code** end` </font>",
        "```\n<b>& " + "x" * 500 + "\n```",
        "*" * 200,
        "\x00\x07 control ​ characters",
        "# " + "very long title " * 40,
        "",
    ],
)
def test_the_pdf_takes_any_answer(config, answer):
    memory = ConversationMemory(config)
    memory.commit("<title>", answer)
    assert render(report_of(memory), ".pdf").startswith(b"%PDF")


def test_the_pdf_composes_the_accents_written_apart(config):
    memory = ConversationMemory(config)
    memory.commit("Who?", "Be\u0301nard paid.")
    text = pdf_text(render(report_of(memory), ".pdf"))
    assert "Bénard paid." in text and "not in the font" not in text


@pytest.mark.parametrize(
    "answer", ["word " * 6000, "```\n" + "line\n" * 400 + "```", "- item\n" * 300], ids=["paragraph", "code", "list"],
)
def test_a_block_longer_than_a_page_is_cut(config, answer):
    memory = ConversationMemory(config)
    memory.commit("Who?", answer)
    assert len(PdfReader(BytesIO(render(report_of(memory), ".pdf"))).pages) > 1


def test_a_long_conversation_goes_on_over_the_pages(config):
    memory = ConversationMemory(config)
    for number in range(60):
        memory.commit(f"Question {number}", f"Answer {number}\n\n- " + "word " * 80)
    data = render(report_of(memory), ".pdf")
    assert len(PdfReader(BytesIO(data)).pages) > 3
    assert "Answer 59" in pdf_text(data)


def test_the_pdf_of_the_document_has_its_sections():
    document = Document()
    document.apply([Edit("append", "Timeline", "- 4 avril : virement")])
    text = pdf_text(render(document_report(document), ".pdf"))
    assert "Investigation" in text and "Timeline" in text and "4 avril : virement" in text


def test_the_pdf_alone_needs_reportlab(memory, monkeypatch):
    monkeypatch.setitem(sys.modules, "sensai.core.export.pdf", None)
    with pytest.raises(ExportError, match="pip install -r requirements.txt"):
        render(report_of(memory), ".pdf")
    assert render(report_of(memory), ".md")


def test_a_report_without_parts_is_only_a_title():
    assert to_markdown(Report("T", "D", [])) == "# T\n\n*D*\n"
