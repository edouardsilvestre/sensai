import pytest

from sensai.core.document.document import DEFAULT_TEMPLATE, MAX_EDITS, Document, Edit

CASE = """\
# Alvor Consulting

## People
- Marc Vidal, CFO of Alvor Consulting [mail of 4 April 2023]

## Timeline
- 4 April 2023: Marc Vidal approves the transfer of 48 250 EUR
"""


@pytest.fixture
def document() -> Document:
    return Document(CASE)


def reasons(report) -> list[str]:
    return [rejected.reason for rejected in report.rejected]


def test_markdown_is_read_and_written_back_unchanged(document):
    assert document.title == "Alvor Consulting"
    assert document.names() == ["People", "Timeline"]
    assert document.markdown() == CASE


def test_the_default_document_is_empty_with_its_sections():
    document = Document()
    assert document.is_empty
    assert document.names() == ["Summary", "People and companies", "Timeline", "Open questions"]
    assert document.markdown() == DEFAULT_TEMPLATE


def test_append_adds_lines_at_the_end_of_one_section_only(document):
    report = document.apply([Edit("append", "timeline", "- 5 April 2023: the invoice is missing")])
    assert report.applied and not report.rejected
    assert document.section("Timeline").body.endswith("48 250 EUR\n- 5 April 2023: the invoice is missing")
    assert document.section("People").body == "- Marc Vidal, CFO of Alvor Consulting [mail of 4 April 2023]"


def test_replace_corrects_one_passage(document):
    document.apply([Edit("replace", "People", "CEO", find="CFO")])
    assert document.section("People").body == "- Marc Vidal, CEO of Alvor Consulting [mail of 4 April 2023]"


@pytest.mark.parametrize(
    ("find", "reason"),
    [("CTO", "the text to find is not in the section"), ("", "the text to find is not in the section"),
     ("Marc Vidal", "the text to find is 2 times in the section: quote a longer passage")],
)
def test_replace_needs_a_passage_found_once(find, reason):
    document = Document(CASE.replace("48 250 EUR", "48 250 EUR, Marc Vidal again"))
    before = document.markdown()
    report = document.apply([Edit("replace", "Timeline", "X", find=find)])
    assert reasons(report) == [reason]
    assert document.markdown() == before


def test_replace_in_an_empty_section_fills_it():
    document = Document()
    report = document.apply([Edit("replace", "Summary", "Alvor paid Kestrel without invoice", find="(nothing)")])
    assert report.applied and document.section("Summary").body == "Alvor paid Kestrel without invoice"


def test_rewrite_replaces_one_section(document):
    document.apply([Edit("rewrite", "Timeline", "- 2023: the transfer")])
    assert document.section("Timeline").body == "- 2023: the transfer"
    assert "Marc Vidal, CFO" in document.markdown()


def test_add_section_appends_a_new_section(document):
    document.apply([Edit("add_section", "Money flows", "- 48 250 EUR to Kestrel Ltd")])
    assert document.names() == ["People", "Timeline", "Money flows"]
    assert document.markdown().endswith("## Money flows\n- 48 250 EUR to Kestrel Ltd\n")


@pytest.mark.parametrize(
    ("edit", "reason"),
    [
        (Edit("append", "Suspects", "x"), "no section 'Suspects'; the sections are: People, Timeline"),
        (Edit("add_section", "timeline", "x"), "the section 'Timeline' already exists"),
        (Edit("delete", "Timeline", "x"), "unknown operation 'delete', expected one of append, replace, rewrite, add_section"),
        (Edit("append", "", "x"), "no section given"),
        (Edit("append", "Timeline", "  "), "the text is empty"),
        (Edit("rewrite", "Timeline", ""), "the text is empty"),
        (Edit("append", "People", "- marc vidal,  CFO of Alvor Consulting"), "the section already says it"),
        (Edit("append", "People", "- Marc Vidal, CFO of Alvor Consulting (email of 4 April)"), "the section already says it"),
    ],
)
def test_invalid_edits_are_rejected_with_the_reason(document, edit, reason):
    report = document.apply([edit])
    assert reasons(report) == [reason] and not report.applied
    assert document.markdown() == CASE


@pytest.mark.parametrize(
    ("known", "new"),
    [("- Marc Vidal (CEO)", "- Marc Vidal (CFO)"), ("- Transfer (EUR 2M)", "- Transfer (EUR 5M)")],
)
def test_a_parenthesis_that_is_not_a_citation_is_part_of_the_fact(known, new):
    document = Document(f"## People\n{known}")
    report = document.apply([Edit("append", "People", new)])
    assert report.applied and document.section("People").body == f"{known}\n{new}"


@pytest.mark.parametrize("citation", ["(email of 4 April)", "(source: Claire)", "(alvor_invoices.pdf, p. 3)"])
def test_a_fact_cited_in_a_parenthesis_is_a_duplicate(citation):
    document = Document("## People\n- Marc Vidal, CFO [mail of 4 April 2023]")
    report = document.apply([Edit("append", "People", f"- Marc Vidal, CFO {citation}")])
    assert reasons(report) == ["the section already says it"]


def test_a_new_fact_close_to_a_known_one_is_added(document):
    report = document.apply([Edit("append", "Timeline", "- 6 April 2023: Marc Vidal approves the transfer of 48 250 EUR")])
    assert report.applied


def test_a_rejected_edit_does_not_stop_the_others(document):
    report = document.apply([Edit("append", "Suspects", "x"), Edit("append", "Timeline", "- 6 April: payment")])
    assert len(report.applied) == 1 and len(report.rejected) == 1
    assert "6 April" in document.section("Timeline").body


def test_headings_in_a_text_cannot_split_the_document(document):
    document.apply([Edit("append", "Timeline", "## Hidden\n# Title")])
    assert document.names() == ["People", "Timeline"]
    assert Document(document.markdown()).names() == ["People", "Timeline"]
    assert "### Hidden\n### Title" in document.section("Timeline").body


def test_too_many_edits_in_one_turn_are_rejected(document):
    edits = [Edit("append", "Timeline", f"- event {number}") for number in range(MAX_EDITS + 2)]
    report = document.apply(edits)
    assert len(report.applied) == MAX_EDITS
    assert reasons(report) == [f"more than {MAX_EDITS} edits in one turn"] * 2


def test_undo_restores_the_document_before_the_last_turn(document):
    document.apply([Edit("append", "Timeline", "- one"), Edit("append", "People", "- two")])
    document.apply([Edit("append", "Timeline", "- three")])
    assert document.undo()
    assert "- one" in document.markdown() and "- three" not in document.markdown()
    assert document.undo()
    assert document.markdown() == CASE
    assert not document.undo()


def test_a_turn_without_applied_edit_is_not_an_undo_step(document):
    document.apply([Edit("append", "Suspects", "x")])
    assert not document.undo()
