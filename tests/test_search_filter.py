import pytest

from sensai.core.rag.models import SearchFilter, parse_filter


def test_a_filter_is_read_from_the_command_line_syntax():
    search_filter = parse_filter("after:2023-03 before:2023-07-15 from:Dupont to:marc kind:EMAIL source:leak/mails/")
    assert search_filter == SearchFilter(
        after="2023-03", before="2023-07-15", author="Dupont", recipient="marc", kind="email", source="leak/mails/"
    )
    assert parse_filter(search_filter.describe()) == search_filter


def test_a_value_with_spaces_is_quoted():
    search_filter = parse_filter('from:"Jean Dupont" kind:pdf')
    assert search_filter.author == "Jean Dupont"
    assert search_filter.describe() == 'from:"Jean Dupont" kind:pdf'


def test_an_empty_filter():
    assert parse_filter("").is_empty()
    assert SearchFilter().describe() == ""
    assert not SearchFilter(kind="pdf").is_empty()


@pytest.mark.parametrize(
    "text, message",
    [
        ("after:2023-13", "after"),
        ("before:March", "before"),
        ("after:2023-07 before:2023-03", "earlier"),
        ("kind:video", "email, html, pdf, spreadsheet, text"),
        ("author:dupont", "Unknown filter 'author'"),
        ("dupont", "KEY:VALUE"),
        ("from:", "KEY:VALUE"),
    ],
)
def test_an_invalid_filter_says_what_is_wrong(text, message):
    with pytest.raises(ValueError, match=message):
        parse_filter(text)


def test_a_name_with_an_apostrophe_survives_a_round_trip():
    search_filter = SearchFilter(author="Sean O'Brien")
    assert parse_filter(search_filter.describe()) == search_filter
