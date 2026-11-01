import json
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

from sensai.core.export.report import HIDDEN_SOURCE, YOU, Part, Report, conversation_report, prepare_publication, to_html
from sensai.core.export.shares import Share, ShareError, ShareStore, public_link
from sensai.core.guardrails.rules import DEFAULT_RULES, Guardrails, parse_rules
from sensai.core.memory.conversation import ConversationMemory
from sensai.interfaces.web.public import PAGE_HEADERS, create_public_app

SOURCE = {"name": "source", "terms": ["Claire Benoit"], "on": ["output"], "replacement": "[SOURCE]"}
EMAIL = "c.benoit@alvor.fr"


def guardrails(*rules: dict) -> Guardrails:
    return Guardrails(parse_rules(list(rules)))


def report(text: str = "Claire Benoit spoke.", citation: str = "mail.eml, page 1") -> Report:
    return Report(
        "Who is Claire Benoit?", "Model m", [Part(YOU, "Who is Claire Benoit?"), Part("Sensai", text, [(1, citation)])],
        {"turns": "the whole session"},
    )


@pytest.fixture
def shares(tmp_path) -> ShareStore:
    return ShareStore(tmp_path / "shared")


@pytest.fixture
def public(shares) -> TestClient:
    return TestClient(create_public_app(shares))


def test_a_page_is_found_by_its_link_only(shares):
    share = shares.add("conversation", "Title", "<p>page</p>")
    assert len(share.token) == 32 and share.token == share.token.lower()
    assert shares.page(share.token) == "<p>page</p>"
    assert shares.page("0" * 32) is None


def test_two_links_are_different(shares):
    assert shares.add("document", "A", "a").token != shares.add("document", "A", "a").token


@pytest.mark.parametrize("token", ["../secret", "..\\secret", "x", "0" * 31, "0" * 33, "G" * 32, "A" * 32])
def test_nothing_but_a_link_reads_a_file(shares, token):
    shares.directory.mkdir()
    (shares.directory.parent / "secret.json").write_text(json.dumps({"page": "secret"}), encoding="utf-8")
    assert shares.page(token) is None and shares.remove(token) is False


def test_the_links_are_listed_from_the_most_recent(shares):
    first, second = shares.add("document", "First", "a"), shares.add("conversation", "Second", "b")
    assert {share.token for share in shares.list()} == {first.token, second.token}
    assert shares.list()[0].created >= shares.list()[1].created
    assert ShareStore(shares.directory / "nothing").list() == []


def test_an_unreadable_file_is_left_out(shares):
    share = shares.add("document", "Kept", "a")
    (shares.directory / f"{'0' * 32}.json").write_text("{not json", encoding="utf-8")
    (shares.directory / "notes.json").write_text('{"kind": "x", "title": "y", "created": "2026-01-01", "page": "z"}', encoding="utf-8")
    assert [listed.token for listed in shares.list()] == [share.token]


def test_a_removed_link_stops_answering(shares):
    share = shares.add("document", "A", "a")
    assert shares.remove(share.token) is True
    assert shares.page(share.token) is None and shares.remove(share.token) is False


def test_a_folder_that_cannot_be_written_is_an_error(tmp_path):
    (tmp_path / "file").write_text("", encoding="utf-8")
    with pytest.raises(ShareError, match="Cannot write the page"):
        ShareStore(tmp_path / "file" / "shared").add("document", "A", "a")


def test_the_link_starts_with_the_public_url(config):
    config.public_url = "https://sensai.example.org/"
    assert public_link(config, Share("abc", "document", "A", None)) == "https://sensai.example.org/shared/abc"


def test_without_a_public_url_the_link_uses_this_machine(config, monkeypatch):
    monkeypatch.setattr("sensai.core.export.shares.local_address", lambda: "192.168.1.20")
    config.public_port = 9000
    assert public_link(config, Share("abc", "document", "A", None)) == "http://192.168.1.20:9000/shared/abc"


def test_the_public_server_shows_a_page_to_anyone(shares, public):
    share = shares.add("conversation", "Title", "<p>frozen copy</p>")
    response = public.get(f"/shared/{share.token}")
    assert response.status_code == 200 and response.text == "<p>frozen copy</p>"
    assert response.headers["content-type"].startswith("text/html")
    assert {name: response.headers[name] for name in PAGE_HEADERS} == PAGE_HEADERS
    assert "script-src" not in response.headers["content-security-policy"]


def test_the_public_server_shows_nothing_else(shares, public):
    shares.add("conversation", "Title", "<p>page</p>")
    (shares.directory.parent / "sessions").mkdir()
    for path in ("/", "/shared", "/shared/", f"/shared/{'0' * 32}", "/shared/..%2Fsessions", "/api/state", "/docs", "/openapi.json"):
        response = public.get(path)
        assert response.status_code == 404, path
        assert "page" not in response.text.replace("This page does not exist", ""), path


def test_a_taken_down_link_says_so(shares, public):
    share = shares.add("conversation", "Title", "<p>page</p>")
    shares.remove(share.token)
    response = public.get(f"/shared/{share.token}")
    assert response.status_code == 404 and "no longer shared" in response.text


def test_the_public_server_only_reads(shares, public):
    share = shares.add("conversation", "Title", "<p>page</p>")
    assert public.post(f"/shared/{share.token}").status_code == 405
    assert public.delete(f"/shared/{share.token}").status_code == 405
    assert shares.page(share.token)


def test_the_public_server_needs_no_session():
    # It must start without the chat: what it can reach is what can leak.
    code = "import sys, sensai.interfaces.web.public; print('sensai.interfaces.cli.main' in sys.modules)"
    assert subprocess.check_output([sys.executable, "-c", code], text=True).strip() == "False"


def test_a_rule_redacts_the_page():
    publication = prepare_publication(report(), guardrails(SOURCE))
    page = to_html(publication.report)
    assert "Claire Benoit" not in page and "[SOURCE] spoke." in page
    assert publication.report.title == "Who is [SOURCE]?"
    assert [(hit.rule, hit.count, hit.found) for hit in publication.hits] == [("source", 2, "SOURCE")]


def test_a_title_cut_in_a_protected_name_leaves_no_piece_of_it(config):
    memory = ConversationMemory(config)
    memory.commit("x" * 50 + " Claire Benoit paid", "No.")
    report = conversation_report(memory, "m", {})
    assert report.title.endswith("Claire...")
    publication = prepare_publication(report, guardrails(SOURCE))
    assert "Claire" not in to_html(publication.report) and "Claire" not in publication.report.title


def test_the_rules_of_the_messages_count_too():
    rule = {"name": "input", "terms": ["Claire Benoit"], "on": ["input"], "action": "redact"}
    assert "Claire Benoit" not in to_html(prepare_publication(report(), guardrails(rule)).report)


def test_personal_data_is_redacted_by_default():
    publication = prepare_publication(report(f"Write to {EMAIL}."), Guardrails(parse_rules(DEFAULT_RULES)))
    assert EMAIL not in to_html(publication.report) and "[EMAIL]" in to_html(publication.report)


def test_a_rule_for_the_tools_is_not_about_the_page():
    rule = {"name": "tool", "terms": ["Claire Benoit"], "on": ["tools"], "action": "block"}
    assert prepare_publication(report(), guardrails(rule)).blocked is None


def test_a_block_rule_stops_the_publication():
    rule = {"name": "stop", "terms": ["Claire Benoit"], "on": ["publish"], "action": "block", "message": "Protected."}
    publication = prepare_publication(report(), guardrails(rule))
    assert publication.blocked.rule == "stop" and publication.blocked.message == "Protected."


def test_a_warning_leaves_the_text():
    rule = {"name": "check", "terms": ["Claire Benoit"], "on": ["output"], "action": "warn", "message": "Check."}
    publication = prepare_publication(report(), guardrails(rule))
    assert "Claire Benoit" in to_html(publication.report)
    assert publication.hits[0].action == "warn" and publication.blocked is None


def test_the_judge_is_not_asked():
    def judge(policy: str, text: str) -> str:
        raise AssertionError("The judge is about what the user may ask")

    rule = {"name": "judged", "judge": "anything", "on": ["input"], "action": "block"}
    assert prepare_publication(report(), Guardrails(parse_rules([rule]), judge)).blocked is None


@pytest.mark.parametrize(
    ("citation", "shown"),
    [
        (r"C:\Users\alexa\Enquete Alvor\mail.eml, page 1", "mail.eml, page 1"),
        ("/home/alexa/enquete-alvor/mail.eml", "mail.eml"),
        (r"C:\Enquete, Alvor\mail.eml, page 1", "mail.eml, page 1"),
        ("mail.eml", "mail.eml"),
    ],
)
def test_the_folders_of_the_sources_stay_on_this_machine(citation, shown):
    publication = prepare_publication(report(citation=citation), guardrails())
    assert publication.sources == [shown]
    assert "alexa" not in to_html(publication.report).lower().replace("alexandre", "")


@pytest.mark.parametrize("name", ["claire_benoit.eml", "Claire-Benoit-interview.pdf", "claire.benoit.txt"])
def test_a_file_named_after_a_protected_name_is_hidden(name):
    publication = prepare_publication(report(citation=f"sources/{name}, page 1"), guardrails(SOURCE))
    page = to_html(publication.report).lower()
    assert publication.sources == [HIDDEN_SOURCE] and "benoit" not in page.replace("[source]", "")


def test_only_the_parts_are_published():
    publication = prepare_publication(report(), guardrails())
    assert publication.report.data == {} and "whole session" not in to_html(publication.report)
