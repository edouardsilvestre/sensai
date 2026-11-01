import json
from io import BytesIO

import pytest
from pypdf import PdfReader

from conftest import FakeClient
from sensai.core.agent import Agent
from sensai.core.document.document import Edit
from sensai.core.export.shares import ShareStore
from sensai.core.guardrails.rules import Guardrails, parse_rules
from sensai.core.memory.sessions import SessionStore
from sensai.interfaces.cli.commands.dispatch import handle_command
from sensai.interfaces.cli.main import main, parse_args
from sensai.interfaces.cli.session import Session

SOURCE = {"name": "source", "terms": ["Marc Vidal"], "on": ["output"], "replacement": "[SOURCE]"}
BLOCK = {"name": "stop", "terms": ["Marc Vidal"], "on": ["publish"], "action": "block", "message": "Never name him."}


@pytest.fixture
def session(config) -> Session:
    agent = Agent(config, FakeClient("Hello"))
    agent.memory.commit("Who approved it?", "Marc Vidal approved it.")
    return Session(config, agent, store=SessionStore(config.sessions_dir), shares=ShareStore(config.shares_dir))


@pytest.fixture
def sharing(session):
    session.config.features["sharing"] = True
    session.config.public_url = "http://sensai.test"
    return session


def answer(monkeypatch, *answers: str) -> None:
    replies = iter(answers)

    def reply(question: str = "") -> str:
        value = next(replies, None)
        if value is None:
            raise EOFError
        return value

    monkeypatch.setattr("builtins.input", reply)


@pytest.mark.parametrize("name", ["out.md", "out.json", "out.pdf", "out.html", "OUT.PDF"])
def test_export_writes_the_format_of_the_file_name(session, tmp_path, capsys, name):
    path = tmp_path / "folder" / name
    handle_command(session, f'/export "{path}"')
    assert path.read_bytes() and f"Conversation written to {path}." in capsys.readouterr().out


def test_export_writes_each_format(session, tmp_path):
    for name in ("out.md", "out.json", "out.pdf", "out.html"):
        handle_command(session, f"/export {tmp_path / name}")
    assert "Marc Vidal approved it." in (tmp_path / "out.md").read_text(encoding="utf-8")
    assert json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))["turns"][0]["user"] == "Who approved it?"
    assert "Marc Vidal" in PdfReader(BytesIO((tmp_path / "out.pdf").read_bytes())).pages[0].extract_text()
    assert (tmp_path / "out.html").read_text(encoding="utf-8").startswith("<!doctype html>")


def test_export_refuses_an_unknown_format(session, tmp_path, capsys):
    path = tmp_path / "out.docx"
    handle_command(session, f"/export {path}")
    assert ".md, .json, .pdf, .html" in capsys.readouterr().err and not path.exists()


def test_export_needs_a_path(session, capsys):
    handle_command(session, "/export")
    assert "Usage: /export PATH" in capsys.readouterr().err


def test_export_needs_a_message(session, tmp_path, capsys):
    session.agent.memory.clear()
    handle_command(session, f"/export {tmp_path / 'out.md'}")
    assert "No message to export yet." in capsys.readouterr().out and not (tmp_path / "out.md").exists()


def test_export_says_when_a_folder_cannot_be_written(session, tmp_path, capsys):
    (tmp_path / "file").write_text("", encoding="utf-8")
    handle_command(session, f"/export {tmp_path / 'file' / 'out.md'}")
    assert "Cannot write" in capsys.readouterr().err


def test_export_says_a_branch_is_alone_in_the_file(session, tmp_path, capsys):
    session.agent.memory.fork(1)
    session.agent.memory.commit("Other", "Answer")
    handle_command(session, f"/export {tmp_path / 'out.md'}")
    assert "branch 2 only, out of 2: a .json file holds all of them" in capsys.readouterr().out
    handle_command(session, f"/export {tmp_path / 'out.json'}")
    assert "only" not in capsys.readouterr().out


def test_document_export_writes_a_pdf(session, tmp_path, capsys):
    session.agent.memory.document.apply([Edit("append", "Summary", "Paid in April.")])
    path = tmp_path / "file.pdf"
    handle_command(session, f"/document export {path}")
    assert "Paid in April." in PdfReader(BytesIO(path.read_bytes())).pages[0].extract_text()
    assert f"Document written to {path}." in capsys.readouterr().out


def test_share_is_off_by_default(session, capsys, monkeypatch):
    answer(monkeypatch)
    handle_command(session, "/share conversation")
    assert "The sharing feature is off" in capsys.readouterr().out and session.shares.list() == []


def test_share_publishes_after_a_yes(sharing, capsys, monkeypatch):
    answer(monkeypatch, "y")
    handle_command(sharing, "/share conversation")
    out = capsys.readouterr().out
    share = sharing.shares.list()[0]
    assert f"Published: http://sensai.test/shared/{share.token}" in out
    assert "Page: Who approved it? (1 message)" in out and "without logging in" in out
    assert "python -m sensai publish" in out
    assert "Marc Vidal approved it." in sharing.shares.page(share.token) and share.kind == "conversation"


@pytest.mark.parametrize("answers", [("n",), ()], ids=["no", "end of input"])
def test_share_publishes_nothing_without_a_yes(sharing, capsys, monkeypatch, answers):
    answer(monkeypatch, *answers)
    handle_command(sharing, "/share conversation")
    assert "Nothing was published." in capsys.readouterr().out and sharing.shares.list() == []


def test_share_says_what_the_rules_changed(sharing, capsys, monkeypatch):
    sharing.agent.guardrails = Guardrails(parse_rules([SOURCE]))
    answer(monkeypatch, "y")
    handle_command(sharing, "/share conversation")
    assert "Rule 'source': 1 redacted (SOURCE)." in capsys.readouterr().out
    page = sharing.shares.page(sharing.shares.list()[0].token)
    assert "Marc Vidal" not in page and "[SOURCE] approved it." in page


def test_share_says_when_nothing_was_checked(sharing, capsys, monkeypatch):
    sharing.agent.guardrails = Guardrails(parse_rules([SOURCE]))
    sharing.config.features["guardrails"] = False
    answer(monkeypatch, "n")
    handle_command(sharing, "/share conversation")
    assert "The guardrails are off: nothing was checked or redacted." in capsys.readouterr().out


def test_share_names_the_sources(sharing, capsys, monkeypatch):
    sharing.agent.memory.commit("Where?", "In the file [1].", [r"C:\Enquete\mail.eml, page 2"])
    answer(monkeypatch, "n")
    handle_command(sharing, "/share conversation")
    assert "Sources named: mail.eml, page 2" in capsys.readouterr().out


def test_a_block_rule_stops_the_share(sharing, capsys, monkeypatch):
    sharing.agent.guardrails = Guardrails(parse_rules([BLOCK]))
    answer(monkeypatch)
    handle_command(sharing, "/share conversation")
    assert "Not published: the rule 'stop' found STOP. Never name him." in capsys.readouterr().err
    assert sharing.shares.list() == []


def test_share_document(sharing, capsys, monkeypatch):
    sharing.agent.memory.document.apply([Edit("append", "Summary", "Paid in April.")])
    answer(monkeypatch, "o")
    handle_command(sharing, "/share document")
    share = sharing.shares.list()[0]
    assert share.kind == "document" and share.title == "Investigation"
    assert "Paid in April." in sharing.shares.page(share.token) and "Who approved it?" not in sharing.shares.page(share.token)


def test_share_refuses_an_empty_document(sharing, capsys):
    handle_command(sharing, "/share document")
    assert "The document is empty" in capsys.readouterr().out


def test_share_refuses_an_empty_conversation(sharing, capsys):
    sharing.agent.memory.clear()
    handle_command(sharing, "/share conversation")
    assert "No message to publish yet." in capsys.readouterr().out


def test_share_does_not_follow_the_conversation(sharing, monkeypatch):
    answer(monkeypatch, "y")
    handle_command(sharing, "/share conversation")
    sharing.agent.memory.commit("A later secret", "Later answer")
    assert "later" not in sharing.shares.page(sharing.shares.list()[0].token).lower()


def test_share_lists_the_links(sharing, capsys, monkeypatch):
    handle_command(sharing, "/share")
    assert "No public link yet." in capsys.readouterr().out
    answer(monkeypatch, "y")
    handle_command(sharing, "/share conversation")
    capsys.readouterr()
    handle_command(sharing, "/share")
    out = capsys.readouterr().out
    assert "conversation  Who approved it?" in out and f"http://sensai.test/shared/{sharing.shares.list()[0].token}" in out


def test_a_link_can_be_taken_down_with_the_feature_off(sharing, capsys, monkeypatch):
    answer(monkeypatch, "y")
    handle_command(sharing, "/share conversation")
    token = sharing.shares.list()[0].token
    sharing.config.features["sharing"] = False
    capsys.readouterr()
    handle_command(sharing, "/share remove 1")
    assert "Link 1 is down: Who approved it?" in capsys.readouterr().out
    assert sharing.shares.page(token) is None


@pytest.mark.parametrize(
    ("command", "message"),
    [
        ("/share remove", "Usage: /share"),
        ("/share remove x", "Usage: /share"),
        ("/share remove 3", "No link 3"),
        ("/share everything", "Usage: /share"),
        ("/share conversation now", "Usage: /share"),
    ],
)
def test_share_errors(sharing, capsys, command, message):
    handle_command(sharing, command)
    assert message in capsys.readouterr().err


def test_publish_listens_on_every_address_by_default():
    assert parse_args(["publish"]).host == "0.0.0.0"


def test_publish_stops_when_the_feature_is_off(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("SENSAI_FEATURE_SHARING", raising=False)
    env = tmp_path / "test.env"
    env.write_text("", encoding="utf-8")
    assert main(["-e", str(env), "publish"]) == 1
    assert "SENSAI_FEATURE_SHARING=true" in capsys.readouterr().err
