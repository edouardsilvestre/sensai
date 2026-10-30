import json

import pytest

from sensai.core.agent import Agent
from sensai.core.document.document import Document, Edit
from sensai.core.memory.conversation import ConversationMemory
from sensai.core.memory.sessions import SessionStore
from sensai.interfaces.cli.main import Session, chat_turn, describe_edits, handle_command, load_memory

TIMELINE = {"op": "append", "section": "Timeline", "text": "- 4 April 2023: Marc Vidal approves the transfer"}


@pytest.fixture
def session(config, client):
    session = Session(config, Agent(config, client))
    load_memory(session, None)
    return session


def test_the_changes_are_printed_after_the_answer(session, client, capsys):
    client.edits = json.dumps({"edits": [TIMELINE, {"op": "append", "section": "Suspects", "text": "- Marc Vidal"}]})
    chat_turn(session, "When was the transfer approved?")
    out, err = capsys.readouterr()
    assert "[document] added to Timeline. /document to read it." in out
    assert "document edit rejected (append in 'Suspects'): no section 'Suspects'" in err


def test_changes_are_grouped_by_section():
    edits = [Edit("append", "Timeline", "a"), Edit("append", "Timeline", "b"), Edit("replace", "People", "c", "d")]
    assert describe_edits(edits) == "2 added to Timeline, corrected in People"


def test_a_failed_update_is_a_warning(session, client, capsys):
    client.edits = "{"
    chat_turn(session, "Hi")
    assert "The document was not updated: the edits are not valid JSON" in capsys.readouterr().err


def test_document_shows_the_document(session, client, capsys):
    client.edits = json.dumps({"edits": [TIMELINE]})
    chat_turn(session, "When?")
    capsys.readouterr()
    handle_command(session, "/document")
    out = capsys.readouterr().out
    assert out.startswith("# Investigation\n") and "## Timeline\n- 4 April 2023: Marc Vidal" in out


def test_an_empty_document_lists_its_sections(session, capsys):
    handle_command(session, "/document")
    assert "The document is empty" in capsys.readouterr().out


def test_document_says_when_the_feature_is_off(session, capsys):
    session.config.features["document"] = False
    handle_command(session, "/document")
    assert "The document feature is off" in capsys.readouterr().out


def test_undo_reverts_the_last_answer_and_is_saved(session, client, capsys):
    client.edits = json.dumps({"edits": [TIMELINE]})
    chat_turn(session, "When?")
    handle_command(session, "/document undo")
    assert session.agent.memory.document.is_empty
    assert "undone" in capsys.readouterr().out
    memory = ConversationMemory(session.config)
    SessionStore(session.config.sessions_dir).load(session.session_id, memory)
    assert memory.document.is_empty
    handle_command(session, "/document undo")
    assert "Nothing to undo." in capsys.readouterr().out


def test_export_writes_the_markdown(session, client, tmp_path, capsys):
    client.edits = json.dumps({"edits": [TIMELINE]})
    chat_turn(session, "When?")
    path = tmp_path / "out" / "enquete alvor.md"
    handle_command(session, f'/document export "{path}"')
    assert path.read_text(encoding="utf-8") == session.agent.memory.document.markdown()
    assert f"Document written to {path}." in capsys.readouterr().out


@pytest.mark.parametrize("argument", ["export", "delete"])
def test_invalid_document_command(session, capsys, argument):
    handle_command(session, f"/document {argument}")
    assert "Usage: /document" in capsys.readouterr().err


def test_the_document_is_saved_and_resumed_with_the_conversation(session, client, config):
    client.edits = json.dumps({"edits": [TIMELINE]})
    chat_turn(session, "When?")
    resumed = Session(config, Agent(config, client))
    load_memory(resumed, session.session_id)
    assert resumed.agent.memory.document.markdown() == session.agent.memory.document.markdown()


def test_a_new_conversation_starts_with_an_empty_document(session, client):
    client.edits = json.dumps({"edits": [TIMELINE]})
    chat_turn(session, "When?")
    handle_command(session, "/new")
    assert session.agent.memory.document.is_empty


def test_a_session_saved_before_the_document_starts_with_an_empty_one(config, tmp_path):
    path = config.sessions_dir / "old.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"version": 1, "updated": "2026-10-27T15:30:12", "turns": []}), encoding="utf-8")
    memory = ConversationMemory(config)
    memory.document = Document("# Other\n\n## A\nx\n")
    SessionStore(config.sessions_dir).load("old", memory)
    assert memory.document.markdown() == Document().markdown()
