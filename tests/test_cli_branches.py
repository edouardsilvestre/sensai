import pytest

from sensai.core.agent import Agent
from sensai.core.document.document import Edit
from sensai.core.guardrails.rules import Guardrails, parse_rules
from sensai.core.llm.ollama import LLMError
from sensai.core.memory.sessions import SessionStore
from sensai.interfaces.cli.main import Session, chat_turn, handle_command, load_memory


@pytest.fixture
def session(config, client):
    session = Session(config, Agent(config, client))
    load_memory(session, None)
    for question in ("First question", "Second question", "Third question"):
        chat_turn(session, question)
    return session


def questions(session) -> list[str]:
    return [turn.user for turn in session.agent.memory.turns]


def test_history_numbers_the_messages_of_the_branch(session, capsys):
    capsys.readouterr()
    handle_command(session, "/history")
    out = capsys.readouterr().out
    assert "Branch 1:" in out
    assert "  1. First question" in out and "  3. Third question" in out
    assert "Hello there" in out


def test_history_of_an_empty_conversation(config, client, capsys):
    handle_command(Session(config, Agent(config, client)), "/history")
    assert "No message yet." in capsys.readouterr().out


def test_edit_answers_in_a_new_branch(session, client, capsys):
    handle_command(session, "/edit 2 Another question")
    assert questions(session) == ["First question", "Another question"]
    sent = [message["content"] for message in client.chats[-1][0]]
    assert "Second question" not in sent and "Another question" in sent
    assert "Branch 2, from message 2 of branch 1, which stays as it was." in capsys.readouterr().out


def test_edit_keeps_the_original_branch(session, capsys):
    handle_command(session, "/edit 2 Another question")
    handle_command(session, "/branch 1")
    assert questions(session) == ["First question", "Second question", "Third question"]
    assert "On branch 1, 3 turns. Last answer: Hello there" in capsys.readouterr().out


def test_edit_asks_for_the_text_when_it_is_not_given(session, monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda text: "Typed question")
    handle_command(session, "/edit 1")
    assert questions(session) == ["Typed question"]
    assert "Message 1: First question" in capsys.readouterr().out


@pytest.mark.parametrize("answer", ["", "   "])
def test_edit_can_be_cancelled(session, monkeypatch, answer):
    monkeypatch.setattr("builtins.input", lambda text: answer)
    handle_command(session, "/edit 1")
    assert len(session.agent.memory.branches()) == 1 and len(questions(session)) == 3


@pytest.mark.parametrize(
    ("argument", "message"),
    [
        ("", "Usage: /edit N"),
        ("two text", "Usage: /edit N"),
        ("\u00b2 text", "Usage: /edit N"),
        ("0 text", "No message 0"),
        ("9 text", "No message 9"),
    ],
)
def test_invalid_edit(session, capsys, argument, message):
    handle_command(session, f"/edit {argument}")
    assert message in capsys.readouterr().err
    assert len(session.agent.memory.branches()) == 1


def test_an_edit_the_guardrails_block_makes_no_branch(config, client, capsys):
    rule = {"name": "private", "pattern": "adresse", "on": ["input"], "action": "block"}
    session = Session(config, Agent(config, client, guardrails=Guardrails(parse_rules([rule]))))
    chat_turn(session, "First question")
    handle_command(session, "/edit 1 Son adresse ?")
    assert session.agent.memory.branch.number == 1 and questions(session) == ["First question"]
    assert len(session.agent.memory.branches()) == 1
    assert "Nothing was answered: back on branch 1." in capsys.readouterr().out


def test_an_edit_the_guardrails_redact_keeps_its_branch(config, client):
    rule = {"name": "source", "terms": ["Claire"], "on": ["input"], "replacement": "[SOURCE]"}
    session = Session(config, Agent(config, client, guardrails=Guardrails(parse_rules([rule]))))
    chat_turn(session, "First question")
    handle_command(session, "/edit 1 Who is Claire?")
    assert session.agent.memory.branch.number == 2 and questions(session) == ["Who is [SOURCE]?"]


def test_the_guardrails_also_apply_to_the_name_of_the_branch(config, client, tmp_path, capsys):
    rule = {"name": "source", "terms": ["Claire"], "on": ["input"], "replacement": "[SOURCE]"}
    store = SessionStore(tmp_path)
    session = Session(config, Agent(config, client, guardrails=Guardrails(parse_rules([rule]))), store=store)
    chat_turn(session, "First question")
    handle_command(session, "/edit 1 Who is Claire?")
    handle_command(session, "/branches")
    assert "Who is [SOURCE]?" in capsys.readouterr().out
    assert "Claire" not in (tmp_path / f"{session.session_id}.json").read_text(encoding="utf-8")


def test_a_failed_edit_makes_no_branch(session, client, monkeypatch, capsys):
    def failing(*args, **kwargs):
        raise LLMError("Lost connection")
        yield

    monkeypatch.setattr(client, "chat_stream", failing)
    handle_command(session, "/edit 2 Another question")
    assert session.agent.memory.branch.number == 1 and len(questions(session)) == 3
    assert "Lost connection" in capsys.readouterr().err


def test_branches_lists_them(session, capsys):
    handle_command(session, "/edit 2 Another question")
    capsys.readouterr()
    handle_command(session, "/branches")
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split() == ["1", "3", "turns", "original"]
    assert lines[1].split() == ["*", "2", "2", "turns", "from", "message", "2", "of", "branch", "1:", "Another", "question"]


def test_branch_shows_the_active_branch(session, capsys):
    handle_command(session, "/branch")
    assert "On branch 1." in capsys.readouterr().out
    handle_command(session, "/branch 1")
    assert "Already on branch 1." in capsys.readouterr().out


@pytest.mark.parametrize(
    ("argument", "message"), [("7", "No branch 7"), ("x", "Usage: /branch N"), ("\u00b2", "Usage: /branch N")]
)
def test_invalid_branch(session, capsys, argument, message):
    handle_command(session, f"/branch {argument}")
    assert message in capsys.readouterr().err


def test_the_model_gets_the_history_and_the_document_of_the_active_branch(session, client):
    memory = session.agent.memory
    memory.document.apply([Edit("append", "Timeline", "- in the original branch")])
    handle_command(session, "/edit 2 Another question")
    assert "in the original branch" not in str(client.chats[-1][0])
    handle_command(session, "/branch 1")
    chat_turn(session, "And then?")
    assert "in the original branch" in str(client.chats[-1][0])
    assert "Another question" not in str(client.chats[-1][0])


def test_new_starts_again_with_one_branch(session):
    handle_command(session, "/edit 2 Another question")
    handle_command(session, "/new")
    assert len(session.agent.memory.branches()) == 1 and session.agent.memory.branch.number == 1


def test_branches_are_resumed(session, config, client, capsys):
    handle_command(session, "/edit 2 Another question")
    capsys.readouterr()
    resumed = Session(config, Agent(config, client))
    load_memory(resumed, session.session_id)
    assert "On branch 2 of 2" in capsys.readouterr().out
    assert questions(resumed) == ["First question", "Another question"]


def test_switching_branches_is_saved(session, config):
    handle_command(session, "/edit 2 Another question")
    handle_command(session, "/branch 1")
    assert SessionStore(config.sessions_dir).list()[0].turns == 3


@pytest.mark.parametrize("command", ["/edit 1 Text", "/branches", "/branch 1"])
def test_the_commands_need_the_feature(session, capsys, command):
    session.config.features["branching"] = False
    handle_command(session, command)
    assert "The branching feature is off" in capsys.readouterr().out
    assert len(session.agent.memory.branches()) == 1
