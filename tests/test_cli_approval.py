import pytest

from sensai.core.agent import Agent
from sensai.core.approval import Approval
from sensai.core.llm.ollama import ToolCall
from sensai.core.tools.registry import ToolRegistry, requires_approval
from sensai.interfaces.cli.commands.settings import approval_command
from sensai.interfaces.cli.session import Session, ask_approval

CALL = ToolCall("web_search", {"query": "Orion Delta Consulting"})
REASON = "the query is sent to an external search engine"


@pytest.fixture
def session(config, client):
    return Session(config, Agent(config, client))


def answer(monkeypatch, *answers):
    replies = iter(answers)
    questions = []

    def fake_input(prompt=""):
        questions.append(prompt)
        reply = next(replies)
        if reply is EOFError:
            raise EOFError
        return reply

    monkeypatch.setattr("builtins.input", fake_input)
    return questions


@pytest.mark.parametrize("reply", ["y", "YES", " oui "])
def test_yes_allows_the_call(session, monkeypatch, reply):
    questions = answer(monkeypatch, reply)
    assert ask_approval(session, CALL, REASON) == Approval(True)
    assert questions == ["Allow web_search? The query is sent to an external search engine. "
                         "[y]es / [n]o / [a]lways this session: "]


def test_no_refuses_with_the_reason_given_for_the_model(session, monkeypatch):
    answer(monkeypatch, "n", "keep the company name secret")
    assert ask_approval(session, CALL, REASON) == Approval(False, "keep the company name secret")


def test_the_reason_of_a_refusal_is_optional(session, monkeypatch):
    answer(monkeypatch, "no", "")
    assert ask_approval(session, CALL, REASON) == Approval(False, "")


def test_always_allows_the_tool_until_the_end_of_the_session(session, monkeypatch, capsys):
    questions = answer(monkeypatch, "a", "n", "")
    assert ask_approval(session, CALL, REASON) == Approval(True)
    assert ask_approval(session, CALL, REASON) == Approval(True)
    assert "[web_search] allowed for this session" in capsys.readouterr().out
    assert ask_approval(session, ToolCall("run_python", {"code": "1"}), "the code runs") == Approval(False, "")
    assert len(questions) == 3


def test_an_unclear_answer_asks_again(session, monkeypatch):
    questions = answer(monkeypatch, "", "maybe", "y")
    assert ask_approval(session, CALL, REASON) == Approval(True)
    assert len(questions) == 3


def test_no_answer_refuses(session, monkeypatch):
    answer(monkeypatch, EOFError)
    assert ask_approval(session, CALL, REASON) == Approval(False, "no answer")


def test_approval_command_shows_the_mode(session, capsys):
    approval_command(session, "")
    assert "Approval: critical" in capsys.readouterr().out


def test_approval_command_changes_the_mode_and_forgets_the_allowed_tools(session, capsys):
    session.always_allowed.add("web_search")
    approval_command(session, "all")
    assert session.config.approval == "all"
    assert session.always_allowed == set()
    assert "Approval: all, Sensai asks before every tool call." in capsys.readouterr().out


def test_approval_command_rejects_unknown_modes(session, capsys):
    approval_command(session, "sometimes")
    assert session.config.approval == "critical"
    assert "Usage: /approval [critical|all|off]" in capsys.readouterr().err


def test_the_cli_approver_is_used_by_the_agent(session, client, monkeypatch):
    ran = []

    @requires_approval(REASON)
    def web_search(query: str) -> str:
        """
        Search the web.

        Args:
          query: Keywords

        Returns:
          The results
        """
        ran.append(query)
        return "results"

    session.agent.tools = ToolRegistry([web_search])
    session.agent.approver = lambda call, reason: ask_approval(session, call, reason)
    client.tool_rounds = [[CALL]]
    answer(monkeypatch, "n", "not this name")

    list(session.agent.run("Who owns Orion?"))

    assert ran == []
    assert "not this name" in client.chats[1][0][-1]["content"]
