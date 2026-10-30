import pytest

from conftest import FakeClient
from sensai.core.agent import Agent
from sensai.core.config import DEFAULT_PERSONAS_DIR
from sensai.core.events import AgentStartedEvent, DoneEvent, ReviewDoneEvent
from sensai.core.guardrails.rules import Guardrails, parse_rules
from sensai.core.llm.ollama import ToolCall
from sensai.core.orchestrator import NO_EVIDENCE, Review, ReviewError, approved
from sensai.core.persona import load_personas
from sensai.core.rag.models import SearchResult
from sensai.core.tools.registry import ToolRegistry
from sensai.interfaces.cli.main import Session, handle_command

PERSONAS = load_personas(DEFAULT_PERSONAS_DIR)
RESULT = SearchResult(
    chunk_id=1, text="Alice approved the transfer.", location="p. 2", source="leak/mail.eml",
    description="From: alice@corp.example", score=0.9,
)


class FakeRetriever:
    def __init__(self, results: list[SearchResult]):
        self.results = results

    def search(self, query: str, k: int) -> list[SearchResult]:
        return self.results


class ScriptedClient(FakeClient):
    """Answers each chat that does not call a tool with the next answer of the script."""

    def __init__(self, answers: list[str]):
        super().__init__()
        self.answers = list(answers)

    def chat_stream(self, *args, **kwargs):
        if not (kwargs.get("tools") and self.tool_rounds):
            self.answer = self.answers.pop(0)
        return super().chat_stream(*args, **kwargs)


def lookup(query: str) -> str:
    """
    Look something up.

    Args:
      query: What to look up
    """
    return query


def speakers(events) -> list[str]:
    return [event.persona.id for event in events if isinstance(event, AgentStartedEvent)]


def review_done(events) -> ReviewDoneEvent:
    return next(event for event in events if isinstance(event, ReviewDoneEvent))


@pytest.mark.parametrize(
    ("verdict", "expected"),
    [
        ("Claim 1: supported.\nVERDICT: APPROVED", True),
        ("Claim 1: supported.\n**Verdict: approved.**\n", True),
        ("Claim 1: supported.\n**Verdict:** APPROVED", True),
        ("Claim 1: supported.\nVERDICT:  `APPROVED`", True),
        ("Claim 1: unsupported.\n**Verdict:** REJECTED", False),
        ("Claim 1: unsupported.\nVERDICT: REJECTED", False),
        ("VERDICT: APPROVED\nClaim 1: unsupported.", False),
        ("Looks good to me.", False),
        ("", False),
    ],
)
def test_only_an_explicit_approval_counts(verdict, expected):
    assert approved(verdict) is expected


def test_approved_answer_goes_to_the_editor(config):
    client = ScriptedClient(["Orion paid Delta [1].", "Claim 1: supported.\nVERDICT: APPROVED", "Orion paid Delta."])
    agent = Agent(config, client)
    events = list(Review(agent, PERSONAS).run("Who paid Delta?"))

    assert speakers(events) == ["investigator", "fact_checker", "editor"]
    assert review_done(events) == ReviewDoneEvent(True, "Orion paid Delta. ", 1)
    check = client.chats[1][0][-1]["content"]
    assert "Question: Who paid Delta?" in check and "Orion paid Delta [1]." in check
    edit = client.chats[2][0][-1]["content"]
    assert "Orion paid Delta [1]." in edit and "VERDICT" not in edit


def test_rejected_answer_is_corrected_then_checked_again(config):
    client = ScriptedClient([
        "Orion paid Delta 5M.", "Claim 1: contradicted, the transfer is 2M.\nVERDICT: REJECTED",
        "Orion paid Delta 2M.", "VERDICT: APPROVED", "Orion paid Delta 2M.",
    ])
    events = list(Review(Agent(config, client), PERSONAS).run("How much?"))

    assert speakers(events) == ["investigator", "fact_checker", "investigator", "fact_checker", "editor"]
    assert review_done(events).approved and review_done(events).rounds == 2
    fix = client.chats[2][0]
    assert "the transfer is 2M" in fix[-1]["content"]
    # The investigator keeps its own first answer, and corrects it.
    assert {"role": "assistant", "content": "Orion paid Delta 5M. "} in fix


def test_review_stops_without_editor_after_the_last_round(config):
    config.max_review_rounds = 2
    client = ScriptedClient(["Answer 1", "VERDICT: REJECTED", "Answer 2", "Not sure."])
    events = list(Review(Agent(config, client), PERSONAS).run("Who?"))

    assert speakers(events) == ["investigator", "fact_checker", "investigator", "fact_checker"]
    assert review_done(events) == ReviewDoneEvent(False, "Answer 2 ", 2)
    assert client.answers == []


def test_fact_checker_only_sees_the_question_and_the_answer(config):
    client = ScriptedClient(["Earlier answer", "The answer", "VERDICT: APPROVED", "The text"])
    agent = Agent(config, client)
    list(agent.run("Earlier question"))
    list(Review(agent, PERSONAS).run("Who?"))

    investigator, checker = client.chats[1][0], client.chats[2][0]
    assert {"role": "user", "content": "Earlier question"} in investigator
    assert [message["role"] for message in checker] == ["system", "user"]
    assert checker[0]["content"] == PERSONAS["fact_checker"].system_prompt()


def test_fact_checker_sees_what_the_investigator_found(config):
    client = ScriptedClient(["Alice approved it [1].", "VERDICT: APPROVED", "The text"])
    client.tool_rounds = [[ToolCall("lookup", {"query": "Delta paid 2M"})]]
    retriever = FakeRetriever([RESULT, RESULT])
    agent = Agent(config, client, retriever=retriever, tools=ToolRegistry([lookup]))
    list(Review(agent, PERSONAS).run("Who approved it?"))

    check = client.chats[2][0][-1]["content"]
    assert '[1] leak/mail.eml, p. 2\nFrom: alice@corp.example\n\nAlice approved the transfer.' in check
    assert check.count("Alice approved the transfer.") == 1
    assert "[lookup] Delta paid 2M" in check


def test_fact_checker_is_told_when_nothing_was_found(config):
    client = ScriptedClient(["Delta was paid 5M [1].", "VERDICT: REJECTED", "Unknown", "VERDICT: REJECTED"])
    list(Review(Agent(config, client), PERSONAS).run("Who paid Delta?"))

    for chat in (client.chats[1], client.chats[3]):
        assert NO_EVIDENCE in chat[0][-1]["content"]


def test_only_the_question_and_the_final_text_are_kept(config):
    client = ScriptedClient(["The answer", "VERDICT: APPROVED", "The text"])
    agent = Agent(config, client)
    list(Review(agent, PERSONAS).run("Who?"))
    assert agent.messages == [{"role": "user", "content": "Who?"}, {"role": "assistant", "content": "The text "}]


def test_only_the_investigator_can_call_tools(config):
    client = ScriptedClient(["Found it", "VERDICT: APPROVED", "The text"])
    client.tool_rounds = [[ToolCall("lookup", {"query": "Delta"})]]
    agent = Agent(config, client, tools=ToolRegistry([lookup]))
    list(Review(agent, PERSONAS).run("Who?"))

    assert [bool(tools) for tools in client.tools_seen] == [True, True, False, False]


def test_blocked_question_stops_the_review(config):
    client = ScriptedClient([])
    guardrails = Guardrails(parse_rules([{"name": "source", "terms": ["Marie"], "on": ["input"], "action": "block"}]))
    agent = Agent(config, client, guardrails=guardrails)
    events = list(Review(agent, PERSONAS).run("Where does Marie live?"))

    assert speakers(events) == ["investigator"]
    assert not any(isinstance(event, (DoneEvent, ReviewDoneEvent)) for event in events)
    assert agent.messages == []


def test_missing_personas_are_named(config, client):
    with pytest.raises(ReviewError, match="fact_checker, editor"):
        Review(Agent(config, client), {"investigator": PERSONAS["investigator"]})


@pytest.fixture
def session(config):
    client = ScriptedClient(["The answer", "VERDICT: APPROVED", "The text"])
    return Session(config, Agent(config, client), personas=PERSONAS)


def test_review_command_shows_each_agent(session, capsys):
    handle_command(session, "/review Who paid Delta?")
    out = capsys.readouterr().out
    assert out.index("── investigator") < out.index("── fact_checker") < out.index("── editor")
    assert "[review] approved by the fact-checker after 1 check" in out


def test_review_command_needs_a_question(session, capsys):
    handle_command(session, "/review")
    assert "Usage: /review QUESTION" in capsys.readouterr().err


def test_review_command_is_refused_when_the_feature_is_off(session, capsys):
    handle_command(session, "/feature orchestration off")
    handle_command(session, "/review Who?")
    assert "/feature orchestration on" in capsys.readouterr().out
    assert session.agent.messages == []


def test_review_command_names_the_missing_personas(session, capsys):
    session.personas = {}
    handle_command(session, "/review Who?")
    assert "investigator, fact_checker, editor" in capsys.readouterr().err
