import json

import pytest

from conftest import FakeClient
from sensai.core.agent import Agent
from sensai.core.config import DEFAULT_PERSONAS_DIR
from sensai.core.document.document import Edit
from sensai.core.events import AgentStartedEvent, DocumentEvent, DoneEvent, ReviewDoneEvent
from sensai.core.guardrails.rules import Guardrails, parse_rules
from sensai.core.llm.ollama import ToolCall
from sensai.core.orchestrator import NO_EVIDENCE, VERDICT_SCHEMA, Review, ReviewError, approved, remarks
from sensai.core.persona import load_personas
from sensai.core.rag.models import SearchResult
from sensai.core.tools.registry import ToolRegistry
from sensai.interfaces.cli.main import Session, handle_command

PERSONAS = load_personas(DEFAULT_PERSONAS_DIR)
APPROVED = '{"claims": [{"claim": "Orion paid Delta", "verdict": "supported", "reason": "the mail says so"}]}'
REJECTED = '{"claims": [{"claim": "Orion paid Delta 5M", "verdict": "contradicted", "reason": "the transfer is 2M"}]}'
RESULT = SearchResult(
    chunk_id=1, text="Alice approved the transfer.", location="p. 2", source="leak/mail.eml",
    description="From: alice@corp.example", score=0.9,
)


class FakeRetriever:
    def __init__(self, results: list[SearchResult]):
        self.results = results

    def search(self, query: str, k: int, filters=(), report=None) -> list[SearchResult]:
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


def claims(*verdicts: str) -> str:
    return json.dumps({"claims": [{"claim": f"Claim {i}", "verdict": v, "reason": "why"} for i, v in enumerate(verdicts)]})


@pytest.mark.parametrize(
    ("verdict", "expected"),
    [
        (claims("supported"), True),
        (claims("supported", "common knowledge"), True),
        (claims("supported", "unsupported"), False),
        (claims("contradicted"), False),
        (claims(), False),
        (claims("probably fine"), False),
        ('{"claims": [{"verdict": "supported"}]}', False),
        ('{"verdict": "APPROVED"}', False),
        ("Claim 1: supported.\nVERDICT: APPROVED", False),
        ("", False),
    ],
)
def test_only_accepted_claims_approve(verdict, expected):
    assert approved(verdict) is expected


def test_remarks_list_only_the_claims_to_correct():
    text = remarks(claims("supported", "contradicted"))
    assert text == "- Claim 1: contradicted, why"
    assert "could not check" in remarks("not json")


def test_approved_answer_goes_to_the_editor(config):
    client = ScriptedClient(["Orion paid Delta [1].", APPROVED, "Orion paid Delta."])
    agent = Agent(config, client)
    events = list(Review(agent, PERSONAS).run("Who paid Delta?"))

    assert speakers(events) == ["investigator", "fact_checker", "editor"]
    assert review_done(events) == ReviewDoneEvent(True, "Orion paid Delta. ", 1)
    check = client.chats[1][0][-1]["content"]
    assert "Question: Who paid Delta?" in check and "Orion paid Delta [1]." in check
    edit = client.chats[2][0][-1]["content"]
    assert "Orion paid Delta [1]." in edit and "claims" not in edit


def test_only_the_fact_checker_answers_with_the_schema(config):
    client = ScriptedClient(["Orion paid Delta [1].", APPROVED, "Orion paid Delta."])
    list(Review(Agent(config, client), PERSONAS).run("Who paid Delta?"))
    assert client.formats_seen == [None, VERDICT_SCHEMA, None]


def test_rejected_answer_is_corrected_then_checked_again(config):
    client = ScriptedClient([
        "Orion paid Delta 5M.", REJECTED,
        "Orion paid Delta 2M.", APPROVED, "Orion paid Delta 2M.",
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
    client = ScriptedClient(["Answer 1", REJECTED, "Answer 2", "Not sure."])
    events = list(Review(Agent(config, client), PERSONAS).run("Who?"))

    assert speakers(events) == ["investigator", "fact_checker", "investigator", "fact_checker"]
    assert review_done(events) == ReviewDoneEvent(False, "Answer 2 ", 2)
    assert client.answers == []


def test_fact_checker_only_sees_the_question_and_the_answer(config):
    client = ScriptedClient(["Earlier answer", "The answer", APPROVED, "The text"])
    agent = Agent(config, client)
    list(agent.run("Earlier question"))
    list(Review(agent, PERSONAS).run("Who?"))

    investigator, checker = client.chats[1][0], client.chats[2][0]
    assert {"role": "user", "content": "Earlier question"} in investigator
    assert [message["role"] for message in checker] == ["system", "user"]
    assert checker[0]["content"] == PERSONAS["fact_checker"].system_prompt()


def test_fact_checker_sees_what_the_investigator_found(config):
    client = ScriptedClient(["Alice approved it [1].", APPROVED, "The text"])
    client.tool_rounds = [[ToolCall("lookup", {"query": "Delta paid 2M"})]]
    retriever = FakeRetriever([RESULT, RESULT])
    agent = Agent(config, client, retriever=retriever, tools=ToolRegistry([lookup]))
    list(Review(agent, PERSONAS).run("Who approved it?"))

    check = client.chats[2][0][-1]["content"]
    assert '[1] leak/mail.eml, p. 2\nFrom: alice@corp.example\n\nAlice approved the transfer.' in check
    assert check.count("Alice approved the transfer.") == 1
    assert "[lookup] Delta paid 2M" in check


def test_fact_checker_is_told_when_nothing_was_found(config):
    client = ScriptedClient(["Delta was paid 5M [1].", REJECTED, "Unknown", REJECTED])
    list(Review(Agent(config, client), PERSONAS).run("Who paid Delta?"))

    for chat in (client.chats[1], client.chats[3]):
        assert NO_EVIDENCE in chat[0][-1]["content"]


def test_only_the_claims_about_the_case_need_a_source(config):
    client = ScriptedClient(["Paris is the capital of France.", APPROVED, "The text"])
    list(Review(Agent(config, client), PERSONAS).run("Where is Orion based?"))

    check = client.chats[1][0][-1]["content"]
    assert "A claim about the case (a person, an organization, an amount, a date, an event, an accusation)" in check
    assert "common knowledge" in check and "If in doubt, treat it as a claim about the case." in check


def test_only_the_question_and_the_final_text_are_kept(config):
    client = ScriptedClient(["The answer", APPROVED, "The text"])
    agent = Agent(config, client)
    list(Review(agent, PERSONAS).run("Who?"))
    assert agent.messages == [{"role": "user", "content": "Who?"}, {"role": "assistant", "content": "The text "}]


APPEND = '{"edits": [{"op": "append", "section": "Summary", "text": "- Orion paid Delta"}]}'


def test_only_the_approved_text_goes_to_the_document(config):
    client = ScriptedClient(["Orion paid Delta [1].", APPROVED, "Orion paid Delta."])
    client.edits = APPEND
    agent = Agent(config, client)
    events = list(Review(agent, PERSONAS).run("Who paid Delta?"))

    assert len(client.edit_requests) == 1 and "Orion paid Delta. " in str(client.edit_requests[0])
    assert [type(event) for event in events[-2:]] == [ReviewDoneEvent, DocumentEvent]
    assert agent.memory.document.section("Summary").body == "- Orion paid Delta"


def test_a_rejected_text_does_not_go_to_the_document(config):
    config.max_review_rounds = 1
    client = ScriptedClient(["Orion paid Delta 5M.", REJECTED])
    client.edits = APPEND
    agent = Agent(config, client)
    list(Review(agent, PERSONAS).run("How much?"))
    assert client.edit_requests == [] and agent.memory.document.is_empty


def test_the_investigator_reads_the_working_document(config):
    client = ScriptedClient(["The answer", APPROVED, "The text"])
    agent = Agent(config, client)
    agent.memory.document.apply([Edit("append", "Timeline", "- 4 April: the transfer")])
    list(Review(agent, PERSONAS).run("When?"))
    assert "- 4 April: the transfer" in client.chats[0][0][0]["content"]
    assert "- 4 April: the transfer" not in str(client.chats[1][0])


def test_only_the_investigator_can_call_tools(config):
    client = ScriptedClient(["Found it", APPROVED, "The text"])
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
    client = ScriptedClient(["The answer", APPROVED, "The text"])
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
