import json

import pytest

from conftest import FakeClient
from sensai.core.agent import INTERRUPTED, Agent
from sensai.core.config import DEFAULT_PERSONAS_DIR
from sensai.core.events import TokenEvent
from sensai.core.guardrails.rules import Guardrails, parse_rules
from sensai.core.llm.ollama import LLMError, ToolCall
from sensai.core.memory.sessions import SessionStore
from sensai.core.orchestrator import Review
from sensai.core.persona import load_personas
from sensai.core.rag.models import SearchResult
from sensai.core.structured import ANSWER_SCHEMA
from sensai.core.tools.registry import ToolRegistry
from sensai.interfaces.cli.main import Session, chat_turn, handle_command, json_command


# Long enough for the guardrails filter, which holds back the last 80 characters, to show the beginning.
LONG_ANSWER = " ".join(f"word{number}" for number in range(60))
PERSONAS = load_personas(DEFAULT_PERSONAS_DIR)
APPROVED = '{"claims": [{"claim": "Orion paid Delta", "verdict": "supported", "reason": "the mail says so"}]}'


class InterruptedClient(FakeClient):
    """The user presses Ctrl+C after `words` words of the next answer."""

    def __init__(self, words: int):
        super().__init__(LONG_ANSWER)
        self.words = words

    def chat_stream(self, *args, **kwargs):
        stream = super().chat_stream(*args, **kwargs)
        for number, chunk in enumerate(stream):
            if self.words is not None and number == self.words:
                self.words = None
                raise KeyboardInterrupt
            yield chunk


class SecondRoundClient(FakeClient):
    """Writes a text and calls a tool, then the user presses Ctrl+C in the middle of the next answer."""

    def __init__(self):
        super().__init__()
        self.rounds = 0

    def chat_stream(self, *args, tool_calls=None, **kwargs):
        self.rounds += 1
        if self.rounds == 1:
            tool_calls.append(ToolCall("lookup", {"query": "Alvor"}))
            yield "Let me check."
        elif self.rounds == 2:
            yield "second " * 30
            raise KeyboardInterrupt
        else:
            yield from super().chat_stream(*args, tool_calls=tool_calls, **kwargs)


class ReviewClient(FakeClient):
    """Each chat gets the next answer. The user presses Ctrl+C in the middle of the last one."""

    def __init__(self, answers: list[str]):
        super().__init__()
        self.answers = list(answers)

    def chat_stream(self, *args, **kwargs):
        self.answer = self.answers.pop(0)
        yield from super().chat_stream(*args, **kwargs)
        if not self.answers:
            raise KeyboardInterrupt


class FailingClient(FakeClient):
    def chat_stream(self, *args, **kwargs):
        yield from super().chat_stream(*args, **kwargs)
        raise LLMError("Lost connection")


def lookup(query: str) -> str:
    """
    Look something up.

    Args:
      query: What to look up
    """
    return f"{query} is registered in Lisbon."


class Excerpts:
    """The first search finds the statutes, the reasoning finds the statement."""

    STATUTES = SearchResult(chunk_id=1, text="Orion statutes", location="p. 1", source="statutes.pdf", description="", score=0.9)
    STATEMENT = SearchResult(chunk_id=2, text="Vidal statement", location="p. 2", source="statement.pdf", description="", score=0.9)

    def search(self, query: str, k: int) -> list[SearchResult]:
        return [self.STATEMENT if "Vidal" in query else self.STATUTES]


def step(thought: str, action: str = "answer", **arguments) -> str:
    return json.dumps({"thought": thought, "action": action, "arguments": arguments})


def run_until_interrupted(agent: Agent, question: str, schema: dict | None = None) -> list:
    events = []
    with pytest.raises(KeyboardInterrupt):
        for event in agent.run(question, schema):
            events.append(event)
    return events


def shown(events: list) -> str:
    return "".join(event.text for event in events if isinstance(event, TokenEvent))


def test_the_interrupted_answer_stays_in_the_history(config):
    agent = Agent(config, InterruptedClient(30))
    events = run_until_interrupted(agent, "Who approved?")
    assert agent.interrupt()
    assert shown(events).startswith("word0 word1 ")
    assert agent.memory.turns[0].user == "Who approved?"
    assert agent.memory.turns[0].assistant == f"{shown(events)}\n{INTERRUPTED}"


def test_the_model_gets_exactly_what_the_user_saw(config):
    agent = Agent(config, InterruptedClient(3))
    events = run_until_interrupted(agent, "Who approved?")
    agent.interrupt()
    assert shown(events) == "" and agent.memory.turns[0].assistant == INTERRUPTED


def test_the_next_message_redirects_the_answer(config):
    client = InterruptedClient(30)
    agent = Agent(config, client)
    events = run_until_interrupted(agent, "Summarize the leak")
    agent.interrupt()
    list(agent.run("Shorter, and only the amounts"))
    assert client.chats[-1][0] == [
        {"role": "user", "content": "Summarize the leak"},
        {"role": "assistant", "content": f"{shown(events)}\n{INTERRUPTED}"},
        {"role": "user", "content": "Shorter, and only the amounts"},
    ]


def test_an_answer_interrupted_before_its_first_word_keeps_the_question(config):
    agent = Agent(config, InterruptedClient(0))
    run_until_interrupted(agent, "Who approved?")
    agent.interrupt()
    assert agent.memory.turns[0].assistant == INTERRUPTED


def test_only_what_the_guardrails_let_through_is_kept(config):
    client = InterruptedClient(30)
    client.answer = "It was Claire Benoit who talked " + LONG_ANSWER
    rule = {"name": "source", "terms": ["Claire Benoit"], "on": ["output"], "replacement": "[SOURCE]"}
    agent = Agent(config, client, guardrails=Guardrails(parse_rules([rule])))
    events = run_until_interrupted(agent, "Who talked?")
    agent.interrupt()
    assert agent.memory.turns[0].assistant == f"{shown(events)}\n{INTERRUPTED}"
    assert "[SOURCE]" in shown(events) and "Claire" not in agent.memory.turns[0].assistant


def test_the_turn_keeps_what_came_before_the_tool_call(config):
    agent = Agent(config, SecondRoundClient(), tools=ToolRegistry([lookup]))
    run_until_interrupted(agent, "Where is Alvor?")
    agent.interrupt()
    kept = agent.memory.turns[0].assistant
    assert kept.startswith("Let me check.\n[lookup] Alvor is registered in Lisbon.\n")
    assert "second second" in kept and kept.endswith(INTERRUPTED)


def test_the_next_message_can_use_the_tool_result(config):
    client = SecondRoundClient()
    agent = Agent(config, client, tools=ToolRegistry([lookup]))
    run_until_interrupted(agent, "Where is Alvor?")
    agent.interrupt()
    list(agent.run("Go on"))
    assert "Alvor is registered in Lisbon." in client.chats[-1][0][-2]["content"]


def test_a_kept_tool_result_goes_through_the_guardrails(config):
    rule = {"name": "city", "terms": ["Lisbon"], "on": ["output"], "replacement": "[CITY]"}
    guardrails = Guardrails(parse_rules([rule]))
    agent = Agent(config, SecondRoundClient(), tools=ToolRegistry([lookup]), guardrails=guardrails)
    run_until_interrupted(agent, "Where is Alvor?")
    agent.interrupt()
    kept = agent.memory.turns[0].assistant
    assert "Alvor is registered in [CITY]." in kept and "Lisbon" not in kept


def test_a_tool_result_a_guardrail_blocks_is_not_kept(config):
    rule = {"name": "city", "terms": ["Lisbon"], "on": ["output"], "action": "block"}
    guardrails = Guardrails(parse_rules([rule]))
    agent = Agent(config, SecondRoundClient(), tools=ToolRegistry([lookup]), guardrails=guardrails)
    run_until_interrupted(agent, "Where is Alvor?")
    agent.interrupt()
    kept = agent.memory.turns[0].assistant
    assert "[lookup] result withheld by a guardrail" in kept and "registered" not in kept


def test_the_tool_results_of_the_reasoning_are_kept(config):
    config.features["reasoning"] = True
    client = InterruptedClient(30)
    client.steps = [step("I need the address.", "lookup", query="Alvor"), step("I have it.")]
    agent = Agent(config, client, tools=ToolRegistry([lookup]))
    run_until_interrupted(agent, "Where is Alvor?")
    agent.interrupt()
    kept = agent.memory.turns[0].assistant
    assert "[lookup] Alvor is registered in Lisbon." in kept and "word0 word1 " in kept and kept.endswith(INTERRUPTED)


def test_ctrl_c_between_two_steps_of_the_reasoning_keeps_what_was_found(config):
    config.features["reasoning"] = True
    client = FakeClient()
    client.steps = [step("I need the address.", "lookup", query="Alvor")]
    original = client.chat

    def chat(*args, **kwargs):
        if not client.steps:
            raise KeyboardInterrupt
        return original(*args, **kwargs)

    client.chat = chat
    agent = Agent(config, client, tools=ToolRegistry([lookup]))
    run_until_interrupted(agent, "Where is Alvor?")
    agent.interrupt()
    assert agent.memory.turns[0].assistant.startswith("[lookup] Alvor is registered in Lisbon.")


def test_an_interrupted_turn_cites_the_excerpts_of_the_reasoning(config):
    config.features["reasoning"] = True
    client = InterruptedClient(30)
    client.steps = [step("Who is he?", "search_documents", query="Vidal"), step("I have it.")]
    agent = Agent(config, client, retriever=Excerpts())
    run_until_interrupted(agent, "Who manages Orion?")
    agent.interrupt()
    assert agent.memory.turns[0].citations == [Excerpts.STATUTES.citation, Excerpts.STATEMENT.citation]


def test_a_failed_answer_leaves_nothing_to_interrupt(config):
    agent = Agent(config, FailingClient())
    with pytest.raises(LLMError):
        list(agent.run("Who approved?"))
    assert agent.answering is None and agent.shown == []
    assert not agent.interrupt() and agent.memory.turns == []


def test_an_interrupted_answer_does_not_update_the_document(config):
    client = InterruptedClient(1)
    client.edits = '{"edits": [{"op": "append", "section": "Summary", "text": "- half a fact"}]}'
    agent = Agent(config, client)
    run_until_interrupted(agent, "Who approved?")
    agent.interrupt()
    assert client.edit_requests == [] and agent.memory.document.is_empty


def test_a_json_answer_cut_halfway_is_not_kept(config):
    agent = Agent(config, InterruptedClient(1))
    run_until_interrupted(agent, "Who approved?", ANSWER_SCHEMA)
    assert not agent.interrupt() and agent.memory.turns == []


def test_nothing_to_keep_after_a_finished_answer(config, client):
    agent = Agent(config, client)
    list(agent.run("Who approved?"))
    assert not agent.interrupt()
    assert len(agent.memory.turns) == 1


def test_the_cli_keeps_the_answer_and_says_how_to_go_on(config, capsys):
    session = Session(config, Agent(config, InterruptedClient(30)))
    chat_turn(session, "Who approved?")
    out = capsys.readouterr().out
    assert "[interrupted] What was written is kept" in out
    assert session.agent.memory.turns[0].assistant.startswith("word0 word1 ")
    assert session.agent.memory.turns[0].assistant.endswith(INTERRUPTED)


def test_ctrl_c_while_the_cli_prints_keeps_the_answer_too(config, monkeypatch):
    session = Session(config, Agent(config, FakeClient(LONG_ANSWER)))
    printed = []

    def interrupted_print(*args, **kwargs):
        printed.append(args)
        if len(printed) == 2:
            raise KeyboardInterrupt

    monkeypatch.setattr("builtins.print", interrupted_print)
    chat_turn(session, "Who approved?")
    monkeypatch.undo()
    assert session.agent.memory.turns[0].assistant == f"{printed[0][0]}{printed[1][0]}\n{INTERRUPTED}"


def test_with_the_feature_off_an_interrupted_turn_is_dropped(config, capsys):
    config.features["branching"] = False
    session = Session(config, Agent(config, InterruptedClient(1)))
    chat_turn(session, "Who approved?")
    assert session.agent.memory.turns == []
    assert "kept" not in capsys.readouterr().out


def test_the_interrupted_turn_is_saved(config, tmp_path):
    store = SessionStore(tmp_path / "saved")
    session = Session(config, Agent(config, InterruptedClient(1)), store=store)
    chat_turn(session, "Who approved?")
    assert store.list()[0].turns == 1


def test_an_interrupted_review_keeps_what_the_investigator_wrote(config, capsys):
    session = Session(config, Agent(config, InterruptedClient(30)), personas=PERSONAS)
    handle_command(session, "/review Who approved?")
    turn = session.agent.memory.turns[0]
    assert turn.user == "Who approved?"
    assert turn.assistant.startswith("word0 word1 ") and turn.assistant.endswith(INTERRUPTED)
    assert "[interrupted] What was written is kept" in capsys.readouterr().out


def test_a_review_interrupted_during_the_check_keeps_the_answer_of_the_investigator(config):
    client = ReviewClient(["Orion paid Delta.", APPROVED])
    session = Session(config, Agent(config, client), personas=PERSONAS)
    handle_command(session, "/review Who paid?")
    assert session.agent.memory.turns[0].assistant == f"Orion paid Delta. \n{INTERRUPTED}"


def test_a_review_interrupted_during_the_edit_keeps_the_text_of_the_editor(config):
    client = ReviewClient(["Orion paid Delta.", APPROVED, LONG_ANSWER])
    session = Session(config, Agent(config, client), personas=PERSONAS)
    handle_command(session, "/review Who paid?")
    assistant = session.agent.memory.turns[0].assistant
    assert assistant.startswith("word0 word1 ") and assistant.endswith(INTERRUPTED)
    assert len(session.agent.memory.turns) == 1 and client.edit_requests == []


def test_an_interrupted_review_keeps_the_question_as_the_guardrails_left_it(config):
    rule = {"name": "source", "terms": ["Claire"], "on": ["input"], "replacement": "[SOURCE]"}
    agent = Agent(config, InterruptedClient(30), guardrails=Guardrails(parse_rules([rule])))
    session = Session(config, agent, personas=PERSONAS)
    handle_command(session, "/review Who is Claire?")
    assert agent.memory.turns[0].user == "Who is [SOURCE]?"


def test_a_finished_review_leaves_nothing_to_interrupt(config):
    agent = Agent(config, FakeClient(APPROVED))
    review = Review(agent, PERSONAS)
    list(review.run("Who paid?"))
    assert not review.interrupt() and len(agent.memory.turns) == 1


def test_with_the_feature_off_an_interrupted_review_keeps_nothing(config, capsys):
    config.features["branching"] = False
    session = Session(config, Agent(config, InterruptedClient(30)), personas=PERSONAS)
    handle_command(session, "/review Who approved?")
    assert session.agent.memory.turns == []
    assert "kept" not in capsys.readouterr().out


def test_a_failed_turn_is_not_kept_by_a_review_interrupted_later(config):
    client = InterruptedClient(30)
    session = Session(config, Agent(config, client), personas=PERSONAS)
    session.agent.client = FailingClient()
    chat_turn(session, "First question")
    session.agent.client = client
    handle_command(session, "/review Second question")
    assert [turn.user for turn in session.agent.memory.turns] == ["Second question"]


def test_an_interrupted_json_command_keeps_nothing(config, capsys):
    session = Session(config, Agent(config, InterruptedClient(1)))
    json_command(session, "Who approved?")
    assert session.agent.memory.turns == []
    assert "[interrupted]\n" in capsys.readouterr().out
