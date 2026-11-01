import json

import pytest

from conftest import FakeClient
from sensai.core.agent import KEPT_CHARS, WITHHELD, Agent
from sensai.core.approval import Approval
from sensai.core.config import DEFAULT_PERSONAS_DIR
from sensai.core.guardrails.rules import Guardrails, parse_rules
from sensai.core.llm.ollama import LLMError, ToolCall
from sensai.core.memory.conversation import ConversationMemory, ToolUse, Turn
from sensai.core.memory.sessions import SessionError, SessionStore
from sensai.core.orchestrator import Review
from sensai.core.persona import load_personas
from sensai.core.tools.registry import ToolRegistry, ToolResult, requires_approval

PERSONAS = load_personas(DEFAULT_PERSONAS_DIR)
APPROVED = '{"claims": [{"claim": "Orion paid Delta", "verdict": "supported", "reason": "the mail says so"}]}'
LONG = "word " * 200


def write_note(path: str, content: str) -> ToolResult:
    """
    Write a note.

    Args:
      path: Where to write
      content: What to write
    """
    return ToolResult(f"{path} created ({len(content)} characters).", summary=f"{path} created")


def read_note(path: str) -> str:
    """
    Read a note.

    Args:
      path: What to read
    """
    return LONG


@requires_approval("the query leaves the machine")
def send(query: str) -> str:
    """
    Send a query.

    Args:
      query: The query
    """
    return f"sent {query}"


def agent_with(config, client, **options) -> Agent:
    return Agent(config, client, tools=ToolRegistry([write_note, read_note, send]), **options)


def rules(on: str, action: str, term: str) -> Guardrails:
    return Guardrails(parse_rules([{"name": "source", "terms": [term], "on": [on], "action": action}]))


CALL = ToolCall("write_note", {"path": "drafts/note.md", "content": "Hello"})


def test_the_next_message_shows_the_tool_call_of_the_last_turn(config, client):
    agent = agent_with(config, client)
    client.tool_rounds = [[CALL]]
    list(agent.run("Save Hello"))
    list(agent.run("Save Secret"))
    assert client.chats[-1][0] == [
        {"role": "user", "content": "Save Hello"},
        {"role": "assistant", "content": "", "tool_calls": [CALL.message()]},
        {"role": "tool", "tool_name": "write_note", "content": "drafts/note.md created"},
        {"role": "assistant", "content": "Hello there "},
        {"role": "user", "content": "Save Secret"},
    ]


def test_a_turn_without_tools_stays_a_question_and_an_answer(config, client):
    agent = agent_with(config, client)
    list(agent.run("Hi"))
    assert [message["role"] for message in agent.messages] == ["user", "assistant"]


def test_every_call_of_the_turn_is_kept_in_order(config, client):
    agent = agent_with(config, client)
    client.tool_rounds = [[CALL, ToolCall("read_note", {"path": "a.md"})], [ToolCall("read_note", {"path": "b.md"})]]
    list(agent.run("Do it"))
    assert [tool.arguments.get("path") for tool in agent.memory.turns[0].tools] == ["drafts/note.md", "a.md", "b.md"]


def test_a_result_without_summary_is_kept_short(config, client):
    agent = agent_with(config, client)
    client.tool_rounds = [[ToolCall("read_note", {"path": "a.md"})]]
    list(agent.run("Read it"))
    kept = agent.memory.turns[0].tools[0].result
    assert kept.endswith("...") and len(kept) <= KEPT_CHARS + 3
    # The model has the whole result for the answer of this turn.
    assert client.chats[-1][0][-1]["content"] == LONG


def test_a_long_argument_is_kept_short(config, client):
    agent = agent_with(config, client)
    client.tool_rounds = [[ToolCall("write_note", {"path": "a.md", "content": LONG})]]
    list(agent.run("Save it"))
    assert len(agent.memory.turns[0].tools[0].arguments["content"]) <= KEPT_CHARS + 3


def test_an_argument_that_is_not_text_is_kept_as_text(config, client):
    client.tool_rounds = [[ToolCall("read_note", {"path": ["a", "b"]})]]
    agent = agent_with(config, client)
    list(agent.run("Read them"))
    assert agent.memory.turns[0].tools[0].arguments == {"path": '["a", "b"]'}


def test_a_failed_call_is_kept_with_its_error(config, client):
    agent = agent_with(config, client)
    client.tool_rounds = [[ToolCall("unknown", {"query": "x"})]]
    list(agent.run("Go"))
    assert agent.memory.turns[0].tools[0].result.startswith("Error: ")


def test_a_refused_call_is_kept_as_refused(config, client):
    agent = agent_with(config, client, approver=lambda call, reason: Approval(False))
    client.tool_rounds = [[ToolCall("send", {"query": "Orion"})]]
    list(agent.run("Send it"))
    assert agent.memory.turns[0].tools == [ToolUse("send", {"query": "Orion"}, "refused")]


def test_what_a_rule_redacts_stays_out_of_the_kept_call(config, client):
    agent = agent_with(config, client, guardrails=rules("output", "redact", "Claire Benoit"))
    client.tool_rounds = [[ToolCall("write_note", {"path": "a.md", "content": "Claire Benoit said so"})]]
    list(agent.run("Save it"))
    assert "Claire Benoit" not in json.dumps(agent.messages)


def test_a_result_a_rule_blocks_is_withheld(config, client):
    agent = agent_with(config, client, guardrails=rules("output", "block", "drafts/note.md"))
    client.tool_rounds = [[CALL]]
    list(agent.run("Save it"))
    kept = agent.memory.turns[0].tools[0]
    assert kept.result == WITHHELD and "drafts/note.md" not in json.dumps(agent.messages)


def test_a_call_blocked_by_a_rule_is_not_kept(config, client):
    agent = agent_with(config, client, guardrails=rules("tools", "block", "Claire Benoit"))
    client.tool_rounds = [[ToolCall("send", {"query": "Where is Claire Benoit?"})]]
    list(agent.run("Search it"))
    assert agent.memory.turns[0].tools == []
    assert "Claire Benoit" not in json.dumps(agent.messages)


def test_the_tools_of_the_last_turn_do_not_reach_the_next_one(config, client):
    agent = agent_with(config, client)
    client.tool_rounds = [[CALL]]
    list(agent.run("Save Hello"))
    list(agent.run("Thanks"))
    assert len(agent.memory.turns[0].tools) == 1 and agent.memory.turns[1].tools == []


def test_a_failed_turn_keeps_nothing(config, client, monkeypatch):
    agent = agent_with(config, client)
    original = client.chat_stream

    def fail_after_the_tool(*args, **kwargs):
        if any(message["role"] == "tool" for message in args[1]):
            raise LLMError("Ollama stopped")
        return original(*args, **kwargs)

    monkeypatch.setattr(client, "chat_stream", fail_after_the_tool)
    client.tool_rounds = [[CALL]]
    with pytest.raises(LLMError):
        list(agent.run("Save Hello"))
    monkeypatch.undo()
    list(agent.run("Thanks"))
    assert agent.memory.turns[0].tools == [] and agent.used == []


def test_an_interrupted_turn_keeps_its_tool_calls(config, client):
    agent = agent_with(config, client)
    client.tool_rounds = [[CALL]]
    events = agent.run("Save Hello")
    next(event for event in events if type(event).__name__ == "ToolResultEvent")
    events.close()
    assert agent.interrupt()
    assert agent.memory.turns[0].tools == [ToolUse("write_note", CALL.arguments, "drafts/note.md created")]


def step(action: str, **arguments) -> str:
    return json.dumps({"thought": "I need it.", "action": action, "arguments": arguments})


def test_the_tools_of_the_reasoning_are_kept(config, client):
    config.features["reasoning"] = True
    agent = agent_with(config, client)
    client.steps = [step("write_note", path="a.md", content="Hello")]
    list(agent.run("Save Hello"))
    assert [tool.name for tool in agent.memory.turns[0].tools] == ["write_note"]


def test_a_model_without_tool_calling_gets_no_tool_messages(config, client):
    config.features["reasoning"] = True
    client.capable = False
    agent = agent_with(config, client)
    client.steps = [step("write_note", path="a.md", content="Hello")]
    list(agent.run("Save Hello"))
    assert [message["role"] for message in agent.messages] == ["user", "assistant"]


class ReviewClient(FakeClient):
    """The investigator calls a tool, then each agent answers from the script."""

    def __init__(self):
        super().__init__()
        self.answers = ["Orion paid Delta.", APPROVED, "The text."]

    def chat_stream(self, *args, **kwargs):
        if not (kwargs.get("tools") and self.tool_rounds):
            self.answer = self.answers.pop(0)
        return super().chat_stream(*args, **kwargs)


def test_a_review_keeps_the_tool_calls_of_the_investigator(config):
    client = ReviewClient()
    client.tool_rounds = [[CALL]]
    agent = agent_with(config, client)
    list(Review(agent, PERSONAS).run("Who paid?"))
    assert [tool.name for tool in agent.memory.turns[0].tools] == ["write_note"]


def test_an_interrupted_review_keeps_the_tool_calls_too(config):
    client = ReviewClient()
    client.tool_rounds = [[CALL]]
    agent = agent_with(config, client)
    review = Review(agent, PERSONAS)
    events = review.run("Who paid?")
    next(event for event in events if type(event).__name__ == "ToolResultEvent")
    events.close()
    assert review.interrupt()
    assert [tool.name for tool in agent.memory.turns[0].tools] == ["write_note"]


def memory_with_a_tool(config, client) -> ConversationMemory:
    memory = ConversationMemory(config, client)
    memory.commit("Save Hello", "Saved.", tools=[ToolUse("write_note", {"path": "a.md"}, "a.md created")])
    return memory


def test_a_session_keeps_the_tool_calls(config, client):
    store = SessionStore(config.sessions_dir)
    store.save("note", memory_with_a_tool(config, client), "m")
    memory = ConversationMemory(config, client)
    store.load("note", memory)
    assert memory.turns[0].tools == [ToolUse("write_note", {"path": "a.md"}, "a.md created")]
    assert memory.messages()[2] == {"role": "tool", "tool_name": "write_note", "content": "a.md created"}


def test_a_session_saved_before_the_tool_calls_still_opens(config, client):
    store = SessionStore(config.sessions_dir)
    store.save("old", memory_with_a_tool(config, client), "m")
    path = config.sessions_dir / "old.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    del data["turns"][0]["tools"]
    path.write_text(json.dumps(data), encoding="utf-8")
    memory = ConversationMemory(config, client)
    store.load("old", memory)
    assert memory.turns == [Turn("Save Hello", "Saved.")]


@pytest.mark.parametrize("tools", [[{"name": "x"}], "nope", [{"name": "x", "arguments": 3, "result": "r"}]])
def test_a_session_with_invalid_tool_calls_is_refused(config, client, tools):
    store = SessionStore(config.sessions_dir)
    store.save("broken", memory_with_a_tool(config, client), "m")
    path = config.sessions_dir / "broken.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["turns"][0]["tools"] = tools
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(SessionError):
        store.load("broken", ConversationMemory(config, client))


def test_a_branch_keeps_the_tool_calls_of_the_turns_before_it(config, client):
    memory = memory_with_a_tool(config, client)
    memory.commit("Second", "Answer.")
    memory.fork(2)
    assert memory.turns[0].tools and memory.branch.number == 2


def test_the_tool_calls_go_away_with_the_turns_that_are_condensed(config, client):
    memory = memory_with_a_tool(config, client)
    memory.commit("Second", "Answer.")
    config.keep_recent_turns = 1
    memory.compact(lambda summary, turns: "Summary")
    assert [message["role"] for message in memory.messages()] == ["system", "user", "assistant"]
