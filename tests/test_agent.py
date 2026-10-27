import pytest

from sensai.core.agent import Agent
from sensai.core.approval import ALL_TOOLS_REASON, NO_APPROVER_REASON, Approval
from sensai.core.events import (
    ApprovalEvent,
    CompactionEvent,
    CompactionFailedEvent,
    CompactionStartedEvent,
    DoneEvent,
    SourcesEvent,
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from sensai.core.llm.ollama import LLMError, ToolCall
from sensai.core.rag.models import SearchResult
from sensai.core.tools.registry import ToolRegistry, ToolResult, requires_approval


class FakeRetriever:
    def __init__(self, results: list[SearchResult]):
        self.results = results
        self.queries: list[tuple[str, int]] = []

    def search(self, query: str, k: int) -> list[SearchResult]:
        self.queries.append((query, k))
        return self.results


RESULT = SearchResult(
    chunk_id=1, text="Alice approved the transfer.", location="p. 2", source="leak/mail.eml",
    description="From: alice@corp.example", score=0.9,
)


def test_plain_chat_without_retriever(config, client):
    agent = Agent(config, client)
    events = list(agent.run("Hi"))

    assert isinstance(events[0], TokenEvent)
    assert events[-1] == DoneEvent("Hello there ")
    messages, options = client.chats[0]
    assert messages == [{"role": "user", "content": "Hi"}]
    assert options == {"num_ctx": config.num_ctx}


def test_thinking_setting_is_passed_to_the_model(config, client):
    config.think = False
    list(Agent(config, client).run("Hi"))
    assert client.think is False


def test_excerpts_are_added_to_the_prompt_only(config, client):
    retriever = FakeRetriever([RESULT])
    agent = Agent(config, client, retriever)
    events = list(agent.run("Who approved it?"))

    assert events[0] == SourcesEvent([RESULT])
    assert retriever.queries == [("Who approved it?", config.rag_top_k)]
    prompt = client.chats[0][0][-1]["content"]
    assert '<excerpt id="1" source="leak/mail.eml" location="p. 2">' in prompt
    assert "Alice approved the transfer." in prompt
    assert prompt.endswith("Message: Who approved it?")
    assert agent.messages[0] == {"role": "user", "content": "Who approved it?"}


def test_no_results_means_a_plain_prompt(config, client):
    agent = Agent(config, client, FakeRetriever([]))
    events = list(agent.run("Hi"))

    assert not any(isinstance(event, SourcesEvent) for event in events)
    assert client.chats[0][0][-1]["content"] == "Hi"


def test_disabled_rag_does_not_search(config, client):
    retriever = FakeRetriever([RESULT])
    agent = Agent(config, client, retriever)
    agent.rag_enabled = False
    list(agent.run("Hi"))
    assert retriever.queries == []


def test_failed_turn_leaves_history_unchanged(config, client):
    def broken_stream(model, messages, options=None, think=None, usage=None, **kwargs):
        yield "partial"
        raise LLMError("connection lost")

    client.chat_stream = broken_stream
    agent = Agent(config, client, FakeRetriever([RESULT]))
    with pytest.raises(LLMError):
        list(agent.run("Who approved it?"))
    assert agent.messages == []


def history(agent: Agent, turns: int) -> None:
    for number in range(turns):
        agent.memory.commit(f"question {number}", f"answer {number}")


def over_budget(config, client) -> None:
    config.keep_recent_turns = 2
    client.usage = (config.num_ctx * config.compact_threshold // 100, 0)


def test_usage_of_the_last_turn_is_recorded(config, client):
    agent = Agent(config, client)
    list(agent.run("Hi"))
    assert agent.memory.last_usage.total_tokens == 110


def test_citations_of_the_turn_are_kept_for_the_summary(config, client):
    agent = Agent(config, client, FakeRetriever([RESULT]))
    list(agent.run("Who approved it?"))
    assert agent.memory.turns[0].citations == ["leak/mail.eml, p. 2"]


def test_no_compaction_under_the_threshold(config, client):
    agent = Agent(config, client)
    history(agent, 5)
    events = list(agent.run("Hi"))
    assert not any(isinstance(event, CompactionStartedEvent) for event in events)
    assert client.summaries == []


def test_history_is_compacted_after_the_turn_that_crossed_the_threshold(config, client):
    over_budget(config, client)
    agent = Agent(config, client)
    history(agent, 4)

    events = list(agent.run("And then?"))

    kinds = [type(event) for event in events]
    assert kinds.index(DoneEvent) < kinds.index(CompactionStartedEvent) < kinds.index(CompactionEvent)
    assert events[-1] == CompactionEvent(turns=3, tokens=client.usage[0])
    assert [turn.user for turn in agent.memory.turns] == ["question 3", "And then?"]
    assert "question 0" in client.summaries[0][0]["content"]

    list(agent.run("Next"))
    system = client.chats[-1][0][0]
    assert system["role"] == "system" and "Alice approved the transfer." in system["content"]


def test_failed_compaction_keeps_the_turn_and_the_history(config, client):
    over_budget(config, client)
    client.summary = "  "
    agent = Agent(config, client)
    history(agent, 4)

    events = list(agent.run("And then?"))

    assert isinstance(events[-1], CompactionFailedEvent)
    assert len(agent.memory.turns) == 5
    assert agent.memory.summary == ""



class Lookup:
    def __init__(self):
        self.queries: list[str] = []

        def lookup(query: str) -> str:
            """
            Look a company up.

            Args:
              query: The company name

            Returns:
              Where the company is registered
            """
            self.queries.append(query)
            return f"{query} is registered in Lisbon."

        self.function = lookup


def tool_agent(config, client) -> tuple[Agent, Lookup]:
    tool = Lookup()
    return Agent(config, client, tools=ToolRegistry([tool.function])), tool


def test_without_tools_no_schema_is_sent(config, client):
    list(Agent(config, client).run("Hi"))
    assert client.tools_seen == [None]


def test_tool_call_is_run_and_its_result_sent_back(config, client):
    agent, tool = tool_agent(config, client)
    client.tool_rounds = [[ToolCall("lookup", {"query": "Alvor"})]]

    events = list(agent.run("Where is Alvor based?"))

    call = ToolCall("lookup", {"query": "Alvor"})
    assert events[0] == ToolCallEvent(call)
    assert events[1] == ToolResultEvent(call, ToolResult("Alvor is registered in Lisbon."))
    assert events[-1] == DoneEvent("Hello there ")
    assert tool.queries == ["Alvor"]
    assert client.tools_seen[0] == [tool.function]
    messages = client.chats[1][0]
    assert messages[-2] == {
        "role": "assistant", "content": "", "tool_calls": [{"function": {"name": "lookup", "arguments": {"query": "Alvor"}}}],
    }
    assert messages[-1] == {"role": "tool", "tool_name": "lookup", "content": "Alvor is registered in Lisbon."}


def test_only_the_final_answer_is_kept_in_memory(config, client):
    agent, _ = tool_agent(config, client)
    client.tool_rounds = [[ToolCall("lookup", {"query": "Alvor"})]]
    list(agent.run("Where is Alvor based?"))
    assert agent.messages == [
        {"role": "user", "content": "Where is Alvor based?"},
        {"role": "assistant", "content": "Hello there "},
    ]


def test_parallel_calls_each_get_a_result(config, client):
    agent, tool = tool_agent(config, client)
    client.tool_rounds = [[ToolCall("lookup", {"query": "Alvor"}), ToolCall("lookup", {"query": "Vidal"})]]
    list(agent.run("Alvor and Vidal?"))
    assert tool.queries == ["Alvor", "Vidal"]
    assert [m["role"] for m in client.chats[1][0][-3:]] == ["assistant", "tool", "tool"]


def test_failed_tool_call_is_sent_back_to_the_model(config, client):
    agent, _ = tool_agent(config, client)
    client.tool_rounds = [[ToolCall("unknown", {})]]
    events = list(agent.run("Hi"))
    result = next(event for event in events if isinstance(event, ToolResultEvent)).result
    assert result.error
    assert client.chats[1][0][-1]["content"] == result.content


def test_tool_rounds_are_capped(config, client):
    config.max_tool_rounds = 2
    agent, tool = tool_agent(config, client)
    client.tool_rounds = [[ToolCall("lookup", {"query": str(n)})] for n in range(5)]

    events = list(agent.run("Search forever"))

    assert len(tool.queries) == 2
    assert client.tools_seen[-1] is None
    assert events[-1] == DoneEvent("Hello there ")


def test_disabled_tools_are_not_offered(config, client):
    agent, _ = tool_agent(config, client)
    agent.tools_enabled = False
    list(agent.run("Hi"))
    assert client.tools_seen == [None]


class FakeApprover:
    def __init__(self, approval: Approval):
        self.approval = approval
        self.asked: list[tuple[ToolCall, str]] = []

    def __call__(self, call: ToolCall, reason: str) -> Approval:
        self.asked.append((call, reason))
        return self.approval


SECRET = "the company name leaves the machine"
CALL = ToolCall("lookup", {"query": "Alvor"})


def approval_agent(config, client, approval: Approval | None, critical: bool = True):
    tool = Lookup()
    if critical:
        requires_approval(SECRET)(tool.function)
    approver = FakeApprover(approval) if approval else None
    client.tool_rounds = [[CALL]]
    return Agent(config, client, tools=ToolRegistry([tool.function]), approver=approver), tool, approver


def test_approved_critical_call_is_run(config, client):
    agent, tool, approver = approval_agent(config, client, Approval(True))

    events = list(agent.run("Where is Alvor based?"))

    assert approver.asked == [(CALL, SECRET)]
    assert events[:3] == [
        ToolCallEvent(CALL),
        ApprovalEvent(CALL, Approval(True)),
        ToolResultEvent(CALL, ToolResult("Alvor is registered in Lisbon.")),
    ]
    assert tool.queries == ["Alvor"]


def test_refused_critical_call_is_not_run_and_the_model_gets_the_reason(config, client):
    agent, tool, _ = approval_agent(config, client, Approval(False, "keep the source secret"))

    events = list(agent.run("Where is Alvor based?"))

    assert tool.queries == []
    result = next(event for event in events if isinstance(event, ToolResultEvent)).result
    assert result.summary == "refused" and not result.error
    sent = client.chats[1][0][-1]
    assert sent["role"] == "tool" and "refused" in sent["content"] and "keep the source secret" in sent["content"]
    assert events[-1] == DoneEvent("Hello there ")


def test_refusal_is_not_kept_in_memory(config, client):
    agent, _, _ = approval_agent(config, client, Approval(False))
    list(agent.run("Where is Alvor based?"))
    assert agent.messages == [
        {"role": "user", "content": "Where is Alvor based?"},
        {"role": "assistant", "content": "Hello there "},
    ]


def test_tools_that_are_not_critical_run_without_asking(config, client):
    agent, tool, approver = approval_agent(config, client, Approval(False), critical=False)
    events = list(agent.run("Where is Alvor based?"))
    assert approver.asked == []
    assert not any(isinstance(event, ApprovalEvent) for event in events)
    assert tool.queries == ["Alvor"]


def test_all_mode_asks_before_every_tool(config, client):
    config.approval = "all"
    agent, tool, approver = approval_agent(config, client, Approval(True), critical=False)
    list(agent.run("Where is Alvor based?"))
    assert approver.asked == [(CALL, ALL_TOOLS_REASON)]
    assert tool.queries == ["Alvor"]


def test_off_mode_never_asks(config, client):
    config.approval = "off"
    agent, tool, approver = approval_agent(config, client, Approval(False))
    list(agent.run("Where is Alvor based?"))
    assert approver.asked == []
    assert tool.queries == ["Alvor"]


def test_critical_calls_are_refused_without_an_approver(config, client):
    agent, tool, _ = approval_agent(config, client, None)
    events = list(agent.run("Where is Alvor based?"))
    assert ApprovalEvent(CALL, Approval(False, NO_APPROVER_REASON)) in events
    assert tool.queries == []


def test_unknown_tools_are_not_submitted_for_approval(config, client):
    config.approval = "all"
    agent, _, approver = approval_agent(config, client, Approval(True))
    client.tool_rounds = [[ToolCall("unknown", {})]]
    events = list(agent.run("Hi"))
    assert approver.asked == []
    assert next(event for event in events if isinstance(event, ToolResultEvent)).result.error
