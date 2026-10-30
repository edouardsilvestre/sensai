import json

import pytest

from sensai.core.agent import Agent
from sensai.core.approval import Approval
from sensai.core.events import (
    DoneEvent,
    ReasoningStoppedEvent,
    ThoughtEvent,
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from sensai.core.guardrails.rules import Guardrails, parse_rules
from sensai.core.reasoning import ANSWER, Step
from sensai.core.tools.registry import ToolRegistry, ToolResult, requires_approval

FOUND = "statuts_orion.pdf: Marc Vidal is the manager of Orion Delta Consulting."
SOURCE = {"name": "source", "terms": ["Claire Benoit"], "on": ["output", "tools"], "replacement": "[SOURCE]"}
IBAN = {"name": "personal-data", "detectors": ["iban"], "on": ["tools"], "action": "block"}


def step(thought: str, action: str = ANSWER, **arguments) -> str:
    return json.dumps({"thought": thought, "action": action, "arguments": arguments})


def documents(queries: list[str]):
    def search_documents(query: str) -> str:
        """
        Search the journalist's indexed documents.

        Args:
          query: What to look for

        Returns:
          The excerpts found
        """
        queries.append(query)
        return FOUND

    return search_documents


@pytest.fixture
def reasoning_config(config):
    config.features["reasoning"] = True
    return config


def make_agent(config, client, queries, *rules, tool=None, approver=None) -> Agent:
    tools = ToolRegistry([tool or documents(queries)])
    return Agent(config, client, tools=tools, guardrails=Guardrails(parse_rules(list(rules))), approver=approver)


def thoughts(events) -> list[tuple[int, str]]:
    return [(event.number, event.thought) for event in events if isinstance(event, ThoughtEvent)]


def text(events) -> str:
    return "".join(event.text for event in events if isinstance(event, TokenEvent))


def stopped(events) -> list[str]:
    return [event.reason for event in events if isinstance(event, ReasoningStoppedEvent)]


def test_without_the_feature_the_agent_does_not_reason(config, client):
    agent = make_agent(config, client, [])
    events = list(agent.run("Who manages Orion?"))
    assert thoughts(events) == [] and client.steps_seen == [] and agent.reasoning == []
    assert text(events) == "Hello there "


def test_each_step_is_shown_before_the_answer(reasoning_config, client):
    queries = []
    agent = make_agent(reasoning_config, client, queries)
    client.steps = [step("I need the manager.", "search_documents", query="Orion manager"), step("Marc Vidal.")]
    client.answer = "Marc Vidal dirige Orion."

    events = list(agent.run("Qui dirige Orion ?"))

    assert queries == ["Orion manager"]
    assert thoughts(events) == [(1, "I need the manager."), (2, "Marc Vidal.")]
    kinds = [type(event) for event in events if not isinstance(event, TokenEvent)]
    assert kinds[:4] == [ThoughtEvent, ToolCallEvent, ToolResultEvent, ThoughtEvent]
    assert text(events) == "Marc Vidal dirige Orion. "
    assert isinstance(events[-1], DoneEvent)


def test_the_model_reads_each_result_before_the_next_step(reasoning_config, client):
    agent = make_agent(reasoning_config, client, [])
    client.steps = [step("I need the manager.", "search_documents", query="Orion manager")]

    list(agent.run("Qui dirige Orion ?"))

    first, second = (messages for messages, _ in client.steps_seen)
    assert first[-2]["role"] == "system" and "search_documents(query)" in first[-2]["content"]
    assert first[-1] == {"role": "user", "content": "Qui dirige Orion ?"}
    assert second[-1] == {"role": "user", "content": f"Result of search_documents: {FOUND}"}


def test_the_answer_is_written_from_the_reasoning_without_tools(reasoning_config, client):
    agent = make_agent(reasoning_config, client, [])
    client.steps = [step("I need the manager.", "search_documents", query="Orion manager")]

    list(agent.run("Qui dirige Orion ?"))

    messages, _ = client.chats[-1]
    assert client.tools_seen[-1] is None
    assert messages[-1]["content"] == "Qui dirige Orion ?"
    assert "Step 1. Thought: I need the manager." in messages[-2]["content"]
    assert f"Result: {FOUND}" in messages[-2]["content"]


def test_the_steps_only_offer_the_enabled_tools(reasoning_config, client):
    agent = make_agent(reasoning_config, client, [])
    list(agent.run("Qui dirige Orion ?"))
    reasoning_config.features["tools"] = False
    list(agent.run("Et Nordbank ?"))
    enums = [schema["properties"]["action"]["enum"] for _, schema in client.steps_seen]
    assert enums == [["search_documents", ANSWER], [ANSWER]]


def test_a_critical_tool_waits_for_the_users_approval(reasoning_config, client):
    queries = []

    @requires_approval("the query is sent to an external search engine")
    def web_search(query: str) -> str:
        """
        Search the web.

        Args:
          query: Keywords
        """
        queries.append(query)
        return "results"

    agent = make_agent(
        reasoning_config, client, queries, tool=web_search, approver=lambda call, reason: Approval(False, "secret")
    )
    client.steps = [step("I search the web.", "web_search", query="Orion Delta Consulting")]

    list(agent.run("Qui dirige Orion ?"))

    assert queries == []
    assert agent.reasoning[0].steps[0].result.summary == "refused"
    assert "The user refused this tool call: secret." in client.steps_seen[1][0][-1]["content"]


def test_a_tool_call_blocked_by_a_guardrail_is_not_run(reasoning_config, client):
    queries = []
    agent = make_agent(reasoning_config, client, queries, IBAN)
    client.steps = [step("I look for the account.", "search_documents", query="FR76 3000 6000 0112 3456 7890 189")]

    list(agent.run("À qui est ce compte ?"))

    assert queries == []
    assert agent.reasoning[0].steps[0].result.error


def test_a_thought_is_redacted_before_it_is_shown(reasoning_config, client):
    agent = make_agent(reasoning_config, client, [], SOURCE)
    client.steps = [step("Claire Benoit wrote the email.")]
    events = list(agent.run("Qui a écrit ?"))
    assert thoughts(events) == [(1, "[SOURCE] wrote the email.")]
    assert agent.reasoning[0].steps[0].thought == "[SOURCE] wrote the email."


def test_an_invalid_step_stops_the_reasoning_but_the_agent_answers(reasoning_config, client):
    agent = make_agent(reasoning_config, client, [])
    client.steps = ['{"thought": "I need']
    events = list(agent.run("Qui dirige Orion ?"))
    assert stopped(events)[0].startswith("The step is not valid JSON")
    assert text(events) == "Hello there "


def test_the_reasoning_stops_after_the_step_limit(reasoning_config, client):
    queries = []
    reasoning_config.max_tool_rounds = 2
    agent = make_agent(reasoning_config, client, queries)
    client.steps = [step(f"Search {n}.", "search_documents", query=f"q{n}") for n in range(3)]

    events = list(agent.run("Qui dirige Orion ?"))

    assert queries == ["q0", "q1"]
    assert stopped(events) == ["no answer after 2 steps"]
    assert text(events) == "Hello there "


def test_the_session_keeps_the_reasoning_of_each_question_but_not_the_history(reasoning_config, client):
    agent = make_agent(reasoning_config, client, [])
    client.steps = [step("I need the manager.", "search_documents", query="Orion"), step("Marc Vidal.")]
    list(agent.run("Qui dirige Orion ?"))
    list(agent.run("Merci"))

    assert [reasoning.question for reasoning in agent.reasoning] == ["Qui dirige Orion ?", "Merci"]
    assert agent.reasoning[0].steps == [
        Step("I need the manager.", "search_documents", {"query": "Orion"}, ToolResult(FOUND)),
        Step("Marc Vidal.", ANSWER),
    ]
    assert all("I need the manager" not in message["content"] for message in agent.messages)
