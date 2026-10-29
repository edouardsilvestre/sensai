from sensai.core.agent import STOPPED, Agent
from sensai.core.approval import Approval
from sensai.core.events import ApprovalEvent, DoneEvent, GuardrailEvent, TokenEvent, ToolResultEvent
from sensai.core.guardrails.rules import Guardrails, parse_rules
from sensai.core.llm.ollama import ToolCall
from sensai.core.tools.registry import ToolRegistry

SOURCE = {"name": "source", "terms": ["Claire Benoit"], "on": ["output", "tools"], "replacement": "[SOURCE]"}
IBAN = {"name": "personal-data", "detectors": ["iban"], "on": ["tools"], "tools": ["web_search"], "action": "block"}


def agent_with(config, client, *rules, judge=None, tools=None) -> Agent:
    return Agent(config, client, tools=tools, guardrails=Guardrails(parse_rules(list(rules)), judge))


def text(events) -> str:
    return "".join(event.text for event in events if isinstance(event, TokenEvent))


def hits(events) -> list:
    return [event.hit for event in events if isinstance(event, GuardrailEvent)]


def test_blocked_message_never_reaches_the_model(config, client):
    rule = {"name": "private-life", "judge": "private addresses", "on": ["input"], "action": "block"}
    agent = agent_with(config, client, rule, judge=lambda policy, message: "asks for a home address")
    events = list(agent.run("Where does Marc Vidal live?"))
    assert hits(events)[0].rule == "private-life"
    assert client.chats == [] and agent.memory.turns == []
    assert not any(isinstance(event, DoneEvent) for event in events)


def test_answer_is_redacted_before_it_is_shown_and_kept(config, client):
    client.answer = "Claire Benoit a refusé de valider le virement."
    agent = agent_with(config, client, SOURCE)
    events = list(agent.run("Qui a refusé ?"))
    assert text(events) == "[SOURCE] a refusé de valider le virement. "
    assert agent.memory.turns[0].assistant == text(events)
    assert hits(events)[0].count == 1


def test_blocked_answer_is_stopped(config, client):
    client.answer = "Le compte FR7630006000011234567890189 a reçu les fonds."
    rule = {"name": "iban", "detectors": ["iban"], "action": "block"}
    agent = agent_with(config, client, rule)
    events = list(agent.run("Quel compte ?"))
    assert text(events) == "Le compte "
    assert agent.memory.turns[0].assistant == f"Le compte \n{STOPPED}"


def search_agent(config, client, *rules):
    queries = []

    def web_search(query: str) -> str:
        """
        Search the web.

        Args:
          query: Keywords
        """
        queries.append(query)
        return "results"

    return agent_with(config, client, *rules, tools=ToolRegistry([web_search])), queries


def test_tool_call_with_personal_data_is_blocked_and_the_model_is_told(config, client):
    agent, queries = search_agent(config, client, IBAN)
    client.tool_rounds = [[ToolCall("web_search", {"query": "FR76 3000 6000 0112 3456 7890 189"})]]
    events = list(agent.run("À qui est ce compte ?"))
    result = next(event.result for event in events if isinstance(event, ToolResultEvent))
    assert queries == []
    assert result.error and "blocked by the guardrail 'personal-data' (IBAN)" in result.content
    assert "blocked by the guardrail" in client.chats[1][0][-1]["content"]


def test_tool_call_is_redacted_before_it_runs(config, client):
    agent, queries = search_agent(config, client, SOURCE)
    client.tool_rounds = [[ToolCall("web_search", {"query": "Claire Benoit Alvor Consulting"})]]
    list(agent.run("Cherche"))
    assert queries == ["[SOURCE] Alvor Consulting"]


def test_blocked_tool_call_is_not_run_nor_submitted_for_approval(config, client):
    agent, queries = search_agent(config, client, IBAN)
    asked = []
    agent.approver = lambda call, reason: asked.append(call) or Approval(True)
    config.approval = "all"
    client.tool_rounds = [[ToolCall("web_search", {"query": "FR76 3000 6000 0112 3456 7890 189"})]]
    events = list(agent.run("À qui est ce compte ?"))
    assert queries == [] and asked == []
    assert not any(isinstance(event, ApprovalEvent) for event in events)


def test_approved_tool_call_runs_once_with_the_redacted_arguments(config, client):
    agent, queries = search_agent(config, client, SOURCE)
    asked = []
    agent.approver = lambda call, reason: asked.append(call) or Approval(True)
    config.approval = "all"
    client.tool_rounds = [[ToolCall("web_search", {"query": "Claire Benoit Alvor Consulting"})]]
    list(agent.run("Cherche"))
    assert queries == ["[SOURCE] Alvor Consulting"]
    assert asked == [ToolCall("web_search", {"query": "[SOURCE] Alvor Consulting"})]


def test_guardrails_feature_off_checks_nothing(config, client):
    client.answer = "Claire Benoit a refusé."
    agent = agent_with(config, client, SOURCE)
    config.features["guardrails"] = False
    assert text(list(agent.run("Qui ?"))) == "Claire Benoit a refusé. "
