import json
from datetime import datetime, timedelta, timezone

import pytest

from conftest import FakeClient
from sensai.core.agent import Agent, EmptyInputError
from sensai.core.cache import SemanticCache
from sensai.core.config import DEFAULT_PERSONAS_DIR
from sensai.core.events import LogFailedEvent, TokenEvent
from sensai.core.guardrails.rules import Guardrails, parse_rules
from sensai.core.llm.ollama import LLMError, ToolCall
from sensai.core.monitoring.log import InteractionLog
from sensai.core.orchestrator import Review
from sensai.core.persona import load_personas
from sensai.core.rag.models import SearchResult
from sensai.core.tools.registry import ToolRegistry

QUESTION = "Qui dirige Orion ?"
PERSONAS = load_personas(DEFAULT_PERSONAS_DIR)
APPROVED = '{"claims": [{"claim": "Marc Vidal runs Orion", "verdict": "supported", "reason": "the statutes say so"}]}'


class Retriever:
    def search(self, query: str, k: int) -> list[SearchResult]:
        return [SearchResult(1, "Gérant : Marc Vidal.", "p. 2", "statuts_orion.pdf", "", 0.9)]


class BrokenClient(FakeClient):
    def chat_stream(self, *args, **kwargs):
        raise LLMError("Cannot reach Ollama at http://localhost:11434. Is `ollama serve` running?")


class ScriptedClient(FakeClient):
    def __init__(self, answers: list[str]):
        super().__init__()
        self.answers = list(answers)

    def chat_stream(self, *args, **kwargs):
        self.answer = self.answers.pop(0)
        return super().chat_stream(*args, **kwargs)


def lookup(query: str) -> str:
    """
    Look something up.

    Args:
      query: What to look up
    """
    return "Marc Vidal"


@pytest.fixture
def log(config):
    log = InteractionLog(config.log_path)
    yield log
    log.close()


def make_agent(config, client, log, **kwargs) -> Agent:
    config.features["document"] = False
    client.answer = "Marc Vidal dirige Orion [1]."
    return Agent(config, client, log=log, **kwargs)


def logged(log) -> list:
    return log.since(datetime.now(timezone.utc) - timedelta(minutes=1))


def test_an_answer_is_logged_with_its_prompt_and_its_measures(config, client, log):
    agent = make_agent(config, client, log, retriever=Retriever())
    agent.session_id = "20261101-100000"
    list(agent.run(QUESTION))

    [record] = logged(log)
    assert (record.model, record.session_id, record.persona) == ("chat-model", "20261101-100000", "")
    assert (record.question, record.answer, record.outcome) == (QUESTION, "Marc Vidal dirige Orion [1]. ", "answered")
    assert record.prompt == client.chats[-1][0]
    assert record.prompt[-1]["content"].endswith(f"Message: {QUESTION}")
    assert (record.prompt_tokens, record.completion_tokens, record.sources) == (100, 10, 1)
    assert record.seconds >= record.first_token_seconds >= 0
    assert len(record.prompt_version) == 8


def test_the_question_is_logged_after_the_guardrails(config, client, log):
    rule = {"name": "source", "terms": ["Claire Martin"], "on": ["input", "output"], "replacement": "[SOURCE]"}
    agent = make_agent(config, client, log, guardrails=Guardrails(parse_rules([rule])))
    list(agent.run("Que sait Claire Martin d'Orion ?"))

    [record] = logged(log)
    assert record.question == "Que sait [SOURCE] d'Orion ?" and record.guardrails == ["source"]
    assert "Claire Martin" not in json.dumps(record.prompt, ensure_ascii=False)


def test_a_blocked_message_is_logged_as_blocked(config, client, log):
    rule = {"name": "private-life", "judge": "private addresses", "on": ["input"], "action": "block"}
    guardrails = Guardrails(parse_rules([rule]), lambda policy, message: "asks for a home address")
    list(make_agent(config, client, log, guardrails=guardrails).run("Où habite Marc Vidal ?"))

    [record] = logged(log)
    assert (record.outcome, record.guardrails, record.answer, record.prompt) == ("blocked", ["private-life"], "", [])
    assert client.chats == []


def test_a_stopped_answer_is_logged_as_stopped(config, client, log):
    rule = {"name": "iban", "detectors": ["iban"], "action": "block"}
    agent = make_agent(config, client, log, guardrails=Guardrails(parse_rules([rule])))
    client.answer = "Le compte FR7630006000011234567890189 a reçu les fonds."
    list(agent.run(QUESTION))
    assert [(record.outcome, record.guardrails) for record in logged(log)] == [("stopped", ["iban"])]


def test_a_cached_answer_is_logged_without_tokens(config, client, log, tmp_path):
    cache = SemanticCache(tmp_path / "cache.db", FakeClient.vector, threshold=0.9)
    agent = make_agent(config, client, log, cache=cache, confirm_cache=lambda cached: True)
    list(agent.run(QUESTION))
    agent.memory.turns = []
    list(agent.run(QUESTION))

    first, second = logged(log)
    assert (first.outcome, second.outcome) == ("answered", "cached")
    assert (second.prompt_tokens, second.completion_tokens, second.answer) == (0, 0, first.answer)
    cache.close()


def test_an_error_is_logged_then_raised(config, log):
    agent = make_agent(config, BrokenClient(), log)
    with pytest.raises(LLMError):
        list(agent.run(QUESTION))
    [record] = logged(log)
    assert (record.outcome, record.question) == ("error", QUESTION)
    assert record.error.startswith("Cannot reach Ollama")


def test_an_interruption_is_logged_as_an_error(config, client, log):
    events = make_agent(config, client, log).run(QUESTION)
    next(events)
    with pytest.raises(KeyboardInterrupt):
        events.throw(KeyboardInterrupt())
    assert [(record.outcome, record.error) for record in logged(log)] == [("error", "KeyboardInterrupt")]


def test_the_tools_called_are_logged(config, client, log):
    agent = make_agent(config, client, log, tools=ToolRegistry([lookup]))
    client.tool_rounds = [[ToolCall("lookup", {"query": "Orion"}), ToolCall("lookup", {"query": "Vidal"})]]
    list(agent.run(QUESTION))
    [record] = logged(log)
    assert record.tools == ["lookup", "lookup"]
    assert [message["role"] for message in record.prompt][-3:] == ["assistant", "tool", "tool"]


def test_the_reasoning_steps_are_logged(config, client, log):
    config.features["reasoning"] = True
    client.steps = [
        json.dumps({"thought": "I look Orion up.", "action": "lookup", "arguments": {"query": "Orion"}}),
        json.dumps({"thought": "I can answer.", "action": "answer", "arguments": {}}),
    ]
    list(make_agent(config, client, log, tools=ToolRegistry([lookup])).run(QUESTION))
    [record] = logged(log)
    assert (record.reasoning_steps, record.tools) == (2, ["lookup"])


def test_without_content_only_the_measures_are_logged(config, client, log):
    config.log_content = False
    list(make_agent(config, client, log, retriever=Retriever()).run(QUESTION))
    [record] = logged(log)
    assert (record.question, record.prompt, record.answer) == ("", [], "")
    assert (record.outcome, record.prompt_tokens, record.sources) == ("answered", 100, 1)


def test_the_logging_feature_turns_it_off(config, client, log):
    config.features["logging"] = False
    list(make_agent(config, client, log).run(QUESTION))
    assert logged(log) == []


def test_an_empty_message_is_not_logged(config, client, log):
    with pytest.raises(EmptyInputError):
        list(make_agent(config, client, log).run("   "))
    assert logged(log) == []


def test_the_prompt_version_follows_the_system_prompt(config, client, log):
    agent = make_agent(config, client, log)
    version = agent.prompt_version()
    assert make_agent(config, FakeClient(), log).prompt_version() == version
    agent.set_persona(PERSONAS["fact_checker"])
    assert agent.prompt_version() != version
    list(agent.run(QUESTION))
    [record] = logged(log)
    assert (record.persona, record.prompt_version) == ("fact_checker", agent.prompt_version())


def test_a_log_that_cannot_be_written_does_not_stop_the_answer(config, client, log):
    log.close()
    events = list(make_agent(config, client, log).run(QUESTION))
    assert "".join(event.text for event in events if isinstance(event, TokenEvent)) == "Marc Vidal dirige Orion [1]. "
    [failed] = [event for event in events if isinstance(event, LogFailedEvent)]
    assert failed.message.startswith("Cannot write to the log")


def test_each_agent_of_a_review_is_logged_with_its_persona(config, log):
    client = ScriptedClient(["Marc Vidal dirige Orion.", APPROVED, "Marc Vidal dirige Orion, d'après les statuts."])
    agent = make_agent(config, client, log)
    agent.session_id = "20261101-100000"
    list(Review(agent, PERSONAS).run(QUESTION))
    records = logged(log)
    assert [record.persona for record in records] == ["investigator", "fact_checker", "editor"]
    assert {record.session_id for record in records} == {"20261101-100000"}
