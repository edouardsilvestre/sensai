import json

import pytest

from conftest import FakeClient
from sensai.core.agent import Agent
from sensai.core.cache import SemanticCache
from sensai.core.evaluation.judge import CLAIM_SCHEMA, SCORE_SCHEMA
from sensai.core.events import CheckEvent, CheckFailedEvent, CheckStartedEvent, DoneEvent
from sensai.core.guardrails.rules import Guardrails, parse_rules
from sensai.core.llm.ollama import LLMError, ToolCall
from sensai.core.rag.models import SearchResult
from sensai.core.tools.registry import ToolRegistry

QUESTION = "Qui dirige Orion ?"
ANSWER = "Marc Vidal dirige Orion [1]."
STATUTES = SearchResult(
    1, "Gérance : Orion Delta Consulting est gérée par M. Marc Vidal.", "p. 2", "statuts.pdf", "", 0.9
)


class Retriever:
    def search(self, query: str, k: int) -> list[SearchResult]:
        return [STATUTES]


class JudgeClient(FakeClient):
    def __init__(self, *verdicts: str):
        super().__init__(answer=ANSWER)
        self.verdicts = list(verdicts)
        self.judged: list[str] = []
        self.broken = False

    def chat(self, model, messages, options=None, think=None, usage=None, format=None) -> str:
        if format not in (CLAIM_SCHEMA, SCORE_SCHEMA):
            return super().chat(model, messages, options, think, usage, format)
        if self.broken:
            raise LLMError("Cannot reach Ollama")
        self.judged.append(messages[0]["content"])
        if format == SCORE_SCHEMA:
            return json.dumps({"relevance": 5, "coherence": 4})
        return json.dumps({"reason": "the statutes say so", "verdict": self.verdicts.pop(0)})


def lookup(query: str) -> str:
    """
    Look a company up in the registry.

    Args:
      query: The company
    """
    return "Orion Delta Consulting: SARL, Lyon"


@pytest.fixture
def client() -> JudgeClient:
    return JudgeClient("supported")


def make_agent(config, client, **kwargs) -> Agent:
    config.features.update(document=False, fact_check=True)
    return Agent(config, client, retriever=Retriever(), **kwargs)


def checks(events) -> list:
    return [event.judgement for event in events if isinstance(event, CheckEvent)]


def test_the_answer_is_checked_against_its_excerpts_after_it_is_given(config, client):
    events = list(make_agent(config, client).run(QUESTION))

    done = next(index for index, event in enumerate(events) if isinstance(event, DoneEvent))
    started = next(index for index, event in enumerate(events) if isinstance(event, CheckStartedEvent))
    assert done < started and events[started] == CheckStartedEvent(1, "chat-model")
    [judgement] = checks(events)
    assert ([claim.verdict for claim in judgement.claims], judgement.relevance) == (["supported"], 5)
    assert "[1] statuts.pdf, p. 2:\nGérance : Orion Delta Consulting" in client.judged[0]
    assert f"Sentence: {ANSWER}" in client.judged[0]


def test_the_judge_model_is_the_one_set(config, client):
    config.judge_model = "qwen3:1.7b"
    events = list(make_agent(config, client).run(QUESTION))
    assert CheckStartedEvent(1, "qwen3:1.7b") in events


def test_the_results_of_the_tools_are_evidence_too(config, client):
    agent = make_agent(config, client, tools=ToolRegistry([lookup]))
    client.tool_rounds = [[ToolCall("lookup", {"query": "Orion"})]]
    list(agent.run(QUESTION))
    assert "Tool result:\nOrion Delta Consulting: SARL, Lyon" in client.judged[0]


def test_the_results_found_while_reasoning_are_evidence_too(config, client):
    config.features["reasoning"] = True
    client.steps = [
        json.dumps({"thought": "I look Orion up.", "action": "lookup", "arguments": {"query": "Orion"}}),
        json.dumps({"thought": "I can answer.", "action": "answer", "arguments": {}}),
    ]
    list(make_agent(config, client, tools=ToolRegistry([lookup])).run(QUESTION))
    assert "Tool result:\nOrion Delta Consulting: SARL, Lyon" in client.judged[0]


def test_a_failed_check_keeps_the_answer(config, client):
    client.broken = True
    events = list(make_agent(config, client).run(QUESTION))
    assert DoneEvent(f"{ANSWER} ") in events
    assert [event.message for event in events if isinstance(event, CheckFailedEvent)] == ["Cannot reach Ollama"]


def test_the_check_is_off_by_default(config, client):
    config.features["document"] = False
    events = list(Agent(config, client, retriever=Retriever()).run(QUESTION))
    assert checks(events) == [] and client.judged == []


def test_a_json_answer_is_not_checked(config, client):
    events = list(make_agent(config, client).run(QUESTION, format={"type": "object"}))
    assert checks(events) == []


def test_a_stopped_answer_is_not_checked(config, client):
    rule = {"name": "iban", "detectors": ["iban"], "action": "block"}
    client.answer = "Le compte FR7630006000011234567890189 a reçu les fonds."
    events = list(make_agent(config, client, guardrails=Guardrails(parse_rules([rule]))).run(QUESTION))
    assert checks(events) == []


def test_a_cached_answer_is_not_checked_again(config, client, tmp_path):
    cache = SemanticCache(tmp_path / "cache.db", FakeClient.vector, threshold=0.9)
    agent = make_agent(config, client, cache=cache, confirm_cache=lambda cached: True)
    list(agent.run(QUESTION))
    agent.memory.turns = []
    events = list(agent.run(QUESTION))
    assert checks(events) == [] and len(client.judged) == 2
    cache.close()


def test_the_agents_of_a_review_are_not_checked(config, client):
    agent = make_agent(config, client)
    agent.keeps_document = False
    assert checks(list(agent.run(QUESTION))) == []
