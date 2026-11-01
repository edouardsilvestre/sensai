import json
import re

import pytest

from sensai.core.agent import Agent
from sensai.core.cache import SemanticCache
from sensai.core.events import CacheFailedEvent, CacheHitEvent, DoneEvent, SourcesEvent, TokenEvent
from sensai.core.examples import Example, ExampleLibrary
from sensai.core.guardrails.rules import Guardrails, parse_rules
from sensai.core.llm.ollama import LLMError, ToolCall
from sensai.core.persona import Persona
from sensai.core.rag.models import SearchFilter, SearchResult
from sensai.core.tools.registry import ToolRegistry

WORDS = ["qui", "dirige", "la", "société", "orion", "gérant", "quel", "montant", "vidal"]
QUESTION = "Qui dirige Orion ?"
CLOSE = "Qui dirige la société Orion ?"
ANSWER = "Marc Vidal dirige Orion [1]."
STATUTES = SearchResult(1, "Gérant : Marc Vidal.", "p. 2", "statuts_orion.pdf", "", 0.9)


def embed(text: str) -> list[float]:
    words = re.findall(r"\w+", text.lower())
    return [float(words.count(word)) for word in WORDS]


class Retriever:
    def __init__(self):
        self.documents = "v1"
        self.queries: list[str] = []

    def search(self, query: str, k: int, filters=(), report=None) -> list[SearchResult]:
        self.queries.append(query)
        return [STATUTES]

    def fingerprint(self) -> str:
        return self.documents


class Confirm:
    def __init__(self, answer: bool = True):
        self.answer = answer
        self.asked: list = []

    def __call__(self, cached) -> bool:
        self.asked.append(cached)
        return self.answer


@pytest.fixture
def cache(tmp_path):
    cache = SemanticCache(tmp_path / "cache.db", embed, threshold=0.7)
    yield cache
    cache.close()


def make_agent(config, client, cache, confirm=None, **kwargs) -> Agent:
    config.features["document"] = False
    client.answer = ANSWER
    return Agent(config, client, cache=cache, confirm_cache=confirm, **kwargs)


def new_conversation(agent: Agent) -> None:
    agent.memory.turns = []
    agent.memory.summary = ""


def text(events) -> str:
    return "".join(event.text for event in events if isinstance(event, TokenEvent))


def hits(events) -> list:
    return [event.cached for event in events if isinstance(event, CacheHitEvent)]


def test_a_close_question_is_answered_from_the_cache_once_confirmed(config, client, cache):
    confirm = Confirm()
    agent = make_agent(config, client, cache, confirm)
    list(agent.run(QUESTION))
    new_conversation(agent)

    events = list(agent.run(CLOSE))

    assert len(client.chats) == 1
    assert [cached.question for cached in confirm.asked] == [QUESTION]
    assert hits(events)[0].similarity == pytest.approx(confirm.asked[0].similarity)
    assert text(events) == f"{ANSWER} " and events[-1] == DoneEvent(f"{ANSWER} ")
    assert (agent.memory.turns[-1].user, agent.memory.turns[-1].assistant) == (CLOSE, f"{ANSWER} ")


def test_the_sources_come_back_with_the_cached_answer(config, client, cache):
    agent = make_agent(config, client, cache, Confirm(), retriever=Retriever())
    list(agent.run(QUESTION))
    new_conversation(agent)

    events = list(agent.run(CLOSE))

    assert [event.sources for event in events if isinstance(event, SourcesEvent)] == [[STATUTES]]
    assert agent.retriever.queries == [QUESTION]
    assert agent.memory.turns[-1].citations == ["statuts_orion.pdf, p. 2"]


def test_a_refused_answer_is_given_by_the_model_and_kept_too(config, client, cache):
    agent = make_agent(config, client, cache, Confirm(False))
    list(agent.run(QUESTION))
    new_conversation(agent)

    events = list(agent.run(CLOSE))

    assert hits(events) == [] and len(client.chats) == 2
    assert cache.stats().answers == 2


def test_without_a_way_to_confirm_the_cache_is_not_used(config, client, cache):
    agent = make_agent(config, client, cache)
    list(agent.run(QUESTION))
    new_conversation(agent)
    list(agent.run(QUESTION))
    assert len(client.chats) == 2 and cache.stats().answers == 0


def test_a_follow_up_question_is_not_answered_from_another_conversation(config, client, cache):
    confirm = Confirm()
    agent = make_agent(config, client, cache, confirm)
    list(agent.run(QUESTION))
    list(agent.run(QUESTION))
    assert confirm.asked == [] and len(client.chats) == 2


def test_another_persona_is_another_context(config, client, cache):
    confirm = Confirm()
    agent = make_agent(config, client, cache, confirm)
    list(agent.run(QUESTION))
    new_conversation(agent)
    agent.set_persona(Persona("fact_checker", "Fact-checker", "checks the claims", "strict", "claims"))
    list(agent.run(QUESTION))
    assert confirm.asked == []


def test_changed_documents_are_another_context(config, client, cache):
    confirm = Confirm()
    agent = make_agent(config, client, cache, confirm, retriever=Retriever())
    list(agent.run(QUESTION))
    new_conversation(agent)
    agent.retriever.documents = "v2"
    list(agent.run(QUESTION))
    assert confirm.asked == []


def test_another_filter_is_another_context(config, client, cache):
    confirm = Confirm()
    agent = make_agent(config, client, cache, confirm, retriever=Retriever())
    list(agent.run(QUESTION))
    new_conversation(agent)
    agent.search_filter = SearchFilter(after="2023")
    list(agent.run(QUESTION))
    assert confirm.asked == []


SOURCE_RULE = {"name": "source", "terms": ["Marc Vidal"], "on": ["output"], "replacement": "[SOURCE]"}


def test_changed_guardrails_are_another_context(config, client, cache):
    confirm = Confirm()
    agent = make_agent(config, client, cache, confirm)
    list(agent.run(QUESTION))
    new_conversation(agent)
    agent.guardrails = Guardrails(parse_rules([SOURCE_RULE]))
    events = list(agent.run(QUESTION))
    assert confirm.asked == [] and text(events) == "[SOURCE] dirige Orion [1]. "


def test_turning_the_guardrails_off_is_another_context(config, client, cache):
    confirm = Confirm()
    agent = make_agent(config, client, cache, confirm, guardrails=Guardrails(parse_rules([SOURCE_RULE])))
    list(agent.run(QUESTION))
    new_conversation(agent)
    config.features["guardrails"] = False
    list(agent.run(QUESTION))
    assert confirm.asked == []


@pytest.fixture
def examples(tmp_path):
    return ExampleLibrary(tmp_path / "examples.json", embed)


def test_a_new_example_is_another_context(config, client, cache, examples):
    confirm = Confirm()
    agent = make_agent(config, client, cache, confirm, examples=examples)
    list(agent.run(QUESTION))
    new_conversation(agent)
    examples.add(Example("Quel montant a reçu Marc Vidal ?", "Les documents ne le disent pas."))
    list(agent.run(QUESTION))
    assert confirm.asked == []


@pytest.mark.parametrize("change", ["feature off", "k changed"])
def test_showing_the_examples_differently_is_another_context(config, client, cache, examples, change):
    examples.add(Example("Quel montant a reçu Marc Vidal ?", "Les documents ne le disent pas."))
    confirm = Confirm()
    agent = make_agent(config, client, cache, confirm, examples=examples)
    list(agent.run(QUESTION))
    new_conversation(agent)
    if change == "feature off":
        config.features["examples"] = False
    else:
        config.examples_k = 1
    list(agent.run(QUESTION))
    assert confirm.asked == []


@pytest.mark.parametrize("setting", ["empty library", "feature off"])
def test_examples_never_shown_do_not_change_the_context(config, client, cache, examples, setting):
    if setting == "feature off":
        examples.add(Example("Quel montant a reçu Marc Vidal ?", "Les documents ne le disent pas."))
        config.features["examples"] = False
    confirm = Confirm()
    agent = make_agent(config, client, cache, confirm)
    list(agent.run(QUESTION))
    new_conversation(agent)
    agent.examples = examples
    list(agent.run(QUESTION))
    assert len(confirm.asked) == 1


def test_an_answer_that_used_a_tool_is_not_kept(config, client, cache):
    def lookup(query: str) -> str:
        """
        Look something up.

        Args:
          query: What to look up
        """
        return "Marc Vidal"

    agent = make_agent(config, client, cache, Confirm(), tools=ToolRegistry([lookup]))
    client.tool_rounds = [[ToolCall("lookup", {"query": "Orion"})]]
    list(agent.run(QUESTION))
    assert cache.stats().answers == 0


def test_an_answer_that_searched_while_reasoning_is_not_kept(config, client, cache):
    config.features["reasoning"] = True
    agent = make_agent(config, client, cache, Confirm(), retriever=Retriever())
    client.steps = [json.dumps({"thought": "I search.", "action": "search_documents", "arguments": {"query": "Orion"}})]
    list(agent.run(QUESTION))
    assert cache.stats().answers == 0


def test_a_stopped_answer_is_not_kept(config, client, cache):
    rule = {"name": "iban", "detectors": ["iban"], "action": "block"}
    agent = make_agent(config, client, cache, Confirm(), guardrails=Guardrails(parse_rules([rule])))
    client.answer = "Le compte FR7630006000011234567890189 a reçu les fonds."
    list(agent.run(QUESTION))
    assert cache.stats().answers == 0


def test_a_json_answer_is_neither_looked_up_nor_kept(config, client, cache):
    confirm = Confirm()
    agent = make_agent(config, client, cache, confirm)
    list(agent.run(QUESTION))
    new_conversation(agent)
    list(agent.run(QUESTION, format={"type": "object"}))
    assert confirm.asked == [] and cache.stats().answers == 1


def test_a_blocked_message_never_reaches_the_cache(config, client, cache):
    rule = {"name": "private-life", "judge": "private addresses", "on": ["input"], "action": "block"}
    guardrails = Guardrails(parse_rules([rule]), lambda policy, message: "asks for a home address")
    confirm = Confirm()
    agent = make_agent(config, client, cache, confirm, guardrails=guardrails)
    list(agent.run("Où habite Marc Vidal ?"))
    assert confirm.asked == [] and cache.stats().answers == 0


def test_a_cache_failure_does_not_stop_the_answer(config, client, tmp_path):
    def broken(text: str) -> list[float]:
        raise LLMError("Model 'qwen3-embedding:0.6b' not found")

    cache = SemanticCache(tmp_path / "cache.db", broken, threshold=0.7)
    events = list(make_agent(config, client, cache, Confirm()).run(QUESTION))
    assert [event.message for event in events if isinstance(event, CacheFailedEvent)] == [
        "Model 'qwen3-embedding:0.6b' not found"
    ]
    assert text(events) == f"{ANSWER} "
    cache.close()


def test_the_cache_feature_turns_it_off(config, client, cache):
    config.features["cache"] = False
    confirm = Confirm()
    agent = make_agent(config, client, cache, confirm)
    list(agent.run(QUESTION))
    new_conversation(agent)
    list(agent.run(QUESTION))
    assert confirm.asked == [] and cache.stats().answers == 0
