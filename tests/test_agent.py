import pytest

from sensai.core.agent import Agent
from sensai.core.events import DoneEvent, SourcesEvent, TokenEvent
from sensai.core.llm.ollama import LLMError
from sensai.core.rag.models import SearchResult


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
    def broken_stream(model, messages, options=None, think=None):
        yield "partial"
        raise LLMError("connection lost")

    client.chat_stream = broken_stream
    agent = Agent(config, client, FakeRetriever([RESULT]))
    with pytest.raises(LLMError):
        list(agent.run("Who approved it?"))
    assert agent.messages == []
