import re

import pytest

from sensai.core.agent import Agent
from sensai.core.events import ExamplesEvent, ExamplesFailedEvent, TokenEvent
from sensai.core.examples import Example, ExampleLibrary, Excerpt
from sensai.core.llm.ollama import LLMError
from sensai.core.rag.models import SearchResult

WORDS = ["salaire", "bonus", "marc", "vidal", "orion", "siège", "bonjour"]
SALARY = "Quel salaire touche Marc Vidal ?"
BONUS = Example(
    "Quel bonus touche Marc Vidal ?",
    "Les documents ne mentionnent pas de bonus. Ils indiquent seulement qu'il est gérant [2].",
    [Excerpt(2, "statuts_orion.pdf", "p. 2", "Gérant : Marc Vidal.")],
)
SEAT = Example("Où est le siège d'Orion ?", "À Lyon [1].")


def embed(text: str) -> list[float]:
    words = re.findall(r"\w+", text.lower())
    return [float(words.count(word)) for word in WORDS] + [0.01]


class Retriever:
    def search(self, query: str, k: int, filters=(), report=None) -> list[SearchResult]:
        return [SearchResult(1, "Gérant : Marc Vidal.", "p. 2", "statuts_orion.pdf", "", 0.9)]


@pytest.fixture
def library(tmp_path):
    library = ExampleLibrary(tmp_path / "examples.json", embed)
    library.add(BONUS)
    library.add(SEAT)
    return library


def make_agent(config, client, library, **kwargs) -> Agent:
    config.features["document"] = False
    return Agent(config, client, examples=library, **kwargs)


def sent(client, call: int = -1) -> list[dict]:
    return client.chats[call][0]


def test_the_closest_examples_are_sent_before_the_message(config, client, library):
    events = list(make_agent(config, client, library).run(SALARY))

    messages = sent(client)
    assert messages[-1] == {"role": "user", "content": SALARY}
    assert messages[-2]["role"] == "system"
    assert "Message: Quel bonus touche Marc Vidal ?" in messages[-2]["content"]
    assert '<excerpt id="2" source="statuts_orion.pdf" location="p. 2">' in messages[-2]["content"]
    assert "Où est le siège" not in messages[-2]["content"]
    assert [event.examples for event in events if isinstance(event, ExamplesEvent)] == [[BONUS]]


def test_the_examples_come_after_the_history_and_never_join_it(config, client, library):
    agent = make_agent(config, client, library)
    list(agent.run("Bonjour"))
    list(agent.run(SALARY))

    messages = sent(client)
    assert [message["role"] for message in messages] == ["user", "assistant", "system", "user"]
    assert all("Quel bonus" not in message["content"] for message in agent.memory.messages())


def test_the_examples_are_sent_with_the_excerpts_of_the_documents(config, client, library):
    list(make_agent(config, client, library, retriever=Retriever()).run(SALARY))
    messages = sent(client)
    assert "Message: Quel bonus" in messages[-2]["content"]
    assert "<excerpts>" in messages[-1]["content"] and messages[-1]["content"].endswith(f"Message: {SALARY}")


def test_the_number_of_examples_follows_the_setting(config, client, library):
    config.examples_k = 2
    list(make_agent(config, client, library).run("Quel salaire et quel bonus touche Marc Vidal au siège d'Orion ?"))
    assert sent(client)[-2]["content"].count("<example>") == 2


def test_no_example_is_close_enough(config, client, library):
    events = list(make_agent(config, client, library).run("Bonjour"))
    assert [message["role"] for message in sent(client)] == ["user"]
    assert not any(isinstance(event, ExamplesEvent) for event in events)


@pytest.mark.parametrize("setting", ["feature off", "k zero", "json"])
def test_no_examples_are_sent(config, client, library, setting):
    if setting == "feature off":
        config.features["examples"] = False
    if setting == "k zero":
        config.examples_k = 0
    run = make_agent(config, client, library).run(SALARY, format={"type": "object"} if setting == "json" else None)
    events = list(run)
    assert all(message["role"] != "system" for message in sent(client))
    assert not any(isinstance(event, ExamplesEvent) for event in events)


def test_without_a_library_nothing_changes(config, client):
    list(make_agent(config, client, None).run(SALARY))
    assert [message["role"] for message in sent(client)] == ["user"]


def test_a_failed_embedding_does_not_stop_the_answer(config, client, tmp_path):
    def broken(text: str) -> list[float]:
        raise LLMError("Model 'qwen3-embedding:0.6b' not found")

    library = ExampleLibrary(tmp_path / "examples.json", broken)
    library.add(BONUS)
    events = list(make_agent(config, client, library).run(SALARY))

    assert [event.message for event in events if isinstance(event, ExamplesFailedEvent)] == [
        "Model 'qwen3-embedding:0.6b' not found"
    ]
    assert "".join(event.text for event in events if isinstance(event, TokenEvent)) == "Hello there "


def test_the_reasoning_steps_get_the_examples(config, client, library):
    config.features["reasoning"] = True
    list(make_agent(config, client, library).run(SALARY))
    step_messages = client.steps_seen[0][0]
    assert any("Message: Quel bonus" in message["content"] for message in step_messages if message["role"] == "system")
