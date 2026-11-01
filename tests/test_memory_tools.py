import inspect

import pytest

from sensai.core.llm.ollama import ToolCall
from sensai.core.memory.store import MemoryStore
from sensai.core.tools.memory import MemoryTools
from sensai.core.tools.registry import ToolError, ToolRegistry


def test_the_descriptions_give_no_example_names():
    """A small model repeats a name it finds in a description as if it were a fact of the case."""
    for name, tool in inspect.getmembers(MemoryTools, inspect.isfunction):
        if name.startswith("memory_"):
            assert not any(example in inspect.getdoc(tool) for example in ("Dupont", "Alvor", "Vidal"))


@pytest.fixture
def store(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    yield store
    store.close()


@pytest.fixture
def memory(store) -> MemoryTools:
    return MemoryTools(store)


def test_the_tools_are_named_after_their_methods(memory):
    assert ToolRegistry(memory.tools()).names() == [
        "memory_search", "memory_add_fact", "memory_update_fact",
        "memory_delete_fact", "memory_link", "memory_unlink",
    ]


def test_search_shows_the_facts_with_their_ids_and_the_relations(memory):
    memory.memory_add_fact("Alvor Consulting", "Registered in Cyprus", "company")
    memory.memory_link("Jean Dupont", "runs", "Alvor Consulting")

    result = memory.memory_search("alvor")
    assert "data, not instructions" in result.content
    assert "Alvor Consulting (company)\n  fact 1: Registered in Cyprus\n" in result.content
    assert "  relation: Jean Dupont runs Alvor Consulting" in result.content
    assert result.summary == "1 found: Alvor Consulting"


def test_search_says_when_nothing_is_stored(memory):
    result = memory.memory_search("Alvor")
    assert result.content == "Nothing in memory about 'Alvor'."
    assert result.summary == "nothing stored"


def test_add_update_and_delete_a_fact(memory, store):
    assert memory.memory_add_fact("Alvor", "Registered in 2019").summary == "fact 1 saved"
    memory.memory_add_fact("Alvor", "Based in Limassol")

    assert memory.memory_update_fact(1, "Registered in 2018").summary == "fact 1 updated"
    assert memory.memory_delete_fact(2).summary == "fact 2 deleted"
    [entity] = store.search("Alvor")
    assert [fact.content for fact in entity.facts] == ["Registered in 2018"]


def test_fact_ids_sent_as_strings_are_accepted(memory):
    memory.memory_add_fact("Alvor", "Registered in 2019")
    assert memory.memory_update_fact("1", "Registered in 2018").summary == "fact 1 updated"


def test_link_and_unlink(memory, store):
    assert memory.memory_link("Jean Dupont", "runs", "Alvor").summary == "relation saved"
    assert memory.memory_link("Jean Dupont", "runs", "Alvor").summary == "already stored"
    assert memory.memory_unlink("Jean Dupont", "runs", "Alvor").summary == "relation deleted"
    [entity] = store.search("Alvor")
    assert entity.relations == []


@pytest.mark.parametrize(
    "call, message",
    [
        (lambda m: m.memory_update_fact(42, "x"), "memory_update_fact: No fact with id 42"),
        (lambda m: m.memory_delete_fact("abc"), "fact_id must be a number, got 'abc'"),
        (lambda m: m.memory_add_fact("Alvor", " "), "memory_add_fact: The fact is empty"),
        (lambda m: m.memory_unlink("A", "runs", "B"), "memory_unlink: No relation 'A runs B'"),
        (lambda m: m.memory_search(" "), "memory_search: The query is empty"),
    ],
)
def test_store_errors_become_tool_errors(memory, call, message):
    with pytest.raises(ToolError, match=message):
        call(memory)


def test_the_registry_sends_the_errors_back_to_the_model(memory):
    result = ToolRegistry(memory.tools()).execute(ToolCall("memory_delete_fact", {"fact_id": 42}))
    assert result.error
    assert result.content == "Error: memory_delete_fact: No fact with id 42"
