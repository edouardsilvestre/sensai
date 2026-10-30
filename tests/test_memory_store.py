import pytest

from sensai.core.memory.store import Fact, MemoryStore, MemoryStoreError, Relation


@pytest.fixture
def store(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    yield store
    store.close()


def find(store: MemoryStore, name: str):
    [entity] = store.search(name)
    return entity


def test_adding_a_fact_creates_the_entity(store):
    fact_id = store.add_fact("Alvor Consulting", "Registered in Cyprus in 2019", "company")

    entity = find(store, "Alvor Consulting")
    assert entity.type == "company"
    assert entity.facts == [Fact(fact_id, "Registered in Cyprus in 2019")]


def test_entity_names_ignore_case(store):
    store.add_fact("Alvor Consulting", "first")
    store.add_fact("  alvor consulting ", "second")

    assert [entity.name for entity in store.search("alvor")] == ["Alvor Consulting"]
    assert [fact.content for fact in find(store, "ALVOR CONSULTING").facts] == ["first", "second"]


def test_the_type_is_only_filled_in_when_unknown(store):
    store.add_fact("Alvor", "first")
    store.add_fact("Alvor", "second", "company")
    store.add_fact("Alvor", "third", "person")

    assert find(store, "Alvor").type == "company"


def test_update_and_delete_a_fact(store):
    first = store.add_fact("Alvor", "Registered in 2019")
    second = store.add_fact("Alvor", "Based in Limassol")

    store.update_fact(first, "Registered in 2018")
    store.delete_fact(second)

    assert find(store, "Alvor").facts == [Fact(first, "Registered in 2018")]


def test_unknown_fact_ids_are_errors(store):
    with pytest.raises(MemoryStoreError, match="No fact with id 42"):
        store.update_fact(42, "anything")
    with pytest.raises(MemoryStoreError, match="No fact with id 42"):
        store.delete_fact(42)


def test_empty_values_are_errors(store):
    with pytest.raises(MemoryStoreError, match="entity name is empty"):
        store.add_fact(" ", "a fact")
    with pytest.raises(MemoryStoreError, match="fact is empty"):
        store.add_fact("Alvor", " ")
    assert store.search("Alvor") == []


def test_link_creates_the_entities_and_is_seen_from_both_sides(store):
    assert store.link("Jean Dupont", "runs", "Alvor Consulting")

    relation = Relation("Jean Dupont", "runs", "Alvor Consulting")
    assert find(store, "Jean Dupont").relations == [relation]
    assert find(store, "Alvor Consulting").relations == [relation]


def test_the_same_relation_is_stored_once(store):
    store.link("Jean Dupont", "runs", "Alvor")
    assert not store.link("jean dupont", "RUNS", "alvor")

    assert len(find(store, "Alvor").relations) == 1


def test_unlink(store):
    store.link("Jean Dupont", "runs", "Alvor")
    store.unlink("jean dupont", "runs", "Alvor")

    assert find(store, "Alvor").relations == []
    with pytest.raises(MemoryStoreError, match="No relation"):
        store.unlink("Jean Dupont", "runs", "Alvor")


def test_search_matches_part_of_a_name_or_a_fact(store):
    store.add_fact("Alvor Consulting", "Registered in Cyprus")
    store.add_fact("Delta Holding", "Paid Alvor 2 million euros")
    store.add_fact("Jean Dupont", "Journalist")

    assert [entity.name for entity in store.search("alvor")] == ["Alvor Consulting", "Delta Holding"]
    assert [entity.name for entity in store.search("CYPRUS")] == ["Alvor Consulting"]
    assert store.search("nothing") == []


def test_the_memory_survives_a_restart(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    store.add_fact("Alvor", "Registered in 2019")
    store.link("Jean Dupont", "runs", "Alvor")
    store.close()

    store = MemoryStore(tmp_path / "memory.db")
    entity = find(store, "Alvor")
    store.close()
    assert [fact.content for fact in entity.facts] == ["Registered in 2019"]
    assert entity.relations == [Relation("Jean Dupont", "runs", "Alvor")]


def test_an_invalid_file_is_an_error(tmp_path):
    path = tmp_path / "memory.db"
    path.write_text("not a database")

    with pytest.raises(MemoryStoreError, match="Cannot use"):
        MemoryStore(path)
