import json

import pytest

from sensai.core.config import Config
from sensai.core.document.document import Edit
from sensai.core.memory.conversation import ConversationMemory
from sensai.core.memory.sessions import SessionStore


@pytest.fixture
def memory() -> ConversationMemory:
    memory = ConversationMemory(Config())
    for question in ("first", "second", "third"):
        memory.commit(question, f"answer to {question}")
        memory.document.apply([Edit("append", "Timeline", f"- {question}")])
    return memory


def questions(memory: ConversationMemory) -> list[str]:
    return [turn.user for turn in memory.turns]


def test_a_branch_keeps_the_turns_before_the_edited_one(memory):
    memory.fork(2, "other second")
    assert questions(memory) == ["first"]
    assert (memory.branch.number, memory.branch.parent, memory.branch.turn) == (2, 1, 2)


def test_the_original_branch_stays_as_it_was(memory):
    memory.fork(2, "other second")
    memory.commit("other second", "other answer")
    assert memory.switch(1)
    assert questions(memory) == ["first", "second", "third"]
    assert memory.switch(2)
    assert questions(memory) == ["first", "other second"]


def test_a_branch_starts_from_the_document_before_the_edited_turn(memory):
    memory.fork(2, "other second")
    assert memory.document.section("Timeline").body == "- first"


def test_the_document_of_a_branch_does_not_reach_the_others(memory):
    memory.fork(2, "other second")
    memory.document.apply([Edit("append", "Timeline", "- only here")])
    memory.switch(1)
    assert memory.document.section("Timeline").body == "- first\n- second\n- third"


def test_the_undo_history_of_a_document_follows_its_branch(memory):
    memory.fork(2, "other second")
    memory.switch(1)
    assert memory.document.undo()
    assert memory.document.section("Timeline").body == "- first\n- second"


def test_a_branch_keeps_the_summary_of_the_older_turns(memory):
    memory.summary = "Older facts"
    memory.fork(1, "other first")
    assert memory.summary == "Older facts" and memory.turns == []


def test_the_compaction_of_a_branch_leaves_the_others_alone(memory):
    memory.fork(3, "other third")
    memory.compact(lambda previous, turns: "Condensed")
    memory.switch(1)
    assert memory.summary == "" and questions(memory) == ["first", "second", "third"]


def test_branches_are_numbered_and_can_fork_from_any_branch(memory):
    memory.fork(2, "b")
    memory.commit("b", "answer")
    memory.fork(1, "c")
    memory.switch(1)
    memory.fork(3, "d")
    assert [branch.number for branch in memory.branches()] == [1, 2, 3, 4]
    assert [branch.parent for branch in memory.branches()] == [None, 1, 2, 1]


def test_an_unknown_branch_changes_nothing(memory):
    assert not memory.switch(7)
    assert not memory.switch(1)
    assert questions(memory) == ["first", "second", "third"]


def test_a_discarded_branch_is_forgotten(memory):
    memory.fork(2, "other second")
    memory.discard_branch()
    assert memory.branch.number == 1 and questions(memory) == ["first", "second", "third"]
    assert [branch.number for branch in memory.branches()] == [1]


def test_clear_forgets_the_branches(memory):
    memory.fork(2, "other second")
    memory.clear()
    assert [branch.number for branch in memory.branches()] == [1] and memory.turns == []


def test_branches_are_saved_and_resumed(memory, tmp_path):
    store = SessionStore(tmp_path)
    memory.fork(2, "other second")
    memory.commit("other second", "other answer")
    store.save("case", memory, "m")

    resumed = ConversationMemory(Config())
    assert store.load("case", resumed) == 2
    assert (resumed.branch.number, resumed.branch.parent, resumed.branch.label) == (2, 1, "other second")
    assert questions(resumed) == ["first", "other second"]
    assert resumed.switch(1)
    assert questions(resumed) == ["first", "second", "third"]
    assert resumed.document.section("Timeline").body == "- first\n- second\n- third"
    # The document of each turn is saved too, for a branch made after resuming.
    resumed.fork(3, "other third")
    assert resumed.document.section("Timeline").body == "- first\n- second"


def test_a_session_saved_before_the_branches_is_resumed_with_one_branch(memory, tmp_path):
    store = SessionStore(tmp_path)
    store.save("old", memory, "m")
    path = tmp_path / "old.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    del data["branch"], data["other_branches"]
    for turn in data["turns"]:
        del turn["document"]
    path.write_text(json.dumps(data), encoding="utf-8")

    resumed = ConversationMemory(Config())
    store.load("old", resumed)
    assert [branch.number for branch in resumed.branches()] == [1]
    resumed.fork(2, "other")
    assert resumed.document.is_empty
