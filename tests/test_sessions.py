import json
import os

import pytest

from sensai.core.config import Config
from sensai.core.memory.conversation import ConversationMemory, Turn
from sensai.core.memory.sessions import SessionError, SessionStore


@pytest.fixture
def store(tmp_path) -> SessionStore:
    return SessionStore(tmp_path / "sessions")


def memory_with(*turns: Turn, summary: str = "") -> ConversationMemory:
    memory = ConversationMemory(Config())
    memory.turns, memory.summary = list(turns), summary
    return memory


def test_saved_conversation_is_restored(store):
    turn = Turn("Who approved the transfer?", "Marc Vidal [1].", ["leak/mail.eml"])
    store.save("case", memory_with(turn, summary="Alvor is a shell company."), "qwen3")

    memory = ConversationMemory(Config())
    assert store.load("case", memory) == 1
    assert memory.turns == [turn]
    assert memory.summary == "Alvor is a shell company."


def test_file_is_readable_json(store):
    store.save("case", memory_with(Turn("Qui a signé ?", "Marc Vidal.")), "qwen3")
    data = json.loads((store.directory / "case.json").read_text(encoding="utf-8"))
    assert data["model"] == "qwen3" and data["turns"][0]["user"] == "Qui a signé ?"


def test_saving_again_replaces_the_conversation(store):
    store.save("case", memory_with(Turn("a", "b")), "m")
    store.save("case", memory_with(Turn("a", "b"), Turn("c", "d")), "m")
    assert [info.turns for info in store.list()] == [2]


def test_sessions_are_listed_most_recent_first(store):
    store.save("old", memory_with(Turn("First question about the leak", "answer")), "m")
    store.save("new", memory_with(Turn("Second", "answer")), "m")
    old = store.directory / "old.json"
    data = json.loads(old.read_text(encoding="utf-8"))
    data["updated"] = "2020-01-01T10:00:00"
    old.write_text(json.dumps(data), encoding="utf-8")

    assert [info.id for info in store.list()] == ["new", "old"]
    assert store.list()[1].title == "First question about the leak"
    assert store.latest() == "new"


def test_long_title_is_shortened(store):
    store.save("case", memory_with(Turn("word " * 40, "answer")), "m")
    title = store.list()[0].title
    assert len(title) == 60 and title.endswith("...")


def test_no_session_yet(store):
    assert store.list() == [] and store.latest() is None


def test_unreadable_file_is_skipped_in_the_list(store):
    store.save("good", memory_with(Turn("a", "b")), "m")
    (store.directory / "broken.json").write_text("{not json", encoding="utf-8")
    assert [info.id for info in store.list()] == ["good"]
    with pytest.raises(SessionError, match="Cannot read"):
        store.load("broken", ConversationMemory(Config()))


def test_unknown_session_leaves_the_memory_unchanged(store):
    memory = memory_with(Turn("a", "b"))
    with pytest.raises(SessionError, match="No saved conversation 'missing'"):
        store.load("missing", memory)
    assert memory.turns == [Turn("a", "b")]


@pytest.mark.parametrize("session_id", ["../secret", "a/b", ""])
def test_ids_cannot_leave_the_folder(store, session_id):
    with pytest.raises(SessionError, match="Invalid session id"):
        store.load(session_id, ConversationMemory(Config()))


def test_new_ids_do_not_overwrite_a_session(store):
    first = store.new_id()
    store.save(first, memory_with(Turn("a", "b")), "m")
    assert store.new_id() != first


def test_failed_save_is_reported(store, monkeypatch):
    def fail(source, target):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(SessionError, match="disk full"):
        store.save("case", memory_with(Turn("a", "b")), "m")
