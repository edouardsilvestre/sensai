import time
from datetime import datetime, timezone

import pytest

from sensai.core.agent import Agent
from sensai.core.cache import CachedAnswer, SemanticCache
from sensai.core.llm.ollama import LLMError, ModelNotFoundError
from sensai.core.rag.models import SearchResult
from sensai.interfaces.cli.main import (
    HELP,
    Session,
    chat_turn,
    confirm_cached,
    handle_command,
    load_cache,
    new_command,
)

QUESTION = "Qui dirige Orion ?"
CLOSE = "Qui dirige la société Orion ?"
ANSWER = "Marc Vidal dirige Orion."
STATUTES = SearchResult(1, "Gérant : Marc Vidal.", "p. 2", "statuts_orion.pdf", "", 0.9)
CACHED = CachedAnswer(QUESTION, ANSWER, [], datetime(2026, 10, 31, 15, 0, tzinfo=timezone.utc), 0.934)


def answer(monkeypatch, *answers):
    replies = iter(answers)
    questions = []

    def fake_input(prompt=""):
        questions.append(prompt)
        reply = next(replies)
        if reply is EOFError:
            raise EOFError
        return reply

    monkeypatch.setattr("builtins.input", fake_input)
    return questions


@pytest.fixture
def session(config, client):
    config.features["document"] = False
    config.cache_threshold = 70
    client.answer = ANSWER
    agent = Agent(config, client, confirm_cache=confirm_cached)
    session = Session(config, agent)
    session.cache = agent.cache = load_cache(config, client)
    yield session
    if session.cache is not None:
        session.cache.close()


@pytest.fixture
def paris(monkeypatch):
    monkeypatch.setenv("TZ", "Europe/Paris")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


@pytest.mark.parametrize("reply", ["y", "YES", " oui "])
def test_yes_uses_the_cached_answer(monkeypatch, reply):
    questions = answer(monkeypatch, reply)
    assert confirm_cached(CACHED) is True
    assert questions == ["Use this answer? [y]es / [n]o: "]


@pytest.mark.parametrize("reply", ["n", "No", "non"])
def test_no_lets_the_model_answer(monkeypatch, reply):
    answer(monkeypatch, reply)
    assert confirm_cached(CACHED) is False


def test_an_unclear_answer_asks_again(monkeypatch):
    questions = answer(monkeypatch, "", "maybe", "y")
    assert confirm_cached(CACHED) is True
    assert len(questions) == 3


def test_no_answer_lets_the_model_answer(monkeypatch):
    answer(monkeypatch, EOFError)
    assert confirm_cached(CACHED) is False


def test_the_cached_question_is_shown_with_its_local_date_and_similarity(monkeypatch, capsys, paris):
    answer(monkeypatch, "y")
    confirm_cached(CACHED)
    assert capsys.readouterr().out == (
        f'[cache] A close question was answered on 2026-10-31 at 16:00: "{QUESTION}" (similarity 93%)\n'
    )


def test_a_confirmed_close_question_is_answered_without_the_model(session, client, monkeypatch, capsys):
    chat_turn(session, QUESTION)
    new_command(session)
    answer(monkeypatch, "y")
    capsys.readouterr()

    chat_turn(session, CLOSE)

    out = capsys.readouterr().out
    assert f'"{QUESTION}"' in out and "[cache] Answer reused, the model was not called." in out
    assert ANSWER in out
    assert len(client.chats) == 1


def test_a_refused_cached_answer_is_given_by_the_model(session, client, monkeypatch, capsys):
    chat_turn(session, QUESTION)
    new_command(session)
    answer(monkeypatch, "n")

    chat_turn(session, CLOSE)

    assert "Answer reused" not in capsys.readouterr().out
    assert len(client.chats) == 2


def test_a_cache_failure_is_a_warning(session, client, capsys):
    def broken(text: str) -> list[float]:
        raise LLMError("Cannot reach Ollama")

    session.cache.embed = broken
    chat_turn(session, QUESTION)
    out, err = capsys.readouterr()
    assert "[warning] The cache could not be used: Cannot reach Ollama" in err
    assert ANSWER in out


def test_cache_command_shows_the_stats(session, config, monkeypatch, capsys):
    chat_turn(session, QUESTION)
    new_command(session)
    answer(monkeypatch, "y")
    chat_turn(session, CLOSE)
    capsys.readouterr()

    handle_command(session, "/cache")

    assert capsys.readouterr().out == (
        f"Cache: 1 answer kept, reused 1 time ({config.cache_path}, threshold 70%).\n"
    )


def test_cache_clear_removes_every_answer(session, capsys):
    session.cache.add(QUESTION, [1.0], "context", ANSWER, [])
    session.cache.add(CLOSE, [1.0], "context", ANSWER, [])

    handle_command(session, "/cache clear")

    assert capsys.readouterr().out == "Cache cleared: 2 answers removed.\n"
    assert session.cache.stats().answers == 0


def test_cache_command_says_when_the_feature_is_off(session, config, capsys):
    config.features["cache"] = False
    handle_command(session, "/cache")
    assert "The cache feature is off: /feature cache on to reuse them." in capsys.readouterr().out


def test_cache_command_without_a_cache(session, config, capsys):
    session.cache.close()
    session.cache = None
    handle_command(session, "/cache")
    assert capsys.readouterr().out == "The cache could not be opened: /feature cache on to try again.\n"
    config.features["cache"] = False
    handle_command(session, "/cache")
    assert "/feature cache on to reuse the answers to close questions" in capsys.readouterr().out


def test_cache_command_rejects_other_arguments(session, capsys):
    handle_command(session, "/cache empty")
    assert "[error] Usage: /cache [clear]" in capsys.readouterr().err


def test_feature_cache_on_opens_the_cache(config, client, capsys):
    config.features["cache"] = False
    session = Session(config, Agent(config, client, confirm_cache=confirm_cached))
    handle_command(session, "/feature cache on")
    assert isinstance(session.cache, SemanticCache) and session.agent.cache is session.cache
    session.cache.close()


def test_a_missing_embedding_model_disables_the_cache(config, client, capsys):
    def ensure_model(model: str) -> None:
        raise ModelNotFoundError(model, ["chat-model"])

    client.ensure_model = ensure_model
    assert load_cache(config, client) is None
    assert "[warning] Cache disabled: " in capsys.readouterr().err


def test_an_unreadable_cache_disables_it(config, client, capsys):
    config.cache_path.mkdir()
    assert load_cache(config, client) is None
    assert f"[warning] Cache disabled: Cannot open the cache at {config.cache_path}" in capsys.readouterr().err


class Retriever:
    def search(self, query: str, k: int, filters=(), report=None) -> list[SearchResult]:
        return [STATUTES]

    def fingerprint(self) -> str:
        return "v1"


def test_the_cached_sources_are_listed_again(session, client, monkeypatch, capsys):
    session.agent.retriever = Retriever()
    client.answer = "Marc Vidal dirige Orion [1]."
    chat_turn(session, QUESTION)
    new_command(session)
    answer(monkeypatch, "y")
    capsys.readouterr()

    chat_turn(session, CLOSE)

    assert "  [1] statuts_orion.pdf, p. 2" in capsys.readouterr().out
    assert session.last_sources == [STATUTES] and len(client.chats) == 1


def test_help_lists_the_cache_command():
    assert "/cache [clear]" in HELP
