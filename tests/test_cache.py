import re
from datetime import datetime, timezone

import pytest

from sensai.core.cache import CacheError, CacheStats, SemanticCache
from sensai.core.rag.models import SearchResult

WORDS = ["qui", "dirige", "la", "société", "orion", "quel", "montant", "a", "reçu", "vidal", "gérant"]
CONTEXT = "persona=investigator"
STATUTES = SearchResult(1, "Gérant : Marc Vidal.", "p. 2", "statuts_orion.pdf", "Title: Statuts", 0.9)


def embed(text: str) -> list[float]:
    words = re.findall(r"\w+", text.lower())
    return [float(words.count(word)) for word in WORDS]


@pytest.fixture
def cache(tmp_path):
    cache = SemanticCache(tmp_path / "cache" / "cache.db", embed, threshold=0.7)
    yield cache
    cache.close()


def remember(cache, question="Qui dirige Orion ?", context=CONTEXT, answer="Marc Vidal [1].", sources=(STATUTES,)):
    cache.add(question, cache.embed(question), context, answer, list(sources))


def test_an_empty_cache_finds_nothing(cache):
    assert cache.find(embed("Qui dirige Orion ?"), CONTEXT) is None


def test_the_same_question_is_found_with_its_answer(cache):
    remember(cache)
    found = cache.find(embed("Qui dirige Orion ?"), CONTEXT)
    assert (found.question, found.answer) == ("Qui dirige Orion ?", "Marc Vidal [1].")
    assert found.similarity == pytest.approx(1.0)
    assert found.created_at.tzinfo == timezone.utc and found.created_at <= datetime.now(timezone.utc)


def test_a_close_question_is_found_but_not_a_different_one(cache):
    remember(cache)
    assert cache.find(embed("Qui dirige la société Orion ?"), CONTEXT).answer == "Marc Vidal [1]."
    assert cache.find(embed("Quel montant a reçu Vidal ?"), CONTEXT) is None


def test_the_closest_question_is_kept(cache):
    remember(cache, "Qui dirige la société ?", answer="Unclear.")
    remember(cache)
    assert cache.find(embed("Qui dirige Orion ?"), CONTEXT).answer == "Marc Vidal [1]."


def test_the_same_question_in_another_context_is_not_found(cache):
    remember(cache)
    assert cache.find(embed("Qui dirige Orion ?"), "persona=fact_checker") is None


def test_the_sources_come_back_as_they_were(cache):
    remember(cache)
    assert cache.find(embed("Qui dirige Orion ?"), CONTEXT).sources == [STATUTES]


def test_each_use_is_counted(cache):
    remember(cache)
    remember(cache, "Quel montant a reçu Vidal ?", answer="480 000 EUR.", sources=())
    cache.find(embed("Qui dirige Orion ?"), CONTEXT)
    cache.find(embed("Qui dirige la société Orion ?"), CONTEXT)
    assert cache.stats() == CacheStats(answers=2, uses=2)


def test_clear_empties_the_cache(cache):
    remember(cache)
    remember(cache, "Quel montant a reçu Vidal ?")
    assert cache.clear() == 2
    assert cache.stats() == CacheStats(answers=0, uses=0)
    assert cache.find(embed("Qui dirige Orion ?"), CONTEXT) is None


def test_the_cache_is_kept_between_two_runs(tmp_path):
    path = tmp_path / "cache.db"
    first = SemanticCache(path, embed, threshold=0.7)
    remember(first)
    first.close()
    second = SemanticCache(path, embed, threshold=0.7)
    assert second.find(embed("Qui dirige Orion ?"), CONTEXT).answer == "Marc Vidal [1]."
    second.close()


def test_a_corrupted_file_is_refused(tmp_path):
    path = tmp_path / "cache.db"
    path.write_bytes(b"not a database" * 100)
    with pytest.raises(CacheError, match="Cannot open the cache"):
        SemanticCache(path, embed, threshold=0.7)
