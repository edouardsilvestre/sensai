import sys
from types import SimpleNamespace

import pytest

from sensai.core.rag.models import SearchResult
from sensai.core.rag.reranker import Reranker, RerankError


def result(chunk_id: int, text: str = "", description: str = "") -> SearchResult:
    return SearchResult(chunk_id, text or f"chunk {chunk_id}", "", "doc.txt", description, score=0.5)


def test_the_candidates_are_sorted_by_relevance():
    reranker = Reranker("fake", scorer=lambda pairs: [-2.0, 3.0, 0.0])
    reranked = reranker.rerank("question", [result(1), result(2), result(3)])
    assert [r.chunk_id for r in reranked] == [2, 3, 1]
    assert [round(r.relevance, 2) for r in reranked] == [0.95, 0.5, 0.12]
    assert {r.score for r in reranked} == {0.5}


def test_the_reranker_reads_the_question_with_the_header_and_the_text():
    read = []
    reranker = Reranker("fake", scorer=lambda pairs: read.extend(pairs) or [0.0] * len(pairs))
    reranker.rerank("the email of Dupont", [result(1, "The transfer is approved.", "From: Dupont")])
    assert read == [("the email of Dupont", "From: Dupont\n\nThe transfer is approved.")]


def test_extreme_scores_stay_from_0_to_1():
    reranker = Reranker("fake", scorer=lambda pairs: [1000.0, -1000.0])
    assert [r.relevance for r in reranker.rerank("question", [result(1), result(2)])] == [1.0, 0.0]


def test_no_candidates_do_not_load_the_model(monkeypatch):
    monkeypatch.setattr(Reranker, "_load", lambda self: pytest.fail("the model was loaded"))
    assert Reranker("model").rerank("question", []) == []


def test_the_model_is_loaded_once(monkeypatch):
    loads = []
    monkeypatch.setattr(Reranker, "_load", lambda self: loads.append(1) or (lambda pairs: [0.0] * len(pairs)))
    reranker = Reranker("model")
    reranker.rerank("question", [result(1)])
    reranker.rerank("question", [result(2)])
    assert loads == [1]


def test_a_wrong_number_of_scores_is_an_error():
    with pytest.raises(RerankError, match="1 scores for 2 excerpts"):
        Reranker("fake", scorer=lambda pairs: [0.0]).rerank("question", [result(1), result(2)])


@pytest.fixture
def real_load(monkeypatch):
    # Undoes the fake model of conftest, then fakes the libraries instead: importing torch takes seconds.
    monkeypatch.undo()
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "torch.nn", SimpleNamespace(Identity=lambda: "identity"))
    return monkeypatch


def test_a_model_that_cannot_be_loaded_is_a_rerank_error(real_load):
    def missing(name, max_length):
        raise OSError("Repository Not Found")

    real_load.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=missing))
    with pytest.raises(RerankError, match="Cannot load the rerank model 'nobody/nothing': Repository Not Found"):
        Reranker("nobody/nothing").rerank("question", [result(1)])


def test_the_raw_scores_of_the_model_are_asked(real_load):
    asked = {}

    class CrossEncoder:
        def __init__(self, name, max_length):
            asked.update(name=name, max_length=max_length)

        def predict(self, pairs, activation_fn, show_progress_bar):
            asked["activation_fn"] = activation_fn
            return [2.0 for _ in pairs]

    real_load.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=CrossEncoder))
    [reranked] = Reranker("some/model", max_tokens=256).rerank("question", [result(1)])
    assert asked == {"name": "some/model", "max_length": 256, "activation_fn": "identity"}
    assert reranked.relevance == pytest.approx(0.88, abs=0.01)


def test_without_sentence_transformers_the_error_says_what_to_install(real_load):
    real_load.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(RerankError, match="pip install -r requirements.txt"):
        Reranker("model").rerank("question", [result(1)])
