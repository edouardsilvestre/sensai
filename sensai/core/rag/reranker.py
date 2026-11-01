import math
from collections.abc import Callable
from dataclasses import replace

from sensai.core.rag.models import SearchResult

# Scores each (question, passage) pair: the higher, the better the passage answers the question.
Scorer = Callable[[list[tuple[str, str]]], list[float]]


class RerankError(Exception):
    pass


class Reranker:
    """
    Reads the question and each candidate together with a cross-encoder, and scores from 0 to 1 whether the candidate
    answers the question. Much slower than comparing vectors, so it only reads the candidates of the search.
    """

    def __init__(self, model: str, max_tokens: int = 512, scorer: Scorer | None = None):
        self.model = model
        self.max_tokens = max_tokens
        # Loaded at the first search: torch takes seconds to import, and the model to load.
        self._scorer = scorer

    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        if not results:
            return []
        # The passage with its title and people, like the model reads it: "the email of Dupont" is in the header.
        pairs = [(query, "\n\n".join(filter(None, [result.description, result.text]))) for result in results]
        scores = self._score(pairs)
        if len(scores) != len(results):
            raise RerankError(f"The rerank model returned {len(scores)} scores for {len(results)} excerpts")
        scored = [replace(result, relevance=_sigmoid(score)) for result, score in zip(results, scores)]
        return sorted(scored, key=lambda result: result.relevance, reverse=True)

    def _score(self, pairs: list[tuple[str, str]]) -> list[float]:
        if self._scorer is None:
            self._scorer = self._load()
        return self._scorer(pairs)

    def _load(self) -> Scorer:
        try:
            from sentence_transformers import CrossEncoder
            from torch.nn import Identity
        except ImportError as e:
            raise RerankError("The rerank feature needs sentence-transformers: pip install -r requirements.txt") from e
        try:
            model = CrossEncoder(self.model, max_length=self.max_tokens)
        except OSError as e:
            raise RerankError(f"Cannot load the rerank model '{self.model}': {e}") from e

        # The raw scores of every model, made 0 to 1 by _sigmoid: some models apply a sigmoid themselves, others not.
        def score(pairs: list[tuple[str, str]]) -> list[float]:
            return [float(value) for value in model.predict(pairs, activation_fn=Identity(), show_progress_bar=False)]

        return score


def _sigmoid(score: float) -> float:
    # Written for both signs: math.exp(-score) overflows for a very negative score.
    if score >= 0:
        return 1 / (1 + math.exp(-score))
    return math.exp(score) / (1 + math.exp(score))
