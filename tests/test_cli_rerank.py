from sensai.core.agent import Agent
from sensai.core.rag.models import SearchResult
from sensai.interfaces.cli.main import HELP, Session, chat_turn, handle_command

KEPT = SearchResult(1, "Alice approved the transfer of March.", "", "leak/march.eml", "", score=0.9, relevance=0.97)
NEAR_MISS = SearchResult(2, "Bob approved the transfer of May.", "", "leak/may.eml", "", score=0.8, relevance=0.04)


class Retriever:
    def __init__(self, candidates: int, kept: list[SearchResult], dropped: list[SearchResult]):
        self.candidates, self.kept, self.dropped = candidates, kept, dropped

    def search(self, query: str, k: int, filters=(), report=None) -> list[SearchResult]:
        report.candidates, report.kept, report.dropped = self.candidates, len(self.kept), list(self.dropped)
        return list(self.kept)


def session(config, client, retriever: Retriever) -> Session:
    return Session(config, Agent(config, client, retriever))


def test_the_counts_of_the_reranker_are_shown(config, client, capsys):
    config.rag_top_k = 1
    chat_turn(session(config, client, Retriever(30, [KEPT], [NEAR_MISS] * 9)), "Who approved the transfer of March?")
    assert (
        "[rerank] 30 candidates: 1 kept, 20 left out after the 1 best, 9 dropped under 20% relevance (/sources dropped)"
        in capsys.readouterr().out
    )


def test_only_the_parts_that_happened_are_shown(config, client, capsys):
    chat_turn(session(config, client, Retriever(1, [KEPT], [])), "Who approved the transfer of March?")
    assert "[rerank] 1 candidate: 1 kept\n" in capsys.readouterr().out


class UnrankedRetriever:
    def search(self, query: str, k: int, filters=(), report=None) -> list[SearchResult]:
        report.rerank_error = "The rerank feature needs sentence-transformers: pip install -r requirements.txt"
        return [KEPT]


def test_a_search_not_reranked_is_said_and_answered(config, client, capsys):
    chat_turn(session(config, client, UnrankedRetriever()), "Who approved the transfer of March?")
    captured = capsys.readouterr()
    assert (
        "Excerpts not reranked until Sensai restarts: "
        "The rerank feature needs sentence-transformers: pip install -r requirements.txt" in captured.err
    )
    assert "[rerank]" not in captured.out
    assert "[search] 1 excerpts from 1 documents" in captured.out


def test_sources_dropped_shows_the_excerpts_the_model_did_not_get(config, client, capsys):
    chatting = session(config, client, Retriever(2, [KEPT], [NEAR_MISS]))
    chat_turn(chatting, "Who approved the transfer of March?")
    capsys.readouterr()

    handle_command(chatting, "/sources dropped")
    out = capsys.readouterr().out
    assert "- leak/may.eml (relevance 0.04, similarity 0.80)" in out
    assert "Bob approved the transfer of May." in out
    assert "march.eml" not in out

    handle_command(chatting, "/sources")
    assert "[1] leak/march.eml (relevance 0.97, similarity 0.90)" in capsys.readouterr().out


def test_a_new_message_forgets_the_dropped_excerpts(config, client, capsys):
    retriever = Retriever(2, [KEPT], [NEAR_MISS])
    chatting = session(config, client, retriever)
    chat_turn(chatting, "Who approved the transfer of March?")
    retriever.dropped = []
    chat_turn(chatting, "And who signed it?")
    capsys.readouterr()

    handle_command(chatting, "/sources dropped")
    assert capsys.readouterr().out == "No excerpt was dropped for the last message.\n"


def test_sources_with_another_argument_shows_the_usage(config, client, capsys):
    handle_command(session(config, client, Retriever(0, [], [])), "/sources everything")
    assert "Usage: /sources [dropped]" in capsys.readouterr().err


def test_help_lists_sources_dropped():
    assert "/sources dropped" in HELP
