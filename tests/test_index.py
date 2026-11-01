import os

import pytest

from sensai.core.rag.chunker import Chunker, join_chunks
from sensai.core.rag.index import DocumentIndex
from sensai.core.rag.models import Document, Section
from sensai.core.rag.models import SearchFilter, SearchReport
from sensai.core.rag.reranker import Reranker
from sensai.core.rag.store import IndexMismatchError


@pytest.fixture
def corpus(tmp_path):
    folder = tmp_path / "leak"
    (folder / "emails").mkdir(parents=True)
    (folder / "emails" / "transfer.txt").write_text("Alice approved the transfer to Delta Holding.", encoding="utf-8")
    (folder / "minutes.md").write_text("# Board minutes\nThe board discussed the annual budget.", encoding="utf-8")
    (folder / "accounts.csv").write_text("owner,bank\nDelta Holding,Geneva Private Bank\n", encoding="utf-8")
    (folder / "photo.png").write_bytes(b"\x89PNG")
    (folder / ".hidden.txt").write_text("should not be indexed", encoding="utf-8")
    return folder


@pytest.fixture
def index(config, client):
    index = DocumentIndex.open(config, client)
    yield index
    index.close()


def test_ingest_indexes_supported_visible_files(index, corpus):
    report = index.ingest([corpus])
    assert (report.added, report.unsupported, report.failed) == (3, 1, [])
    assert index.stats().documents == 3


def test_search_finds_the_relevant_document(index, corpus):
    index.ingest([corpus])
    results = index.search("who approved the transfer to Delta?", k=1)
    assert results[0].source == "leak/emails/transfer.txt"


def test_only_the_chunk_text_is_embedded(index, corpus, client):
    index.ingest([corpus])
    assert "Alice approved the transfer to Delta Holding." in client.embedded
    assert not any("leak/" in text for text in client.embedded)


def test_unchanged_files_are_not_embedded_again(index, corpus, client):
    index.ingest([corpus])
    embedded = len(client.embedded)

    report = index.ingest([corpus])
    assert (report.added, report.unchanged) == (0, 3)
    assert len(client.embedded) == embedded


def test_touched_file_with_same_content_is_unchanged(index, corpus, client):
    index.ingest([corpus])
    embedded = len(client.embedded)
    os.utime(corpus / "minutes.md", (1, 1))

    assert index.ingest([corpus]).unchanged == 3
    assert len(client.embedded) == embedded


def test_modified_file_is_reindexed(index, corpus):
    index.ingest([corpus])
    (corpus / "minutes.md").write_text("The board approved a loan to Omega.", encoding="utf-8")

    report = index.ingest([corpus])
    assert (report.updated, report.unchanged) == (1, 2)
    assert "Omega" in index.search("loan to Omega", k=1)[0].text


def test_the_fingerprint_changes_when_a_file_changes(index, corpus):
    index.ingest([corpus])
    before = index.fingerprint()
    index.ingest([corpus])
    assert index.fingerprint() == before
    (corpus / "minutes.md").write_text("The board approved a loan to Omega.", encoding="utf-8")
    index.ingest([corpus])
    assert index.fingerprint() != before


def test_deleted_file_is_removed_from_the_index(index, corpus):
    index.ingest([corpus])
    (corpus / "accounts.csv").unlink()

    assert index.ingest([corpus]).removed == 1
    assert index.stats().documents == 2


def test_unreadable_file_is_reported_without_stopping(index, corpus):
    (corpus / "broken.pdf").write_bytes(b"garbage")
    report = index.ingest([corpus])
    assert report.added == 3
    assert [name for name, _ in report.failed] == ["leak/broken.pdf"]


def test_single_files_and_missing_paths(index, corpus, tmp_path):
    report = index.ingest([corpus / "minutes.md", corpus / "photo.png", tmp_path / "nope"])
    assert report.added == 1
    assert len(report.failed) == 2
    assert index.search("budget", k=1)[0].source == "minutes.md"


def test_progress_is_reported_per_file(index, corpus):
    seen = []
    index.ingest([corpus], on_progress=seen.append)
    assert [(p.position, p.total) for p in seen] == [(1, 3), (2, 3), (3, 3)]
    assert {p.status for p in seen} == {"added"}


def test_embedding_window_fits_a_chunk(index, corpus, client, config):
    index.ingest([corpus])
    assert client.embed_options == {"num_ctx": config.chunk_size}
    index.search("budget", k=1)
    assert client.embed_options == {"num_ctx": config.chunk_size}


def test_empty_index_search_does_not_embed(index, client):
    assert index.search("anything", k=3) == []
    assert client.embedded == []


def test_changing_the_embedding_model_needs_a_rebuild(config, client, corpus):
    index = DocumentIndex.open(config, client)
    index.ingest([corpus])
    index.close()

    config.embed_model = "another-model"
    with pytest.raises(IndexMismatchError):
        DocumentIndex.open(config, client)
    rebuilt = DocumentIndex.open(config, client, rebuild=True)
    assert rebuilt.stats().documents == 0
    rebuilt.close()


def test_a_filter_limits_the_search_to_its_documents(index, corpus, client, config):
    # The spreadsheet does not answer the question: the threshold would drop it.
    config.rerank_threshold = 0
    index.ingest([corpus])
    results = index.search("who approved the transfer to Delta?", k=3, filters=[SearchFilter(kind="spreadsheet")])
    assert [result.source for result in results] == ["leak/accounts.csv"]
    assert index.count_documents([SearchFilter(source="leak/emails/")]) == 1
    assert index.count_documents([]) == 3

    client.embedded.clear()
    assert index.search("transfer", k=3, filters=[SearchFilter(author="nobody")]) == []
    assert client.embedded == []


@pytest.fixture
def accounts(tmp_path):
    folder = tmp_path / "accounts"
    folder.mkdir()
    (folder / "a.txt").write_text("transfer transfer transfer", encoding="utf-8")
    (folder / "b.txt").write_text("Account FR7630004000 opened at the Geneva branch for the holding company", encoding="utf-8")
    (folder / "c.txt").write_text("the transfer was approved", encoding="utf-8")
    return folder


def test_the_hybrid_search_finds_what_the_vectors_miss(config, client, accounts):
    config.rag_candidates = 2
    index = DocumentIndex.open(config, client)
    index.ingest([accounts])
    assert "accounts/b.txt" in [r.source for r in index.search("FR7630004000 transfer", k=2)]

    config.features["hybrid"] = False
    assert [r.source for r in index.search("FR7630004000 transfer", k=2)] == ["accounts/a.txt", "accounts/c.txt"]
    index.close()


def test_the_reranker_puts_first_the_chunk_that_answers(index, corpus, config):
    index.ingest([corpus])
    index.reranker = Reranker("fake", scorer=lambda pairs: [5.0 if "Geneva" in passage else -5.0 for _, passage in pairs])
    [best] = index.search("who approved the transfer to Delta?", k=1)
    assert best.source == "leak/accounts.csv"
    assert best.relevance > 0.99

    config.features["rerank"] = False
    [best] = index.search("who approved the transfer to Delta?", k=1)
    assert (best.source, best.relevance) == ("leak/emails/transfer.txt", None)


def test_the_reranker_reads_every_candidate_but_keeps_k(index, corpus, config):
    index.ingest([corpus])
    read = []
    index.reranker = Reranker("fake", scorer=lambda pairs: read.extend(pairs) or [0.0] * len(pairs))
    config.rag_candidates = 3
    assert len(index.search("transfer budget bank", k=1)) == 1
    assert len(read) == 3


def test_the_chunks_under_the_threshold_are_dropped_and_reported(index, corpus, config):
    index.ingest([corpus])
    index.reranker = Reranker("fake", scorer=lambda pairs: [5.0 if "Alice" in passage else -5.0 for _, passage in pairs])
    report = SearchReport()
    results = index.search("who approved the transfer to Delta?", k=3, report=report)
    assert [result.source for result in results] == ["leak/emails/transfer.txt"]
    assert report.candidates == 3
    assert sorted(result.source for result in report.dropped) == ["leak/accounts.csv", "leak/minutes.md"]
    assert all(result.relevance < 0.2 for result in report.dropped)


def test_a_threshold_of_0_keeps_every_chunk(index, corpus, config):
    index.ingest([corpus])
    index.reranker = Reranker("fake", scorer=lambda pairs: [-20.0] * len(pairs))
    config.rerank_threshold = 0
    report = SearchReport()
    assert len(index.search("transfer", k=3, report=report)) == 3
    assert report.dropped == []


def test_without_the_reranker_nothing_is_dropped(index, corpus, config):
    index.ingest([corpus])
    index.reranker = Reranker("fake", scorer=lambda pairs: [-20.0] * len(pairs))
    config.features["rerank"] = False
    report = SearchReport()
    assert len(index.search("transfer", k=3, report=report)) == 3
    assert (report.candidates, report.dropped) == (0, [])


# Sentences shorter than the overlap of the test config, so that the chunks overlap.
STORY = " ".join(f"Event {i} on day {i}." for i in range(60))


@pytest.fixture
def story(tmp_path, config):
    """One document of many chunks, and those chunks as the index cuts them."""
    folder = tmp_path / "story"
    folder.mkdir()
    (folder / "story.txt").write_text(STORY, encoding="utf-8")
    return folder, Chunker(config.chunk_size, config.chunk_overlap).split(Document("text", [Section(STORY)]))


def answers(*texts: str):
    """A reranker for which only the chunks `texts` answer."""
    return Reranker("fake", scorer=lambda pairs: [5.0 if passage.endswith(texts) else -5.0 for _, passage in pairs])


def test_an_excerpt_comes_with_the_chunks_before_and_after_it(index, story, config):
    folder, chunks = story
    assert len(chunks) > 6
    index.ingest([folder])
    index.reranker = answers(chunks[3].text)

    [excerpt] = index.search("what happened?", k=3)
    assert excerpt.text == join_chunks(chunks[2:5])
    assert excerpt.text in STORY
    assert excerpt.relevance > 0.99


def test_the_first_chunk_has_no_chunk_before_it(index, story, config):
    folder, chunks = story
    index.ingest([folder])
    index.reranker = answers(chunks[0].text)
    [excerpt] = index.search("what happened?", k=3)
    assert excerpt.text == join_chunks(chunks[0:2])


def test_close_excerpts_of_a_document_make_one_passage(index, story, config):
    folder, chunks = story
    index.ingest([folder])
    index.reranker = answers(chunks[3].text, chunks[5].text)

    results = index.search("what happened?", k=3)
    assert [result.text for result in results] == [join_chunks(chunks[2:7])]


def test_distant_excerpts_stay_apart_in_their_order(index, story, config):
    folder, chunks = story
    index.ingest([folder])
    scores = {chunks[1].text: 3.0, chunks[6].text: 5.0}
    index.reranker = Reranker("fake", scorer=lambda pairs: [next((v for c, v in scores.items() if p.endswith(c)), -5.0) for _, p in pairs])

    results = index.search("what happened?", k=3)
    assert [result.text for result in results] == [
        join_chunks(chunks[5:8]), join_chunks(chunks[0:3])
    ]


def test_an_excerpt_of_a_document_reindexed_during_the_search_is_kept_as_found(index, story):
    folder, chunks = story
    index.ingest([folder])
    document = index.store.get_document(str(folder / "story.txt"))

    def score(pairs):
        # The document is reindexed between the search of its chunks and their widening.
        index.store.delete_document(document.id)
        return [5.0 if passage.endswith(chunks[3].text) else -5.0 for _, passage in pairs]

    index.reranker = Reranker("fake", scorer=score)
    [excerpt] = index.search("what happened?", k=3)
    assert excerpt.text == chunks[3].text


def test_no_neighbors_sends_the_chunk_alone(index, story, config):
    folder, chunks = story
    index.ingest([folder])
    index.reranker = answers(chunks[3].text)
    config.rag_neighbors = 0
    [excerpt] = index.search("what happened?", k=3)
    assert excerpt.text == chunks[3].text


def test_the_dropped_chunks_are_reported_without_their_neighbors(index, story):
    folder, chunks = story
    index.ingest([folder])
    index.reranker = answers(chunks[3].text)
    report = SearchReport()
    index.search("what happened?", k=3, report=report)
    assert all(result.text in [chunk.text for chunk in chunks] for result in report.dropped)
