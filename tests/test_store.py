import sqlite3

import pytest

from sensai.core.rag.models import Chunk
from sensai.core.rag.store import DocumentRecord, IndexMismatchError, StoreError, VectorStore
from tests.conftest import FakeClient


def record(path: str, sha: str = "sha") -> DocumentRecord:
    return DocumentRecord(
        path=path, name=path.rsplit("/", 1)[-1], kind="text", title="", metadata={"author": "Alice"},
        sha256=sha, size=1, mtime=1.0,
    )


def save(store: VectorStore, path: str, texts: list[str], model: str = "m") -> None:
    chunks = [Chunk(text, f"p. {i + 1}") for i, text in enumerate(texts)]
    store.save_document(record(path), chunks, [FakeClient.vector(text) for text in texts], model)


@pytest.fixture
def store(tmp_path):
    store = VectorStore(tmp_path / "index.db")
    yield store
    store.close()


def test_search_returns_closest_chunks_first(store):
    save(store, "/docs/a.txt", ["bank transfer to delta holding", "weather is sunny today"])
    save(store, "/docs/b.txt", ["minutes of the board meeting"])

    results = store.search(FakeClient.vector("transfer to delta"), k=2)
    assert [result.text for result in results][0] == "bank transfer to delta holding"
    assert results[0].source == "a.txt"
    assert results[0].location == "p. 1"
    assert results[0].description == "Author: Alice"
    assert results[0].score > results[1].score


def test_the_fingerprint_changes_with_the_documents(store):
    def add(path: str, sha: str) -> None:
        store.save_document(record(path, sha), [Chunk(path)], [FakeClient.vector(path)], "m")

    empty = store.fingerprint()
    add("/docs/a.txt", "sha-a")
    first = store.fingerprint()
    assert first != empty and store.fingerprint() == first
    add("/docs/a.txt", "sha-a-corrected")
    corrected = store.fingerprint()
    assert corrected != first
    add("/docs/b.txt", "sha-b")
    assert store.fingerprint() != corrected
    store.delete_document(store.get_document("/docs/b.txt").id)
    assert store.fingerprint() == corrected


def test_empty_index(store):
    assert store.is_empty()
    assert store.search(FakeClient.vector("anything"), k=3) == []


def test_saving_again_replaces_the_chunks(store):
    save(store, "/docs/a.txt", ["old text one", "old text two"])
    save(store, "/docs/a.txt", ["new text"])

    assert store.stats().documents == 1
    assert store.stats().chunks == 1
    assert [result.text for result in store.search(FakeClient.vector("old text"), k=5)] == ["new text"]


def test_deleting_a_document_removes_its_vectors(store):
    save(store, "/docs/a.txt", ["alpha"])
    save(store, "/docs/b.txt", ["beta"])
    store.delete_document(store.get_document("/docs/a.txt").id)

    assert [result.source for result in store.search(FakeClient.vector("alpha"), k=5)] == ["b.txt"]


def test_documents_under_matches_the_folder_only(store):
    save(store, "/docs/a.txt", ["a"])
    save(store, "/docs/sub/b.txt", ["b"])
    save(store, "/docs-other/c.txt", ["c"])
    assert sorted(document.name for document in store.documents_under("/docs")) == ["a.txt", "b.txt"]


def test_index_is_bound_to_its_embedding_model(store):
    save(store, "/docs/a.txt", ["alpha"], model="model-a")
    with pytest.raises(IndexMismatchError):
        store.check_model("model-b")
    with pytest.raises(IndexMismatchError):
        save(store, "/docs/b.txt", ["beta"], model="model-b")


def test_vectors_must_match_the_index_dimensions(store):
    save(store, "/docs/a.txt", ["alpha"])
    with pytest.raises(IndexMismatchError):
        store.search([0.1, 0.2], k=1)


def test_rebuild_allows_another_model(tmp_path):
    first = VectorStore(tmp_path / "index.db")
    save(first, "/docs/a.txt", ["alpha"], model="model-a")
    first.close()
    store = VectorStore(tmp_path / "index.db", rebuild=True)
    save(store, "/docs/a.txt", ["alpha"], model="model-b")
    assert store.embed_model == "model-b"
    assert store.stats().chunks == 1
    store.close()


def test_unusable_path_is_a_store_error(tmp_path):
    (tmp_path / "file").write_text("", encoding="utf-8")
    with pytest.raises(StoreError):
        VectorStore(tmp_path / "file" / "index.db")


def test_non_index_file_is_a_store_error(tmp_path):
    path = tmp_path / "notes.db"
    path.write_text("not a database" * 100, encoding="utf-8")
    with pytest.raises(StoreError):
        VectorStore(path)


def test_index_survives_reopening(tmp_path):
    first = VectorStore(tmp_path / "index.db")
    save(first, "/docs/a.txt", ["alpha"])
    first.close()

    second = VectorStore(tmp_path / "index.db")
    assert second.search(FakeClient.vector("alpha"), k=1)[0].text == "alpha"
    second.close()


def columns(path, name: str) -> tuple:
    with sqlite3.connect(path) as db:
        return db.execute("SELECT date, author, sender, recipients FROM documents WHERE name = ?", (name,)).fetchone()


def test_the_metadata_used_by_the_filters_gets_its_own_columns(tmp_path):
    store = VectorStore(tmp_path / "index.db")
    documents = {
        "mail.eml": {"from": "Jean Dupont <jd@alvor.fr>", "to": "Marc", "cc": "Lea", "date": "2023-03-14T10:05+01:00"},
        "report.pdf": {"author": "Alice", "date": "2023-02-01"},
        "ledger.xlsx": {"author": "Bob"},
        "old.eml": {"from": "Marc", "date": "sometime in March"},
    }
    for name, metadata in documents.items():
        document = DocumentRecord(f"/docs/{name}", name, "text", "", metadata, "sha", 1, 1.0)
        store.save_document(document, [Chunk(name)], [FakeClient.vector(name)], "m")
    store.close()

    assert columns(tmp_path / "index.db", "mail.eml") == ("2023-03-14T10:05+01:00", "", "Jean Dupont <jd@alvor.fr>", "Marc, Lea")
    assert columns(tmp_path / "index.db", "report.pdf") == ("2023-02-01", "Alice", "", "")
    assert columns(tmp_path / "index.db", "ledger.xlsx") == (None, "Bob", "", "")
    # An unreadable date cannot be compared: no date filter keeps the document.
    assert columns(tmp_path / "index.db", "old.eml") == (None, "", "Marc", "")


def test_an_index_of_an_older_format_needs_a_rebuild(tmp_path):
    path = tmp_path / "index.db"
    old = VectorStore(path)
    save(old, "/docs/a.txt", ["alpha"])
    old.close()
    with sqlite3.connect(path) as db:
        db.execute("UPDATE meta SET value = '1' WHERE key = 'schema_version'")
        db.execute("DROP INDEX documents_date")
        for column in ("date", "author", "sender", "recipients"):
            db.execute(f"ALTER TABLE documents DROP COLUMN {column}")

    with pytest.raises(StoreError, match="--rebuild"):
        VectorStore(path)
    rebuilt = VectorStore(path, rebuild=True)
    assert rebuilt.is_empty()
    save(rebuilt, "/docs/a.txt", ["alpha"])
    assert rebuilt.search(FakeClient.vector("alpha"), k=1)[0].text == "alpha"
    rebuilt.close()
