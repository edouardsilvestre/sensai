import hashlib
import json
import os
import re
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import sqlite_vec

from sensai.core.rag.models import Chunk, SearchFilter, SearchResult, describe

SCHEMA_VERSION = "2"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    metadata TEXT NOT NULL DEFAULT '{}',
    -- Copied from the metadata for the search filters; date is ISO, or NULL when unreadable.
    date TEXT,
    author TEXT NOT NULL DEFAULT '',
    sender TEXT NOT NULL DEFAULT '',
    recipients TEXT NOT NULL DEFAULT '',
    sha256 TEXT NOT NULL,
    size INTEGER NOT NULL,
    mtime REAL NOT NULL,
    indexed_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    location TEXT NOT NULL DEFAULT '',
    text TEXT NOT NULL,
    -- The characters at the start of the text that repeat the end of the previous chunk of the document.
    overlap INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS chunks_document ON chunks(document_id);
CREATE INDEX IF NOT EXISTS documents_date ON documents(date);
-- The words of each chunk for the keyword search, with the title, people and attachments of its document.
-- rowid is the id of the chunk.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, header, tokenize = 'unicode61 remove_diacritics 2');
"""

# The ranking of the keyword search. The header weighs less than the text: a name in the From of an email
# matters less than in its body. Set as the rank of the table, which FTS5 sorts by faster than by bm25().
_RANK = "bm25(1.0, 0.5)"
_HEADER_FIELDS = ("from", "to", "cc", "author", "attachments")
# More words make a slower query, and add little after the first ones.
_MAX_QUERY_WORDS = 32


class StoreError(Exception):
    pass


class IndexMismatchError(StoreError):
    pass


@dataclass
class DocumentRecord:
    path: str
    name: str
    kind: str
    title: str
    metadata: dict[str, str]
    sha256: str
    size: int
    mtime: float


@dataclass
class StoredDocument:
    id: int
    path: str
    name: str
    sha256: str
    size: int
    mtime: float


@dataclass
class IndexStats:
    documents: int
    chunks: int
    embed_model: str | None
    dimensions: int | None


class VectorStore:
    def __init__(self, path: Path, rebuild: bool = False):
        self.path = path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(path)
        except (OSError, sqlite3.Error) as e:
            raise StoreError(f"Cannot open the index at {path}: {e}") from e
        try:
            self._db.enable_load_extension(True)
            sqlite_vec.load(self._db)
            self._db.enable_load_extension(False)
        except (AttributeError, sqlite3.OperationalError) as e:
            self._db.close()
            raise StoreError(f"Cannot load sqlite-vec (this Python's sqlite3 may not support extensions): {e}") from e
        try:
            self._db.execute("PRAGMA foreign_keys = ON")
            # SQLite's lower() and LIKE only fold ASCII: "élodie" would not find "Élodie".
            self._db.create_function("casefold", 1, lambda text: text.casefold() if text else text, deterministic=True)
            self._db.execute("PRAGMA journal_mode = WAL")
            if rebuild:
                self._drop_tables()
            # Checked before the schema runs: the tables of an older format may lack its columns.
            has_meta = self._db.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'meta'").fetchone()
            version = self._get_meta("schema_version") if has_meta else None
            if version is not None and version != SCHEMA_VERSION:
                self._db.close()
                raise StoreError(
                    f"{path} uses index format v{version}, expected v{SCHEMA_VERSION}: "
                    f"rebuild it with `python -m sensai ingest --rebuild PATH`"
                )
            has_keywords = self._db.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'chunks_fts'"
            ).fetchone()
            self._db.executescript(_SCHEMA)
            with self._db:
                if version is None:
                    self._set_meta("schema_version", SCHEMA_VERSION)
                if not has_keywords:
                    self._db.execute("INSERT INTO chunks_fts (chunks_fts, rank) VALUES ('rank', ?)", (_RANK,))
                    # An index made before the keyword search: its chunks get their words now.
                    self._fill_keywords()
                if "overlap" not in {row[1] for row in self._db.execute("PRAGMA table_info(chunks)")}:
                    # An index made before the neighbors: its chunks are joined as if they did not overlap,
                    # repeating a few sentences, until `ingest --rebuild`.
                    self._db.execute("ALTER TABLE chunks ADD COLUMN overlap INTEGER NOT NULL DEFAULT 0")
        except sqlite3.DatabaseError as e:
            self._db.close()
            raise StoreError(f"Cannot use {path} as an index: {e}") from e

    @property
    def embed_model(self) -> str | None:
        return self._get_meta("embed_model")

    @property
    def dimensions(self) -> int | None:
        value = self._get_meta("dimensions")
        return int(value) if value else None

    def check_model(self, model: str) -> None:
        indexed = self.embed_model
        if indexed is not None and indexed != model:
            raise IndexMismatchError(
                f"The index was built with '{indexed}' but the configured embedding model is '{model}'. "
                f"Set SENSAI_EMBED_MODEL={indexed}, or rebuild the index."
            )

    def _bind_vectors(self, model: str, dimensions: int) -> None:
        self.check_model(model)
        if self.dimensions is None:
            self._db.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0("
                f"embedding float[{dimensions}] distance_metric=cosine, document_id integer)"
            )
            self._set_meta("embed_model", model)
            self._set_meta("dimensions", str(dimensions))
        elif self.dimensions != dimensions:
            raise IndexMismatchError(f"Got {dimensions}-dimension vectors for an index of {self.dimensions} dimensions")

    def get_document(self, path: str) -> StoredDocument | None:
        row = self._db.execute(
            "SELECT id, path, name, sha256, size, mtime FROM documents WHERE path = ?", (path,)
        ).fetchone()
        return StoredDocument(*row) if row else None

    def documents_under(self, folder: str) -> list[StoredDocument]:
        prefix = folder.rstrip(os.sep) + os.sep
        rows = self._db.execute(
            "SELECT id, path, name, sha256, size, mtime FROM documents WHERE substr(path, 1, ?) = ?",
            (len(prefix), prefix),
        ).fetchall()
        return [StoredDocument(*row) for row in rows]

    def save_document(self, record: DocumentRecord, chunks: list[Chunk], vectors: list[list[float]], model: str) -> None:
        if len(chunks) != len(vectors) or not chunks:
            raise ValueError("a document needs one vector per chunk, and at least one chunk")
        with self._db:
            self._bind_vectors(model, len(vectors[0]))
            existing = self.get_document(record.path)
            values = (
                record.name, record.kind, record.title, json.dumps(record.metadata, ensure_ascii=False),
                *_filter_columns(record.metadata),
                record.sha256, record.size, record.mtime, datetime.now(timezone.utc).isoformat(timespec="seconds"),
            )
            if existing:
                self._delete_chunks(existing.id)
                self._db.execute(
                    "UPDATE documents SET name = ?, kind = ?, title = ?, metadata = ?, date = ?, author = ?, "
                    "sender = ?, recipients = ?, sha256 = ?, size = ?, mtime = ?, indexed_at = ? WHERE id = ?",
                    (*values, existing.id),
                )
                document_id = existing.id
            else:
                cursor = self._db.execute(
                    "INSERT INTO documents (name, kind, title, metadata, date, author, sender, recipients, "
                    "sha256, size, mtime, indexed_at, path) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (*values, record.path),
                )
                document_id = cursor.lastrowid
            for position, (chunk, vector) in enumerate(zip(chunks, vectors)):
                cursor = self._db.execute(
                    "INSERT INTO chunks (document_id, position, location, text, overlap) VALUES (?, ?, ?, ?, ?)",
                    (document_id, position, chunk.location, chunk.text, chunk.overlap),
                )
                self._db.execute(
                    "INSERT INTO vec_chunks (rowid, embedding, document_id) VALUES (?, ?, ?)",
                    (cursor.lastrowid, sqlite_vec.serialize_float32(vector), document_id),
                )
                self._db.execute(
                    "INSERT INTO chunks_fts (rowid, text, header) VALUES (?, ?, ?)",
                    (cursor.lastrowid, chunk.text, _header(record.title, record.metadata)),
                )

    def update_file_info(self, document_id: int, size: int, mtime: float) -> None:
        with self._db:
            self._db.execute("UPDATE documents SET size = ?, mtime = ? WHERE id = ?", (size, mtime, document_id))

    def delete_document(self, document_id: int) -> None:
        with self._db:
            self._delete_chunks(document_id)
            self._db.execute("DELETE FROM documents WHERE id = ?", (document_id,))

    def _delete_chunks(self, document_id: int) -> None:
        ids = [(row[0],) for row in self._db.execute("SELECT id FROM chunks WHERE document_id = ?", (document_id,))]
        if self.dimensions is not None:
            self._db.executemany("DELETE FROM vec_chunks WHERE rowid = ?", ids)
        self._db.executemany("DELETE FROM chunks_fts WHERE rowid = ?", ids)
        self._db.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))

    def _fill_keywords(self) -> None:
        rows = self._db.execute(
            "SELECT c.id, c.text, d.title, d.metadata FROM chunks c JOIN documents d ON d.id = c.document_id"
        )
        self._db.executemany(
            "INSERT INTO chunks_fts (rowid, text, header) VALUES (?, ?, ?)",
            [(chunk_id, text, _header(title, json.loads(metadata))) for chunk_id, text, title, metadata in rows],
        )

    def chunk_positions(self, chunk_ids: list[int]) -> dict[int, tuple[int, int]]:
        """The document id and the position in it of each chunk."""
        rows = self._db.execute(
            "SELECT id, document_id, position FROM chunks WHERE id IN (SELECT value FROM json_each(?))",
            (json.dumps(chunk_ids),),
        )
        return {chunk_id: (document_id, position) for chunk_id, document_id, position in rows}

    def chunks_between(self, document_id: int, first: int, last: int) -> list[Chunk]:
        """The chunks of a document from position `first` to `last`, in their order."""
        rows = self._db.execute(
            "SELECT text, location, overlap FROM chunks WHERE document_id = ? AND position BETWEEN ? AND ? "
            "ORDER BY position",
            (document_id, first, last),
        )
        return [Chunk(text, location, overlap) for text, location, overlap in rows]

    def is_empty(self) -> bool:
        return self._db.execute("SELECT NOT EXISTS (SELECT 1 FROM chunks)").fetchone()[0] == 1

    def allowed_documents(self, filters: Iterable[SearchFilter]) -> list[int] | None:
        """The ids of the documents that match every filter, or None when no filter limits the search."""
        where: list[str] = []
        params: list[str | int] = []
        for search_filter in filters:
            if search_filter.after:
                where.append("date >= ?")
                params.append(search_filter.after)
            if search_filter.before:
                where.append("date < ?")
                params.append(search_filter.before)
            if search_filter.author:
                where.append("(instr(casefold(author), ?) OR instr(casefold(sender), ?))")
                params += [search_filter.author.casefold()] * 2
            if search_filter.recipient:
                where.append("instr(casefold(recipients), ?)")
                params.append(search_filter.recipient.casefold())
            if search_filter.kind:
                where.append("kind = ?")
                params.append(search_filter.kind)
            if search_filter.source:
                where.append("substr(name, 1, ?) = ?")
                params += [len(search_filter.source), search_filter.source]
        if not where:
            return None
        rows = self._db.execute(f"SELECT id FROM documents WHERE {' AND '.join(where)}", params)
        return [row[0] for row in rows]

    def search(self, vector: list[float], k: int, document_ids: list[int] | None = None) -> list[SearchResult]:
        if self.dimensions is None or document_ids == []:
            return []
        if len(vector) != self.dimensions:
            raise IndexMismatchError(f"Query vector has {len(vector)} dimensions, the index has {self.dimensions}")
        # The filter runs inside the KNN: the k closest chunks are taken among the allowed documents.
        allowed = ""
        params: list = [sqlite_vec.serialize_float32(vector), k]
        if document_ids is not None:
            allowed = "AND document_id IN (SELECT value FROM json_each(?))"
            params.append(json.dumps(document_ids))
        rows = self._db.execute(
            f"""
            WITH knn AS (
                SELECT rowid, distance FROM vec_chunks WHERE embedding MATCH ? AND k = ? {allowed}
            )
            SELECT c.id, c.text, c.location, d.name, d.title, d.metadata, knn.distance
            FROM knn
            JOIN chunks c ON c.id = knn.rowid
            JOIN documents d ON d.id = c.document_id
            ORDER BY knn.distance
            """,
            params,
        ).fetchall()
        return [_result(*row) for row in rows]

    def keyword_search(
        self, query: str, k: int, vector: list[float], document_ids: list[int] | None = None
    ) -> list[SearchResult]:
        """
        The k chunks that match the words of the query best by BM25: the rarer a word in the index, the more it counts.
        `vector` is the one of the query, to give each result its similarity like the vector search does.
        """
        match = _keyword_query(query)
        if not match or self.dimensions is None or document_ids == []:
            return []
        allowed = ""
        params: list = [match]
        if document_ids is not None:
            allowed = "AND c.document_id IN (SELECT value FROM json_each(?))"
            params.append(json.dumps(document_ids))
        params += [k, sqlite_vec.serialize_float32(vector)]
        # The k best chunks first, then the rest of their columns: a common word matches most of the index,
        # and reading the vector of every match would take seconds.
        rows = self._db.execute(
            f"""
            WITH best AS (
                SELECT f.rowid AS id, f.rank AS rank
                FROM chunks_fts f JOIN chunks c ON c.id = f.rowid
                WHERE chunks_fts MATCH ? {allowed}
                ORDER BY f.rank
                LIMIT ?
            )
            SELECT c.id, c.text, c.location, d.name, d.title, d.metadata, vec_distance_cosine(v.embedding, ?)
            FROM best
            JOIN chunks c ON c.id = best.id
            JOIN documents d ON d.id = c.document_id
            JOIN vec_chunks v ON v.rowid = c.id
            ORDER BY best.rank
            """,
            params,
        ).fetchall()
        return [_result(*row) for row in rows]

    def stats(self) -> IndexStats:
        documents, chunks = self._db.execute(
            "SELECT (SELECT COUNT(*) FROM documents), (SELECT COUNT(*) FROM chunks)"
        ).fetchone()
        return IndexStats(documents, chunks, self.embed_model, self.dimensions)

    def fingerprint(self) -> str:
        documents = self._db.execute("SELECT path, sha256 FROM documents ORDER BY path").fetchall()
        return hashlib.sha256(json.dumps(documents).encode()).hexdigest()

    def _drop_tables(self) -> None:
        # Dropped rather than emptied, so that an index of an older format gets the current tables.
        for table in ("vec_chunks", "chunks_fts", "chunks", "documents", "meta"):
            self._db.execute(f"DROP TABLE IF EXISTS {table}")

    def close(self) -> None:
        self._db.close()

    def _get_meta(self, key: str) -> str | None:
        row = self._db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def _set_meta(self, key: str, value: str) -> None:
        self._db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))


def _filter_columns(metadata: dict[str, str]) -> tuple[str | None, str, str, str]:
    date = metadata.get("date") or None
    try:
        if date is not None:
            datetime.fromisoformat(date)
    except ValueError:
        date = None
    recipients = ", ".join(filter(None, [metadata.get("to"), metadata.get("cc")]))
    return date, metadata.get("author", ""), metadata.get("from", ""), recipients


def _result(chunk_id: int, text: str, location: str, name: str, title: str, metadata: str, distance: float) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        text=text,
        location=location,
        source=name,
        description=describe(title, json.loads(metadata)),
        score=1.0 - distance,
    )


def _header(title: str, metadata: dict[str, str]) -> str:
    return " ".join(filter(None, [title, *(metadata.get(key, "") for key in _HEADER_FIELDS)]))


def _keyword_query(text: str) -> str:
    # Each word in quotes, so that nothing in the query is read as FTS5 syntax (AND, NEAR, *, ^, "...").
    # One-letter words, like the l of "l'entreprise", match too much to help.
    words = dict.fromkeys(word for word in re.findall(r"\w+", text.casefold()) if len(word) > 1)
    return " OR ".join(f'"{word}"' for word in list(words)[:_MAX_QUERY_WORDS])
