import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import sqlite_vec

from sensai.core.rag.models import Chunk, SearchResult, describe

SCHEMA_VERSION = "1"

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
    text TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_document ON chunks(document_id);
"""


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
    def __init__(self, path: Path):
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
            self._db.execute("PRAGMA journal_mode = WAL")
            self._db.executescript(_SCHEMA)
        except sqlite3.DatabaseError as e:
            self._db.close()
            raise StoreError(f"Cannot use {path} as an index: {e}") from e
        version = self._get_meta("schema_version")
        if version is None:
            with self._db:
                self._set_meta("schema_version", SCHEMA_VERSION)
        elif version != SCHEMA_VERSION:
            self._db.close()
            raise StoreError(f"{path} uses index format v{version}, expected v{SCHEMA_VERSION}: rebuild it")

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
                record.sha256, record.size, record.mtime, datetime.now(timezone.utc).isoformat(timespec="seconds"),
            )
            if existing:
                self._delete_chunks(existing.id)
                self._db.execute(
                    "UPDATE documents SET name = ?, kind = ?, title = ?, metadata = ?, sha256 = ?, size = ?, "
                    "mtime = ?, indexed_at = ? WHERE id = ?",
                    (*values, existing.id),
                )
                document_id = existing.id
            else:
                cursor = self._db.execute(
                    "INSERT INTO documents (name, kind, title, metadata, sha256, size, mtime, indexed_at, path) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (*values, record.path),
                )
                document_id = cursor.lastrowid
            for position, (chunk, vector) in enumerate(zip(chunks, vectors)):
                cursor = self._db.execute(
                    "INSERT INTO chunks (document_id, position, location, text) VALUES (?, ?, ?, ?)",
                    (document_id, position, chunk.location, chunk.text),
                )
                self._db.execute(
                    "INSERT INTO vec_chunks (rowid, embedding, document_id) VALUES (?, ?, ?)",
                    (cursor.lastrowid, sqlite_vec.serialize_float32(vector), document_id),
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
        self._db.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))

    def is_empty(self) -> bool:
        return self._db.execute("SELECT NOT EXISTS (SELECT 1 FROM chunks)").fetchone()[0] == 1

    def search(self, vector: list[float], k: int) -> list[SearchResult]:
        if self.dimensions is None:
            return []
        if len(vector) != self.dimensions:
            raise IndexMismatchError(f"Query vector has {len(vector)} dimensions, the index has {self.dimensions}")
        rows = self._db.execute(
            """
            WITH knn AS (
                SELECT rowid, distance FROM vec_chunks WHERE embedding MATCH ? AND k = ?
            )
            SELECT c.id, c.text, c.location, d.name, d.title, d.metadata, knn.distance
            FROM knn
            JOIN chunks c ON c.id = knn.rowid
            JOIN documents d ON d.id = c.document_id
            ORDER BY knn.distance
            """,
            (sqlite_vec.serialize_float32(vector), k),
        ).fetchall()
        return [
            SearchResult(
                chunk_id=chunk_id,
                text=text,
                location=location,
                source=name,
                description=describe(title, json.loads(metadata)),
                score=1.0 - distance,
            )
            for chunk_id, text, location, name, title, metadata, distance in rows
        ]

    def stats(self) -> IndexStats:
        documents, chunks = self._db.execute(
            "SELECT (SELECT COUNT(*) FROM documents), (SELECT COUNT(*) FROM chunks)"
        ).fetchone()
        return IndexStats(documents, chunks, self.embed_model, self.dimensions)

    def fingerprint(self) -> str:
        documents = self._db.execute("SELECT path, sha256 FROM documents ORDER BY path").fetchall()
        return hashlib.sha256(json.dumps(documents).encode()).hexdigest()

    def reset(self) -> None:
        with self._db:
            self._db.execute("DROP TABLE IF EXISTS vec_chunks")
            self._db.execute("DELETE FROM chunks")
            self._db.execute("DELETE FROM documents")
            self._db.execute("DELETE FROM meta WHERE key IN ('embed_model', 'dimensions')")

    def close(self) -> None:
        self._db.close()

    def _get_meta(self, key: str) -> str | None:
        row = self._db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def _set_meta(self, key: str, value: str) -> None:
        self._db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))
