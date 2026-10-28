import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# Names and relations are compared without case: "Alvor Consulting" and "alvor consulting" are the same entity.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS entities (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE COLLATE NOCASE,
    type TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY,
    entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS relations (
    id INTEGER PRIMARY KEY,
    subject_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    predicate TEXT NOT NULL COLLATE NOCASE,
    object_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    UNIQUE (subject_id, predicate, object_id)
);
CREATE INDEX IF NOT EXISTS facts_entity ON facts(entity_id);
"""


class MemoryStoreError(Exception):
    pass


@dataclass
class Fact:
    id: int
    content: str


@dataclass
class Relation:
    subject: str
    predicate: str
    object: str


@dataclass
class Entity:
    name: str
    type: str
    facts: list[Fact] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)


class MemoryStore:
    """Long-term memory: entities, the facts known about them and the relations between them."""

    def __init__(self, path: Path):
        self.path = path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(path)
        except (OSError, sqlite3.Error) as e:
            raise MemoryStoreError(f"Cannot open the memory at {path}: {e}") from e
        try:
            self._db.execute("PRAGMA foreign_keys = ON")
            self._db.executescript(_SCHEMA)
        except sqlite3.DatabaseError as e:
            self._db.close()
            raise MemoryStoreError(f"Cannot use {path} as a memory: {e}") from e

    def add_fact(self, entity: str, content: str, entity_type: str = "") -> int:
        """Stores a fact about `entity`, created if it is new, and returns the fact's id."""
        content = _required(content, "fact")
        with self._db:
            entity_id = self._entity_id(entity, entity_type)
            now = _now()
            cursor = self._db.execute(
                "INSERT INTO facts (entity_id, content, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (entity_id, content, now, now),
            )
        return cursor.lastrowid

    def update_fact(self, fact_id: int, content: str) -> None:
        content = _required(content, "fact")
        with self._db:
            cursor = self._db.execute(
                "UPDATE facts SET content = ?, updated_at = ? WHERE id = ?", (content, _now(), fact_id)
            )
        if cursor.rowcount == 0:
            raise MemoryStoreError(f"No fact with id {fact_id}")

    def delete_fact(self, fact_id: int) -> None:
        with self._db:
            cursor = self._db.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
        if cursor.rowcount == 0:
            raise MemoryStoreError(f"No fact with id {fact_id}")

    def link(self, subject: str, predicate: str, object: str) -> bool:
        """Stores `subject predicate object`, creating the missing entities. False if it was already stored."""
        predicate = _required(predicate, "relation")
        with self._db:
            subject_id = self._entity_id(subject)
            object_id = self._entity_id(object)
            cursor = self._db.execute(
                "INSERT OR IGNORE INTO relations (subject_id, predicate, object_id, created_at) VALUES (?, ?, ?, ?)",
                (subject_id, predicate, object_id, _now()),
            )
        return cursor.rowcount == 1

    def unlink(self, subject: str, predicate: str, object: str) -> None:
        with self._db:
            cursor = self._db.execute(
                """
                DELETE FROM relations WHERE predicate = ?
                AND subject_id = (SELECT id FROM entities WHERE name = ?)
                AND object_id = (SELECT id FROM entities WHERE name = ?)
                """,
                (predicate.strip(), subject.strip(), object.strip()),
            )
        if cursor.rowcount == 0:
            raise MemoryStoreError(f"No relation '{subject} {predicate} {object}'")

def _required(value: str, what: str) -> str:
    value = value.strip()
    if not value:
        raise MemoryStoreError(f"The {what} is empty")
    return value


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
