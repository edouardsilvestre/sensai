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

