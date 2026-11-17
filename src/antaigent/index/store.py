"""One SQLite file per project holding files, chunks, embeddings and a keyword index.

Vectors are searched by brute force in NumPy. That stays fast up to a few hundred
thousand chunks and avoids native extensions; swap in sqlite-vec or LanceDB if a
project ever outgrows it.
"""

from __future__ import annotations

import re
import sqlite3
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..ingest.chunking import Chunk

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS files (
    path TEXT PRIMARY KEY,
    size INTEGER NOT NULL,
    mtime REAL NOT NULL,
    sha1 TEXT NOT NULL,
    status TEXT NOT NULL,
    n_chunks INTEGER NOT NULL DEFAULT 0,
    indexed_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    kind TEXT NOT NULL,
    symbols TEXT NOT NULL,
    text TEXT NOT NULL,
    embedding BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_by_path ON chunks(path);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(path, symbols, terms, text);
"""

# bm25 column weights, in chunks_fts column order: path, symbols, terms, text.
_BM25 = "bm25(chunks_fts, 2.0, 4.0, 1.0, 1.0)"

_WORD = re.compile(r"[A-Za-z0-9_]+")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")
_STOPWORDS = frozenset(
    """a an and are as at be but by can could do does did for from has have how i if in into is it
    its me my of on or our should so than that the their them then there these this those to
    us was we were what when where which who why will with would you your about any all also
    explain tell show give find please""".split()
)


def split_identifier(word: str) -> list[str]:
    parts: list[str] = []
    for piece in word.split("_"):
        parts.extend(p.lower() for p in _CAMEL.findall(piece))
    return parts


def identifier_terms(text: str) -> str:
    """camelCase parts, so `getUserName` is also findable as get/user/name.
    (FTS5 already splits snake_case on underscores.)"""
    terms: set[str] = set()
    for word in set(_WORD.findall(text)):
        if word != word.lower() and word != word.upper():
            parts = split_identifier(word)
            if len(parts) > 1:
                terms.update(parts)
    return " ".join(sorted(terms))


def fts_query(text: str) -> str | None:
    tokens: list[str] = []
    for word in _WORD.findall(text):
        for token in (word.lower(), *split_identifier(word)):
            if len(token) > 1 and token not in _STOPWORDS and token not in tokens:
                tokens.append(token)
    return " OR ".join(f'"{t}"' for t in tokens) if tokens else None


@dataclass(frozen=True)
class FileRecord:
    path: str
    size: int
    mtime: float
    sha1: str
    status: str  # "indexed", "empty", "skipped: <reason>" or "error: <message>"
    n_chunks: int = 0


@dataclass(frozen=True)
class StoredChunk:
    id: int
    path: str
    start_line: int
    end_line: int
    kind: str
    symbols: list[str]
    text: str

    @property
    def location(self) -> str:
        return f"{self.path}:{self.start_line}-{self.end_line}"


class Store:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(_SCHEMA)
        self._vectors: tuple[np.ndarray, np.ndarray] | None = None

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self.conn:
            self.conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))

    def files(self) -> dict[str, FileRecord]:
        rows = self.conn.execute("SELECT path, size, mtime, sha1, status, n_chunks FROM files")
        return {r[0]: FileRecord(*r) for r in rows}

    def touch_file(self, path: str, size: int, mtime: float) -> None:
        with self.conn:
            self.conn.execute("UPDATE files SET size = ?, mtime = ? WHERE path = ?", (size, mtime, path))

    def put_file(self, record: FileRecord, chunks: Sequence[Chunk] = (), embeddings: np.ndarray | None = None) -> None:
        """Replace everything stored for one file in a single transaction."""
        if chunks and (embeddings is None or len(embeddings) != len(chunks)):
            raise ValueError("every chunk needs an embedding")
        with self.conn:
            self._delete_chunks(record.path)
            self.conn.execute(
                "INSERT OR REPLACE INTO files (path, size, mtime, sha1, status, n_chunks, indexed_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (record.path, record.size, record.mtime, record.sha1, record.status, len(chunks), time.time()),
            )
            for chunk, vector in zip(chunks, embeddings if embeddings is not None else []):
                symbols = ", ".join(chunk.symbols)
                cur = self.conn.execute(
                    "INSERT INTO chunks (path, start_line, end_line, kind, symbols, text, embedding) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (record.path, chunk.start_line, chunk.end_line, chunk.kind, symbols, chunk.text,
                     np.asarray(vector, dtype=np.float32).tobytes()),
                )
                self.conn.execute(
                    "INSERT INTO chunks_fts (rowid, path, symbols, terms, text) VALUES (?, ?, ?, ?, ?)",
                    (cur.lastrowid, record.path, symbols, identifier_terms(chunk.text + " " + symbols), chunk.text),
                )
        self._vectors = None

    def delete_file(self, path: str) -> None:
        with self.conn:
            self._delete_chunks(path)
            self.conn.execute("DELETE FROM files WHERE path = ?", (path,))
        self._vectors = None

    def _delete_chunks(self, path: str) -> None:
        self.conn.execute("DELETE FROM chunks_fts WHERE rowid IN (SELECT id FROM chunks WHERE path = ?)", (path,))
        self.conn.execute("DELETE FROM chunks WHERE path = ?", (path,))

    def clear(self) -> None:
        with self.conn:
            for table in ("chunks_fts", "chunks", "files", "meta"):
                self.conn.execute(f"DELETE FROM {table}")
        self._vectors = None

    def stats(self) -> dict[str, object]:
        def one(sql: str) -> object:
            return self.conn.execute(sql).fetchone()[0]

        return {
            "files_indexed": one("SELECT COUNT(*) FROM files WHERE status = 'indexed'"),
            "files_skipped": one("SELECT COUNT(*) FROM files WHERE status != 'indexed'"),
            "chunks": one("SELECT COUNT(*) FROM chunks"),
            "last_indexed": one("SELECT MAX(indexed_at) FROM files"),
        }

    def skipped_files(self, limit: int = 20) -> list[tuple[str, str]]:
        rows = self.conn.execute(
            "SELECT path, status FROM files WHERE status != 'indexed' ORDER BY path LIMIT ?", (limit,)
        )
        return list(rows)

    def vectors(self) -> tuple[np.ndarray, np.ndarray]:
        if self._vectors is None:
            rows = self.conn.execute("SELECT id, embedding FROM chunks ORDER BY id").fetchall()
            if not rows:
                self._vectors = (np.empty(0, dtype=np.int64), np.empty((0, 0), dtype=np.float32))
            else:
                ids = np.fromiter((r[0] for r in rows), dtype=np.int64, count=len(rows))
                matrix = np.frombuffer(b"".join(r[1] for r in rows), dtype=np.float32).reshape(len(rows), -1)
                self._vectors = (ids, matrix)
        return self._vectors

    def keyword_search(self, query: str, limit: int) -> list[int]:
        match = fts_query(query)
        if match is None:
            return []
        try:
            rows = self.conn.execute(
                f"SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY {_BM25} LIMIT ?", (match, limit)
            )
            return [r[0] for r in rows]
        except sqlite3.OperationalError:
            return []

    def chunks_by_id(self, ids: Sequence[int]) -> dict[int, StoredChunk]:
        if not ids:
            return {}
        marks = ",".join("?" * len(ids))
        rows = self.conn.execute(
            f"SELECT id, path, start_line, end_line, kind, symbols, text FROM chunks WHERE id IN ({marks})", list(ids)
        )
        return {
            r[0]: StoredChunk(r[0], r[1], r[2], r[3], r[4], [s for s in r[5].split(", ") if s], r[6]) for r in rows
        }
