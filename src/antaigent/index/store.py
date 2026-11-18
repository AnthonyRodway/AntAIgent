"""One SQLite file per project holding files, chunks, embeddings, a keyword index,
the code graph (definitions, references, imports) and generated summaries.

Vectors are searched by brute force in NumPy. That stays fast up to a few hundred
thousand chunks and avoids native extensions; swap in sqlite-vec or LanceDB if a
project ever outgrows it.
"""

from __future__ import annotations

import re
import sqlite3
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..ingest.chunking import Chunk
from ..ingest.graph import FileGraph, ImportResolver

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
CREATE TABLE IF NOT EXISTS symbols (
    path TEXT NOT NULL,
    name TEXT NOT NULL,
    qualname TEXT NOT NULL,
    kind TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS symbols_by_name ON symbols(name);
CREATE INDEX IF NOT EXISTS symbols_by_qualname ON symbols(qualname);
CREATE INDEX IF NOT EXISTS symbols_by_path ON symbols(path);
CREATE TABLE IF NOT EXISTS refs (
    path TEXT NOT NULL,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,
    line INTEGER NOT NULL,
    caller TEXT NOT NULL,
    snippet TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS refs_by_name ON refs(name);
CREATE INDEX IF NOT EXISTS refs_by_path ON refs(path, caller);
CREATE TABLE IF NOT EXISTS imports (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL,
    line INTEGER NOT NULL,
    module TEXT NOT NULL,
    names TEXT NOT NULL,
    language TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS imports_by_path ON imports(path);
CREATE TABLE IF NOT EXISTS import_targets (import_id INTEGER NOT NULL, source TEXT NOT NULL, target TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS import_targets_by_source ON import_targets(source);
CREATE INDEX IF NOT EXISTS import_targets_by_target ON import_targets(target);
CREATE TABLE IF NOT EXISTS summaries (
    scope TEXT NOT NULL,
    key TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    model TEXT NOT NULL,
    text TEXT NOT NULL,
    embedding BLOB,
    created_at REAL NOT NULL,
    PRIMARY KEY (scope, key)
);
"""

_TABLES = ("chunks_fts", "chunks", "files", "meta", "symbols", "refs", "imports", "import_targets", "summaries")

# bm25 column weights, in chunks_fts column order: path, symbols, terms, text.
_BM25 = "bm25(chunks_fts, 2.0, 4.0, 1.0, 1.0)"

_WORD = re.compile(r"[A-Za-z0-9_]+")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")
STOPWORDS = frozenset(
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
            if len(token) > 1 and token not in STOPWORDS and token not in tokens:
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


@dataclass(frozen=True)
class SymbolRow:
    path: str
    name: str
    qualname: str
    kind: str
    start_line: int
    end_line: int

    @property
    def location(self) -> str:
        return f"{self.path}:{self.start_line}-{self.end_line}"


@dataclass(frozen=True)
class RefRow:
    path: str
    name: str
    kind: str
    line: int
    caller: str
    snippet: str

    @property
    def location(self) -> str:
        return f"{self.path}:{self.line}"


@dataclass(frozen=True)
class ImportRow:
    path: str
    line: int
    module: str
    target: str | None  # the project file it resolves to, if any


@dataclass(frozen=True)
class Summary:
    scope: str  # "file", "dir" or "project"
    key: str  # file or folder path ("" for the project root)
    fingerprint: str
    model: str
    text: str
    created_at: float


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

    def file(self, path: str) -> FileRecord | None:
        row = self.conn.execute(
            "SELECT path, size, mtime, sha1, status, n_chunks FROM files WHERE path = ?", (path,)
        ).fetchone()
        return FileRecord(*row) if row else None

    def touch_file(self, path: str, size: int, mtime: float) -> None:
        with self.conn:
            self.conn.execute("UPDATE files SET size = ?, mtime = ? WHERE path = ?", (size, mtime, path))

    def put_file(
        self,
        record: FileRecord,
        chunks: Sequence[Chunk] = (),
        embeddings: np.ndarray | None = None,
        graph: FileGraph | None = None,
    ) -> None:
        """Replace everything stored for one file in a single transaction.

        The file's summary is dropped too, since it described the old content."""
        if chunks and (embeddings is None or len(embeddings) != len(chunks)):
            raise ValueError("every chunk needs an embedding")
        with self.conn:
            self._delete_chunks(record.path)
            self._delete_graph(record.path)
            self.conn.execute("DELETE FROM summaries WHERE scope = 'file' AND key = ?", (record.path,))
            if graph is not None:
                self._insert_graph(record.path, graph)
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
            self._delete_graph(path)
            self.conn.execute("DELETE FROM summaries WHERE scope = 'file' AND key = ?", (path,))
            self.conn.execute("DELETE FROM files WHERE path = ?", (path,))
        self._vectors = None

    def _delete_chunks(self, path: str) -> None:
        self.conn.execute("DELETE FROM chunks_fts WHERE rowid IN (SELECT id FROM chunks WHERE path = ?)", (path,))
        self.conn.execute("DELETE FROM chunks WHERE path = ?", (path,))

    def _delete_graph(self, path: str) -> None:
        self.conn.execute("DELETE FROM symbols WHERE path = ?", (path,))
        self.conn.execute("DELETE FROM refs WHERE path = ?", (path,))
        self.conn.execute("DELETE FROM import_targets WHERE source = ?", (path,))
        self.conn.execute("DELETE FROM imports WHERE path = ?", (path,))

    def _insert_graph(self, path: str, graph: FileGraph) -> None:
        self.conn.executemany(
            "INSERT INTO symbols (path, name, qualname, kind, start_line, end_line) VALUES (?, ?, ?, ?, ?, ?)",
            [(path, d.name, d.qualname, d.kind, d.start_line, d.end_line) for d in graph.definitions],
        )
        self.conn.executemany(
            "INSERT INTO refs (path, name, kind, line, caller, snippet) VALUES (?, ?, ?, ?, ?, ?)",
            [(path, r.name, r.kind, r.line, r.caller, r.snippet) for r in graph.references],
        )
        self.conn.executemany(
            "INSERT INTO imports (path, line, module, names, language) VALUES (?, ?, ?, ?, ?)",
            [(path, i.line, i.module, ",".join(i.names), graph.language) for i in graph.imports],
        )

    def clear(self) -> None:
        with self.conn:
            for table in _TABLES:
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
            "symbols": one("SELECT COUNT(*) FROM symbols"),
            "references": one("SELECT COUNT(*) FROM refs"),
            "imports": one("SELECT COUNT(*) FROM imports"),
            "local_imports": one("SELECT COUNT(*) FROM import_targets"),
            "file_summaries": one("SELECT COUNT(*) FROM summaries WHERE scope = 'file'"),
            "dir_summaries": one("SELECT COUNT(*) FROM summaries WHERE scope = 'dir'"),
            "has_overview": bool(one("SELECT COUNT(*) FROM summaries WHERE scope = 'project'")),
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

    _CHUNK_COLUMNS = "id, path, start_line, end_line, kind, symbols, text"

    @staticmethod
    def _chunk(r) -> StoredChunk:
        return StoredChunk(r[0], r[1], r[2], r[3], r[4], [s for s in r[5].split(", ") if s], r[6])

    def chunks_by_id(self, ids: Sequence[int]) -> dict[int, StoredChunk]:
        if not ids:
            return {}
        marks = ",".join("?" * len(ids))
        rows = self.conn.execute(f"SELECT {self._CHUNK_COLUMNS} FROM chunks WHERE id IN ({marks})", list(ids))
        return {r[0]: self._chunk(r) for r in rows}

    def chunks_for_file(self, path: str) -> list[StoredChunk]:
        rows = self.conn.execute(
            f"SELECT {self._CHUNK_COLUMNS} FROM chunks WHERE path = ? ORDER BY start_line, id", (path,)
        )
        return [self._chunk(r) for r in rows]

    def chunk_at(self, path: str, line: int) -> StoredChunk | None:
        """The chunk that starts closest before `line` and contains it."""
        row = self.conn.execute(
            f"SELECT {self._CHUNK_COLUMNS} FROM chunks WHERE path = ? AND start_line <= ? AND end_line >= ? "
            "ORDER BY start_line DESC LIMIT 1",
            (path, line, line),
        ).fetchone()
        return self._chunk(row) if row else None

    # --- Code graph ----------------------------------------------------------------

    _SYMBOL_COLUMNS = "path, name, qualname, kind, start_line, end_line"
    _REF_COLUMNS = "path, name, kind, line, caller, snippet"

    def find_definitions(self, name: str, limit: int = 50) -> list[SymbolRow]:
        """Definitions whose name or qualified name is `name`; case-insensitive if nothing matches exactly."""
        rows = self.conn.execute(
            f"SELECT {self._SYMBOL_COLUMNS} FROM symbols WHERE name = ? OR qualname = ? "
            "ORDER BY path, start_line LIMIT ?",
            (name, name, limit),
        ).fetchall()
        if not rows:
            rows = self.conn.execute(
                f"SELECT {self._SYMBOL_COLUMNS} FROM symbols WHERE name = ? COLLATE NOCASE OR qualname = ? COLLATE NOCASE "
                "ORDER BY path, start_line LIMIT ?",
                (name, name, limit),
            ).fetchall()
        return [SymbolRow(*r) for r in rows]

    def similar_symbols(self, fragment: str, limit: int = 10) -> list[str]:
        pattern = "%" + fragment.replace("%", "").replace("_", "\\_") + "%"
        rows = self.conn.execute(
            "SELECT qualname FROM symbols WHERE qualname LIKE ? ESCAPE '\\' "
            "GROUP BY qualname ORDER BY length(qualname) LIMIT ?",
            (pattern, limit),
        )
        return [r[0] for r in rows]

    def search_symbols(self, fragment: str, limit: int = 200) -> list[SymbolRow]:
        """Definitions whose qualified name contains `fragment`, best matches first."""
        like = fragment.replace("%", "").replace("_", "\\_")
        rows = self.conn.execute(
            f"SELECT {self._SYMBOL_COLUMNS} FROM symbols WHERE qualname LIKE ? ESCAPE '\\' "
            "ORDER BY (name = ? COLLATE NOCASE) DESC, (name LIKE ? ESCAPE '\\') DESC, length(qualname), path LIMIT ?",
            ("%" + like + "%", fragment, like + "%", limit),
        )
        return [SymbolRow(*r) for r in rows]

    def definitions_in(self, path: str) -> list[SymbolRow]:
        rows = self.conn.execute(
            f"SELECT {self._SYMBOL_COLUMNS} FROM symbols WHERE path = ? ORDER BY start_line", (path,)
        )
        return [SymbolRow(*r) for r in rows]

    def references_to(self, name: str, limit: int = 200) -> list[RefRow]:
        """Places that call or mention `name` (the last part, if qualified)."""
        short = name.rsplit(".", 1)[-1]
        rows = self.conn.execute(
            f"SELECT {self._REF_COLUMNS} FROM refs WHERE name = ? ORDER BY path, line LIMIT ?", (short, limit)
        )
        return [RefRow(*r) for r in rows]

    def calls_within(self, path: str, qualname: str, limit: int = 200) -> list[RefRow]:
        """References made inside a definition, including its nested definitions."""
        rows = self.conn.execute(
            f"SELECT {self._REF_COLUMNS} FROM refs WHERE path = ? AND (caller = ? OR caller LIKE ? ESCAPE '\\') "
            "ORDER BY line LIMIT ?",
            (path, qualname, qualname.replace("_", "\\_") + ".%", limit),
        )
        return [RefRow(*r) for r in rows]

    def defined_names(self, candidates: Iterable[str]) -> set[str]:
        """Which of `candidates` are defined somewhere, by name or qualified name."""
        names = list(dict.fromkeys(candidates))
        found: set[str] = set()
        for i in range(0, len(names), 400):
            batch = names[i : i + 400]
            marks = ",".join("?" * len(batch))
            for (name,) in self.conn.execute(
                f"SELECT name FROM symbols WHERE name IN ({marks}) UNION SELECT qualname FROM symbols WHERE qualname IN ({marks})",
                batch + batch,
            ):
                found.add(name)
        return found

    def relink_imports(self) -> int:
        """Resolve every stored import against the current set of files. Returns the number of links."""
        resolver = ImportResolver(self.files())
        rows = self.conn.execute("SELECT id, path, module, names, language FROM imports").fetchall()
        links = [
            (import_id, path, target)
            for import_id, path, module, names, language in rows
            for target in resolver.resolve(path, language, module, [n for n in names.split(",") if n])
        ]
        with self.conn:
            self.conn.execute("DELETE FROM import_targets")
            self.conn.executemany("INSERT INTO import_targets (import_id, source, target) VALUES (?, ?, ?)", links)
        return len(links)

    def imports_of(self, path: str) -> list[ImportRow]:
        rows = self.conn.execute(
            "SELECT i.path, i.line, i.module, t.target FROM imports i "
            "LEFT JOIN import_targets t ON t.import_id = i.id WHERE i.path = ? ORDER BY i.line, t.target",
            (path,),
        )
        return [ImportRow(*r) for r in rows]

    def imported_by(self, path: str) -> list[ImportRow]:
        rows = self.conn.execute(
            "SELECT i.path, i.line, i.module, t.target FROM import_targets t "
            "JOIN imports i ON i.id = t.import_id WHERE t.target = ? ORDER BY i.path, i.line",
            (path,),
        )
        return [ImportRow(*r) for r in rows]

    def match_paths(self, fragment: str, limit: int = 10) -> list[str]:
        """Indexed files named by `fragment`: an exact path, a path suffix, or a substring."""
        fragment = fragment.replace("\\", "/").strip("/")
        if not fragment:
            return []
        exact = self.conn.execute("SELECT path FROM files WHERE path = ?", (fragment,)).fetchall()
        if exact:
            return [exact[0][0]]
        like = fragment.replace("%", "").replace("_", "\\_")
        rows = self.conn.execute(
            "SELECT path FROM files WHERE status = 'indexed' AND (path LIKE ? ESCAPE '\\' OR path LIKE ? ESCAPE '\\') "
            "ORDER BY (path LIKE ? ESCAPE '\\') DESC, length(path) LIMIT ?",
            ("%/" + like, "%" + like + "%", "%/" + like, limit),
        )
        return [r[0] for r in rows]

    # --- Summaries -----------------------------------------------------------------

    def get_summary(self, scope: str, key: str) -> Summary | None:
        row = self.conn.execute(
            "SELECT scope, key, fingerprint, model, text, created_at FROM summaries WHERE scope = ? AND key = ?",
            (scope, key),
        ).fetchone()
        return Summary(*row) if row else None

    def summaries(self, scope: str | None = None) -> dict[tuple[str, str], Summary]:
        sql = "SELECT scope, key, fingerprint, model, text, created_at FROM summaries"
        rows = self.conn.execute(sql + " WHERE scope = ?", (scope,)) if scope else self.conn.execute(sql)
        return {(r[0], r[1]): Summary(*r) for r in rows}

    def put_summary(self, summary: Summary, embedding: np.ndarray | None = None) -> None:
        blob = np.asarray(embedding, dtype=np.float32).tobytes() if embedding is not None else None
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO summaries (scope, key, fingerprint, model, text, embedding, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (summary.scope, summary.key, summary.fingerprint, summary.model, summary.text, blob, summary.created_at),
            )

    def delete_summaries(self, scope: str, keys: Iterable[str]) -> None:
        with self.conn:
            self.conn.executemany("DELETE FROM summaries WHERE scope = ? AND key = ?", [(scope, k) for k in keys])

    def summary_vectors(self) -> tuple[list[tuple[str, str]], np.ndarray]:
        """(scope, key) pairs and embeddings of the file and folder summaries."""
        rows = self.conn.execute(
            "SELECT scope, key, embedding FROM summaries WHERE scope IN ('file', 'dir') AND embedding IS NOT NULL"
        ).fetchall()
        if not rows:
            return [], np.empty((0, 0), dtype=np.float32)
        keys = [(r[0], r[1]) for r in rows]
        return keys, np.frombuffer(b"".join(r[2] for r in rows), dtype=np.float32).reshape(len(rows), -1)
