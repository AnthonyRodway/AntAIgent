"""Saved conversations from the Ask tab, one SQLite file per project.

They live in chats.db beside the project's index.db rather than inside it: the index
is a cache that `learn --full` or a change of embedding model wipes, while chats are
your own history.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .projects import index_dir

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY,
    conversation_id INTEGER NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    thinking TEXT NOT NULL,
    sources TEXT NOT NULL,
    notes TEXT NOT NULL,
    stats TEXT NOT NULL,
    error TEXT NOT NULL,
    stopped INTEGER NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS turns_by_conversation ON turns(conversation_id, id);
"""

TITLE_CHARS = 60
# Only the token counts and timings from Ollama's final message are worth keeping.
_STAT_KEYS = ("prompt_eval_count", "prompt_eval_duration", "eval_count", "eval_duration", "load_duration", "total_duration")


@dataclass(frozen=True)
class Conversation:
    id: int
    title: str
    created_at: float
    updated_at: float
    turns: int


@dataclass
class SavedTurn:
    question: str
    answer: str = ""
    thinking: str = ""
    # Citations as (path, start line, end line): chunk ids change every time a project is re-learned.
    sources: list[tuple[str, int, int]] = field(default_factory=list)
    graph: list[str] = field(default_factory=list)
    summaries: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    stopped: bool = False


def chats_path(home: Path, root: Path) -> Path:
    return index_dir(home, root) / "chats.db"


def make_title(question: str) -> str:
    text = " ".join(question.split())
    return text if len(text) <= TITLE_CHARS else text[: TITLE_CHARS - 3].rstrip() + "..."


class ChatStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.executescript(_SCHEMA)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> ChatStore:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def conversations(self) -> list[Conversation]:
        """Most recently used first."""
        rows = self.conn.execute(
            "SELECT c.id, c.title, c.created_at, c.updated_at, COUNT(t.id) FROM conversations c "
            "LEFT JOIN turns t ON t.conversation_id = c.id GROUP BY c.id ORDER BY c.updated_at DESC, c.id DESC"
        )
        return [Conversation(*r) for r in rows]

    def create(self, title: str) -> int:
        now = time.time()
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO conversations (title, created_at, updated_at) VALUES (?, ?, ?)", (title, now, now)
            )
        return cur.lastrowid

    def rename(self, conversation_id: int, title: str) -> None:
        with self.conn:
            self.conn.execute("UPDATE conversations SET title = ? WHERE id = ?", (title, conversation_id))

    def delete(self, conversation_id: int) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM turns WHERE conversation_id = ?", (conversation_id,))
            self.conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))

    def add_turn(self, conversation_id: int, turn: SavedTurn) -> None:
        now = time.time()
        stats = {k: v for k, v in turn.stats.items() if k in _STAT_KEYS}
        with self.conn:
            self.conn.execute(
                "INSERT INTO turns (conversation_id, question, answer, thinking, sources, notes, stats, error, stopped, "
                "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    conversation_id, turn.question, turn.answer, turn.thinking,
                    json.dumps([list(s) for s in turn.sources]),
                    json.dumps({"graph": turn.graph, "summaries": turn.summaries}),
                    json.dumps(stats), turn.error, int(turn.stopped), now,
                ),
            )
            self.conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id))

    def turns(self, conversation_id: int) -> list[SavedTurn]:
        rows = self.conn.execute(
            "SELECT question, answer, thinking, sources, notes, stats, error, stopped FROM turns "
            "WHERE conversation_id = ? ORDER BY id",
            (conversation_id,),
        )
        out = []
        for question, answer, thinking, sources, notes_json, stats, error, stopped in rows:
            notes = json.loads(notes_json)
            out.append(SavedTurn(
                question, answer, thinking,
                [(p, int(s), int(e)) for p, s, e in json.loads(sources)],
                notes.get("graph", []), notes.get("summaries", []),
                json.loads(stats), error, bool(stopped),
            ))
        return out
