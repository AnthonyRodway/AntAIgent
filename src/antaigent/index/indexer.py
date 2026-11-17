"""Build a project's index, then keep it up to date by reprocessing only changed files."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Protocol

import numpy as np

from ..ingest.chunking import DATA_EXTENSIONS, NOTEBOOK_EXTENSIONS, Chunk, Chunker
from ..ingest.walker import SourceFile, walk
from .store import FileRecord, Store

DATA_PREVIEW_BYTES = 64 * 1024
# Notebooks are mostly embedded output images; only their cell sources get indexed.
MAX_NOTEBOOK_BYTES = 25_000_000


class DocumentEmbedder(Protocol):
    model: str
    batch_size: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...


ProgressFn = Callable[[int, int, str], None]


@dataclass
class LearnReport:
    scanned: int = 0
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    removed: int = 0
    skipped: int = 0
    chunks: int = 0
    rebuilt: bool = False
    seconds: float = 0.0
    errors: list[str] = field(default_factory=list)
    missing_grammars: set[str] = field(default_factory=set)


def embedding_text(path: str, chunk: Chunk) -> str:
    header = f"{path} | {', '.join(chunk.symbols[:8])}" if chunk.symbols else path
    return f"{header}\n{chunk.text}"


def _read(f: SourceFile, max_file_bytes: int) -> tuple[bytes, str | None]:
    """Return (content, skip reason)."""
    ext = PurePosixPath(f.rel).suffix.lower()
    limit = MAX_NOTEBOOK_BYTES if ext in NOTEBOOK_EXTENSIONS else max_file_bytes
    if ext not in DATA_EXTENSIONS and f.size > limit:
        return b"", f"larger than {limit // 1000:,} KB"
    with open(f.path, "rb") as fh:
        data = fh.read(DATA_PREVIEW_BYTES) if ext in DATA_EXTENSIONS else fh.read()
    if b"\x00" in data[:8192]:
        return data, "binary"
    return data, None


def _decode(data: bytes) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace")


def learn(
    root: Path,
    store: Store,
    embedder: DocumentEmbedder,
    *,
    max_file_bytes: int = 1_000_000,
    chunk_chars: int = 2400,
    full: bool = False,
    progress: ProgressFn | None = None,
) -> LearnReport:
    started = time.perf_counter()
    report = LearnReport()

    signature = f"{embedder.model}|{chunk_chars}"
    if full or store.get_meta("signature") != signature:
        report.rebuilt = bool(store.files())
        store.clear()
        store.set_meta("signature", signature)

    known = store.files()
    files = list(walk(root))
    report.scanned = len(files)
    seen = {f.rel for f in files}
    for path in sorted(set(known) - seen):
        store.delete_file(path)
        report.removed += 1

    todo = []
    for f in files:
        rec = known.get(f.rel)
        if rec is not None and rec.size == f.size and rec.mtime == f.mtime:
            report.unchanged += 1
        else:
            todo.append(f)

    chunker = Chunker(chunk_chars)
    pending: list[tuple[FileRecord, list[Chunk], bool]] = []
    pending_chunks = 0

    def flush() -> None:
        nonlocal pending, pending_chunks
        texts = [embedding_text(rec.path, c) for rec, chunks, _ in pending for c in chunks]
        vectors = embedder.embed_documents(texts)
        offset = 0
        for rec, chunks, is_new in pending:
            store.put_file(rec, chunks, vectors[offset : offset + len(chunks)])
            offset += len(chunks)
            report.chunks += len(chunks)
            if is_new:
                report.added += 1
            else:
                report.updated += 1
        pending, pending_chunks = [], 0

    for done, f in enumerate(todo):
        if progress:
            progress(done, len(todo), f.rel)
        prev = known.get(f.rel)
        try:
            data, reason = _read(f, max_file_bytes)
        except OSError as exc:
            report.errors.append(f"{f.rel}: {exc}")
            continue
        sha = hashlib.sha1(data).hexdigest()
        if prev is not None and prev.sha1 == sha:
            store.touch_file(f.rel, f.size, f.mtime)
            report.unchanged += 1
            continue
        if reason:
            store.put_file(FileRecord(f.rel, f.size, f.mtime, sha, f"skipped: {reason}"))
            report.skipped += 1
            continue
        try:
            chunks = chunker.chunk(f.rel, _decode(data))
        except Exception as exc:  # one unparseable file must not stop the whole run
            report.errors.append(f"{f.rel}: {type(exc).__name__}: {exc}")
            store.put_file(FileRecord(f.rel, f.size, f.mtime, sha, f"error: {type(exc).__name__}"))
            continue
        if not chunks:
            store.put_file(FileRecord(f.rel, f.size, f.mtime, sha, "empty"))
            report.skipped += 1
            continue
        pending.append((FileRecord(f.rel, f.size, f.mtime, sha, "indexed", len(chunks)), chunks, prev is None))
        pending_chunks += len(chunks)
        if pending_chunks >= embedder.batch_size * 4:
            flush()

    if pending:
        flush()
    if progress:
        progress(len(todo), len(todo), "")

    store.set_meta("last_learn", str(time.time()))
    report.missing_grammars = chunker.missing_grammars
    report.seconds = time.perf_counter() - started
    return report
