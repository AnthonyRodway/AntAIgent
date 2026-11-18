"""Write short summaries of every file and folder, and an overview of the whole project.

Summaries come from the chat model and are stored with a fingerprint of what they
describe, so re-running only redoes the files that changed and the folders above them.
`ask` uses them for broad questions ("what does this project do?", "where is
validation handled?") that no single excerpt answers.
"""

from __future__ import annotations

import hashlib
import posixpath
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from .indexer import DocumentEmbedder
from .store import Store, StoredChunk, Summary

# Bump when the prompts change enough that old summaries should be redone.
SUMMARY_VERSION = 1
FILE_INPUT_CHARS = 12_000
CHILDREN_INPUT_CHARS = 10_000
README_CHARS = 4_000
MAX_SUMMARY_CHARS = 1_500
MAX_CONSECUTIVE_ERRORS = 3

CompleteFn = Callable[[list[dict[str, str]]], str]
ProgressFn = Callable[[int, int, str], None]
StopFn = Callable[[], bool]

SYSTEM_PROMPT = """\
You write short, factual summaries of the files and folders in a software or data project, \
for a developer who is new to it. Start directly with what the thing does (for example \
"Defines..." or "Holds..."), never with "This file". Name the important classes, functions, \
tables or columns. Plain prose: no headings and no bullet lists unless asked. Don't guess \
beyond what you are shown."""


@dataclass
class SummaryReport:
    files: int = 0
    dirs: int = 0
    project: bool = False
    unchanged: int = 0
    removed: int = 0
    seconds: float = 0.0
    stopped: bool = False
    errors: list[str] = field(default_factory=list)


def _fingerprint(*parts: str) -> str:
    return f"v{SUMMARY_VERSION}:" + hashlib.sha1("\n".join(parts).encode("utf-8")).hexdigest()


def file_text(chunks: list[StoredChunk]) -> str:
    """Reassemble a file from its chunks, dropping the lines that overlapping windows repeat."""
    parts: list[str] = []
    last_end = 0
    for c in chunks:
        if c.kind in ("data", "notebook"):
            parts.append(("\n" if parts else "") + c.text)
            continue
        lines = c.text.split("\n")
        if c.start_line <= last_end:
            lines = lines[last_end - c.start_line + 1 :]
        if lines:
            parts.append("\n".join(lines))
        last_end = max(last_end, c.end_line)
    return "\n".join(parts)


def _outline(store: Store, path: str, limit: int = 60) -> str:
    defs = [d for d in store.definitions_in(path) if d.kind in ("class", "function", "method", "interface", "module")]
    if not defs:
        return ""
    names = [f"{d.qualname} ({d.kind})" for d in defs[:limit]]
    more = f", and {len(defs) - limit} more" if len(defs) > limit else ""
    return "Definitions: " + ", ".join(names) + more


def _clean(text: str) -> str:
    text = text.strip().strip('"').strip()
    return text if len(text) <= MAX_SUMMARY_CHARS else text[: MAX_SUMMARY_CHARS - 3].rstrip() + "..."


def _file_prompt(project: str, path: str, text: str, outline: str, kind: str) -> str:
    truncated = len(text) > FILE_INPUT_CHARS
    body = text[:FILE_INPUT_CHARS]
    lang = PurePosixPath(path).suffix.lstrip(".").lower()
    parts = [f"Project: {project}", f"File: {path}"]
    if outline and (truncated or kind == "code"):
        parts.append(outline)
    parts.append(f"Content{' (first part only)' if truncated else ''}:\n````{lang}\n{body}\n````")
    if kind == "data":
        parts.append("Summarize this data file in 1-3 sentences: what the data appears to describe and its key columns.")
    else:
        parts.append(
            "Summarize this file in 2-4 sentences: what it is for, its main parts, and how it fits into the project."
        )
    return "\n\n".join(parts)


def _children_block(children: list[tuple[str, str, str]]) -> str:
    """children: (scope, key, summary text)."""
    per_child = max(200, CHILDREN_INPUT_CHARS // max(len(children), 1))
    lines = []
    for scope, key, text in children:
        label = f"folder `{key.rsplit('/', 1)[-1]}/`" if scope == "dir" else f"file `{key.rsplit('/', 1)[-1]}`"
        short = text if len(text) <= per_child else text[: per_child - 3] + "..."
        lines.append(f"- {label}: {short}")
    return "\n".join(lines)


def _tree(paths: list[str]) -> dict[str, list[tuple[str, str]]]:
    """Map each folder ("" is the root) to its children as (scope, key)."""
    children: dict[str, list[tuple[str, str]]] = {"": []}
    for path in sorted(paths):
        parent = posixpath.dirname(path)
        children.setdefault(parent, []).append(("file", path))
        while parent:
            grand = posixpath.dirname(parent)
            siblings = children.setdefault(grand, [])
            if ("dir", parent) in siblings:
                break
            siblings.append(("dir", parent))
            children.setdefault(parent, [])
            parent = grand
    return children


def summarize(
    project: str,
    store: Store,
    embedder: DocumentEmbedder,
    complete: CompleteFn,
    *,
    model: str,
    force: bool = False,
    progress: ProgressFn | None = None,
    should_stop: StopFn | None = None,
) -> SummaryReport:
    started = time.perf_counter()
    report = SummaryReport()
    files = {p: r for p, r in store.files().items() if r.status == "indexed"}
    existing = store.summaries()
    tree = _tree(list(files))
    dirs = sorted((d for d in tree if d), key=lambda d: (-d.count("/"), d))

    stale_dirs = [k for (scope, k) in existing if scope == "dir" and k not in tree]
    if stale_dirs:
        store.delete_summaries("dir", stale_dirs)
        report.removed += len(stale_dirs)

    file_todo = [
        p for p in sorted(files)
        if force or (s := existing.get(("file", p))) is None or s.fingerprint != _fingerprint(files[p].sha1)
    ]
    total = len(file_todo) + len(dirs) + 1
    done = 0
    consecutive_errors = 0
    fingerprints: dict[tuple[str, str], str] = {}
    texts: dict[tuple[str, str], str] = {}
    for key, s in existing.items():
        fingerprints[key], texts[key] = s.fingerprint, s.text

    def stopped() -> bool:
        if should_stop is not None and should_stop():
            report.stopped = True
        return report.stopped

    def generate(scope: str, key: str, fingerprint: str, prompt: str) -> bool:
        nonlocal consecutive_errors
        try:
            text = _clean(complete([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]))
        except Exception as exc:
            report.errors.append(f"{key or project}: {exc}")
            consecutive_errors += 1
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                report.stopped = True
            return False
        consecutive_errors = 0
        if not text:
            report.errors.append(f"{key or project}: the model returned an empty summary")
            return False
        label = key or project
        vector = embedder.embed_documents([f"{label} | summary\n{text}"])[0] if scope != "project" else None
        store.put_summary(Summary(scope, key, fingerprint, model, text, time.time()), vector)
        fingerprints[(scope, key)], texts[(scope, key)] = fingerprint, text
        return True

    # Files
    report.unchanged += len(files) - len(file_todo)
    for path in file_todo:
        if stopped():
            break
        if progress:
            progress(done, total, path)
        chunks = store.chunks_for_file(path)
        if chunks:
            kind = chunks[0].kind
            prompt = _file_prompt(project, path, file_text(chunks), _outline(store, path), kind)
            if generate("file", path, _fingerprint(files[path].sha1), prompt):
                report.files += 1
        done += 1

    # Folders, deepest first, then the project overview from the top level.
    for key in [*dirs, ""]:
        if stopped():
            break
        scope = "dir" if key else "project"
        if progress:
            progress(done, total, f"{key}/" if key else "project overview")
        done += 1
        kids = [(s, k) for s, k in tree.get(key, []) if (s, k) in texts]
        if not kids:
            continue
        fingerprint = _fingerprint(*(f"{s}:{k}:{fingerprints[(s, k)]}" for s, k in kids))
        current = existing.get((scope, key))
        if not force and current is not None and current.fingerprint == fingerprint:
            report.unchanged += 1
            continue
        block = _children_block([(s, k, texts[(s, k)]) for s, k in kids])
        if key:
            prompt = (
                f"Project: {project}\nFolder: {key}/\n\nContents:\n{block}\n\n"
                "Summarize what this folder contains and is for in 1-3 sentences."
            )
            if generate("dir", key, fingerprint, prompt):
                report.dirs += 1
        else:
            readme = _readme(store, files)
            readme_part = f"README (excerpt):\n{readme}\n\n" if readme else ""
            prompt = (
                f"Project folder: {project}\n\n{readme_part}Top-level contents:\n{block}\n\n"
                "Write an overview of this project for a developer new to it: one short paragraph on what it is "
                "and does, then a bullet list of its main parts with their paths in backticks. Under 250 words. "
                "Markdown is fine here."
            )
            report.project = generate("project", "", fingerprint, prompt)

    if progress:
        progress(total, total, "")
    report.seconds = time.perf_counter() - started
    return report


def _readme(store: Store, files: dict) -> str:
    for path in sorted(files, key=lambda p: (p.count("/"), p.lower())):
        if "/" not in path and path.lower().startswith("readme"):
            return file_text(store.chunks_for_file(path))[:README_CHARS]
    return ""
