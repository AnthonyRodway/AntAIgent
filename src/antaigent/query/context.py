"""Gather everything worth showing the model for one question.

That is the excerpts from hybrid search, plus (when the index has them) the project
overview, the most relevant file and folder summaries, and code-graph facts about any
symbols or files the question names: where they are defined, what calls them, and
what imports what.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from ..index.store import STOPWORDS, Store, StoredChunk, SymbolRow
from ..ingest.chunking import CODE_LANGUAGES, DOC_EXTENSIONS, NOTEBOOK_EXTENSIONS
from ..ingest.pdf import PDF_EXTENSIONS
from ..ingest.profiling import PROFILE_EXTENSIONS
from .answer import build_messages
from .retrieval import Hit, QueryEmbedder, search

MAX_SYMBOLS = 3
MAX_FILES = 2
MAX_CALLERS = 12
MAX_CALLEES = 20
MAX_SUMMARIES = 6
OVERVIEW_CHARS = 2_500
GRAPH_CHARS = 3_500
SUMMARY_CHARS = 3_000
CODE_KINDS = ("function", "method", "class", "interface", "module", "macro")

_TOKEN = re.compile(r"`?([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)(`|\(\))?")
_FILE_TOKEN = re.compile(r"[\w./\\-]*\w\.[A-Za-z][A-Za-z0-9]{0,6}\b")
_FILE_EXTENSIONS = (
    set(CODE_LANGUAGES) | DOC_EXTENSIONS | NOTEBOOK_EXTENSIONS | PROFILE_EXTENSIONS | PDF_EXTENSIONS
    | {".txt", ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".xml", ".html", ".css", ".rst", ".bat", ".cmd"}
)


def _is_file_token(token: str) -> bool:
    return "." in token and "." + token.rsplit(".", 1)[1].lower() in _FILE_EXTENSIONS


@dataclass
class Context:
    messages: list[dict[str, str]]
    hits: list[Hit]  # excerpts given to the model
    symbols: list[str] = field(default_factory=list)  # code symbols recognised in the question
    files: list[str] = field(default_factory=list)  # files recognised in the question
    summaries: list[str] = field(default_factory=list)  # paths whose summaries were included


def _looks_like_code(token: str, marker: str | None) -> bool:
    return bool(marker) or "_" in token or "." in token or (token != token.lower() and token != token.upper())


def recognise_symbols(store: Store, question: str, limit: int = MAX_SYMBOLS) -> list[str]:
    """Names in the question that are defined in the project, most code-like first."""
    candidates: list[tuple[str, bool]] = []
    for match in _TOKEN.finditer(question):
        token, marker = match.group(1).strip("."), match.group(2)
        if len(token) < 3 or token.lower() in STOPWORDS or _is_file_token(token):
            continue
        candidates.append((token, _looks_like_code(token, marker)))
        if "." in token:
            candidates.append((token.rsplit(".", 1)[-1], True))
    defined = store.defined_names(t for t, _ in candidates)
    chosen: list[str] = []
    for code_like in (True, False):
        for token, is_code in candidates:
            if is_code != code_like or token not in defined or token in chosen:
                continue
            # A plain English word ("validate", "run") only counts if exactly one thing has that name.
            if not is_code and len(store.find_definitions(token, limit=2)) != 1:
                continue
            chosen.append(token)
    # `log` adds nothing once `Logger.log` is chosen.
    chosen = [t for t in chosen if not any(o != t and o.endswith("." + t) for o in chosen)]
    return chosen[:limit]


def recognise_files(store: Store, question: str, limit: int = MAX_FILES) -> list[str]:
    found: list[str] = []
    for token in _FILE_TOKEN.findall(question):
        if not _is_file_token(token):
            continue
        matches = store.match_paths(token, limit=4)
        if matches and len(matches) <= 3 and matches[0] not in found:
            found.append(matches[0])
    return found[:limit]


def _where(ref) -> str:
    return f"{ref.path}:{ref.line} (in {ref.caller or 'module level'})"


def symbol_facts(store: Store, name: str) -> tuple[list[str], list[SymbolRow]]:
    defs = store.find_definitions(name, limit=8)
    if not defs:
        return [], []
    lines: list[str] = []
    if len(defs) == 1:
        d = defs[0]
        lines.append(f"- `{d.qualname}` ({d.kind}) is defined at {d.location}.")
    else:
        listed = ", ".join(f"`{d.qualname}` ({d.kind}) at {d.location}" for d in defs[:5])
        more = f", and {len(defs) - 5} more" if len(defs) > 5 else ""
        lines.append(f"- `{name}` has {len(defs)} definitions: {listed}{more}.")
    refs = store.references_to(name, limit=200)
    if refs:
        shown = ", ".join(_where(r) for r in refs[:MAX_CALLERS])
        more = f", and {len(refs) - MAX_CALLERS} more" if len(refs) > MAX_CALLERS else ""
        lines.append(f"  Referenced or called from {len(refs)} place(s): {shown}{more}.")
    else:
        lines.append("  No calls to it were found by static analysis (it may be unused, an entry point, or called dynamically).")
    if len(defs) == 1 and defs[0].kind in CODE_KINDS:
        d = defs[0]
        callees = list(dict.fromkeys(r.name for r in store.calls_within(d.path, d.qualname)))
        if callees:
            more = f", and {len(callees) - MAX_CALLEES} more" if len(callees) > MAX_CALLEES else ""
            lines.append(f"  It calls: {', '.join(callees[:MAX_CALLEES])}{more}.")
    if len(defs) == 1:
        importers = sorted({i.path for i in store.imported_by(defs[0].path)})
        if importers:
            more = f", and {len(importers) - MAX_CALLERS} more" if len(importers) > MAX_CALLERS else ""
            lines.append(f"  Its file {defs[0].path} is imported by: {', '.join(importers[:MAX_CALLERS])}{more}.")
    return lines, defs


def file_facts(store: Store, path: str) -> list[str]:
    imports = store.imports_of(path)
    resolved = {(i.line, i.module) for i in imports if i.target}
    local = sorted({i.target for i in imports if i.target})
    external = sorted({i.module for i in imports if (i.line, i.module) not in resolved})
    importers = sorted({i.path for i in store.imported_by(path)})
    defs = [d.qualname for d in store.definitions_in(path) if "." not in d.qualname and d.kind in CODE_KINDS]
    lines = [f"- `{path}`:"]
    if defs:
        lines.append(f"  Defines: {', '.join(defs[:25])}{' ...' if len(defs) > 25 else ''}.")
    if local:
        lines.append(f"  Imports project files: {', '.join(local)}.")
    if external:
        lines.append(f"  Imports other modules: {', '.join(external[:25])}.")
    lines.append(f"  Imported by: {', '.join(importers)}." if importers else "  No project file imports it.")
    return lines


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 4].rstrip() + " ..."


def summary_matches(store: Store, vector: np.ndarray | None, k: int = 4) -> list[tuple[str, str]]:
    if vector is None:
        return []
    keys, matrix = store.summary_vectors()
    if not keys or matrix.shape[1] != len(vector):
        return []
    sims = matrix @ vector
    return [keys[i] for i in np.argsort(-sims)[:k]]


_REFERRING = frozenset("it its it's this that these those they them their there he she".split())


def search_text(question: str, history: Sequence[tuple[str, str]] = ()) -> str:
    """What to search for. A short or referring follow-up ("which files import it?") means
    little on its own, so it is searched together with the previous question."""
    if not history:
        return question
    words = re.findall(r"[a-z']+", question.lower())
    if len(words) <= 12 or _REFERRING.intersection(words):
        return f"{history[-1][0]}\n{question}"
    return question


def gather(
    store: Store,
    embedder: QueryEmbedder,
    question: str,
    *,
    project: str,
    k: int,
    context_chars: int,
    history: Sequence[tuple[str, str]] = (),
) -> Context:
    query = search_text(question, history)
    ids, _ = store.vectors()
    vector = embedder.embed_query(query) if len(ids) else None
    hits = search(store, embedder, query, k, vector=vector)

    symbols = recognise_symbols(store, query)
    files = recognise_files(store, query)
    graph_lines: list[str] = []
    definition_chunks: list[StoredChunk] = []
    for name in symbols:
        lines, defs = symbol_facts(store, name)
        graph_lines += lines
        for d in defs[:2]:
            chunk = store.chunk_at(d.path, d.start_line)
            if chunk is not None:
                definition_chunks.append(chunk)
    for path in files:
        graph_lines += file_facts(store, path)

    # Definitions of the symbols asked about go right after the best match.
    seen = {h.chunk.id for h in hits}
    extra = [Hit(c, 0.0, None, None) for c in definition_chunks if c.id not in seen and not seen.add(c.id)]
    if extra:
        hits = hits[:1] + extra + hits[1:]

    notes: list[tuple[str, str]] = []
    overview = store.get_summary("project", "")
    if overview is not None:
        notes.append(("Project overview", _truncate(overview.text, OVERVIEW_CHARS)))
    if graph_lines:
        notes.append(("Code graph facts (static analysis; may be incomplete)", _truncate("\n".join(graph_lines), GRAPH_CHARS)))

    summary_keys = summary_matches(store, vector)
    for hit in hits[:4]:
        summary_keys.append(("file", hit.chunk.path))
    summary_lines: list[str] = []
    used_summaries: list[str] = []
    for scope, key in dict.fromkeys(summary_keys):
        if len(used_summaries) >= MAX_SUMMARIES:
            break
        s = store.get_summary(scope, key)
        if s is None:
            continue
        label = f"{key}/" if scope == "dir" else key
        summary_lines.append(f"- `{label}`: {s.text}")
        used_summaries.append(label)
    if summary_lines:
        notes.append(("Summaries of relevant files and folders", _truncate("\n".join(summary_lines), SUMMARY_CHARS)))

    messages, used = build_messages(question, hits, project, context_chars, notes=notes, history=history)
    return Context(messages, used, symbols, files, used_summaries)
