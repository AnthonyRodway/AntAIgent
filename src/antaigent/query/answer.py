"""Turn retrieved material and a question into a prompt for the chat model."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import PurePosixPath

from .retrieval import Hit

SYSTEM_PROMPT = """\
You are AntAIgent, a local assistant that answers questions about one project folder \
using material retrieved from it: excerpts of its files, and when available a project \
overview, summaries of files and folders, and code-graph facts found by static analysis \
(where things are defined, what calls them, what imports what).

- Base the answer on this material. If it doesn't contain enough to answer, say what is \
missing rather than guessing, and suggest files or terms that might help.
- Cite the excerpts you rely on inline as `path:start-end`, copied from their headers. \
Cite summaries and graph facts by the file path or location they give.
- When you show code, quote it exactly as it appears in the excerpts.
- Be direct and concise. Use Markdown."""

MAX_HISTORY_ANSWER = 2_000


def format_hit(hit: Hit) -> str:
    c = hit.chunk
    unit = {"notebook": "cells", "data": "data profile", "pdf": "pages"}.get(c.kind, "lines")
    symbols = f" | {', '.join(c.symbols[:6])}" if c.symbols else ""
    lang = PurePosixPath(c.path).suffix.lstrip(".").lower()
    # Four-backtick fences survive excerpts that themselves contain ``` blocks.
    return f"### {c.location} ({unit}){symbols}\n````{lang}\n{c.text}\n````"


def build_messages(
    question: str,
    hits: list[Hit],
    project: str,
    context_chars: int,
    *,
    notes: Sequence[tuple[str, str]] = (),
    history: Sequence[tuple[str, str]] = (),
) -> tuple[list[dict[str, str]], list[Hit]]:
    """Pack the notes, then the best excerpts that fit in the context budget.

    `notes` are (title, text) sections such as the project overview; `history` is
    earlier (question, answer) pairs from the same conversation.
    Returns the messages and the hits that were used."""
    sections = [f"Project: {project}"]
    total = 0
    for title, text in notes:
        if text.strip():
            sections.append(f"{title}:\n{text.strip()}")
            total += len(text)

    blocks: list[str] = []
    used: list[Hit] = []
    for hit in hits:
        block = format_hit(hit)
        if used and total + len(block) > context_chars:
            break
        blocks.append(block)
        used.append(hit)
        total += len(block)
    if blocks:
        sections.append("Excerpts, most relevant first:\n\n" + "\n\n".join(blocks))
    sections.append(f"Question: {question}")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for earlier_question, earlier_answer in history:
        answer = earlier_answer if len(earlier_answer) <= MAX_HISTORY_ANSWER else earlier_answer[:MAX_HISTORY_ANSWER] + "..."
        messages.append({"role": "user", "content": earlier_question})
        messages.append({"role": "assistant", "content": answer})
    messages.append({"role": "user", "content": "\n\n".join(sections)})
    return messages, used
