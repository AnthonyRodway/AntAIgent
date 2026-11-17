"""Turn retrieved excerpts and a question into a prompt for the chat model."""

from __future__ import annotations

from pathlib import PurePosixPath

from .retrieval import Hit

SYSTEM_PROMPT = """\
You are AntAIgent, a local assistant that answers questions about one project folder \
using excerpts retrieved from its files.

- Base the answer on the excerpts. If they don't contain enough to answer, say what is \
missing rather than guessing, and suggest files or terms that might help.
- Cite the excerpts you rely on inline as `path:start-end`, copied from their headers.
- When you show code, quote it exactly as it appears in the excerpts.
- Be direct and concise. Use Markdown."""


def format_hit(hit: Hit) -> str:
    c = hit.chunk
    unit = "cells" if c.kind == "notebook" else "lines"
    symbols = f" | {', '.join(c.symbols[:6])}" if c.symbols else ""
    lang = PurePosixPath(c.path).suffix.lstrip(".").lower()
    # Four-backtick fences survive excerpts that themselves contain ``` blocks.
    return f"### {c.location} ({unit}){symbols}\n````{lang}\n{c.text}\n````"


def build_messages(
    question: str, hits: list[Hit], project: str, context_chars: int
) -> tuple[list[dict[str, str]], list[Hit]]:
    """Pack the best excerpts that fit in the context budget; return messages and the hits used."""
    blocks: list[str] = []
    used: list[Hit] = []
    total = 0
    for hit in hits:
        block = format_hit(hit)
        if used and total + len(block) > context_chars:
            break
        blocks.append(block)
        used.append(hit)
        total += len(block)
    excerpts = "\n\n".join(blocks)
    user = f"Project: {project}\n\nExcerpts, most relevant first:\n\n{excerpts}\n\nQuestion: {question}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}], used
