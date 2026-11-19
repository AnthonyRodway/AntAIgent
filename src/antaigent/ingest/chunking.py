"""Split files into retrievable chunks that keep their line numbers.

Code is split along syntax-tree boundaries (functions, classes, methods) using
tree-sitter, Markdown along headings, notebooks by cell, and everything else into
overlapping line windows. Data files are normally profiled instead (profiling.py);
the CSV/TSV preview here is the fallback when that fails.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath

CODE_LANGUAGES = {
    ".py": "python", ".pyi": "python",
    ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "tsx",
    ".java": "java", ".kt": "kotlin", ".kts": "kotlin", ".scala": "scala",
    ".c": "c", ".h": "c", ".cc": "cpp", ".cpp": "cpp", ".cxx": "cpp", ".hpp": "cpp", ".hh": "cpp",
    ".cs": "csharp", ".go": "go", ".rs": "rust", ".rb": "ruby", ".php": "php", ".swift": "swift",
    ".r": "r", ".jl": "julia", ".lua": "lua",
    ".sh": "bash", ".bash": "bash", ".ps1": "powershell", ".psm1": "powershell",
    ".sql": "sql",
}
DOC_EXTENSIONS = frozenset({".md", ".markdown", ".mdx"})
DATA_EXTENSIONS = frozenset({".csv", ".tsv"})
NOTEBOOK_EXTENSIONS = frozenset({".ipynb"})

DATA_PREVIEW_ROWS = 20


@dataclass
class Chunk:
    text: str
    start_line: int  # 1-based and inclusive; cell numbers for notebooks
    end_line: int
    kind: str  # code | doc | text | data | notebook
    symbols: list[str] = field(default_factory=list)


@dataclass
class _Unit:
    start: int  # 0-based row, inclusive
    end: int
    symbol: str


def split_lines(text: str) -> list[str]:
    # Not str.splitlines(): it also breaks on form feeds and Unicode line
    # separators, which would put line numbers out of step with tree-sitter's rows.
    lines = [line.rstrip("\r") for line in text.split("\n")]
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def line_windows(lines: list[str], a: int, b: int, max_chars: int, kind: str, overlap: int = 2) -> list[Chunk]:
    out: list[Chunk] = []
    i = a
    while i <= b:
        j, size = i, 0
        while j <= b and (j == i or size + len(lines[j]) + 1 <= max_chars):
            size += len(lines[j]) + 1
            j += 1
        out.append(Chunk("\n".join(lines[i:j])[:max_chars], i + 1, j, kind))
        if j > b:
            break
        i = max(j - overlap, i + 1)
    return out


def _merge_units(units: list[_Unit], lines: list[str], max_chars: int, kind: str) -> list[Chunk]:
    """Greedily pack consecutive units into chunks of at most max_chars.

    Chunks are contiguous line spans, so comments and blank lines between units
    stay attached to the following unit and every line lands in some chunk.
    """
    if not lines:
        return []
    if not units:
        return line_windows(lines, 0, len(lines) - 1, max_chars, kind)

    prefix = [0]
    for line in lines:
        prefix.append(prefix[-1] + len(line) + 1)

    def size(a: int, b: int) -> int:
        return prefix[b + 1] - prefix[a]

    chunks: list[Chunk] = []

    def flush(a: int, b: int, symbols: list[str]) -> None:
        if b < a:
            return
        if size(a, b) <= max_chars:
            pieces = [Chunk("\n".join(lines[a : b + 1]), a + 1, b + 1, kind)]
        else:
            pieces = line_windows(lines, a, b, max_chars, kind)
        unique = list(dict.fromkeys(s for s in symbols if s))
        for piece in pieces:
            piece.symbols = list(unique)
        chunks.extend(pieces)

    last = len(lines) - 1
    start, end, symbols = 0, -1, []
    for unit in sorted(units, key=lambda u: (u.start, u.end)):
        unit_end = min(unit.end, last)
        if unit_end <= end:
            symbols.append(unit.symbol)
            continue
        if end >= start and size(start, unit_end) > max_chars:
            flush(start, end, symbols)
            start, symbols = end + 1, []
        end = unit_end
        symbols.append(unit.symbol)
    flush(start, last, symbols)
    return chunks


_DEF_WORDS = ("function", "method", "class", "struct", "interface", "enum", "impl", "trait", "module", "namespace", "constructor")
_NOT_DEF_WORDS = ("call", "body", "parameter", "argument", "expression", "modifier", "_list")


def _symbol(node) -> str:
    kind = node.type
    if kind == "decorated_definition":
        inner = node.child_by_field_name("definition")
        return _symbol(inner) if inner is not None else ""
    if not any(w in kind for w in _DEF_WORDS) or any(w in kind for w in _NOT_DEF_WORDS):
        return ""
    name = node.child_by_field_name("name")
    if name is None or name.text is None:
        return ""
    return name.text.decode("utf-8", "replace")


def _end_row(node) -> int:
    row, col = node.end_point
    return row - 1 if col == 0 and row > node.start_point[0] else row


def _collect_units(node, max_bytes: int, prefix: str, out: list[_Unit]) -> None:
    """Descend into syntax nodes that are too big to be one chunk on their own."""
    for child in node.children:
        sym = _symbol(child)
        qualified = f"{prefix}.{sym}" if prefix and sym else sym
        if child.end_byte - child.start_byte > max_bytes and child.child_count:
            before = len(out)
            _collect_units(child, max_bytes, qualified or prefix, out)
            if len(out) == before:
                out.append(_Unit(child.start_point[0], _end_row(child), qualified))
            elif qualified and not out[before].symbol:
                out[before].symbol = qualified
        else:
            out.append(_Unit(child.start_point[0], _end_row(child), qualified))


# Closing #s and spaces are trimmed in Python: doing it in the pattern, as
# (.*?)\s*#*\s*$, backtracks so badly that one long line can stall learning for hours.
_HEADING = re.compile(r"^(#{1,6})[ \t]+(.*)$")


def _markdown(lines: list[str], max_chars: int) -> list[Chunk]:
    sections: list[_Unit] = []
    stack: list[tuple[int, str]] = []
    in_fence = False
    start, crumb = 0, ""
    for i, line in enumerate(lines):
        if line.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
            continue
        match = None if in_fence else _HEADING.match(line)
        if match:
            if i > start:
                sections.append(_Unit(start, i - 1, crumb))
            level = len(match.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, match.group(2).rstrip().rstrip("#").rstrip()))
            start, crumb = i, " > ".join(title for _, title in stack)
    sections.append(_Unit(start, len(lines) - 1, crumb))
    return _merge_units(sections, lines, max_chars, "doc")


def _notebook(text: str, max_chars: int) -> list[Chunk]:
    cells = json.loads(text).get("cells", [])
    blocks: list[tuple[int, str]] = []
    for number, cell in enumerate(cells, 1):
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        if source.strip():
            blocks.append((number, f"# --- cell {number} [{cell.get('cell_type', 'code')}] ---\n{source.strip()}"))

    chunks: list[Chunk] = []
    group: list[tuple[int, str]] = []
    size = 0

    def flush() -> None:
        if group:
            chunks.append(Chunk("\n\n".join(b for _, b in group), group[0][0], group[-1][0], "notebook"))

    for number, block in blocks:
        if len(block) > max_chars:
            flush()
            group, size = [], 0
            block_lines = block.split("\n")
            for piece in line_windows(block_lines, 0, len(block_lines) - 1, max_chars, "notebook"):
                piece.start_line = piece.end_line = number
                chunks.append(piece)
            continue
        if group and size + len(block) + 2 > max_chars:
            flush()
            group, size = [], 0
        group.append((number, block))
        size += len(block) + 2
    flush()
    return chunks


def _data_preview(rel: str, text: str, max_chars: int) -> list[Chunk]:
    rows = split_lines(text)[: DATA_PREVIEW_ROWS + 1]
    if not rows:
        return []
    delimiter = "\t" if rel.lower().endswith(".tsv") else ","
    try:
        columns = next(csv.reader([rows[0]], delimiter=delimiter))
    except (csv.Error, StopIteration):
        columns = []
    columns = [c.strip() for c in columns if c.strip()]
    body = (
        f"Data file: {rel}\n"
        f"Columns ({len(columns)}): {', '.join(columns)}\n"
        f"First {len(rows) - 1} rows:\n" + "\n".join(rows)
    )
    return [Chunk(body[:max_chars], 1, len(rows), "data", columns[:50])]


class Chunker:
    def __init__(self, max_chars: int = 2400):
        self.max_chars = max_chars
        self.missing_grammars: set[str] = set()
        self._parsers: dict[str, object] = {}
        self._installed: set[str] | None = None

    def parser(self, language: str):
        """The tree-sitter parser for `language`, or None if its grammar isn't installed."""
        if language not in self._parsers:
            parser = None
            try:
                import tree_sitter_language_pack as tslp

                if self._installed is None:
                    self._installed = set(tslp.downloaded_languages())
                # Only grammars already on disk: get_parser() would otherwise download
                # one, and learning must work offline. `antai setup` fetches them.
                if language in self._installed:
                    parser = tslp.get_parser(language)
            except Exception:
                parser = None
            self._parsers[language] = parser
        return self._parsers[language]

    def chunk(self, rel: str, text: str) -> list[Chunk]:
        ext = PurePosixPath(rel).suffix.lower()
        if ext in DATA_EXTENSIONS:
            chunks = _data_preview(rel, text, self.max_chars)
        elif ext in NOTEBOOK_EXTENSIONS:
            try:
                chunks = _notebook(text, self.max_chars)
            except (ValueError, AttributeError):
                lines = split_lines(text)
                chunks = line_windows(lines, 0, len(lines) - 1, self.max_chars, "text") if lines else []
        else:
            lines = split_lines(text)
            if not lines:
                return []
            if ext in DOC_EXTENSIONS:
                chunks = _markdown(lines, self.max_chars)
            elif ext in CODE_LANGUAGES:
                chunks = self._code(CODE_LANGUAGES[ext], text, lines)
            else:
                chunks = line_windows(lines, 0, len(lines) - 1, self.max_chars, "text")
        return [c for c in chunks if c.text.strip()]

    def _code(self, language: str, text: str, lines: list[str]) -> list[Chunk]:
        parser = self.parser(language)
        if parser is None:
            self.missing_grammars.add(language)
            return line_windows(lines, 0, len(lines) - 1, self.max_chars, "code")
        tree = parser.parse(text.encode("utf-8"))
        units: list[_Unit] = []
        _collect_units(tree.root_node, self.max_chars, "", units)
        return _merge_units(units, lines, self.max_chars, "code")
