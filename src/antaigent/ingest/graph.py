"""Static code facts: where symbols are defined, where they are called from, and what
each file imports.

Definitions and call sites come from the tree-sitter "tags" queries that ship with
each grammar (the convention GitHub's code navigation uses). Imports are found with
per-language patterns and resolved to project files where possible. All of it is best
effort: dynamic calls, re-exports and unusual import styles are missed.
"""

from __future__ import annotations

import posixpath
import re
from bisect import bisect_right
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from .chunking import CODE_LANGUAGES, _end_row, _symbol, split_lines

MAX_SNIPPET = 160


@dataclass(frozen=True)
class Definition:
    name: str
    qualname: str  # with enclosing classes and functions, e.g. "Store.put_file"
    kind: str  # class | function | method | interface | module | constant | type | ...
    start_line: int
    end_line: int


@dataclass(frozen=True)
class Reference:
    name: str
    kind: str  # call | class | type | implementation
    line: int
    caller: str  # qualname of the innermost enclosing definition; "" at module level
    snippet: str


@dataclass(frozen=True)
class Import:
    module: str  # as written in the source, e.g. "os", ".store", "./util", "stdio.h"
    line: int
    names: tuple[str, ...] = ()


@dataclass
class FileGraph:
    language: str
    definitions: list[Definition] = field(default_factory=list)
    references: list[Reference] = field(default_factory=list)
    imports: list[Import] = field(default_factory=list)


_CLASS_KINDS = frozenset({"class", "interface", "struct", "trait", "impl"})
_REFERENCE_KINDS = {"send": "call"}


class GraphExtractor:
    def __init__(self, parser_for: Callable[[str], object | None]):
        self._parser_for = parser_for
        self._queries: dict[str, object | None] = {}

    def _query(self, language: str):
        if language not in self._queries:
            query = None
            try:
                import tree_sitter as ts
                import tree_sitter_language_pack as tslp

                source = tslp.get_tags_query(language)
                if source:
                    query = ts.Query(tslp.get_language(language), source)
            except Exception:
                query = None
            self._queries[language] = query
        return self._queries[language]

    def extract(self, rel: str, text: str) -> FileGraph | None:
        language = CODE_LANGUAGES.get(PurePosixPath(rel).suffix.lower())
        if language is None:
            return None
        graph = FileGraph(language, imports=find_imports(language, text))
        parser = self._parser_for(language)
        if parser is None:
            return graph
        tree = parser.parse(text.encode("utf-8"))
        lines = split_lines(text)
        query = self._query(language)
        if query is not None:
            graph.definitions, graph.references = _tags(query, tree, lines)
        else:
            graph.definitions = _definitions_by_walking(tree)
        return graph


def _text(node) -> str:
    return node.text.decode("utf-8", "replace") if node.text is not None else ""


def _qualify(raw: list[tuple[int, int, str, str, int, int]]) -> tuple[list[Definition], list[tuple[int, int, str]]]:
    """Give each definition its qualified name; return definitions and (start, end, qualname) spans."""
    raw.sort(key=lambda d: (d[0], -d[1]))
    definitions: list[Definition] = []
    spans: list[tuple[int, int, str]] = []
    stack: list[tuple[int, str, str]] = []  # (end_byte, qualname, kind)
    for start, end, name, kind, start_row, end_row in raw:
        while stack and stack[-1][0] <= start:
            stack.pop()
        parent = stack[-1] if stack else None
        qualname = f"{parent[1]}.{name}" if parent else name
        if kind == "function" and parent and parent[2] in _CLASS_KINDS:
            kind = "method"
        definitions.append(Definition(name, qualname, kind, start_row + 1, end_row + 1))
        spans.append((start, end, qualname))
        stack.append((end, qualname, kind))
    return definitions, spans


def _tags(query, tree, lines: list[str]) -> tuple[list[Definition], list[Reference]]:
    from tree_sitter import QueryCursor

    raw_defs: dict[tuple[int, int, str], tuple[int, int, str, str, int, int]] = {}
    raw_refs: dict[tuple[int, str], tuple[int, str, str, int]] = {}
    for _, captures in QueryCursor(query).matches(tree.root_node):
        name_nodes = captures.get("name")
        if not name_nodes:
            continue
        name_node = name_nodes[0]
        name = _text(name_node).strip()
        if not name:
            continue
        for capture, nodes in captures.items():
            if capture.startswith("definition."):
                node = nodes[0]
                key = (node.start_byte, node.end_byte, name)
                raw_defs.setdefault(
                    key,
                    (node.start_byte, node.end_byte, name, capture.split(".", 1)[1], node.start_point[0], _end_row(node)),
                )
                break
            if capture.startswith("reference."):
                kind = capture.split(".", 1)[1]
                raw_refs.setdefault(
                    (name_node.start_byte, name),
                    (name_node.start_byte, name, _REFERENCE_KINDS.get(kind, kind), name_node.start_point[0]),
                )
                break

    definitions, spans = _qualify(list(raw_defs.values()))

    references: list[Reference] = []
    active: list[tuple[int, str]] = []  # (end_byte, qualname) of definitions enclosing the cursor
    i = 0
    for start, name, kind, row in sorted(raw_refs.values()):
        while i < len(spans) and spans[i][0] <= start:
            while active and active[-1][0] <= spans[i][0]:
                active.pop()
            active.append((spans[i][1], spans[i][2]))
            i += 1
        while active and active[-1][0] <= start:
            active.pop()
        snippet = lines[row].strip()[:MAX_SNIPPET] if row < len(lines) else ""
        references.append(Reference(name, kind, row + 1, active[-1][1] if active else "", snippet))
    return definitions, references


def _definitions_by_walking(tree) -> list[Definition]:
    """For grammars without a tags query: definitions only, found by node type."""
    raw: list[tuple[int, int, str, str, int, int]] = []
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        name = _symbol(node)
        if name:
            kind = "class" if "class" in node.type else "function" if "function" in node.type or "method" in node.type else node.type
            raw.append((node.start_byte, node.end_byte, name, kind, node.start_point[0], _end_row(node)))
        stack.extend(reversed(node.named_children))
    return _qualify(raw)[0]


# --- Imports -------------------------------------------------------------------------

_M = re.MULTILINE
_PY_IMPORT = re.compile(r"^[ \t]*import[ \t]+([\w.][\w. \t,]*)", _M)
_PY_FROM = re.compile(r"^[ \t]*from[ \t]+(\.+[\w.]*|[\w.]+)[ \t]+import[ \t]+(\([^)]*\)|[^\n#;]*)", _M)
_JS_FROM = re.compile(r"""^[ \t]*(?:import|export)\b[^'";]*?\bfrom[ \t]*['"]([^'"\n]+)['"]""", _M)
_JS_BARE = re.compile(r"""^[ \t]*import[ \t]*['"]([^'"\n]+)['"]""", _M)
_JS_CALL = re.compile(r"""\b(?:require|import)\s*\(\s*['"]([^'"\n]+)['"]\s*\)""")
_C_INCLUDE = re.compile(r"""^[ \t]*#[ \t]*include[ \t]*[<"]([^>"\n]+)[>"]""", _M)
_JVM_IMPORT = re.compile(r"^[ \t]*import[ \t]+(?:static[ \t]+)?([\w.]+)", _M)
_CS_USING = re.compile(r"^[ \t]*(?:global[ \t]+)?using[ \t]+(?:static[ \t]+)?(?:\w+[ \t]*=[ \t]*)?([\w.]+)[ \t]*;", _M)
_GO_SINGLE = re.compile(r"""^[ \t]*import[ \t]+(?:[\w.]+[ \t]+)?"([^"\n]+)\"""", _M)
_GO_BLOCK = re.compile(r"^[ \t]*import[ \t]*\(([^)]*)\)", _M)
_GO_PATH = re.compile(r'"([^"\n]+)"')
_RUST_USE = re.compile(r"^[ \t]*(?:pub(?:\([^)]*\))?[ \t]+)?use[ \t]+([\w:]+)", _M)
_RUST_MOD = re.compile(r"^[ \t]*(?:pub(?:\([^)]*\))?[ \t]+)?mod[ \t]+(\w+)[ \t]*;", _M)
_RUBY_REQUIRE = re.compile(r"""^[ \t]*(require_relative|require|load)[ \t(]+['"]([^'"\n]+)['"]""", _M)
_PHP_USE = re.compile(r"^[ \t]*use[ \t]+([\w\\]+)", _M)
_PHP_INCLUDE = re.compile(r"""\b(?:require|include)(?:_once)?[ \t(]*['"]([^'"\n]+)['"]""")
_R_LIBRARY = re.compile(r"""\b(?:library|require|requireNamespace)\(\s*["']?([\w.]+)["']?""")
_R_SOURCE = re.compile(r"""\bsource\(\s*["']([^"'\n]+)["']""")
_LUA_REQUIRE = re.compile(r"""\brequire\s*\(?\s*['"]([^'"\n]+)['"]""")
_PS_MODULE = re.compile(r"""^[ \t]*(?:Import-Module[ \t]+(?:-Name[ \t]+)?|using[ \t]+module[ \t]+)['"]?([^\s'";]+)""", _M | re.IGNORECASE)
_PS_DOT = re.compile(r"""^[ \t]*\.[ \t]+['"]?([^\s'"]+\.ps1)""", _M | re.IGNORECASE)
_SH_SOURCE = re.compile(r"""^[ \t]*(?:source|\.)[ \t]+['"]?([^\s'";]+)""", _M)
_JULIA_USING = re.compile(r"^[ \t]*(?:using|import)[ \t]+([\w.]+)", _M)
_JULIA_INCLUDE = re.compile(r'\binclude\(\s*"([^"\n]+)"')
_SWIFT_IMPORT = re.compile(r"^[ \t]*import[ \t]+(?:\w+[ \t]+)?([\w.]+)", _M)

_SIMPLE_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "javascript": (_JS_FROM, _JS_BARE, _JS_CALL),
    "typescript": (_JS_FROM, _JS_BARE, _JS_CALL),
    "tsx": (_JS_FROM, _JS_BARE, _JS_CALL),
    "c": (_C_INCLUDE,),
    "cpp": (_C_INCLUDE,),
    "java": (_JVM_IMPORT,),
    "kotlin": (_JVM_IMPORT,),
    "scala": (_JVM_IMPORT,),
    "csharp": (_CS_USING,),
    "go": (_GO_SINGLE,),
    "rust": (_RUST_USE,),
    "php": (_PHP_USE, _PHP_INCLUDE),
    "r": (_R_LIBRARY, _R_SOURCE),
    "lua": (_LUA_REQUIRE,),
    "powershell": (_PS_MODULE, _PS_DOT),
    "bash": (_SH_SOURCE,),
    "julia": (_JULIA_USING, _JULIA_INCLUDE),
    "swift": (_SWIFT_IMPORT,),
}

_IDENT = re.compile(r"^[A-Za-z_]\w*$")


def _names(raw: str) -> tuple[str, ...]:
    out = []
    for part in raw.strip().strip("()").split(","):
        name = re.split(r"\s+as\s+", part.strip())[0].strip()
        if _IDENT.match(name):
            out.append(name)
    return tuple(out)


def find_imports(language: str, text: str) -> list[Import]:
    starts = [0] + [m.end() for m in re.finditer("\n", text)]

    def line(pos: int) -> int:
        return bisect_right(starts, pos)

    found: list[Import] = []
    if language == "python":
        for m in _PY_IMPORT.finditer(text):
            for part in m.group(1).split(","):
                module = re.split(r"\s+as\s+", part.strip())[0].strip()
                if module:
                    found.append(Import(module, line(m.start())))
        for m in _PY_FROM.finditer(text):
            found.append(Import(m.group(1), line(m.start()), _names(m.group(2))))
    elif language == "ruby":
        for m in _RUBY_REQUIRE.finditer(text):
            module = m.group(2)
            if m.group(1) == "require_relative" and not module.startswith("."):
                module = "./" + module
            found.append(Import(module, line(m.start())))
    else:
        for pattern in _SIMPLE_PATTERNS.get(language, ()):
            for m in pattern.finditer(text):
                found.append(Import(m.group(1).rstrip("."), line(m.start())))
        if language == "go":
            for block in _GO_BLOCK.finditer(text):
                for m in _GO_PATH.finditer(block.group(1)):
                    found.append(Import(m.group(1), line(block.start(1) + m.start())))
        elif language == "rust":
            for m in _RUST_MOD.finditer(text):
                found.append(Import("mod " + m.group(1), line(m.start())))

    unique: dict[tuple[str, int], Import] = {}
    for imp in found:
        unique.setdefault((imp.module, imp.line), imp)
    return sorted(unique.values(), key=lambda i: (i.line, i.module))


# --- Resolving imports to project files ----------------------------------------------

_JS_EXTS = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".vue", ".svelte")


def _clean(path: str) -> str | None:
    path = posixpath.normpath(path.replace("\\", "/")).lstrip("/")
    if path in ("", ".") or path.startswith("../") or path == "..":
        return None
    return path


def _shared_prefix(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a.split("/"), b.split("/")):
        if x != y:
            break
        n += 1
    return n


class ImportResolver:
    """Map an import, as written, to files in the project. Heuristic and language-aware."""

    def __init__(self, paths: Iterable[str]):
        self.paths = set(paths)
        self._by_suffix: dict[str, list[str]] = {}
        for path in self.paths:
            parts = path.split("/")
            for i in range(len(parts)):
                self._by_suffix.setdefault("/".join(parts[i:]), []).append(path)

    def _find(self, candidates: Iterable[str], importer: str, allow_suffix: bool) -> str | None:
        cleaned = [c for c in (_clean(c) for c in candidates) if c]
        for c in cleaned:
            if c in self.paths:
                return c
        if allow_suffix:
            for c in cleaned:
                matches = self._by_suffix.get(c)
                if matches:
                    return max(matches, key=lambda m: (_shared_prefix(m, importer), -len(m), m))
        return None

    def resolve(self, importer: str, language: str, module: str, names: Iterable[str] = ()) -> list[str]:
        here = posixpath.dirname(importer)

        def local(p: str) -> str:
            return posixpath.join(here, p)

        def one(candidates: Iterable[str], allow_suffix: bool) -> list[str]:
            found = self._find(candidates, importer, allow_suffix)
            return [found] if found and found != importer else []

        if language == "python":
            dots = len(module) - len(module.lstrip("."))
            body = module[dots:].replace(".", "/")
            if dots:
                base = here
                for _ in range(dots - 1):
                    base = posixpath.dirname(base)
                stem = posixpath.join(base, body) if body else base
            else:
                stem = body
            targets: list[str] = []
            for name in names:
                targets += one([f"{stem}/{name}.py", f"{stem}/{name}/__init__.py"], not dots)
            if targets:
                return list(dict.fromkeys(targets))
            candidates = [f"{stem}.py", f"{stem}/__init__.py", f"{stem}.pyi"]
            if not dots:
                candidates = [local(f"{body}.py"), local(f"{body}/__init__.py"), *candidates]
            return one(candidates, not dots)

        if language in ("javascript", "typescript", "tsx"):
            if module.startswith("."):
                base, allow = local(module), False
            elif module.startswith(("@/", "~/")):
                base, allow = "src/" + module[2:], True
            else:
                return []
            return one([base, *(base + e for e in _JS_EXTS), *(f"{base}/index{e}" for e in _JS_EXTS)], allow)

        if language in ("c", "cpp"):
            return one([local(module), module], True)

        if language in ("java", "kotlin", "scala"):
            stem = module.replace(".", "/")
            return one([stem + e for e in (".java", ".kt", ".scala")], True)

        if language == "rust":
            if module.startswith("mod "):
                name = module[4:]
                stem = here if PurePosixPath(importer).name in ("mod.rs", "lib.rs", "main.rs") else importer[:-3]
                return one([f"{stem}/{name}.rs", f"{stem}/{name}/mod.rs"], False)
            parts = [p for p in module.split("::") if p not in ("crate", "self", "super", "*", "")]
            for n in range(len(parts), 0, -1):
                stem = "/".join(parts[:n])
                found = one([f"src/{stem}.rs", f"src/{stem}/mod.rs", f"{stem}.rs", f"{stem}/mod.rs"], True)
                if found:
                    return found
            return []

        if language == "ruby":
            if module.startswith("."):
                return one([local(module + ".rb"), local(module)], False)
            return one([module + ".rb", f"lib/{module}.rb"], True)

        if language == "php":
            if "\\" in module:
                return one([module.replace("\\", "/") + ".php"], True)
            return one([local(module), module], True)

        if language == "lua":
            return one([module.replace(".", "/") + ".lua", module.replace(".", "/") + "/init.lua"], True)

        if language == "powershell":
            path = re.sub(r"^\$PSScriptRoot[\\/]", "", module, flags=re.IGNORECASE).replace("\\", "/")
            if "/" in path or path.lower().endswith((".ps1", ".psm1", ".psd1")):
                return one([local(path), path], True)
            return one([f"{path}.psm1", f"{path}/{path}.psm1", f"{path}.psd1", f"{path}/{path}.psd1"], True)

        if language in ("bash", "r", "julia"):
            if "/" in module or "." in PurePosixPath(module).name:
                return one([local(module), module], True)
            return []

        return []
