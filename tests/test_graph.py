import pytest

from antaigent.ingest.chunking import Chunker
from antaigent.ingest.graph import GraphExtractor, ImportResolver, find_imports

from .conftest import python_grammar_installed

PYTHON = '''\
import os, sys as system
from . import helpers
from ..store import Store, FileRecord as Record
from pkg.util import (
    parse,
    render,
)


class Report(Base):
    def build(self, rows):
        return format_rows(rows) + self.footer()

    def footer(self):
        def inner():
            return os.sep
        return inner()


def format_rows(rows):
    return "\\n".join(rows)
'''


@pytest.mark.skipif(not python_grammar_installed(), reason="python tree-sitter grammar not downloaded")
def test_python_definitions_references_and_callers():
    graph = GraphExtractor(Chunker().parser).extract("pkg/report.py", PYTHON)
    defs = {d.qualname: d for d in graph.definitions}
    assert defs["Report"].kind == "class"
    assert defs["Report.build"].kind == "method"
    assert defs["Report.footer.inner"].kind == "function"
    assert (defs["format_rows"].start_line, defs["format_rows"].end_line) == (20, 21)

    refs = {(r.name, r.caller) for r in graph.references}
    assert ("format_rows", "Report.build") in refs
    assert ("footer", "Report.build") in refs
    assert ("inner", "Report.footer") in refs
    call = next(r for r in graph.references if r.name == "format_rows")
    assert call.line == 12 and "format_rows(rows)" in call.snippet


def test_python_imports():
    imports = {(i.module, i.names) for i in find_imports("python", PYTHON)}
    assert ("os", ()) in imports and ("sys", ()) in imports
    assert (".", ("helpers",)) in imports
    assert ("..store", ("Store", "FileRecord")) in imports
    assert ("pkg.util", ("parse", "render")) in imports


def test_imports_in_other_languages():
    js = 'import { a,\n  b } from "./util/helpers";\nimport "./side-effect";\nconst x = require("../lib/thing");\n'
    assert [i.module for i in find_imports("javascript", js)] == ["./util/helpers", "./side-effect", "../lib/thing"]
    c = '#include "util/strings.h"\n#include <stdio.h>\n'
    assert [i.module for i in find_imports("c", c)] == ["util/strings.h", "stdio.h"]
    cs = "using System.IO;\nusing Alias = Foo.Bar;\nnamespace X { }\n"
    assert [i.module for i in find_imports("csharp", cs)] == ["System.IO", "Foo.Bar"]
    go = 'import (\n    "fmt"\n    util "example.com/proj/util"\n)\n'
    assert [(i.module, i.line) for i in find_imports("go", go)] == [("fmt", 2), ("example.com/proj/util", 3)]


def test_resolver_maps_imports_to_project_files():
    resolver = ImportResolver([
        "src/pkg/__init__.py", "src/pkg/helpers.py", "src/pkg/util.py", "src/store.py",
        "web/app.ts", "web/util/helpers.ts", "web/lib/index.js", "native/util/strings.h", "native/main.c",
    ])
    importer = "src/pkg/report.py"
    assert resolver.resolve(importer, "python", ".", ["helpers"]) == ["src/pkg/helpers.py"]
    assert resolver.resolve(importer, "python", "..store", ["Store"]) == ["src/store.py"]
    assert resolver.resolve(importer, "python", "pkg.util", ["parse"]) == ["src/pkg/util.py"]
    assert resolver.resolve(importer, "python", "os", []) == []
    assert resolver.resolve("web/app.ts", "typescript", "./util/helpers") == ["web/util/helpers.ts"]
    assert resolver.resolve("web/app.ts", "typescript", "./lib") == ["web/lib/index.js"]
    assert resolver.resolve("web/app.ts", "typescript", "react") == []
    assert resolver.resolve("native/main.c", "c", "util/strings.h") == ["native/util/strings.h"]
    assert resolver.resolve("native/main.c", "c", "stdio.h") == []
