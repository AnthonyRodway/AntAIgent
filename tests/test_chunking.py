import json

import pytest

from antaigent.ingest.chunking import Chunker, split_lines

from .conftest import python_grammar_installed


def big_python_module() -> str:
    methods = "\n".join(
        f"    def method_{i}(self, value):\n"
        f"        # step {i}\n"
        f"        total = value * {i}\n"
        f"        return total + {i}\n"
        for i in range(12)
    )
    return (
        "import os\n\n\n"
        "def helper(x):\n    return x + 1\n\n\n"
        f"class Big:\n    '''A class too large for one chunk.'''\n\n{methods}\n\n"
        "def tail():\n    return helper(2)\n"
    )


def covered_lines(chunks):
    lines = set()
    for c in chunks:
        lines.update(range(c.start_line, c.end_line + 1))
    return lines


@pytest.mark.skipif(not python_grammar_installed(), reason="python tree-sitter grammar not downloaded")
def test_code_chunks_follow_syntax_and_cover_every_line():
    source = big_python_module()
    chunker = Chunker(max_chars=300)
    chunks = chunker.chunk("pkg/big.py", source)

    assert len(chunks) > 3
    assert all(c.kind == "code" for c in chunks)
    assert all(len(c.text) <= 300 for c in chunks)
    assert covered_lines(chunks) == set(range(1, len(split_lines(source)) + 1))
    symbols = {s for c in chunks for s in c.symbols}
    assert {"helper", "Big.method_0", "Big.method_11", "tail"} <= symbols
    # Each method starts at a chunk boundary or is fully inside one chunk.
    for c in chunks:
        assert c.text == "\n".join(split_lines(source)[c.start_line - 1 : c.end_line])


def test_missing_grammar_falls_back_to_line_windows():
    chunker = Chunker(max_chars=200)
    chunker._installed = set()  # pretend no grammars are on disk
    source = "\n".join(f"line {i}" for i in range(100))
    chunks = chunker.chunk("main.go", source)
    assert chunks and all(c.kind == "code" for c in chunks)
    assert "go" in chunker.missing_grammars
    assert covered_lines(chunks) == set(range(1, 101))


def test_markdown_sections_carry_heading_breadcrumbs():
    doc = "# Guide\nintro\n\n## Install\n" + "pip install x\n" * 30 + "\n## Usage\n```\n# not a heading\n```\nrun it\n"
    chunks = Chunker(max_chars=200).chunk("README.md", doc)
    crumbs = [s for c in chunks for s in c.symbols]
    assert "Guide > Install" in crumbs
    assert "Guide > Usage" in crumbs
    assert not any("not a heading" in s for s in crumbs)


def test_csv_becomes_one_preview_chunk():
    rows = ["site,species,biomass"] + [f"s{i},pine,{i * 1.5}" for i in range(100)]
    (chunk,) = Chunker().chunk("data/plots.csv", "\n".join(rows))
    assert chunk.kind == "data"
    assert chunk.symbols == ["site", "species", "biomass"]
    assert "s0,pine,0.0" in chunk.text
    assert "s99" not in chunk.text


def test_notebook_chunks_reference_cell_numbers():
    nb = {
        "cells": [
            {"cell_type": "markdown", "source": ["# Analysis\n"]},
            {"cell_type": "code", "source": "import pandas as pd\n", "outputs": [{"data": "x" * 5000}]},
            {"cell_type": "code", "source": ""},
            {"cell_type": "code", "source": "df = pd.read_csv('plots.csv')\n"},
        ]
    }
    chunks = Chunker().chunk("analysis.ipynb", json.dumps(nb))
    assert len(chunks) == 1
    assert (chunks[0].start_line, chunks[0].end_line) == (1, 4)
    assert "read_csv" in chunks[0].text and "xxxxx" not in chunks[0].text


def test_plain_text_windows_overlap_and_cover_everything():
    text = "\n".join(f"row {i} " + "y" * 40 for i in range(60))
    chunks = Chunker(max_chars=500).chunk("notes.txt", text)
    assert len(chunks) > 1
    assert covered_lines(chunks) == set(range(1, 61))
    assert chunks[1].start_line <= chunks[0].end_line
