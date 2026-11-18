"""Smoke tests for the desktop app, run without a screen. Skipped if PySide6 isn't installed."""

import os
import time

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from antaigent.config import Settings  # noqa: E402
from antaigent.gui.render import (  # noqa: E402
    LIGHT,  # noqa: E402
    file_link,
    markdown_html,
    parse_link,
)
from antaigent.index.indexer import learn  # noqa: E402
from antaigent.index.store import Store  # noqa: E402
from antaigent.projects import Registry, index_path  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_citations_in_answers_become_links():
    html = markdown_html("See `pkg/a.py:3-9` and `other.py:1`.\n\n```python\nx = 1\n```", LIGHT, {"pkg/a.py"})
    assert file_link("pkg/a.py", 3, 9) in html
    assert "other.py:1" in html and file_link("other.py", 1, 1) not in html
    assert parse_link(file_link("pkg/a b.py", 3, 9)) == ("file", {"p": "pkg/a b.py", "s": "3", "e": "9"})
    bracketed = markdown_html("It logs 【pkg/a.py:12-20】 and 【gone.py:1-2】.", LIGHT, {"pkg/a.py"})
    assert file_link("pkg/a.py", 12, 20) in bracketed
    assert "【gone.py:1-2】" in bracketed


def test_window_opens_a_project_and_navigates(tmp_path, embedder, qapp):
    from antaigent.gui.window import MainWindow

    home, project = tmp_path / "home", tmp_path / "project"
    (project / "pkg").mkdir(parents=True)
    (project / "pkg" / "util.py").write_text("def helper(x):\n    return x + 1\n")
    (project / "pkg" / "main.py").write_text("from .util import helper\n\n\ndef run():\n    return helper(2)\n")
    with Store(index_path(home, project)) as store:
        learn(project, store, embedder)
    Registry(home).touch(project)

    window = MainWindow(Settings(home=home, ollama_url="http://127.0.0.1:9"), project)
    try:
        assert window.root == project.resolve()
        assert window.paths == {"pkg/util.py", "pkg/main.py"}
        assert window.sidebar.tree.topLevelItemCount() == 1  # the pkg folder

        window.follow_link(file_link("pkg/main.py", 4, 5))
        assert window.tabs.currentWidget() is window.file
        assert window.file.title.text() == "pkg/main.py"
        assert "def run()" in window.file.code.toPlainText()
        assert "pkg/util.py" in window.file.info.toPlainText()  # its import, resolved

        window.open_symbol("helper")
        assert window.tabs.currentWidget() is window.symbol
        window.overview.refresh()
        assert "No overview yet" in window.overview.view.toPlainText()

        # Ollama isn't reachable here, so the answer fails, but the chat is still saved.
        window.ask.submit("What does helper do?")
        deadline = time.time() + 30
        while window.ask.task is not None and time.time() < deadline:
            qapp.processEvents()
            time.sleep(0.02)
        assert window.ask.task is None
        assert window.sidebar.chat_list.topLevelItemCount() == 1
        saved = window.sidebar.chat_list.topLevelItem(0)
        assert saved.text(0) == "What does helper do?"

        window.new_chat()
        assert window.ask.turns == [] and window.ask.conversation_id is None
        window.sidebar._open_chat(window.sidebar.chat_list.topLevelItem(0))
        assert [t.question for t in window.ask.turns] == ["What does helper do?"]
        assert window.ask.turns[0].error
        assert "What does helper do?" in window.ask.view.toPlainText()

        window.delete_chat(window.ask.conversation_id)
        assert window.sidebar.chat_list.topLevelItemCount() == 0
        assert window.ask.turns == []
    finally:
        window.close()
