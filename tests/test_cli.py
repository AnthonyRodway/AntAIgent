from typer.testing import CliRunner

from antaigent.cli import app
from antaigent.index.indexer import learn
from antaigent.index.store import Store
from antaigent.projects import Registry, index_path

runner = CliRunner()

COMMANDS = ("learn", "ask", "search", "symbol", "deps", "summarize", "overview", "watch", "gui",
            "status", "projects", "doctor", "setup")


def test_every_command_is_registered():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for name in COMMANDS:
        assert name in result.output


def _learned_project(tmp_path, monkeypatch, embedder):
    home, project = tmp_path / "home", tmp_path / "project"
    (project / "pkg").mkdir(parents=True)
    (project / "pkg" / "util.py").write_text("def helper(x):\n    return x + 1\n")
    (project / "pkg" / "main.py").write_text("from .util import helper\n\n\ndef run():\n    return helper(2)\n")
    monkeypatch.setenv("ANTAI_HOME", str(home))
    monkeypatch.setenv("COLUMNS", "200")
    with Store(index_path(home, project)) as store:
        learn(project, store, embedder)
    Registry(home).touch(project)
    return project


def test_symbol_and_deps_commands(tmp_path, monkeypatch, embedder):
    from .conftest import python_grammar_installed

    project = _learned_project(tmp_path, monkeypatch, embedder)
    deps = runner.invoke(app, ["deps", "util.py", "-p", str(project)])
    assert deps.exit_code == 0, deps.output
    assert "pkg/main.py:1" in deps.output
    if python_grammar_installed():
        symbol = runner.invoke(app, ["symbol", "helper", "-p", str(project)])
        assert symbol.exit_code == 0, symbol.output
        assert "pkg/util.py:1-2" in symbol.output
        assert "pkg/main.py:5" in symbol.output
        missing = runner.invoke(app, ["symbol", "helpr", "-p", str(project)])
        assert missing.exit_code == 1


def test_overview_without_summaries_fails_helpfully(tmp_path, monkeypatch, embedder):
    project = _learned_project(tmp_path, monkeypatch, embedder)
    result = runner.invoke(app, ["overview", "-p", str(project)])
    assert result.exit_code == 1
    assert "summarize" in result.output


def test_projects_with_an_empty_home(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTAI_HOME", str(tmp_path))
    result = runner.invoke(app, ["projects"])
    assert result.exit_code == 0
    assert "No folders learned yet" in result.output


def test_status_without_any_project_fails_helpfully(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTAI_HOME", str(tmp_path))
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 1
