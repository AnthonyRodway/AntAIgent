from typer.testing import CliRunner

from antaigent.cli import app

runner = CliRunner()


def test_every_command_is_registered():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for name in ("learn", "ask", "search", "status", "projects", "doctor", "setup"):
        assert name in result.output


def test_projects_with_an_empty_home(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTAI_HOME", str(tmp_path))
    result = runner.invoke(app, ["projects"])
    assert result.exit_code == 0
    assert "No folders learned yet" in result.output


def test_status_without_any_project_fails_helpfully(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTAI_HOME", str(tmp_path))
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 1
