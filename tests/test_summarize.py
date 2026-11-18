from antaigent.index.indexer import learn
from antaigent.index.store import Store
from antaigent.index.summarize import summarize
from antaigent.query.context import gather


class FakeModel:
    """Stands in for the chat model: answers with what it was asked about."""

    def __init__(self):
        self.calls: list[str] = []

    def __call__(self, messages):
        prompt = messages[-1]["content"]
        subject = next(line.split(": ", 1)[1] for line in prompt.splitlines() if line.startswith(("File:", "Folder:", "Project folder:")))
        self.calls.append(subject)
        return f"Summary of {subject}."


def make_project(root):
    (root / "src" / "core").mkdir(parents=True)
    (root / "src" / "core" / "engine.py").write_text("def run():\n    return 1\n")
    (root / "src" / "cli.py").write_text("from .core.engine import run\n")
    (root / "README.md").write_text("# Demo\n\nA demo project.\n")


def test_summaries_are_built_bottom_up_and_redone_only_when_stale(tmp_path, embedder):
    project = tmp_path / "project"
    make_project(project)
    model = FakeModel()
    with Store(tmp_path / "index.db") as store:
        learn(project, store, embedder)
        report = summarize("project", store, embedder, model, model="fake")
        assert (report.files, report.dirs, report.project) == (3, 2, True)
        assert model.calls[-3:] == ["src/core/", "src/", "project"]
        assert store.get_summary("file", "src/cli.py").text == "Summary of src/cli.py."
        assert store.get_summary("project", "").text == "Summary of project."

        model.calls.clear()
        again = summarize("project", store, embedder, model, model="fake")
        assert (again.files, again.dirs, again.project) == (0, 0, False)
        assert model.calls == []

        (project / "src" / "core" / "engine.py").write_text("def run():\n    return 2\n")
        learn(project, store, embedder)
        assert store.get_summary("file", "src/core/engine.py") is None
        summarize("project", store, embedder, model, model="fake")
        assert model.calls == ["src/core/engine.py", "src/core/", "src/", "project"]

        context = gather(store, embedder, "what does the engine do?", project="project", k=3, context_chars=20_000)
    prompt = context.messages[-1]["content"]
    assert "Project overview:\nSummary of project." in prompt
    assert context.summaries


def test_summarize_stops_when_asked(tmp_path, embedder):
    project = tmp_path / "project"
    make_project(project)
    model = FakeModel()
    with Store(tmp_path / "index.db") as store:
        learn(project, store, embedder)
        report = summarize("project", store, embedder, model, model="fake", should_stop=lambda: len(model.calls) >= 1)
    assert report.stopped and report.files == 1
