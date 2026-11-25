"""Learning reads files ahead on worker threads, embeds one batch while the next is
chunked, and writes in batched transactions. These check that it stays correct when
Ollama fails, when learning is cancelled, and when only some imports are relinked."""

import sqlite3
import threading

import pytest

from antaigent.index.indexer import learn
from antaigent.index.store import Store

from .conftest import FakeEmbedder

FILES = 40


def many_files(root):
    root.mkdir(parents=True)
    for i in range(FILES):
        (root / f"mod{i:02}.py").write_text(f"def function_{i}(x):\n    return x + {i}\n")


class SmallBatches(FakeEmbedder):
    batch_size = 2  # learning sends 8 chunks at a time


class FailingEmbedder(SmallBatches):
    """Fails on its third request, like Ollama going away part way through."""

    def __init__(self):
        self.calls = 0

    def embed_documents(self, texts):
        self.calls += 1
        if self.calls == 3:
            raise RuntimeError("Ollama went away")
        return super().embed_documents(texts)


def worker_threads():
    return [t.name for t in threading.enumerate() if t.name.startswith(("antai-read", "antai-embed"))]


def assert_complete(store):
    """Every stored file has all its chunks: nothing half-written."""
    for path, record in store.files().items():
        assert len(store.chunks_for_file(path)) == record.n_chunks > 0, path


def test_an_embedding_failure_stops_learning_and_the_next_learn_finishes(tmp_path):
    project = tmp_path / "project"
    many_files(project)
    with Store(tmp_path / "index.db") as store:
        with pytest.raises(RuntimeError, match="went away"):
            learn(project, store, FailingEmbedder())
        assert worker_threads() == []
        stored = len(store.files())
        assert 0 < stored < FILES
        assert_complete(store)

        report = learn(project, store, SmallBatches())
        assert report.added == FILES - stored
        assert len(store.files()) == FILES
        assert_complete(store)


def test_cancelling_keeps_the_files_already_stored(tmp_path):
    project = tmp_path / "project"
    many_files(project)

    class Cancelled(Exception):
        pass

    def progress(done, total, current):
        if done == 30:
            raise Cancelled

    with Store(tmp_path / "index.db") as store:
        with pytest.raises(Cancelled):
            learn(project, store, SmallBatches(), progress=progress)
        assert worker_threads() == []
        stored = len(store.files())
        assert 0 < stored < 30
        assert_complete(store)
        assert learn(project, store, SmallBatches()).added == FILES - stored
        assert_complete(store)


def links(store):
    return sorted(store.conn.execute("SELECT source, target FROM import_targets"))


def test_only_changed_files_are_relinked_unless_files_come_or_go(tmp_path, embedder):
    project = tmp_path / "project"
    (project / "pkg").mkdir(parents=True)
    (project / "pkg" / "a.py").write_text("from .b import f\n")
    (project / "pkg" / "b.py").write_text("def f():\n    pass\n")
    (project / "pkg" / "c.py").write_text("from .a import g\n")
    with Store(tmp_path / "index.db") as store:
        learn(project, store, embedder)
        assert links(store) == [("pkg/a.py", "pkg/b.py"), ("pkg/c.py", "pkg/a.py")]

        calls = []
        relink = store.relink_imports
        store.relink_imports = lambda paths=None: calls.append(paths and sorted(paths)) or relink(paths)

        (project / "pkg" / "a.py").write_text("from .c import h\n")
        learn(project, store, embedder)
        assert calls == [["pkg/a.py"]]
        assert links(store) == [("pkg/a.py", "pkg/c.py"), ("pkg/c.py", "pkg/a.py")]
        partial = links(store)
        relink()
        assert links(store) == partial  # the same as resolving everything again

        (project / "pkg" / "d.py").write_text("from .b import f\n")
        learn(project, store, embedder)
        assert calls[-1] is None  # a new file can change where any import leads
        assert ("pkg/d.py", "pkg/b.py") in links(store)

        learn(project, store, embedder)
        assert len(calls) == 2  # nothing changed: no relink at all


def test_transactions_nest_and_commit_or_roll_back_as_one(tmp_path):
    with Store(tmp_path / "index.db") as store:
        with store.transaction():
            store.set_meta("a", "1")
            with store.transaction():
                store.set_meta("b", "2")
            other = sqlite3.connect(store.path)
            assert other.execute("SELECT COUNT(*) FROM meta WHERE key IN ('a', 'b')").fetchone() == (0,)
            other.close()
        assert (store.get_meta("a"), store.get_meta("b")) == ("1", "2")

        with pytest.raises(KeyError):
            with store.transaction():
                store.set_meta("c", "3")
                with store.transaction():
                    store.set_meta("d", "4")
                raise KeyError
        assert (store.get_meta("c"), store.get_meta("d")) == (None, None)
        store.set_meta("e", "5")
        assert store.get_meta("e") == "5"
