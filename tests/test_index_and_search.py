import os

from antaigent.index.indexer import learn
from antaigent.index.store import Store, fts_query, identifier_terms
from antaigent.query.answer import build_messages
from antaigent.query.retrieval import search


def make_project(root):
    (root / "src").mkdir(parents=True)
    (root / "src" / "growth.py").write_text(
        "def compute_growth_curve(age, site_index):\n"
        "    '''Merchantable volume by stand age.'''\n"
        "    return site_index * age ** 0.5\n"
    )
    (root / "src" / "disturbance.py").write_text(
        "class FireEvent:\n"
        "    def apply(self, pools):\n"
        "        pools['deadwood'] += pools.pop('stemwood')\n"
    )
    (root / "README.md").write_text("# Carbon model\n\nSimulates forest carbon pools over time.\n")
    (root / "blob.dat").write_bytes(b"\x00\x01\x02binary")


def test_learn_then_search_finds_the_right_file(tmp_path, embedder):
    project = tmp_path / "project"
    make_project(project)
    with Store(tmp_path / "index.db") as store:
        report = learn(project, store, embedder, chunk_chars=500)
        assert report.added == 3
        assert report.skipped == 1  # blob.dat is binary

        hits = search(store, embedder, "how is the growth curve computed?", k=3)
        assert hits[0].chunk.path == "src/growth.py"
        assert hits[0].keyword_rank is not None and hits[0].vector_rank is not None

        hits = search(store, embedder, "FireEvent", k=3)
        assert hits[0].chunk.path == "src/disturbance.py"


def test_relearn_only_processes_changes(tmp_path, embedder):
    project = tmp_path / "project"
    make_project(project)
    with Store(tmp_path / "index.db") as store:
        learn(project, store, embedder, chunk_chars=500)

        again = learn(project, store, embedder, chunk_chars=500)
        assert (again.added, again.updated, again.removed) == (0, 0, 0)
        assert again.unchanged == 4

        growth = project / "src" / "growth.py"
        growth.write_text(growth.read_text() + "\n\ndef yield_table():\n    return []\n")
        (project / "README.md").unlink()
        touched = project / "src" / "disturbance.py"
        os.utime(touched, (touched.stat().st_atime, touched.stat().st_mtime + 10))  # same content, new mtime

        third = learn(project, store, embedder, chunk_chars=500)
        assert (third.added, third.updated, third.removed) == (0, 1, 1)
        assert "README.md" not in store.files()
        assert search(store, embedder, "yield_table", k=1)[0].chunk.path == "src/growth.py"


def test_changing_chunk_size_rebuilds(tmp_path, embedder):
    project = tmp_path / "project"
    make_project(project)
    with Store(tmp_path / "index.db") as store:
        learn(project, store, embedder, chunk_chars=500)
        report = learn(project, store, embedder, chunk_chars=800)
        assert report.rebuilt and report.added == 3


def test_keyword_query_is_safe_and_splits_identifiers():
    assert fts_query('what does "getUserName(" do?') == '"getusername" OR "get" OR "user" OR "name"'
    assert fts_query("the and of") is None
    assert set(identifier_terms("getUserName parseXMLFile plain snake_case").split()) == {
        "get", "user", "name", "parse", "xml", "file"
    }


def test_build_messages_respects_the_context_budget(tmp_path, embedder):
    project = tmp_path / "project"
    make_project(project)
    with Store(tmp_path / "index.db") as store:
        learn(project, store, embedder, chunk_chars=500)
        hits = search(store, embedder, "carbon pools", k=3)
    messages, used = build_messages("Q?", hits, "project", context_chars=1)
    assert len(used) == 1  # always at least one excerpt
    assert used[0].chunk.location in messages[1]["content"]
    _, used_all = build_messages("Q?", hits, "project", context_chars=100_000)
    assert len(used_all) == len(hits)
