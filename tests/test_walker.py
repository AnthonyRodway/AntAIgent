import os

import pytest

from antaigent.ingest.walker import walk


def write(path, text="x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_walk_honours_ignore_rules(tmp_path):
    write(tmp_path / ".gitignore", "secret.txt\nlogs/\n")
    write(tmp_path / "src" / "app.py", "print('hi')")
    write(tmp_path / "secret.txt")
    write(tmp_path / "logs" / "run.log")
    write(tmp_path / "node_modules" / "lib" / "index.js")
    write(tmp_path / ".git" / "config")
    write(tmp_path / "image.png")
    write(tmp_path / "sub" / ".gitignore", "local.cfg\n")
    write(tmp_path / "sub" / "local.cfg")
    write(tmp_path / "sub" / "keep.cfg")
    write(tmp_path / "other" / "local.cfg")
    write(tmp_path / ".antaiignore", "drafts/\n")
    write(tmp_path / "drafts" / "notes.md")
    write(tmp_path / "out" / "result.gdb" / "a00000001.gdbtable")
    write(tmp_path / "out" / "result.gdb" / "_gdb.host.123.sr.lock")

    rels = {f.rel for f in walk(tmp_path)}

    assert {"src/app.py", "sub/keep.cfg", "other/local.cfg"} <= rels
    assert not any(r.startswith("out/") for r in rels)
    for excluded in ("secret.txt", "logs/run.log", "node_modules/lib/index.js", ".git/config",
                     "image.png", "sub/local.cfg", "drafts/notes.md"):
        assert excluded not in rels


def test_walk_skips_secrets_but_keeps_env_examples(tmp_path):
    for name in (".env", ".env.local", "config/.env.production", "certs/server.key", "certs/server.pem",
                 "home/.ssh/id_rsa", ".npmrc", ".env.example", "app.py"):
        write(tmp_path / name)
    rels = {f.rel for f in walk(tmp_path)}
    assert rels == {".env.example", "app.py"}


def test_walk_does_not_follow_links_out_of_the_project(tmp_path):
    project, outside = tmp_path / "project", tmp_path / "outside"
    write(project / "app.py")
    write(project / "lib" / "util.py")
    write(outside / "credentials.txt", "secret")
    try:
        os.symlink(outside / "credentials.txt", project / "notes.txt")
        os.symlink(outside, project / "linked", target_is_directory=True)
        os.symlink(project / "lib" / "util.py", project / "alias.py")  # stays inside: kept
    except OSError:
        pytest.skip("creating symlinks needs Developer Mode or admin rights on Windows")
    rels = {f.rel for f in walk(project)}
    assert rels == {"app.py", "lib/util.py", "alias.py"}


@pytest.mark.skipif(os.name != "nt", reason="junctions are a Windows feature")
def test_walk_does_not_follow_junctions_out_of_the_project(tmp_path):
    import _winapi

    project, outside = tmp_path / "project", tmp_path / "outside"
    write(project / "app.py")
    write(project / "lib" / "util.py")
    write(outside / "credentials.txt", "secret")
    _winapi.CreateJunction(str(outside), str(project / "linked"))
    _winapi.CreateJunction(str(project / "lib"), str(project / "mirror"))  # stays inside: followed
    rels = [f.rel for f in walk(project)]
    assert rels == ["app.py", "lib/util.py", "mirror/util.py"]


def test_walk_order_is_files_first_then_folders_by_name(tmp_path):
    for name in ("b.txt", "a/z.txt", "a/b/c.txt", "c/d.txt", "a.txt", "a/y.txt"):
        write(tmp_path / name)
    assert [f.rel for f in walk(tmp_path)] == ["a.txt", "b.txt", "a/y.txt", "a/z.txt", "a/b/c.txt", "c/d.txt"]


def test_walk_reports_size_and_forward_slash_paths(tmp_path):
    write(tmp_path / "a" / "b" / "c.txt", "hello")
    (f,) = [f for f in walk(tmp_path) if f.rel.endswith("c.txt")]
    assert f.rel == "a/b/c.txt"
    assert f.size == 5
