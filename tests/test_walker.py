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


def test_walk_reports_size_and_forward_slash_paths(tmp_path):
    write(tmp_path / "a" / "b" / "c.txt", "hello")
    (f,) = [f for f in walk(tmp_path) if f.rel.endswith("c.txt")]
    assert f.rel == "a/b/c.txt"
    assert f.size == 5
