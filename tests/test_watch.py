import threading
import time

import pytest

from antaigent.index.watch import watch

pytest.importorskip("watchfiles")


def test_watch_reports_relevant_changes_then_stops(tmp_path):
    stop = threading.Event()
    seen: list[set[str]] = []

    def on_change(paths: set[str]) -> None:
        seen.append(paths)
        if any(p.endswith("new.py") for p in paths):
            stop.set()

    thread = threading.Thread(target=watch, args=(tmp_path, on_change), kwargs={"stop_event": stop, "debounce_ms": 200})
    thread.start()
    try:
        time.sleep(0.5)
        (tmp_path / "picture.png").write_bytes(b"\x89PNG")
        (tmp_path / "new.py").write_text("x = 1\n")
        thread.join(timeout=15)
    finally:
        stop.set()
        thread.join(timeout=10)
    changed = {p for paths in seen for p in paths}
    assert any(p.endswith("new.py") for p in changed)
    assert not any(p.endswith("picture.png") for p in changed)
