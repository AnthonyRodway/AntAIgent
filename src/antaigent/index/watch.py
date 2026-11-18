"""Notice when files in a learned folder change, so the index can be brought up to date."""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path

from ..ingest.walker import SKIP_EXTENSIONS


def _filter():
    from watchfiles import Change, DefaultFilter

    class LearnFilter(DefaultFilter):
        """watchfiles' defaults (.git, __pycache__, .venv, editor swap files...) plus files learn skips anyway."""

        def __call__(self, change: Change, path: str) -> bool:
            if not super().__call__(change, path):
                return False
            p = Path(path)
            if p.suffix.lower() in SKIP_EXTENSIONS or p.name.endswith(".lock"):
                return False
            return not any(part.lower().endswith(".gdb") for part in p.parts)

    return LearnFilter()


def watch(
    root: Path,
    on_change: Callable[[set[str]], None],
    *,
    stop_event: threading.Event | None = None,
    poll: bool = False,
    debounce_ms: int = 1500,
) -> None:
    """Call `on_change(paths)` after each burst of changes under `root`, until `stop_event` is set.

    `poll` checks for changes on a timer instead of using file-system events, which is
    needed on some network drives."""
    from watchfiles import watch as watch_changes

    for changes in watch_changes(
        root,
        watch_filter=_filter(),
        debounce=debounce_ms,
        stop_event=stop_event,
        force_polling=poll or None,
        raise_interrupt=False,
    ):
        on_change({path for _, path in changes})
