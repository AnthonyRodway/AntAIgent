"""Background jobs for the desktop app.

Each job runs on its own thread with its own database connection (SQLite connections
can't be shared between threads) and reports back to the window through Qt signals.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from PySide6.QtCore import QThread, Signal

from ..config import Settings
from ..index.indexer import learn
from ..index.store import Store
from ..index.summarize import summarize
from ..index.watch import watch
from ..llm.embed import Embedder
from ..llm.ollama import Ollama, has_model
from ..projects import Registry, index_path
from ..query.context import gather
from ..query.retrieval import search


class Cancelled(Exception):
    pass


class Task(QThread):
    """Runs `fn(task)` on a worker thread. `fn` reports through the task's signals."""

    progress = Signal(int, int, str)
    event = Signal(str, object)
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, fn: Callable[[Task], Any], parent=None):
        super().__init__(parent)
        self._fn = fn
        self.stop_event = threading.Event()

    def stop(self) -> None:
        self.stop_event.set()

    @property
    def stopping(self) -> bool:
        return self.stop_event.is_set()

    def run(self) -> None:
        try:
            result = self._fn(self)
        except Cancelled:
            self.succeeded.emit(None)
        except Exception as exc:  # shown to the user rather than lost on a background thread
            self.failed.emit(str(exc) or type(exc).__name__)
        else:
            self.succeeded.emit(result)


def _client(settings: Settings, *models: str) -> Ollama:
    """A client for Ollama, after checking it has `models`. Use it in a `with` block."""
    client = Ollama(settings.ollama_url)
    try:
        available = client.models()
    except Exception:
        client.close()
        raise
    missing = [m for m in dict.fromkeys(models) if not has_model(available, m)]
    if missing:
        client.close()
        raise RuntimeError(
            f"Ollama doesn't have {', '.join(missing)} yet. Download with: "
            + "; ".join(f"ollama pull {m}" for m in missing)
        )
    return client


def _embedder(settings: Settings, client: Ollama) -> Embedder:
    return Embedder(client, settings.embed_model, settings.keep_alive, settings.embed_batch)


def learn_job(settings: Settings, root: Path, full: bool = False):
    def run(task: Task):
        def progress(done: int, total: int, current: str) -> None:
            if task.stopping:
                raise Cancelled
            task.progress.emit(done, total, current)

        with _client(settings, settings.embed_model) as client, Store(index_path(settings.home, root)) as store:
            report = learn(
                root, store, _embedder(settings, client),
                max_file_bytes=settings.max_file_bytes, chunk_chars=settings.chunk_chars,
                full=full, progress=progress,
            )
        Registry(settings.home).touch(root)
        return report

    return run


def summarize_job(settings: Settings, root: Path):
    def run(task: Task):
        with _client(settings, settings.embed_model, settings.summarizer) as client:

            def complete(messages: list[dict[str, str]]) -> str:
                return client.complete(
                    settings.summarizer, messages,
                    think=settings.think, num_ctx=settings.num_ctx, keep_alive=settings.keep_alive,
                )[0]

            with Store(index_path(settings.home, root)) as store:
                return summarize(
                    root.name, store, _embedder(settings, client), complete,
                    model=settings.summarizer,
                    progress=lambda done, total, current: task.progress.emit(done, total, current),
                    should_stop=lambda: task.stopping,
                )

    return run


def ask_job(settings: Settings, root: Path, question: str, history: Sequence[tuple[str, str]], think: str):
    def run(task: Task):
        with _client(settings, settings.embed_model, settings.chat_model) as client:
            with Store(index_path(settings.home, root)) as store:
                context = gather(
                    store, _embedder(settings, client), question,
                    project=root.name, k=settings.top_k, context_chars=settings.context_chars, history=history,
                )
            task.event.emit("context", context)
            if not context.hits:
                raise RuntimeError("The index is empty, so there's nothing to answer from.")
            stream = client.chat(
                settings.chat_model, context.messages,
                think=think, num_ctx=settings.num_ctx, keep_alive=settings.keep_alive,
            )
            try:
                for kind, payload in stream:
                    if task.stopping:
                        break
                    task.event.emit(kind, payload)
            finally:
                stream.close()  # closing the HTTP stream makes Ollama stop generating

    return run


def search_job(settings: Settings, root: Path, query: str, k: int):
    def run(task: Task):
        with _client(settings, settings.embed_model) as client, Store(index_path(settings.home, root)) as store:
            return search(store, _embedder(settings, client), query, k)

    return run


def status_job(settings: Settings):
    def run(task: Task):
        with Ollama(settings.ollama_url, timeout=10) as client:
            return {"version": client.version(), "models": client.models(), "running": client.running()}

    return run


def warm_up(settings: Settings) -> threading.Thread:
    """Start loading the embedding and chat models, so a question typed meanwhile is
    answered without waiting for them: a cold start of a 20B model takes 10-20 seconds.

    A daemon thread rather than a Task: it touches no Qt objects, reports nothing, and
    mustn't keep the app open while Ollama finishes loading."""

    def run() -> None:
        try:
            with Ollama(settings.ollama_url) as client:
                client.embed(settings.embed_model, [], settings.keep_alive)
                client.load(settings.chat_model, keep_alive=settings.keep_alive, num_ctx=settings.num_ctx)
        except Exception:  # only a head start: the question itself loads whatever is missing
            pass

    thread = threading.Thread(target=run, name="antai-warm-up", daemon=True)
    thread.start()
    return thread


def watch_job(root: Path, poll: bool = False):
    def run(task: Task):
        watch(root, lambda paths: task.event.emit("changed", sorted(paths)), stop_event=task.stop_event, poll=poll)

    return run
