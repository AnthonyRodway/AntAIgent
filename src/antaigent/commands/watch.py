"""`antai watch`: keep a project's index up to date while you work."""

import time
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.markup import escape

from ..config import Settings
from ..index.indexer import LearnReport
from ..index.indexer import learn as learn_folder
from ..index.store import Store
from ..index.summarize import summarize as summarize_project
from ..index.watch import watch as watch_folder
from ..llm.embed import Embedder
from ..llm.ollama import Ollama, OllamaError
from ..projects import Registry, index_path
from ._common import console, fail, require_models
from .summarize import make_completer


def _changes(report: LearnReport) -> str:
    parts = [(report.added, "new"), (report.updated, "updated"), (report.removed, "removed"), (report.skipped, "skipped")]
    return ", ".join(f"{n:,} {label}" for n, label in parts if n)


def watch(
    path: Annotated[
        Optional[Path],
        typer.Argument(help="Folder to watch. Defaults to the learned project containing the current folder.",
                       exists=True, file_okay=False, resolve_path=True),
    ] = None,
    summarize: Annotated[bool, typer.Option("--summarize", help="Also refresh summaries after each change.")] = False,
    poll: Annotated[bool, typer.Option("--poll", help="Check on a timer instead of file events (for network drives).")] = False,
) -> None:
    """Re-learn a folder automatically whenever its files change. Stop with Ctrl+C."""
    settings = Settings.from_env()
    registry = Registry(settings.home)
    root = path or registry.resolve(None)
    if root is None:
        fail("No folders learned yet. Run: antai watch <folder>")
    client = Ollama(settings.ollama_url)
    require_models(client, settings.embed_model, *([settings.summarizer] if summarize else []))
    embedder = Embedder(client, settings.embed_model, settings.keep_alive, settings.embed_batch)
    complete = make_completer(client, settings) if summarize else None

    def update() -> None:
        stamp = time.strftime("%H:%M:%S")
        try:
            with Store(index_path(settings.home, root)) as store:
                report = learn_folder(
                    root, store, embedder, max_file_bytes=settings.max_file_bytes,
                    max_pdf_bytes=settings.max_pdf_bytes, chunk_chars=settings.chunk_chars,
                )
                registry.touch(root)
                if report.changed:
                    console.print(f"[dim]{stamp}[/]  {_changes(report)} [dim]({report.seconds:.1f}s)[/]", highlight=False)
                for error in report.errors[:3]:
                    console.print(f"[dim]{stamp}[/]  [yellow]{escape(error)}[/]", highlight=False)
                if complete is not None and report.changed:
                    summary = summarize_project(root.name, store, embedder, complete, model=settings.summarizer)
                    if summary.files or summary.dirs or summary.project:
                        console.print(
                            f"[dim]{stamp}[/]  summaries: {summary.files} files, {summary.dirs} folders"
                            f"{', overview' if summary.project else ''} [dim]({summary.seconds:.0f}s)[/]",
                            highlight=False,
                        )
        except OllamaError as exc:
            console.print(f"[dim]{stamp}[/]  [red]{escape(str(exc))}[/]", highlight=False)

    console.print(f"Watching [bold]{escape(str(root))}[/]. Press Ctrl+C to stop.", highlight=False)
    update()
    try:
        watch_folder(root, lambda _paths: update(), poll=poll)
    except KeyboardInterrupt:
        pass
    console.print("Stopped watching.")
