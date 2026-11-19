"""`antai learn`: index a folder, or bring an existing index up to date."""

from pathlib import Path
from typing import Annotated, Any

import typer
from rich.markup import escape
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from ..config import Settings
from ..index.indexer import LearnReport
from ..index.indexer import learn as learn_folder
from ..index.store import Store
from ..llm.embed import Embedder
from ..llm.ollama import Ollama, OllamaError
from ..projects import Registry, index_path
from ._common import console, fail, require_models, short


def learn(
    path: Annotated[Path, typer.Argument(help="Folder to learn.", exists=True, file_okay=False, resolve_path=True)],
    full: Annotated[bool, typer.Option("--full", help="Rebuild the index from scratch instead of updating it.")] = False,
) -> None:
    """Index a folder. Re-run any time: only new or changed files are processed."""
    settings = Settings.from_env()
    client = Ollama(settings.ollama_url)
    require_models(client, settings.embed_model)
    embedder = Embedder(client, settings.embed_model, settings.keep_alive, settings.embed_batch)
    db = index_path(settings.home, path)
    console.print(f"Learning [bold]{escape(str(path))}[/]", highlight=False)
    console.print(f"[dim]Index: {escape(str(db))}[/]", highlight=False)

    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("{task.fields[current]}", style="dim", markup=False),
        console=console,
        transient=True,
    )
    with Store(db) as store:
        with progress:
            task = progress.add_task("Scanning files", total=None, current="")

            def on_progress(done: int, total: int, current: str) -> None:
                progress.update(task, description="Learning", completed=done, total=total, current=short(current))

            try:
                report = learn_folder(
                    path, store, embedder,
                    max_file_bytes=settings.max_file_bytes,
                    chunk_chars=settings.chunk_chars,
                    full=full,
                    progress=on_progress,
                )
            except KeyboardInterrupt:
                progress.stop()
                fail("Interrupted. Files finished so far are saved; run learn again to continue where it stopped.")
            except OllamaError as exc:
                progress.stop()
                fail(str(exc))
        stats = store.stats()
    Registry(settings.home).touch(path)
    _print_report(report, stats)


def _print_report(report: LearnReport, stats: dict[str, Any]) -> None:
    if report.rebuilt:
        console.print("[yellow]Rebuilt the index from scratch.[/]")
    parts = [f"{report.added:,} new", f"{report.updated:,} updated", f"{report.unchanged:,} unchanged"]
    if report.removed:
        parts.append(f"{report.removed:,} removed")
    if report.skipped:
        parts.append(f"{report.skipped:,} skipped")
    console.print(f"[green]Done[/] in {report.seconds:.1f}s. {report.scanned:,} files scanned: {', '.join(parts)}.")
    console.print(f"The index holds {stats['chunks']:,} chunks from {stats['files_indexed']:,} files.")
    if report.missing_grammars:
        console.print(
            f"[yellow]Used basic line-based splitting for {', '.join(sorted(report.missing_grammars))} code[/] "
            "because those parsers aren't installed. Run [bold]antai setup[/] once (needs internet), "
            "then [bold]antai learn --full[/] for better results."
        )
    if report.errors:
        console.print(f"[yellow]{len(report.errors)} file(s) couldn't be processed:[/]")
        for error in report.errors[:5]:
            console.print(f"  {error}", markup=False, highlight=False)
