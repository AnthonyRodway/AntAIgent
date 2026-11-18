"""`antai summarize`: write summaries of files and folders, and a project overview."""

from typing import Annotated

import typer
from rich.markdown import Markdown
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from ..config import Settings
from ..index.summarize import SummaryReport
from ..index.summarize import summarize as summarize_project
from ..llm.embed import Embedder
from ..llm.ollama import Ollama
from ._common import ProjectOption, console, fail, open_project, require_models, short


def make_completer(client: Ollama, settings: Settings):
    model = settings.summarizer

    def complete(messages: list[dict[str, str]]) -> str:
        return client.complete(
            model, messages, think=settings.think, num_ctx=settings.num_ctx, keep_alive=settings.keep_alive
        )[0]

    return complete


def summarize(
    path: ProjectOption = None,
    force: Annotated[bool, typer.Option("--force", help="Redo every summary, not just the out-of-date ones.")] = False,
) -> None:
    """Summarize each file and folder and write a project overview, using the chat model.

    Takes a few seconds per file the first time; after that only changed files and the
    folders above them are redone. `ask` uses the summaries for broad questions."""
    settings = Settings.from_env()
    client = Ollama(settings.ollama_url)
    require_models(client, settings.embed_model, settings.summarizer)
    root, store = open_project(settings, path)
    embedder = Embedder(client, settings.embed_model, settings.keep_alive, settings.embed_batch)
    console.print(f"Summarizing [bold]{root}[/] with {settings.summarizer}", highlight=False)
    progress = Progress(
        SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
        TextColumn("{task.fields[current]}", style="dim", markup=False),
        console=console, transient=True,
    )
    with store:
        with progress:
            task = progress.add_task("Summarizing", total=None, current="")

            def on_progress(done: int, total: int, current: str) -> None:
                progress.update(task, completed=done, total=total, current=short(current))

            try:
                report = summarize_project(
                    root.name, store, embedder, make_completer(client, settings),
                    model=settings.summarizer, force=force, progress=on_progress,
                )
            except KeyboardInterrupt:
                progress.stop()
                fail("Interrupted. Summaries finished so far are saved; run summarize again to continue.")
        overview = store.get_summary("project", "")
    _print_report(report)
    if overview is not None and report.project:
        console.print()
        console.rule("[bold]Project overview[/]")
        console.print(Markdown(overview.text))


def _print_report(report: SummaryReport) -> None:
    parts = [f"{report.files:,} files", f"{report.dirs:,} folders"]
    if report.project:
        parts.append("the project overview")
    done = ", ".join(parts)
    console.print(f"[green]Done[/] in {report.seconds:.0f}s. Summarized {done}; {report.unchanged:,} already up to date.")
    if report.stopped and report.errors:
        console.print("[yellow]Stopped after repeated errors.[/]")
    for error in report.errors[:5]:
        console.print(f"  [yellow]{error}[/]", highlight=False)
