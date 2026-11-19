"""`antai status`: summarize a project's index and list skipped files."""

import datetime as dt

from rich.markup import escape
from rich.table import Table
from rich.text import Text

from ..config import Settings
from ._common import ProjectOption, console, open_project


def status(path: ProjectOption = None) -> None:
    """Show what's in a project's index."""
    settings = Settings.from_env()
    root, store = open_project(settings, path, check_model=False)
    with store:
        stats = store.stats()
        signature = (store.get_meta("signature") or "|").split("|")
        last_learn = store.get_meta("last_learn")
        skipped = store.skipped_files(10)
    size_mb = store.path.stat().st_size / 1e6
    table = Table(box=None, show_header=False, padding=(0, 2))
    table.add_row("Project", Text(str(root)))
    table.add_row("Index", Text(f"{store.path} ({size_mb:.1f} MB)"))
    table.add_row("Files indexed", f"{stats['files_indexed']:,}")
    table.add_row("Files skipped", f"{stats['files_skipped']:,}")
    table.add_row("Chunks", f"{stats['chunks']:,}")
    table.add_row(
        "Code graph",
        f"{stats['symbols']:,} definitions, {stats['references']:,} references, "
        f"{stats['imports']:,} imports ({stats['local_imports']:,} to project files)",
    )
    if stats["file_summaries"] or stats["dir_summaries"]:
        overview = ", project overview" if stats["has_overview"] else ""
        table.add_row(
            "Summaries", f"{stats['file_summaries']:,} of {stats['files_indexed']:,} files, {stats['dir_summaries']:,} folders{overview}"
        )
    else:
        table.add_row("Summaries", Text("none yet (run antai summarize)", style="dim"))
    table.add_row("Embedding model", signature[0] or "-")
    table.add_row("Chunk size", f"{signature[1]} characters" if len(signature) > 1 and signature[1] else "-")
    if last_learn:
        table.add_row("Last learned", dt.datetime.fromtimestamp(float(last_learn)).strftime("%Y-%m-%d %H:%M"))
    console.print(table)
    if skipped:
        console.print("\n[bold]Skipped files[/] (first 10)")
        for file_path, reason in skipped:
            console.print(f"  {escape(file_path)}  [dim]{escape(reason)}[/]", highlight=False)
