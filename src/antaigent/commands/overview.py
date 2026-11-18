"""`antai overview`: print the project overview and top-level folder summaries."""

from rich.markdown import Markdown
from rich.table import Table
from rich.text import Text

from ..config import Settings
from ._common import ProjectOption, console, fail, open_project


def overview(path: ProjectOption = None) -> None:
    """Show the project overview and what each top-level folder holds."""
    settings = Settings.from_env()
    root, store = open_project(settings, path, check_model=False)
    with store:
        project = store.get_summary("project", "")
        folders = sorted(
            (s for (scope, key), s in store.summaries("dir").items() if "/" not in key), key=lambda s: s.key.lower()
        )
    if project is None:
        fail(f"No overview yet. Run: antai summarize -p \"{root}\"")
    console.rule(f"[bold]{root.name}[/]")
    console.print(Markdown(project.text))
    if folders:
        console.print()
        table = Table(title="Top-level folders", title_justify="left", title_style="bold", box=None, padding=(0, 2))
        table.add_column("Folder", no_wrap=True)
        table.add_column("Summary", overflow="fold")
        for s in folders:
            table.add_row(Text(s.key + "/", style="bold"), Text(s.text))
        console.print(table)
