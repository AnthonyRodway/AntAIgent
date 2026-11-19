"""`antai deps`: what a file imports, and which files import it."""

from typing import Annotated

import typer
from rich.markup import escape
from rich.table import Table
from rich.text import Text

from ..config import Settings
from ._common import ProjectOption, console, fail, open_project


def deps(
    file: Annotated[str, typer.Argument(help="File path, or the end of one (e.g. store.py).")],
    path: ProjectOption = None,
) -> None:
    """Show what a file imports and which files import it."""
    settings = Settings.from_env()
    root, store = open_project(settings, path, check_model=False)
    with store:
        matches = store.match_paths(file)
        if not matches:
            fail(f"No indexed file matches {file!r}.")
        target = matches[0]
        imports = store.imports_of(target)
        importers = store.imported_by(target)
        defs = [d for d in store.definitions_in(target) if "." not in d.qualname]

    if len(matches) > 1 and not target.endswith("/" + file.replace("\\", "/").strip("/")):
        others = escape(", ".join(matches[1:5]))
        console.print(f"[yellow]Several files match; showing {escape(target)}.[/] [dim]Others: {others}[/]", highlight=False)
    console.print(f"[bold]{escape(target)}[/]", highlight=False)
    if defs:
        names = ", ".join(d.qualname for d in defs[:30]) + (" ..." if len(defs) > 30 else "")
        console.print(f"[dim]Defines: {escape(names)}[/]", highlight=False)
    console.print()

    if imports:
        table = Table(title="Imports", title_justify="left", title_style="bold", box=None, padding=(0, 2))
        table.add_column("Line", justify="right", style="dim")
        table.add_column("Module", overflow="fold")
        table.add_column("Project file", overflow="fold")
        for i in imports:
            table.add_row(str(i.line), Text(i.module), Text(i.target or "-", style="" if i.target else "dim"))
        console.print(table)
    else:
        console.print("[bold]Imports:[/] none found")

    console.print()
    if importers:
        table = Table(title=f"Imported by {len({i.path for i in importers})} file(s)", title_justify="left",
                      title_style="bold", box=None, padding=(0, 2))
        table.add_column("Location", overflow="fold")
        table.add_column("As", overflow="fold", style="dim")
        for i in importers:
            table.add_row(Text(f"{i.path}:{i.line}"), Text(i.module))
        console.print(table)
    else:
        console.print("[bold]Imported by:[/] no project file [dim](it may be an entry point or script)[/]")
    console.print(f"\n[dim]{escape(str(root))}[/]", highlight=False)
