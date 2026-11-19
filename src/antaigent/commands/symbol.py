"""`antai symbol`: where a symbol is defined, what calls it, and what it calls."""

from typing import Annotated

import typer
from rich.markup import escape
from rich.table import Table
from rich.text import Text

from ..config import Settings
from ..query.context import CODE_KINDS
from ._common import ProjectOption, console, fail, open_project


def symbol(
    name: Annotated[str, typer.Argument(help="Function, class or other name, optionally qualified (Class.method).")],
    path: ProjectOption = None,
    limit: Annotated[int, typer.Option("--limit", "-n", min=1, help="Most call sites to list.")] = 40,
) -> None:
    """Show where a symbol is defined, what calls it, and what it calls."""
    settings = Settings.from_env()
    root, store = open_project(settings, path, check_model=False)
    with store:
        if not store.stats()["symbols"]:
            fail("This index has no code graph yet. Run `antai learn` again to build it.")
        defs = store.find_definitions(name)
        if not defs:
            similar = store.similar_symbols(name)
            hint = f" Similar names: {', '.join(similar)}" if similar else ""
            fail(f"No definition of {name!r} was found.{hint}")
        refs = store.references_to(name, limit=1000)
        callees = []
        if len(defs) == 1 and defs[0].kind in CODE_KINDS:
            d = defs[0]
            names = list(dict.fromkeys(r.name for r in store.calls_within(d.path, d.qualname)))
            callees = [(n, store.find_definitions(n, limit=2)) for n in names]

    table = Table(title="Defined at", title_justify="left", title_style="bold", box=None, padding=(0, 2))
    table.add_column("Location", overflow="fold")
    table.add_column("Kind", style="dim")
    table.add_column("Name", overflow="fold")
    for d in defs:
        table.add_row(Text(d.location), d.kind, Text(d.qualname))
    console.print(table)

    console.print()
    if refs:
        title = f"Called or referenced from {len(refs):,} place(s)" + (f", first {limit}" if len(refs) > limit else "")
        table = Table(title=title, title_justify="left", title_style="bold", box=None, padding=(0, 2))
        table.add_column("Location", overflow="fold", ratio=2)
        table.add_column("In", overflow="fold", ratio=1)
        table.add_column("Code", overflow="fold", ratio=3, style="dim")
        for r in refs[:limit]:
            table.add_row(Text(r.location), Text(r.caller or "(module level)"), Text(r.snippet))
        console.print(table)
        if len(defs) > 1:
            console.print("[dim]Several definitions share this name, so call sites may belong to any of them.[/]")
    else:
        console.print("[bold]No calls found.[/] [dim]It may be unused, an entry point, or only called dynamically.[/]")

    if callees:
        console.print()
        table = Table(title=f"{defs[0].qualname} calls", title_justify="left", title_style="bold", box=None, padding=(0, 2))
        table.add_column("Name", overflow="fold")
        table.add_column("Defined at", overflow="fold", style="dim")
        for callee, where in callees:
            table.add_row(Text(callee), Text(", ".join(w.location for w in where) or "outside the project"))
        console.print(table)
    console.print(f"\n[dim]{escape(str(root))}. Found by static analysis, so dynamic calls are missed.[/]", highlight=False)
