"""`antai projects`: list every folder that has been learned."""

import datetime as dt

from rich.table import Table
from rich.text import Text

from ..config import Settings
from ..projects import Registry, index_path
from ._common import console


def projects() -> None:
    """List the folders that have been learned."""
    settings = Settings.from_env()
    entries = Registry(settings.home).all()
    if not entries:
        console.print("No folders learned yet. Run: antai learn <folder>")
        return
    table = Table("Folder", "Last used", "Index")
    for root, last_used in entries:
        has_index = index_path(settings.home, root).exists()
        table.add_row(
            Text(str(root)),
            dt.datetime.fromtimestamp(last_used).strftime("%Y-%m-%d %H:%M"),
            "yes" if has_index else "[red]missing[/]",
        )
    console.print(table)
