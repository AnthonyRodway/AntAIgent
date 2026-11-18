"""The `antai` app. Each command lives in its own module under commands/."""

from typing import Annotated

import typer

from . import __version__
from .commands._common import console
from .commands.ask import ask
from .commands.deps import deps
from .commands.doctor import doctor
from .commands.gui import gui
from .commands.learn import learn
from .commands.overview import overview
from .commands.projects import projects
from .commands.search import search
from .commands.setup import setup
from .commands.status import status
from .commands.summarize import summarize
from .commands.symbol import symbol
from .commands.watch import watch

app = typer.Typer(
    help="AntAIgent: learn a folder of code or data, then ask questions about it. Runs entirely on this machine via Ollama.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_show_locals=False,
)


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"antaigent {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool, typer.Option("--version", callback=_version_callback, is_eager=True, help="Show the version and exit.")
    ] = False,
) -> None:
    pass


for command in (
    learn, ask, search, symbol, deps, summarize, overview, watch, gui, status, projects, doctor, setup
):
    app.command()(command)
