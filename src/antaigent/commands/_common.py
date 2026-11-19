"""Helpers shared by the commands: console output, errors, and finding the project's index."""

from pathlib import Path
from typing import Annotated, NoReturn, Optional

import typer
from rich.console import Console
from rich.markup import escape

from ..config import Settings
from ..index.store import Store
from ..llm.ollama import Ollama, OllamaError, has_model
from ..projects import Registry, index_path

console = Console()
err_console = Console(stderr=True)

ProjectOption = Annotated[
    Optional[Path],
    typer.Option(
        "--path", "-p", file_okay=False,
        help="Project folder. Defaults to the learned project containing the current folder, else the most recently used one.",
    ),
]


def fail(message: str) -> NoReturn:
    # Messages carry paths and error text; "[...]" in them must print as-is, not as markup.
    err_console.print(f"[bold red]Error:[/] {escape(message)}", highlight=False)
    raise typer.Exit(1)


def short(path: str, width: int = 60) -> str:
    return path if len(path) <= width else "..." + path[-(width - 3):]


def require_models(client: Ollama, *names: str) -> None:
    try:
        available = client.models()
    except OllamaError as exc:
        fail(str(exc))
    missing = [n for n in dict.fromkeys(names) if not has_model(available, n)]
    if missing:
        pulls = "\n".join(f"  ollama pull {m}" for m in missing)
        fail(f"Ollama doesn't have {', '.join(missing)} yet. Download with:\n{pulls}")


def open_project(settings: Settings, path: Optional[Path], check_model: bool = True) -> tuple[Path, Store]:
    registry = Registry(settings.home)
    root = registry.resolve(path)
    if root is None:
        fail("No folders learned yet. Run: antai learn <folder>")
    db = index_path(settings.home, root)
    if not db.exists():
        fail(f'{root} hasn\'t been learned yet. Run: antai learn "{root}"')
    registry.touch(root)
    store = Store(db)
    index_model = (store.get_meta("signature") or "").split("|")[0]
    if check_model and index_model and index_model != settings.embed_model:
        store.close()
        fail(
            f"This index was built with the embedding model {index_model}, but {settings.embed_model} is configured. "
            f"Re-run learn, or set ANTAI_EMBED_MODEL={index_model}."
        )
    return root, store


def installed_grammars() -> set[str]:
    try:
        import tree_sitter_language_pack as tslp

        return set(tslp.downloaded_languages())
    except Exception:
        return set()
