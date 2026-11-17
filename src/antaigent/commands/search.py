"""`antai search`: show what retrieval finds for a query, without calling the chat model."""

from typing import Annotated

import typer
from rich.table import Table
from rich.text import Text

from ..config import Settings
from ..llm.embed import Embedder
from ..llm.ollama import Ollama, OllamaError
from ..query.retrieval import search as hybrid_search
from ._common import ProjectOption, console, fail, open_project, require_models


def search(
    query: Annotated[str, typer.Argument(help="Words or a question to search for.")],
    path: ProjectOption = None,
    k: Annotated[int, typer.Option("--k", "-k", min=1, help="How many results to show.")] = 10,
) -> None:
    """Show the excerpts a query retrieves, without asking the model. Handy for checking retrieval."""
    settings = Settings.from_env()
    client = Ollama(settings.ollama_url)
    require_models(client, settings.embed_model)
    root, store = open_project(settings, path)
    with store:
        try:
            hits = hybrid_search(store, Embedder(client, settings.embed_model, settings.keep_alive), query, k)
        except OllamaError as exc:
            fail(str(exc))
    table = Table(title=str(root), title_justify="left", caption="Vector/Keyword: rank in each search before fusion")
    table.add_column("#", justify="right")
    table.add_column("Location", overflow="fold", ratio=3)
    table.add_column("Symbols", overflow="fold", ratio=2)
    table.add_column("Vector", justify="right")
    table.add_column("Keyword", justify="right")
    for n, hit in enumerate(hits, 1):
        c = hit.chunk
        table.add_row(
            str(n),
            Text(c.location),
            Text(", ".join(c.symbols[:2]), style="dim"),
            str(hit.vector_rank or "-"),
            str(hit.keyword_rank or "-"),
        )
    console.print(table)
