"""`antai ask`: answer a question from the project's index, streaming the reply."""

from typing import Annotated, Any, Optional

import typer
from rich.live import Live
from rich.markdown import Markdown
from rich.table import Table
from rich.text import Text

from ..config import Settings
from ..llm.embed import Embedder
from ..llm.ollama import Ollama, OllamaError
from ..query.answer import build_messages
from ..query.retrieval import Hit
from ..query.retrieval import search as hybrid_search
from ._common import ProjectOption, console, fail, open_project, require_models


def ask(
    question: Annotated[str, typer.Argument(help="Your question, in quotes.")],
    path: ProjectOption = None,
    k: Annotated[Optional[int], typer.Option("--k", "-k", min=1, help="How many excerpts to retrieve.")] = None,
    model: Annotated[Optional[str], typer.Option("--model", "-m", help="Ollama chat model to answer with.")] = None,
    think: Annotated[Optional[str], typer.Option("--think", help="Reasoning effort: low, medium or high.")] = None,
    show_thinking: Annotated[bool, typer.Option("--show-thinking", help="Print the model's reasoning too.")] = False,
    sources: Annotated[bool, typer.Option("--sources/--no-sources", help="List the excerpts given to the model.")] = True,
) -> None:
    """Answer a question from what was learned about the project."""
    settings = Settings.from_env()
    chat_model = model or settings.chat_model
    client = Ollama(settings.ollama_url)
    require_models(client, settings.embed_model, chat_model)
    root, store = open_project(settings, path)
    with store:
        embedder = Embedder(client, settings.embed_model, settings.keep_alive, settings.embed_batch)
        try:
            with console.status("Searching the index..."):
                hits = hybrid_search(store, embedder, question, k or settings.top_k)
        except OllamaError as exc:
            fail(str(exc))
    if not hits:
        fail("The index is empty, so there's nothing to answer from. Check `antai status`.")

    messages, used = build_messages(question, hits, root.name, settings.context_chars)
    console.print(f"[dim]{root} | {len(used)} excerpts | {chat_model}[/]", highlight=False)
    try:
        stats = _stream_answer(
            client, chat_model, messages, settings, think if think is not None else settings.think, show_thinking
        )
    except OllamaError as exc:
        fail(str(exc))
    if sources:
        _print_sources(used)
    _print_speed(stats)


def _stream_answer(
    client: Ollama, model: str, messages: list[dict[str, str]], settings: Settings, think: str, show_thinking: bool
) -> dict[str, Any]:
    answer = ""
    stats: dict[str, Any] = {}
    reasoning_shown = False
    status = console.status("Thinking...")
    status.start()
    live: Optional[Live] = None
    try:
        for kind, payload in client.chat(
            model, messages, think=think, num_ctx=settings.num_ctx, keep_alive=settings.keep_alive
        ):
            if kind == "thinking" and show_thinking:
                if not reasoning_shown:
                    status.stop()
                    console.rule("[dim]reasoning[/]", style="dim")
                    reasoning_shown = True
                console.print(payload, end="", style="dim", markup=False, highlight=False)
            elif kind == "content":
                if live is None:
                    status.stop()
                    if reasoning_shown:
                        console.print()
                        console.rule(style="dim")
                    live = Live(console=console, refresh_per_second=8, transient=True, vertical_overflow="ellipsis")
                    live.start()
                answer += payload
                live.update(Markdown(answer))
            elif kind == "done":
                stats = payload
    finally:
        status.stop()
        if live is not None:
            live.stop()
    if answer.strip():
        console.print(Markdown(answer))
    else:
        console.print("[yellow]The model returned an empty answer.[/]")
    return stats


def _print_sources(hits: list[Hit]) -> None:
    table = Table(box=None, show_header=False, padding=(0, 1), title="Sources", title_justify="left", title_style="bold")
    table.add_column(justify="right")
    table.add_column(overflow="fold", ratio=3)
    table.add_column(overflow="fold", ratio=2)
    for n, hit in enumerate(hits, 1):
        c = hit.chunk
        table.add_row(Text(f"{n}.", style="dim"), Text(c.location), Text(", ".join(c.symbols[:3]), style="dim"))
    console.print()
    console.print(table)


def _print_speed(stats: dict[str, Any]) -> None:
    if not stats:
        return
    generated = stats.get("eval_count", 0)
    seconds = stats.get("eval_duration", 0) / 1e9
    rate = f" at {generated / seconds:.0f} tok/s" if seconds else ""
    total = stats.get("total_duration", 0) / 1e9
    console.print(
        f"[dim]{stats.get('prompt_eval_count', 0):,} prompt tokens, {generated:,} generated{rate}, {total:.1f}s total[/]"
    )
