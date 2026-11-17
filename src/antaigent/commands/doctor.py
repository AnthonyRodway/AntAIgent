"""`antai doctor`: check Ollama, the models, GPU use and the code parsers."""

import typer

from ..config import Settings
from ..ingest.chunking import CODE_LANGUAGES
from ..llm.ollama import Ollama, OllamaError, has_model
from ._common import console, installed_grammars


def doctor() -> None:
    """Check that Ollama, the models, the GPU and the code parsers are ready."""
    settings = Settings.from_env()
    client = Ollama(settings.ollama_url)
    healthy = True
    try:
        console.print(f"[green]OK[/]   Ollama {client.version()} at {settings.ollama_url}")
        available = client.models()
        running = client.running()
    except OllamaError as exc:
        console.print(f"[red]FAIL[/] {exc}")
        raise typer.Exit(1)

    for role, name in (("chat model", settings.chat_model), ("embedding model", settings.embed_model)):
        if has_model(available, name):
            console.print(f"[green]OK[/]   {role}: {name}")
        else:
            healthy = False
            console.print(f"[red]FAIL[/] {role} {name} is missing. Run: ollama pull {name}")

    if running:
        for m in running:
            size = m.get("size") or 0
            on_gpu = 100 * (m.get("size_vram") or 0) / size if size else 0
            colour, label = ("green", "OK") if on_gpu >= 99 else ("yellow", "WARN")
            console.print(
                f"[{colour}]{label:<4}[/] {m['name']} loaded, {on_gpu:.0f}% on GPU, "
                f"context {m.get('context_length', '?')}"
            )
    else:
        console.print("[dim]--   No models loaded right now. The first question loads one, which takes ~20 s.[/]")

    missing = sorted(set(CODE_LANGUAGES.values()) - installed_grammars())
    if missing:
        console.print(f"[yellow]WARN[/] Code parsers not installed for: {', '.join(missing)}. Run: antai setup")
    else:
        console.print("[green]OK[/]   Code parsers installed")
    raise typer.Exit(0 if healthy else 1)
