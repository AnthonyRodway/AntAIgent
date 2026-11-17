"""`antai setup`: one-time download of code parsers, plus a check for the Ollama models."""

from ..config import Settings
from ..ingest.chunking import CODE_LANGUAGES
from ..llm.ollama import Ollama, OllamaError, has_model
from ._common import console, fail, installed_grammars


def setup() -> None:
    """One-time setup: download code parsers (needs internet) and check Ollama models."""
    import tree_sitter_language_pack as tslp

    settings = Settings.from_env()
    todo = sorted(set(CODE_LANGUAGES.values()) - installed_grammars())
    failed: list[str] = []
    with console.status("Downloading code parsers...") as spinner:
        for language in todo:
            spinner.update(f"Downloading {language} parser...")
            try:
                tslp.get_parser(language)
            except Exception as exc:
                failed.append(f"{language} ({type(exc).__name__})")
    console.print(f"[green]OK[/]   Code parsers: {len(todo) - len(failed)} downloaded, {len(failed)} failed.")
    if failed:
        console.print(f"[yellow]WARN[/] Not available: {', '.join(failed)}. Those files will use line-based splitting.")

    client = Ollama(settings.ollama_url)
    try:
        available = client.models()
    except OllamaError as exc:
        fail(str(exc))
    missing = [n for n in (settings.chat_model, settings.embed_model) if not has_model(available, n)]
    if missing:
        console.print("[yellow]WARN[/] Missing Ollama models. Download with:")
        for name in missing:
            console.print(f"       ollama pull {name}")
    else:
        console.print(f"[green]OK[/]   Ollama models present: {settings.chat_model}, {settings.embed_model}")
