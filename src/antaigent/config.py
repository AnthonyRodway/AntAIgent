"""Settings, each overridable with an ANTAI_<NAME> environment variable."""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    ollama_url: str = "http://127.0.0.1:11434"
    chat_model: str = "gpt-oss:20b"
    embed_model: str = "nomic-embed-text"
    # Model that writes file, folder and project summaries; "" means the chat model.
    summary_model: str = ""
    # Reasoning effort for models that support levels (gpt-oss: low | medium | high).
    # "true"/"false" toggle thinking on other models; "" leaves it to the model's default.
    think: str = "low"
    num_ctx: int = 16384
    keep_alive: str = "30m"
    home: Path = Path.home() / ".antai"
    max_file_bytes: int = 1_000_000
    max_pdf_bytes: int = 2_000_000
    chunk_chars: int = 2400
    embed_batch: int = 32
    top_k: int = 12
    # Must leave room inside num_ctx for the question, the model's reasoning and its answer.
    context_chars: int = 30_000

    @classmethod
    def from_env(cls) -> Settings:
        overrides: dict[str, object] = {}
        for f in dataclasses.fields(cls):
            raw = os.environ.get(f"ANTAI_{f.name.upper()}")
            if raw is None:
                continue
            default = f.default
            if isinstance(default, int):
                overrides[f.name] = int(raw)
            elif isinstance(default, Path):
                overrides[f.name] = Path(raw).expanduser()
            else:
                overrides[f.name] = raw
        return cls(**overrides)

    @property
    def summarizer(self) -> str:
        return self.summary_model or self.chat_model
