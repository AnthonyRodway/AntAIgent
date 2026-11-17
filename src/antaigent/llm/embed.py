"""Turn text into unit-length vectors with an Ollama embedding model."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .ollama import Ollama

# Some embedding models are trained with task prefixes and retrieve noticeably
# worse without them: (prefix for documents, prefix for queries).
_PREFIXES = {
    "nomic-embed": ("search_document: ", "search_query: "),
    "mxbai-embed": ("", "Represent this sentence for searching relevant passages: "),
    "snowflake-arctic-embed": ("", "Represent this sentence for searching relevant passages: "),
}

MAX_INPUT_CHARS = 8000


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class Embedder:
    def __init__(self, client: Ollama, model: str, keep_alive: str = "30m", batch_size: int = 32):
        self.client = client
        self.model = model
        self.keep_alive = keep_alive
        self.batch_size = batch_size
        self._doc_prefix, self._query_prefix = next(
            (p for name, p in _PREFIXES.items() if model.startswith(name)), ("", "")
        )

    def _embed(self, texts: list[str]) -> np.ndarray:
        rows: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = [t[:MAX_INPUT_CHARS] for t in texts[i : i + self.batch_size]]
            rows.extend(self.client.embed(self.model, batch, self.keep_alive))
        return _normalize(np.asarray(rows, dtype=np.float32))

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        return self._embed([self._doc_prefix + t for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([self._query_prefix + text])[0]
