import re
import zlib

import numpy as np
import pytest


class FakeEmbedder:
    """Deterministic bag-of-words vectors, so indexing and search run without Ollama."""

    model = "fake-embed"
    batch_size = 8
    dim = 256

    def _vector(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            v[zlib.crc32(token.encode()) % self.dim] += 1.0
        norm = np.linalg.norm(v)
        return v / norm if norm else v

    def embed_documents(self, texts):
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._vector(t) for t in texts])

    def embed_query(self, text):
        return self._vector(text)


@pytest.fixture
def embedder():
    return FakeEmbedder()


def python_grammar_installed() -> bool:
    try:
        import tree_sitter_language_pack as tslp

        return "python" in tslp.downloaded_languages()
    except Exception:
        return False
