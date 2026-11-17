"""Hybrid retrieval: embedding similarity and keyword search, fused by rank."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ..index.store import Store, StoredChunk

# Reciprocal rank fusion constant; 60 is the value from the original RRF paper.
RRF_K = 60


class QueryEmbedder(Protocol):
    def embed_query(self, text: str) -> np.ndarray: ...


@dataclass(frozen=True)
class Hit:
    chunk: StoredChunk
    score: float
    vector_rank: int | None
    keyword_rank: int | None


def search(store: Store, embedder: QueryEmbedder, query: str, k: int = 12, pool: int = 60) -> list[Hit]:
    ids, matrix = store.vectors()
    vector_ranked: list[int] = []
    if len(ids):
        sims = matrix @ embedder.embed_query(query)
        n = min(pool, len(ids))
        top = np.argpartition(-sims, n - 1)[:n]
        top = top[np.argsort(-sims[top])]
        vector_ranked = [int(ids[i]) for i in top]
    keyword_ranked = store.keyword_search(query, pool)

    scores: dict[int, float] = {}
    vector_rank: dict[int, int] = {}
    keyword_rank: dict[int, int] = {}
    for ranking, ranks in ((vector_ranked, vector_rank), (keyword_ranked, keyword_rank)):
        for rank, chunk_id in enumerate(ranking, 1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)
            ranks[chunk_id] = rank

    best = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
    chunks = store.chunks_by_id(best)
    return [
        Hit(chunks[c], scores[c], vector_rank.get(c), keyword_rank.get(c)) for c in best if c in chunks
    ]
