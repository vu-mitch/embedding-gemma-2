"""A tiny in-memory cosine-similarity vector index."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .extractors import Chunk


@dataclass
class Hit:
    score: float
    chunk: Chunk


@dataclass
class VectorIndex:
    dim: int
    chunks: list[Chunk] = field(default_factory=list)
    vectors: np.ndarray | None = None

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if not chunks:
            return
        assert len(chunks) == len(vectors)
        self.chunks.extend(chunks)
        self.vectors = vectors if self.vectors is None else np.vstack([self.vectors, vectors])

    def __len__(self) -> int:
        return len(self.chunks)

    def search(self, query_vec: np.ndarray, top_k: int = 5, modalities: set[str] | None = None, min_score: float = -1.0) -> list[Hit]:
        if self.vectors is None or not self.chunks:
            return []
        scores = self.vectors @ query_vec
        mask = np.array([modalities is None or c.modality in modalities for c in self.chunks])
        scores = np.where(mask, scores, -np.inf)
        order = np.argsort(-scores)
        hits = []
        for i in order:
            if not np.isfinite(scores[i]) or scores[i] < min_score or len(hits) >= top_k:
                break
            hits.append(Hit(float(scores[i]), self.chunks[i]))
        return hits

    def rank_sources(self, query_vec: np.ndarray, modalities: set[str] | None = None) -> list[tuple[str, str, float]]:
        """Best chunk score per source file: (source, modality, score), descending."""
        if self.vectors is None:
            return []
        scores = self.vectors @ query_vec
        best: dict[str, tuple[str, float]] = {}
        for c, s in zip(self.chunks, scores):
            if modalities is not None and c.modality not in modalities:
                continue
            if c.source not in best or s > best[c.source][1]:
                best[c.source] = (c.modality, float(s))
        return sorted(((src, m, s) for src, (m, s) in best.items()), key=lambda x: -x[2])
