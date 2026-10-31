"""Rows the dashboard shows for stored embeddings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from jira_triage.store.chroma_store import DefectVectorStore, StoredDefect


@dataclass(frozen=True)
class SimilarBug:
    """How close one stored bug is to the bug selected on the dashboard."""

    defect_id: str
    summary: str
    similarity: float


@dataclass(frozen=True)
class EmbeddingRow:
    """One indexed defect prepared for display."""

    defect_id: str
    summary: str
    description: str
    root_cause: str
    dimensions: int
    x: float
    y: float
    embedding: list[float]


class EmbeddingProjector:
    """Place embedding vectors on a 2D chart with principal components."""

    def project(self, vectors: list[list[float]]) -> list[tuple[float, float]]:
        if not vectors:
            return []
        if len(vectors) == 1 or len(vectors[0]) < 2:
            return [(0.0, 0.0) for _ in vectors]
        matrix = np.asarray(vectors, dtype=float)
        centered = matrix - matrix.mean(axis=0)
        _, _, components = np.linalg.svd(centered, full_matrices=False)
        projected = centered @ components[:2].T
        return [(float(point[0]), float(point[1])) for point in projected]


class EmbeddingCatalog:
    """Load every stored embedding and attach a 2D position."""

    def __init__(
        self,
        store: DefectVectorStore,
        projector: EmbeddingProjector | None = None,
    ) -> None:
        self._store = store
        self._projector = projector or EmbeddingProjector()

    def rows(self) -> list[EmbeddingRow]:
        records = self._store.list_all()
        positions = self._projector.project([record.embedding for record in records])
        return [
            _row(record, position) for record, position in zip(records, positions, strict=True)
        ]


def closest_bugs(selected: EmbeddingRow, rows: list[EmbeddingRow], limit: int = 5) -> list[SimilarBug]:
    """Rank the other bugs by cosine similarity of their embeddings."""
    if limit < 1:
        return []
    selected_vector = np.asarray(selected.embedding, dtype=float)
    selected_norm = float(np.linalg.norm(selected_vector))
    ranked: list[SimilarBug] = []
    for row in rows:
        if row.defect_id == selected.defect_id:
            continue
        other = np.asarray(row.embedding, dtype=float)
        denominator = selected_norm * float(np.linalg.norm(other))
        score = float(np.dot(selected_vector, other) / denominator) if denominator else 0.0
        ranked.append(SimilarBug(row.defect_id, row.summary, max(0.0, score)))
    ranked.sort(key=lambda bug: bug.similarity, reverse=True)
    return ranked[:limit]


def _row(record: StoredDefect, position: tuple[float, float]) -> EmbeddingRow:
    summary, _, description = record.text.partition("\n\n")
    return EmbeddingRow(
        defect_id=record.defect_id,
        summary=summary,
        description=description,
        root_cause=record.root_cause,
        dimensions=len(record.embedding),
        x=position[0],
        y=position[1],
        embedding=record.embedding,
    )
