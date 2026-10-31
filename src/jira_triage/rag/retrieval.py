"""Retrieval step: embed one bug and return the closest stored bugs."""

from __future__ import annotations

from dataclasses import dataclass

from jira_triage.domain.defect import Defect
from jira_triage.jira.lookup import JiraDefectLookup
from jira_triage.rag.embeddings import DefectEmbeddingText, Embedder
from jira_triage.store.chroma_store import DefectVectorStore


@dataclass(frozen=True)
class SimilarDefect:
    """One stored defect ranked against the bug being triaged."""

    defect_id: str
    root_cause: str
    document: str
    similarity: float


@dataclass(frozen=True)
class RetrievalResult:
    """The loaded bug and the closest stored defects."""

    defect: Defect
    matches: list[SimilarDefect]


class DefectRetriever:
    """Load one Jira bug, embed its summary and description, and search Chroma."""

    def __init__(
        self,
        lookup: JiraDefectLookup,
        embedder: Embedder,
        store: DefectVectorStore,
        text: DefectEmbeddingText | None = None,
        limit: int = 3,
    ) -> None:
        self._lookup = lookup
        self._embedder = embedder
        self._store = store
        self._text = text or DefectEmbeddingText()
        self._limit = limit

    def retrieve(self, defect_id: str) -> RetrievalResult:
        defect = self._lookup.get(defect_id)
        rendered = self._text.render(defect)
        vectors = self._embedder.embed([rendered])
        if len(vectors) != 1:
            raise ValueError("The embedder did not return a vector for the defect.")
        neighbors = self._store.nearest(vectors[0], limit=self._limit, exclude_id=defect.defect_id)
        return RetrievalResult(
            defect=defect,
            matches=[
                SimilarDefect(
                    defect_id=neighbor.defect_id,
                    root_cause=neighbor.root_cause,
                    document=neighbor.document,
                    similarity=max(0.0, 1.0 - neighbor.distance),
                )
                for neighbor in neighbors
            ],
        )
