"""Embedding step: turn a bug's summary and description into a vector.

The root cause is stored beside the vector and is left out of this text.
"""

from __future__ import annotations

import sys
from typing import Protocol

from jira_triage.domain.defect import Defect


class DefectEmbeddingText:
    """Join the summary and description into one embedding string."""

    def render(self, defect: Defect) -> str:
        summary = defect.summary.strip()
        description = defect.description.strip()
        if summary and description:
            return f"{summary}\n\n{description}"
        return summary or description


class Embedder(Protocol):
    """Turns texts into vectors."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector for each text, in the same order."""


class SentenceTransformerEmbedder:
    """Local embeddings from a sentence-transformers model."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load()
        vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [vector.tolist() for vector in vectors]

    def _load(self):
        if self._model is None:
            print(f"Loading embedding model {self._model_name}.", file=sys.stderr)
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
        return self._model
