"""Store defect embeddings in ChromaDB."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings


@dataclass(frozen=True)
class StoredDefect:
    """One vector plus the defect id and root cause kept beside it."""

    defect_id: str
    text: str
    root_cause: str
    embedding: list[float]


@dataclass(frozen=True)
class Neighbor:
    """A stored defect returned by a similarity search."""

    defect_id: str
    root_cause: str
    document: str
    distance: float


class DefectVectorStore(ABC):
    """Persistence for defect embeddings."""

    @abstractmethod
    def upsert(self, records: list[StoredDefect]) -> None:
        """Insert or replace each defect by id."""

    @abstractmethod
    def count(self) -> int:
        """Return how many defects are stored."""

    @abstractmethod
    def fetch(self, defect_id: str) -> StoredDefect | None:
        """Return one stored defect, or None when the id is absent."""

    @abstractmethod
    def list_all(self) -> list[StoredDefect]:
        """Return every stored defect, ordered by defect id."""

    @abstractmethod
    def nearest(self, embedding: list[float], limit: int = 3, exclude_id: str | None = None) -> list[Neighbor]:
        """Return the closest stored defects. ``exclude_id`` is left out of the results."""


class _PrecomputedEmbeddings(EmbeddingFunction[Documents]):
    """Tells Chroma the vectors are already computed."""

    def __init__(self) -> None:
        return

    @staticmethod
    def name() -> str:
        return "precomputed"

    def __call__(self, input: Documents) -> Embeddings:
        raise RuntimeError("Embeddings are computed before they are stored.")

    def get_config(self) -> dict[str, Any]:
        return {}

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "_PrecomputedEmbeddings":
        return _PrecomputedEmbeddings()

    def default_space(self) -> str:
        return "cosine"


class ChromaDefectVectorStore(DefectVectorStore):
    """A persistent Chroma collection named ``defects``."""

    def __init__(self, path: Path, collection_name: str = "defects") -> None:
        self._path = path
        self._collection_name = collection_name
        self._collection = None

    def upsert(self, records: list[StoredDefect]) -> None:
        if not records:
            return
        self._open().upsert(
            ids=[record.defect_id for record in records],
            embeddings=[record.embedding for record in records],
            documents=[record.text for record in records],
            metadatas=[
                embedding_metadata(record.defect_id, record.root_cause) for record in records
            ],
        )

    def count(self) -> int:
        return int(self._open().count())

    def fetch(self, defect_id: str) -> StoredDefect | None:
        result = self._open().get(ids=[defect_id], include=["documents", "metadatas", "embeddings"])
        records = _records_from(result)
        return records[0] if records else None

    def list_all(self) -> list[StoredDefect]:
        result = self._open().get(include=["documents", "metadatas", "embeddings"])
        return sorted(_records_from(result), key=_defect_sort_key)

    def nearest(self, embedding: list[float], limit: int = 3, exclude_id: str | None = None) -> list[Neighbor]:
        if limit < 1:
            return []
        total = self.count()
        if total == 0:
            return []
        fetch_count = min(total, limit + (1 if exclude_id else 0))
        result = self._open().query(
            query_embeddings=[embedding],
            n_results=fetch_count,
            include=["documents", "metadatas", "distances"],
        )
        neighbors = _neighbors_from(result)
        if exclude_id:
            neighbors = [neighbor for neighbor in neighbors if neighbor.defect_id != exclude_id]
        return neighbors[:limit]

    def _open(self):
        if self._collection is None:
            self._path.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(self._path))
            self._collection = client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=_PrecomputedEmbeddings(),
            )
        return self._collection


def embedding_metadata(defect_id: str, root_cause: str) -> dict[str, str]:
    """Metadata stored beside each embedding. ``defect_id`` is the Jira key."""
    return {"defect_id": defect_id, "root_cause": root_cause}


def _records_from(result: dict[str, Any]) -> list[StoredDefect]:
    ids = result.get("ids") or []
    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []
    embeddings = result.get("embeddings")
    records: list[StoredDefect] = []
    for index, defect_id in enumerate(ids):
        metadata = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
        document = documents[index] if index < len(documents) and documents[index] else ""
        vector = [] if embeddings is None else embeddings[index]
        records.append(
            StoredDefect(
                defect_id=str(metadata.get("defect_id") or defect_id),
                text=str(document or ""),
                root_cause=str(metadata.get("root_cause") or ""),
                embedding=[float(value) for value in vector],
            )
        )
    return records


def _neighbors_from(result: dict[str, Any]) -> list[Neighbor]:
    ids = (result.get("ids") or [[]])[0]
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    neighbors: list[Neighbor] = []
    for index, chroma_id in enumerate(ids):
        metadata = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
        document = documents[index] if index < len(documents) and documents[index] else ""
        distance = distances[index] if index < len(distances) and distances[index] is not None else 1.0
        neighbors.append(
            Neighbor(
                defect_id=str(metadata.get("defect_id") or chroma_id),
                root_cause=str(metadata.get("root_cause") or ""),
                document=str(document or ""),
                distance=float(distance),
            )
        )
    return neighbors


def _defect_sort_key(record: StoredDefect) -> tuple[str, int, str]:
    project, separator, number = record.defect_id.partition("-")
    if separator and number.isdigit():
        return (project, int(number), record.defect_id)
    return (record.defect_id, 0, record.defect_id)
