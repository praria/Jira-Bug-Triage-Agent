"""Read health information from the local Chroma database."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb


@dataclass(frozen=True)
class RecordReport:
    """One stored embedding, its metadata, and its document."""

    chroma_id: str
    defect_id: str
    root_cause: str
    document: str
    dimensions: int
    vector_norm: float

    @property
    def has_root_cause(self) -> bool:
        return bool(self.root_cause.strip())

    @property
    def document_characters(self) -> int:
        return len(self.document)


@dataclass(frozen=True)
class CollectionReport:
    """Health of one Chroma collection."""

    name: str
    record_count: int
    dimensions: int
    distance: str
    metadata_fields: tuple[str, ...]
    empty_defect_id: int
    empty_root_cause: int
    empty_documents: int
    dimension_mismatches: int
    search_ok: bool
    search_ms: float
    records: tuple[RecordReport, ...]


@dataclass(frozen=True)
class DatabaseReport:
    """Health of the Chroma database directory."""

    path: str
    reachable: bool
    heartbeat: int
    disk_bytes: int
    tenant: str
    database: str
    message: str
    collections: tuple[CollectionReport, ...]

    @property
    def collection_count(self) -> int:
        return len(self.collections)

    @property
    def record_count(self) -> int:
        return sum(collection.record_count for collection in self.collections)

    @property
    def issues(self) -> tuple[str, ...]:
        if not self.reachable:
            return (self.message,)
        found: list[str] = []
        if not self.collections:
            found.append("The database has no collections.")
        for collection in self.collections:
            if collection.record_count == 0:
                found.append(f"Collection {collection.name} has no records.")
            if collection.dimension_mismatches:
                found.append(
                    f"Collection {collection.name} has {collection.dimension_mismatches} "
                    "vectors with a different length."
                )
            if collection.empty_defect_id:
                found.append(
                    f"Collection {collection.name} has {collection.empty_defect_id} "
                    "records without defect_id metadata."
                )
            if collection.empty_documents:
                found.append(
                    f"Collection {collection.name} has {collection.empty_documents} "
                    "records with an empty document."
                )
            if collection.empty_root_cause:
                found.append(
                    f"Collection {collection.name} has {collection.empty_root_cause} "
                    "records with an empty root cause."
                )
            if collection.record_count and not collection.search_ok:
                found.append(f"Collection {collection.name} did not return a nearest match.")
        return tuple(found)


class ChromaDatabaseMonitor:
    """Inspect a persistent Chroma database without adding records."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def status(self) -> DatabaseReport:
        disk_bytes = _directory_size(self._path)
        if not self._path.exists():
            return _unavailable(self._path, disk_bytes, "The Chroma directory does not exist.")
        try:
            client = chromadb.PersistentClient(path=str(self._path))
            heartbeat = int(client.heartbeat())
            collections = tuple(_inspect_collection(collection) for collection in client.list_collections())
        except Exception as exc:
            return _unavailable(self._path, disk_bytes, str(exc))
        return DatabaseReport(
            path=str(self._path),
            reachable=True,
            heartbeat=heartbeat,
            disk_bytes=disk_bytes,
            tenant=str(getattr(client, "tenant", "")),
            database=str(getattr(client, "database", "")),
            message="Connected",
            collections=collections,
        )


def _unavailable(path: Path, disk_bytes: int, message: str) -> DatabaseReport:
    return DatabaseReport(
        path=str(path),
        reachable=False,
        heartbeat=0,
        disk_bytes=disk_bytes,
        tenant="",
        database="",
        message=message,
        collections=(),
    )


def _inspect_collection(collection: Any) -> CollectionReport:
    metadata = dict(collection.metadata or {})
    count = int(collection.count())
    distance = str(metadata.get("hnsw:space") or "")
    if count == 0:
        return CollectionReport(
            name=collection.name,
            record_count=0,
            dimensions=0,
            distance=distance,
            metadata_fields=(),
            empty_defect_id=0,
            empty_root_cause=0,
            empty_documents=0,
            dimension_mismatches=0,
            search_ok=True,
            search_ms=0.0,
            records=(),
        )

    result = collection.get(include=["documents", "metadatas", "embeddings"])
    ids = result.get("ids") or []
    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []
    embeddings = result.get("embeddings")
    records: list[RecordReport] = []
    metadata_fields: set[str] = set()
    for index, record_id in enumerate(ids):
        metadata_row = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
        document = documents[index] if index < len(documents) and documents[index] else ""
        vector = [] if embeddings is None else [float(value) for value in embeddings[index]]
        metadata_fields.update(str(key) for key in metadata_row)
        root_cause = str(metadata_row.get("root_cause") or "")
        records.append(
            RecordReport(
                chroma_id=str(record_id),
                defect_id=str(metadata_row.get("defect_id") or ""),
                root_cause=root_cause,
                document=str(document or ""),
                dimensions=len(vector),
                vector_norm=_norm(vector),
            )
        )
    dimensions = records[0].dimensions if records else 0
    first_vector = [] if embeddings is None or not ids else [float(value) for value in embeddings[0]]
    search_ok, search_ms = _probe_search(collection, str(ids[0]), first_vector)
    records = sorted(records, key=_record_sort_key)
    return CollectionReport(
        name=collection.name,
        record_count=count,
        dimensions=dimensions,
        distance=distance,
        metadata_fields=tuple(sorted(metadata_fields)),
        empty_defect_id=sum(1 for record in records if not record.defect_id.strip()),
        empty_root_cause=sum(1 for record in records if not record.has_root_cause),
        empty_documents=sum(1 for record in records if record.document_characters == 0),
        dimension_mismatches=sum(1 for record in records if record.dimensions != dimensions),
        search_ok=search_ok,
        search_ms=search_ms,
        records=tuple(records),
    )


def _probe_search(collection: Any, record_id: str, vector: list[float]) -> tuple[bool, float]:
    if not vector:
        return False, 0.0
    started = time.perf_counter()
    result = collection.query(query_embeddings=[vector], n_results=1)
    elapsed_ms = (time.perf_counter() - started) * 1000
    matched = (result.get("ids") or [[]])[0]
    return (bool(matched) and matched[0] == record_id), elapsed_ms


def _record_sort_key(record: RecordReport) -> tuple[str, int, str]:
    project, separator, number = record.defect_id.partition("-")
    if separator and number.isdigit():
        return (project, int(number), record.defect_id)
    return (record.defect_id, 0, record.defect_id)


def _norm(vector: list[float]) -> float:
    if not vector:
        return 0.0
    return math.sqrt(sum(value * value for value in vector))


def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
