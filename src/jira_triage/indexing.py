"""Embed pulled defects and store the vectors."""

from jira_triage.domain.source import DefectSource
from jira_triage.rag.embeddings import DefectEmbeddingText, Embedder
from jira_triage.store.chroma_store import DefectVectorStore, StoredDefect


class DefectIndexer:
    """Pull defects, embed summary and description, and upsert the vectors."""

    def __init__(
        self,
        source: DefectSource,
        embedder: Embedder,
        store: DefectVectorStore,
        text: DefectEmbeddingText | None = None,
    ) -> None:
        self._source = source
        self._embedder = embedder
        self._store = store
        self._text = text or DefectEmbeddingText()

    def index(self) -> int:
        defects = self._source.pull()
        rendered = [self._text.render(defect) for defect in defects]
        vectors = self._embedder.embed(rendered)
        if len(vectors) != len(defects):
            raise ValueError("The embedder returned a different number of vectors than defects.")
        self._store.upsert(
            [
                StoredDefect(
                    defect_id=defect.defect_id,
                    text=text,
                    root_cause=defect.root_cause,
                    embedding=vector,
                )
                for defect, text, vector in zip(defects, rendered, vectors, strict=True)
            ]
        )
        return len(defects)
