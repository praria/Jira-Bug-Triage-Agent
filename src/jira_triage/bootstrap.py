"""Wire the objects used by the commands."""

from pathlib import Path

from jira_triage.config.settings import Settings
from jira_triage.domain.source import DefectSource
from jira_triage.rag.embeddings import SentenceTransformerEmbedder
from jira_triage.indexing import DefectIndexer
from jira_triage.jira.adf import AdfPlainTextConverter
from jira_triage.jira.client import JiraCloudClient
from jira_triage.jira.defect_source import JiraDefectSource
from jira_triage.jira.lookup import JiraDefectLookup
from jira_triage.jira.mapper import DefectMapper
from jira_triage.jira.query import BugQuery
from jira_triage.jira.root_cause import DescriptionSectionParser
from jira_triage.rag.augmentation import GeminiRecommendationModel, TriageAdvisor
from jira_triage.rag.retrieval import DefectRetriever
from jira_triage.store.chroma_store import ChromaDefectVectorStore


def build_defect_source(settings: Settings) -> DefectSource:
    """Create the Jira-backed source of defects."""
    client = JiraCloudClient(settings)
    converter = AdfPlainTextConverter()
    return JiraDefectSource(
        client=client,
        query=BugQuery(settings.jira_project_key, settings.jira_issue_type),
        mapper=DefectMapper(DescriptionSectionParser(settings.root_cause_label, text_converter=converter)),
    )


def build_defect_retriever(settings: Settings, chroma_path: Path | None = None) -> DefectRetriever:
    """Create the retriever that embeds one bug and searches the local store."""
    path = chroma_path or Path(settings.chroma_path)
    client = JiraCloudClient(settings)
    mapper = DefectMapper(
        DescriptionSectionParser(settings.root_cause_label, text_converter=AdfPlainTextConverter())
    )
    return DefectRetriever(
        lookup=JiraDefectLookup(
            client=client,
            mapper=mapper,
            project_key=settings.jira_project_key,
            issue_type=settings.jira_issue_type,
        ),
        embedder=SentenceTransformerEmbedder(settings.embedding_model),
        store=ChromaDefectVectorStore(path),
    )


def build_triage_advisor(settings: Settings) -> TriageAdvisor:
    """Create the augmentation step. Gemini phrases a clear match when an API key is set."""
    model = None
    if settings.gemini_api_key:
        model = GeminiRecommendationModel(settings.gemini_api_key, settings.gemini_model)
    return TriageAdvisor(model)


def build_defect_indexer(settings: Settings, chroma_path: Path | None = None) -> DefectIndexer:
    """Create the indexer that embeds defects into ChromaDB."""
    path = chroma_path or Path(settings.chroma_path)
    return DefectIndexer(
        source=build_defect_source(settings),
        embedder=SentenceTransformerEmbedder(settings.embedding_model),
        store=ChromaDefectVectorStore(path),
    )
