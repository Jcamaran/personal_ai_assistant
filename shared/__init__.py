# Shared data models used by both brain_server and edge_client.
from .models import (
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SourceDocument,
    HealthCheckResponse,
    DeleteDocumentRequest,
    DeleteDocumentResponse,
    DocumentInfo,
    DocumentListResponse,
    StatsResponse,
)

__version__ = "1.1.0"

__all__ = [
    "IngestRequest",
    "IngestResponse",
    "QueryRequest",
    "QueryResponse",
    "SourceDocument",
    "HealthCheckResponse",
    "DeleteDocumentRequest",
    "DeleteDocumentResponse",
    "DocumentInfo",
    "DocumentListResponse",
    "StatsResponse",
]
