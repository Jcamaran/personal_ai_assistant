# Shared data models package
# This package is used by both brain_server and edge_client

from .models import (
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SourceDocument,
    HealthCheckResponse
)

__version__ = "1.0.0"

__all__ = [
    "IngestRequest",
    "IngestResponse",
    "QueryRequest",
    "QueryResponse",
    "SourceDocument",
    "HealthCheckResponse",
]
