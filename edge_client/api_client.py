"""HTTP client for the brain server API."""
import os
import sys
from typing import Optional, Dict, Any, List
import requests
from dotenv import load_dotenv

# Allow importing the shared package from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from shared.models import (
    QueryRequest,
    QueryResponse,
    IngestRequest,
    IngestResponse,
    DeleteDocumentRequest,
    DeleteDocumentResponse,
    DocumentListResponse,
    StatsResponse,
    HealthCheckResponse,
)

load_dotenv()


class BrainServerClient:
    """Client for the brain server's REST endpoints."""

    def __init__(self, base_url: Optional[str] = None, timeout: int = 60):
        """
        Args:
            base_url: Base URL of the brain server. Defaults to BRAIN_SERVER_URL.
            timeout: Request timeout in seconds.
        """
        base = base_url or os.getenv("BRAIN_SERVER_URL", "http://localhost:8000")
        self.base_url = base.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()

    def check_health(self, verbose: bool = True) -> bool:
        """Return True if the brain server reports a healthy status."""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            response.raise_for_status()

            health = HealthCheckResponse(**response.json())

            if verbose:
                print(f"Brain server status: {health.status}")
                print(f"  Ollama:   {'up' if health.ollama_status else 'down'}")
                print(f"  ChromaDB: {'up' if health.chroma_connected else 'down'}")
                print(f"  Uptime:   {health.uptime:.0f}s")

            return health.status == "healthy"

        except requests.exceptions.Timeout:
            print("Health check timed out - server not responding")
            return False
        except requests.exceptions.ConnectionError:
            print(f"Cannot connect to brain server at {self.base_url}")
            return False
        except Exception as e:
            print(f"Health check failed: {e}")
            return False

    def query(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[dict]] = None,
    ) -> Optional[QueryResponse]:
        """Send a query and return the generated answer with sources."""
        try:
            request_data = QueryRequest(
                query=query,
                top_k=top_k,
                filter_metadata=filter_metadata,
                conversation_history=conversation_history,
            )

            response = self.session.post(
                f"{self.base_url}/query",
                json=request_data.model_dump(),
                timeout=self.timeout,
            )
            response.raise_for_status()

            return QueryResponse(**response.json())

        except requests.exceptions.Timeout:
            print(f"Query timed out after {self.timeout}s")
            return None
        except requests.exceptions.ConnectionError:
            print("Cannot connect to brain server")
            return None
        except requests.exceptions.HTTPError as e:
            print(f"HTTP error: {e.response.status_code} - {e.response.text}")
            return None
        except Exception as e:
            print(f"Query failed: {e}")
            return None

    def ingest(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[IngestResponse]:
        """Ask the brain server to ingest a document."""
        try:
            request_data = IngestRequest(file_path=file_path, metadata=metadata)

            response = self.session.post(
                f"{self.base_url}/ingest",
                json=request_data.model_dump(),
                timeout=30,
            )
            response.raise_for_status()

            return IngestResponse(**response.json())

        except Exception as e:
            print(f"Ingestion failed: {e}")
            return None

    def list_documents(self) -> Optional[DocumentListResponse]:
        """List documents currently indexed on the brain server."""
        try:
            response = self.session.get(f"{self.base_url}/documents", timeout=30)
            response.raise_for_status()
            return DocumentListResponse(**response.json())
        except Exception as e:
            print(f"Failed to list documents: {e}")
            return None

    def delete_document(self, file_path: str) -> Optional[DeleteDocumentResponse]:
        """Remove a document's chunks from the knowledge base."""
        try:
            request_data = DeleteDocumentRequest(file_path=file_path)
            response = self.session.delete(
                f"{self.base_url}/documents",
                json=request_data.model_dump(),
                timeout=30,
            )
            response.raise_for_status()
            return DeleteDocumentResponse(**response.json())
        except Exception as e:
            print(f"Failed to delete document: {e}")
            return None

    def get_stats(self) -> Optional[StatsResponse]:
        """Fetch collection statistics from the brain server."""
        try:
            response = self.session.get(f"{self.base_url}/stats", timeout=30)
            response.raise_for_status()
            return StatsResponse(**response.json())
        except Exception as e:
            print(f"Failed to get stats: {e}")
            return None

    def close(self):
        """Close the underlying HTTP session."""
        self.session.close()
