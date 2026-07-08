"""
HTTP Client for Brain Server API
Handles communication between edge client and brain server
"""
import os
import sys
from typing import Optional, Dict, Any
import requests
from dotenv import load_dotenv

# Add parent directory to path to import shared package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from shared.models import (
    QueryRequest,
    QueryResponse,
    IngestRequest,
    IngestResponse,
    HealthCheckResponse
)

# Load environment variables
load_dotenv()


class BrainServerClient:
    """
    Client for communicating with the Brain Server API.
    Handles requests to /health, /query, and /ingest endpoints.
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: int = 60
    ):
        """
        Initialize the Brain Server client.
        
        Args:
            base_url (str, optional): Base URL of brain server. Defaults to env var.
            timeout (int): Request timeout in seconds. Defaults to 60.
        """
        self.base_url = base_url or os.getenv("BRAIN_SERVER_URL", "http://localhost:8000")
        self.timeout = timeout
        self.session = requests.Session()
        
        # Remove trailing slash from base_url
        self.base_url = self.base_url.rstrip('/')
        
        print(f"🔗 Brain Server Client initialized: {self.base_url}")
    
    def check_health(self) -> bool:
        """
        Check if the brain server is healthy and accessible.
        
        Returns:
            bool: True if server is healthy, False otherwise
        """
        try:
            response = self.session.get(
                f"{self.base_url}/health",
                timeout=5  # Short timeout for health check
            )
            response.raise_for_status()
            
            data = response.json()
            health_response = HealthCheckResponse(**data)
            
            print(f"✅ Brain server is healthy")
            print(f"   Status: {health_response.status}")
            print(f"   Ollama: {'✓' if health_response.ollama_status else '✗'}")
            print(f"   ChromaDB: {'✓' if health_response.chroma_connected else '✗'}")
            print(f"   Uptime: {health_response.uptime:.2f}s")
            
            return health_response.status == "healthy"
            
        except requests.exceptions.Timeout:
            print(f"❌ Health check timed out - server not responding")
            return False
        except requests.exceptions.ConnectionError:
            print(f"❌ Cannot connect to brain server at {self.base_url}")
            return False
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return False
    
    def query(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[list] = None
    ) -> Optional[QueryResponse]:
        """
        Send a query to the brain server and get an AI-generated response.
        
        Args:
            query (str): The question or query text
            top_k (int): Number of relevant documents to retrieve
            filter_metadata (dict, optional): Metadata filters for search
            conversation_history (list, optional): Conversation history for context-aware responses
        Returns:
            QueryResponse: Response with answer and sources, or None if failed
        """
        try:
            # Create request using Pydantic model
            request_data = QueryRequest(
                query=query,
                top_k=top_k,
                filter_metadata=filter_metadata,
                conversation_history=conversation_history
            )
            
            print(f"🧠 Querying brain server: '{query[:50]}...'")
            if conversation_history:
                print(f"📜 Sending {len(conversation_history)} previous interactions for context")
            
            # Send POST request
            response = self.session.post(
                f"{self.base_url}/query",
                json=request_data.model_dump(),  # Pydantic v2 method
                timeout=self.timeout
            )
            response.raise_for_status()
            
            # Parse response using Pydantic model
            data = response.json()
            query_response = QueryResponse(**data)
            
            print(f"✅ Received answer ({len(query_response.sources)} sources)")
            
            return query_response
            
        except requests.exceptions.Timeout:
            print(f"❌ Query timed out after {self.timeout}s")
            return None
        except requests.exceptions.ConnectionError:
            print(f"❌ Cannot connect to brain server")
            return None
        except requests.exceptions.HTTPError as e:
            print(f"❌ HTTP error: {e.response.status_code} - {e.response.text}")
            return None
        except Exception as e:
            print(f"❌ Query failed: {e}")
            return None
    
    def ingest(
        self,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[IngestResponse]:
        """
        Request the brain server to ingest a document.
        
        Args:
            file_path (str): Path to the document file
            metadata (dict, optional): Additional metadata
        
        Returns:
            IngestResponse: Ingestion results, or None if failed
        """
        try:
            # Create request using Pydantic model
            request_data = IngestRequest(
                file_path=file_path,
                metadata=metadata
            )
            
            print(f"📄 Requesting ingestion: {file_path}")
            
            # Send POST request
            response = self.session.post(
                f"{self.base_url}/ingest",
                json=request_data.model_dump(),
                timeout=30
            )
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            ingest_response = IngestResponse(**data)
            
            print(f"✅ Ingestion complete: {ingest_response.chunks_added} chunks")
            
            return ingest_response
            
        except Exception as e:
            print(f"❌ Ingestion failed: {e}")
            return None
    
    def close(self):
        """Close the HTTP session."""
        self.session.close()
        print("🔌 Brain server client closed")


# Convenience function for quick testing
def test_connection():
    """Test connection to brain server"""
    client = BrainServerClient()
    
    print("\n=== Testing Brain Server Connection ===\n")
    
    # Test health check
    if not client.check_health():
        print("\n❌ Failed to connect to brain server")
        return False
    
    # Test query
    print("\n=== Testing Query ===\n")
    response = client.query("What is a RAG application?", top_k=3)
    
    if response:
        print(f"\n📖 Answer:\n{response.answer[:200]}...")
        print(f"\n📚 Sources: {len(response.sources)}")
        for i, source in enumerate(response.sources[:2], 1):
            print(f"\n{i}. {source.metadata.get('file_name', 'unknown')}")
            print(f"   {source.content_snippet[:100]}...")
    
    client.close()
    return True


if __name__ == "__main__":
    # Run test when executed directly
    test_connection()
