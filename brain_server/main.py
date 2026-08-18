"""FastAPI entry point for the brain server."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI
from brain_server.core import config
from brain_server.core.rag_pipeline import (
    ingest_document,
    query_documents,
    list_documents,
    delete_document_by_file_path,
    get_collection_stats,
)
from brain_server.core.rag_agent import run_rag_agent
from brain_server.core.llm_handler import generate_response, check_ollama_health
from brain_server.core.embeddings import get_chromadb_client
from shared.models import (
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    DeleteDocumentRequest,
    DeleteDocumentResponse,
    DocumentListResponse,
    StatsResponse,
    HealthCheckResponse,
    AgentTrace,
)
import time
from datetime import datetime

app = FastAPI(title="Brain Server API", version="1.1.0")

START_TIME = datetime.now()


@app.get("/")
async def root():
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest):
    """Chunk a document and add it to ChromaDB."""
    result = await ingest_document(
        file_path=request.file_path,
        metadata=request.metadata,
    )
    return IngestResponse(**result)


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Answer a question using retrieved context and the LLM."""
    start = time.time()

    if config.AGENTIC_RAG:
        result = await run_rag_agent(
            query=request.query,
            conversation_history=request.conversation_history,
            top_k=request.top_k,
            filter_metadata=request.filter_metadata,
        )
        return QueryResponse(
            query=request.query,
            answer=result["answer"],
            sources=result["sources"],
            processing_time=time.time() - start,
            agent_trace=AgentTrace(**result["agent_trace"]),
        )

    rag_result = await query_documents(
        query=request.query,
        top_k=request.top_k,
        filter_metadata=request.filter_metadata,
    )

    answer = await generate_response(
        query=request.query,
        context=rag_result["context"],
        conversation_history=request.conversation_history,
    )

    return QueryResponse(
        query=request.query,
        answer=answer,
        sources=rag_result["sources"],
        processing_time=time.time() - start,
    )


@app.get("/documents", response_model=DocumentListResponse)
async def get_documents():
    """List all documents currently indexed in the knowledge base."""
    result = await list_documents()
    return DocumentListResponse(**result)


@app.delete("/documents", response_model=DeleteDocumentResponse)
async def delete_document(request: DeleteDocumentRequest):
    """Remove all chunks belonging to a document."""
    result = await delete_document_by_file_path(request.file_path)
    return DeleteDocumentResponse(
        success=result["success"],
        file_path=request.file_path,
        chunks_deleted=result["deleted_chunks"],
        message=result["message"],
    )


@app.get("/stats", response_model=StatsResponse)
async def stats():
    """Return chunk and document counts for the collection."""
    result = await get_collection_stats()
    return StatsResponse(**result)


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Check Ollama and ChromaDB connectivity."""
    ollama_status = await check_ollama_health()

    chroma_connected = False
    try:
        client = get_chromadb_client()
        client.heartbeat()
        chroma_connected = True
    except Exception:
        chroma_connected = False

    uptime_seconds = (datetime.now() - START_TIME).total_seconds()

    return HealthCheckResponse(
        status="healthy" if (ollama_status and chroma_connected) else "degraded",
        start_time=START_TIME.isoformat(),
        uptime=uptime_seconds,
        ollama_status=ollama_status,
        chroma_connected=chroma_connected,
    )
