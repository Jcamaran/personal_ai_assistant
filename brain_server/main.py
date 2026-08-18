# FastAPI entry point for the Brain Server
# Endpoints:
# - POST /ingest: Accepts file path to chunk and add to ChromaDB
# - POST /query: Accepts query string, performs similarity search, returns LLM response

from fastapi import FastAPI
from core.rag_pipeline import ingest_document, query_documents
from core.rag_agent import run_rag_agent
from core.llm_handler import generate_response, check_ollama_health
from core.embeddings import get_chromadb_client
from shared.models import IngestRequest, IngestResponse, QueryRequest, QueryResponse, HealthCheckResponse, AgentTrace
import os
import time
from datetime import datetime


app = FastAPI(title="Brain Server API", version="1.0.0")

# Track server start time for uptime calculation
START_TIME = datetime.now()

def _env_flag(name: str, default: str = "true") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


AGENTIC_RAG = _env_flag("AGENTIC_RAG", "true")


@app.get("/")
async def root():
    return {"status": "ok"}

@app.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest):
    """this endpoint accepts a file path, chunks the document, and adds it to ChromaDB"""
    result = await ingest_document(
        file_path = request.file_path,
        metadata = request.metadata
    )
    return IngestResponse(**result)


@app.post("/query", response_model = QueryResponse)
async def query(request: QueryRequest):
    """Query documents with RAG + LLM"""
    start = time.time()

    if AGENTIC_RAG:
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

    # One-shot path for A/B timing when AGENTIC_RAG=false
    rag_result = await query_documents(
        query = request.query,
        top_k = request.top_k,
        filter_metadata = request.filter_metadata
    )
    
    answer = await generate_response(
        query = request.query,
        context = rag_result['context'],
        conversation_history = request.conversation_history
    )

    return QueryResponse(
        query = request.query,
        answer = answer,
        sources = rag_result['sources'],
        processing_time = time.time() - start
    )


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Comprehensive health check for Brain Server"""
    # Check Ollama connection
    ollama_status = await check_ollama_health()
    
    # Check ChromaDB connection
    chroma_connected = False
    try:
        client = get_chromadb_client()
        client.heartbeat()  # Test connection
        chroma_connected = True
    except Exception:
        chroma_connected = False
    
    # Calculate uptime
    uptime_seconds = (datetime.now() - START_TIME).total_seconds()
    
    return HealthCheckResponse(
        status="healthy" if (ollama_status and chroma_connected) else "degraded",
        start_time=START_TIME.isoformat(),
        uptime=uptime_seconds,
        ollama_status=ollama_status,
        chroma_connected=chroma_connected
    )
