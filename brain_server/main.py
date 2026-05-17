# FastAPI entry point for the Brain Server
# Endpoints:
# - POST /ingest: Accepts file path to chunk and add to ChromaDB
# - POST /query: Accepts query string, performs similarity search, returns LLM response

from fastapi import FastAPI
from core.rag_pipeline import ingest_document, query_documents
from core.llm_handler import generate_response, check_ollama_health
from core.embeddings import get_chromadb_client
from shared.models import IngestRequest, IngestResponse, QueryRequest, QueryResponse, HealthCheckResponse
import time
from datetime import datetime


app = FastAPI(title="Brain Server API", version="1.0.0")

# Track server start time for uptime calculation
START_TIME = datetime.now() 

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

    # Get relevant documents (now async)
    rag_result = await query_documents(
        query = request.query,
        top_k = request.top_k,
        filter_metadata = request.filter_metadata
    )
    
    # Generate response from LLM (now async)
    answer = await generate_response(
        query = request.query,
        context = rag_result['context']
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