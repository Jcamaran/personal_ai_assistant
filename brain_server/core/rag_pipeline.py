# LangChain RAG pipeline logic
# Handles document loading, chunking, and retrieval
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from .embeddings import get_or_create_collection
from .logger import setup_logger
import uuid
import os
from typing import Dict, List, Optional, Any

# Initialize logger
logger = setup_logger(__name__)


def ingest_document(file_path: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load a document, split into chunks, and add to ChromaDB.
    
    Args:
        file_path (str): Path to the document file
        metadata (dict, optional): Additional metadata to store
    
    Returns:
        dict: Ingestion results with success status, document_id, chunks_created
    """
    # Generate unique document ID
    document_id = str(uuid.uuid4())
    
    logger.info(f"Starting ingestion for: {file_path}")
    
    # Load document
    try:
        loader = TextLoader(file_path, encoding='utf-8')
        documents = loader.load()
        logger.debug(f"Loaded {len(documents)} document(s) from {file_path}")
    except Exception as e:
        logger.error(f"Failed to load document from {file_path}: {e}")
        return {
            "success": False,
            "message": f"Failed to load document: {str(e)}",
            "file_path": file_path,
            "chunks_created": 0,
            "document_id": None
        }
    
    # Split into chunks
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Split document into {len(chunks)} chunks")
    except Exception as e:
        logger.error(f"Failed to split document {file_path}: {e}")
        return {
            "success": False,
            "message": f"Failed to split document: {str(e)}",
            "file_path": file_path,
            "chunks_created": 0,
            "document_id": None
        }
    
    # add doc to ChromaDB
    try:
        collection = get_or_create_collection()
        
        # Prepare data for ChromaDB
        chunk_ids = [_generate_chunk_id(document_id, i) for i in range(len(chunks))]
        chunk_texts = [chunk.page_content for chunk in chunks]
        chunk_metadatas = [
            _extract_metadata(file_path, i, len(chunks), metadata) 
            for i in range(len(chunks))
        ]
        
        # Add to collection
        collection.add(
            documents=chunk_texts,
            ids=chunk_ids,
            metadatas=chunk_metadatas
        )
        
        logger.info(f"Successfully ingested {len(chunks)} chunks with document_id: {document_id}")
        
        return {
            "success": True,
            "message": f"Successfully ingested {len(chunks)} chunks",
            "file_path": file_path,
            "chunks_created": len(chunks),
            "document_id": document_id
        }
        
    except Exception as e:
        logger.error(f"Failed to add document to ChromaDB: {e}")
        return {
            "success": False,
            "message": f"Failed to add to database: {str(e)}",
            "file_path": file_path,
            "chunks_created": 0,
            "document_id": None
        }


def query_documents(query: str, top_k: int = 5, filter_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Search ChromaDB for relevant document chunks.
    
    Args:
        query (str): The search query
        top_k (int): Number of results to return
        filter_metadata (dict, optional): Metadata filters
    
    Returns:
        dict: {
            "context": str,  # Combined text for LLM
            "sources": list  # List of source documents with metadata
        }
    """
    logger.info(f"Querying documents with: '{query[:50]}...' (top_k={top_k})")
    
    try:
        collection = get_or_create_collection()
        
        # Query ChromaDB
        results = collection.query(
            query_texts=[query],
            n_results=top_k,
            where=filter_metadata  # Optional metadata filtering
        )
        
        logger.debug(f"Retrieved {len(results['documents'][0])} results")
        
        # Format context for LLM
        context = _format_context(results)
        
        # Format sources for response
        sources = []
        if results['documents'] and len(results['documents'][0]) > 0:
            for i in range(len(results['documents'][0])):
                sources.append({
                    "content": results['documents'][0][i],
                    "file_path": results['metadatas'][0][i].get('file_path', 'unknown'),
                    "similarity_score": float(1 - results['distances'][0][i]) if results.get('distances') else 0.0,
                    "metadata": results['metadatas'][0][i]
                })
        
        logger.info(f"Query successful, returning {len(sources)} sources")
        
        return {
            "context": context,
            "sources": sources
        }
        
    except Exception as e:
        logger.error(f"Failed to query documents: {e}")
        return {
            "context": "",
            "sources": []
        }


def _generate_chunk_id(document_id: str, chunk_index: int) -> str:
    """
    Generate a unique ID for a document chunk.
    
    Args:
        document_id (str): Unique document identifier
        chunk_index (int): Index of the chunk
    
    Returns:
        str: Unique chunk ID
    """
    return f"{document_id}_chunk_{chunk_index}"


def _extract_metadata(file_path: str, chunk_index: int, total_chunks: int, additional_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Build metadata for a document chunk.
    
    Args:
        file_path (str): Path to the document
        chunk_index (int): Index of this chunk
        total_chunks (int): Total number of chunks
        additional_metadata (dict, optional): Additional metadata to include
    
    Returns:
        dict: Complete metadata for the chunk
    """
    metadata = {
        "file_path": file_path,
        "file_name": os.path.basename(file_path),
        "chunk_index": chunk_index,
        "total_chunks": total_chunks
    }
    
    # Merge additional metadata if provided
    if additional_metadata:
        metadata.update(additional_metadata)
    
    return metadata


def _format_context(results: Dict[str, Any]) -> str:
    """
    Format ChromaDB query results into a context string for LLM.
    
    Args:
        results (dict): ChromaDB query results
    
    Returns:
        str: Formatted context string
    """
    if not results['documents'] or len(results['documents'][0]) == 0:
        return "No relevant information found in your notes."
    
    # Combine chunks with source labels
    context_parts = []
    for i, doc in enumerate(results['documents'][0]):
        file_name = results['metadatas'][0][i].get('file_name', 'unknown')
        context_parts.append(f"[From {file_name}]\n{doc}")
    
    return "\n\n".join(context_parts)





