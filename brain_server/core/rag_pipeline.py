# RAG pipeline: document loading, chunking, retrieval, and collection management.
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from .embeddings import get_or_create_collection
from .logger import setup_logger
from . import config
import asyncio
import uuid
import os
from typing import Dict, List, Optional, Any, Set, Tuple

logger = setup_logger(__name__)

NO_NOTES_CONTEXT = "No relevant information found in your notes."


def _failure(message: str, file_path: str, deleted_count: int) -> Dict[str, Any]:
    """Build a standard failure result for ingest_document."""
    return {
        "success": False,
        "message": message,
        "file_path": file_path,
        "chunks_added": 0,
        "chunks_deleted": deleted_count,
        "document_id": None,
        "is_update": False,
    }


async def delete_document_by_file_path(file_path: str) -> Dict[str, Any]:
    """Delete all chunks for a file. Makes re-ingestion idempotent."""
    try:
        collection = get_or_create_collection()
        existing = await asyncio.to_thread(collection.get, where={"file_path": file_path})

        if not existing or not existing["ids"]:
            return {"success": True, "deleted_chunks": 0, "message": "No existing chunks to delete"}

        await asyncio.to_thread(collection.delete, ids=existing["ids"])
        count = len(existing["ids"])
        logger.info(f"Deleted {count} existing chunks for: {file_path}")
        return {"success": True, "deleted_chunks": count, "message": f"Deleted {count} old chunks"}

    except Exception as e:
        logger.error(f"Error deleting chunks for {file_path}: {e}")
        return {"success": False, "deleted_chunks": 0, "message": f"Error during deletion: {e}"}


async def ingest_document(file_path: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load a document, split it into chunks, and store it in ChromaDB.

    Existing chunks for the same file are removed first, so re-ingesting a
    modified file replaces its old content.
    """
    deletion_result = await delete_document_by_file_path(file_path)
    deleted_count = deletion_result.get("deleted_chunks", 0)
    document_id = str(uuid.uuid4())

    logger.info(f"Ingesting: {file_path}")

    try:
        loader = TextLoader(file_path, encoding="utf-8")
        documents = await asyncio.to_thread(loader.load)
    except Exception as e:
        logger.error(f"Failed to load document {file_path}: {e}")
        return _failure(f"Failed to load document: {e}", file_path, deleted_count)

    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = text_splitter.split_documents(documents)
    except Exception as e:
        logger.error(f"Failed to split document {file_path}: {e}")
        return _failure(f"Failed to split document: {e}", file_path, deleted_count)

    try:
        collection = get_or_create_collection()
        chunk_ids = [f"{document_id}_chunk_{i}" for i in range(len(chunks))]
        chunk_texts = [chunk.page_content for chunk in chunks]
        chunk_metadatas = [
            _build_chunk_metadata(file_path, i, len(chunks), metadata)
            for i in range(len(chunks))
        ]

        await asyncio.to_thread(
            collection.add,
            documents=chunk_texts,
            ids=chunk_ids,
            metadatas=chunk_metadatas,
        )

        if deleted_count > 0:
            logger.info(f"Updated {file_path}: {deleted_count} old chunks replaced by {len(chunks)}")
        else:
            logger.info(f"Ingested {file_path}: {len(chunks)} chunks (id: {document_id})")

        return {
            "success": True,
            "message": f"Successfully ingested {len(chunks)} chunks",
            "file_path": file_path,
            "chunks_added": len(chunks),
            "chunks_deleted": deleted_count,
            "document_id": document_id,
            "is_update": deleted_count > 0,
        }

    except Exception as e:
        logger.error(f"Failed to add document to ChromaDB: {e}")
        return _failure(f"Failed to add to database: {e}", file_path, deleted_count)


async def query_documents(query: str, top_k: int = 5, filter_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Search ChromaDB for chunks relevant to a query.

    Returns a dict with "context" (combined text for the LLM) and "sources"
    (per-chunk metadata for the API response).
    """
    logger.info(f"Query: '{query[:50]}' (top_k={top_k})")

    try:
        collection = get_or_create_collection()
        results = await asyncio.to_thread(
            collection.query,
            query_texts=[query],
            n_results=top_k,
            where=filter_metadata,
        )

        sources = []
        if results["documents"] and results["documents"][0]:
            for i in range(len(results["documents"][0])):
                # Chunk IDs have the form {document_id}_chunk_{index}
                chunk_id = results["ids"][0][i] if results.get("ids") else "unknown"
                document_id = chunk_id.rsplit("_chunk_", 1)[0] if "_chunk_" in chunk_id else chunk_id

                sources.append({
                    "document_id": document_id,
                    "content_snippet": results["documents"][0][i],
                    "similarity_score": float(1 - results["distances"][0][i]) if results.get("distances") else 0.0,
                    "metadata": results["metadatas"][0][i],
                })

        return {"context": _format_context(results), "sources": sources}

    except Exception as e:
        logger.error(f"Failed to query documents: {e}")
        return {"context": "", "sources": []}


async def list_documents() -> Dict[str, Any]:
    """List all indexed documents, grouped by file path with chunk counts."""
    try:
        collection = get_or_create_collection()
        records = await asyncio.to_thread(collection.get, include=["metadatas"])

        counts: Dict[str, int] = {}
        for metadata in records.get("metadatas") or []:
            file_path = (metadata or {}).get("file_path", "unknown")
            counts[file_path] = counts.get(file_path, 0) + 1

        documents = [
            {
                "file_path": path,
                "file_name": os.path.basename(path),
                "chunk_count": count,
            }
            for path, count in sorted(counts.items())
        ]
        return {"documents": documents, "total": len(documents)}

    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        return {"documents": [], "total": 0}


async def get_collection_stats() -> Dict[str, Any]:
    """Return chunk and document counts for the active collection."""
    try:
        collection = get_or_create_collection()
        total_chunks = await asyncio.to_thread(collection.count)
        doc_list = await list_documents()

        return {
            "collection_name": collection.name,
            "total_chunks": total_chunks,
            "total_documents": doc_list["total"],
        }

    except Exception as e:
        logger.error(f"Failed to get collection stats: {e}")
        return {
            "collection_name": config.COLLECTION_NAME,
            "total_chunks": 0,
            "total_documents": 0,
        }


def _build_chunk_metadata(file_path: str, chunk_index: int, total_chunks: int, additional_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build the metadata dict stored alongside each chunk."""
    metadata = {
        "file_path": file_path,
        "file_name": os.path.basename(file_path),
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
    }
    if additional_metadata:
        metadata.update(additional_metadata)
    return metadata


def _format_context(results: Dict[str, Any]) -> str:
    """Combine query results into a single context string for the LLM."""
    if not results["documents"] or not results["documents"][0]:
        return NO_NOTES_CONTEXT
    context_parts = []
    for i, doc in enumerate(results["documents"][0]):
        file_name = results["metadatas"][0][i].get("file_name", "unknown")
        context_parts.append(f"[From {file_name}]\n{doc}")

    return "\n\n".join(context_parts)


def format_sources_as_context(sources: List[Dict[str, Any]]) -> str:
    """Format a list of source dicts into the LLM context string."""
    if not sources:
        return NO_NOTES_CONTEXT

    context_parts = []
    for source in sources:
        file_name = (source.get("metadata") or {}).get("file_name", "unknown")
        snippet = source.get("content_snippet") or ""
        context_parts.append(f"[From {file_name}]\n{snippet}")
    return "\n\n".join(context_parts)


def _source_key(source: Dict[str, Any]) -> Tuple[str, Any]:
    """Identity for a chunk: file path plus chunk index when available."""
    metadata = source.get("metadata") or {}
    file_path = metadata.get("file_path") or metadata.get("file_name") or source.get("document_id")
    return (file_path, metadata.get("chunk_index"))


async def get_neighbor_chunks(
    sources: List[Dict[str, Any]],
    max_neighbors: int = 4,
) -> List[Dict[str, Any]]:
    """
    Pull adjacent chunks (chunk_index ± 1) from the same note as each source.

    Deduplicates against the provided sources and caps how many extras are added
    so the LLM context stays bounded.
    """
    if not sources:
        return []

    existing_keys: Set[Tuple[str, Any]] = {_source_key(s) for s in sources}
    neighbors: List[Dict[str, Any]] = []
    seen_neighbor_keys: Set[Tuple[str, Any]] = set()

    files_to_fetch: Dict[str, List[int]] = {}
    for source in sources:
        metadata = source.get("metadata") or {}
        file_path = metadata.get("file_path")
        chunk_index = metadata.get("chunk_index")
        if not file_path or chunk_index is None:
            continue
        files_to_fetch.setdefault(file_path, []).append(int(chunk_index))

    collection = get_or_create_collection()

    for file_path, indexes in files_to_fetch.items():
        wanted = set()
        for idx in indexes:
            wanted.add(idx - 1)
            wanted.add(idx + 1)
        wanted = {i for i in wanted if i >= 0}
        if not wanted:
            continue

        try:
            existing = await asyncio.to_thread(
                collection.get,
                where={"file_path": file_path},
                include=["documents", "metadatas"],
            )
        except Exception as e:
            logger.warning(f"Failed to fetch neighbor chunks for {file_path}: {e}")
            continue

        ids = existing.get("ids") or []
        documents = existing.get("documents") or []
        metadatas = existing.get("metadatas") or []

        for i, meta in enumerate(metadatas):
            if not meta:
                continue
            try:
                idx = int(meta.get("chunk_index"))
            except (TypeError, ValueError):
                continue
            if idx not in wanted:
                continue
            neighbor = {
                "document_id": (
                    ids[i].rsplit("_chunk_", 1)[0]
                    if i < len(ids) and "_chunk_" in ids[i]
                    else (ids[i] if i < len(ids) else "unknown")
                ),
                "content_snippet": documents[i] if i < len(documents) else "",
                "similarity_score": 0.0,
                "metadata": meta,
            }
            key = _source_key(neighbor)
            if key in existing_keys or key in seen_neighbor_keys:
                continue
            seen_neighbor_keys.add(key)
            neighbors.append(neighbor)
            if len(neighbors) >= max_neighbors:
                logger.info(f"Neighbor expansion added {len(neighbors)} chunks (capped)")
                return neighbors

    if neighbors:
        logger.info(f"Neighbor expansion added {len(neighbors)} chunks")
    return neighbors
