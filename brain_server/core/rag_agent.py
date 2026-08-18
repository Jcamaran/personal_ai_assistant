"""Corrective-RAG agent: rewrite, retrieve, grade, retry, then answer."""
from typing import Any, Dict, List, Optional

from .llm_handler import generate_response, grade_documents, rewrite_query
from .logger import setup_logger
from .rag_pipeline import (
    NO_NOTES_CONTEXT,
    format_sources_as_context,
    get_neighbor_chunks,
    query_documents,
)

logger = setup_logger(__name__)

MAX_RETRIEVAL_ROUNDS = 2
MIN_RELEVANT_CHUNKS = 2
FALLBACK_TOP_N = 3
DEFAULT_RETRIEVE_K = 8
MAX_NEIGHBORS = 4


def _should_rewrite(conversation_history: Optional[List], is_retry: bool) -> bool:
    """Skip rewrite on first-turn queries with no history to save an LLM call."""
    if is_retry:
        return True
    return bool(conversation_history)


def _dropped_file_names(sources: List[Dict[str, Any]], grades: List[Dict[str, Any]]) -> List[str]:
    names = []
    seen = set()
    for grade in grades:
        if grade.get("relevant"):
            continue
        idx = grade.get("index")
        if idx is None or idx < 0 or idx >= len(sources):
            continue
        file_name = (sources[idx].get("metadata") or {}).get("file_name", "unknown")
        if file_name not in seen:
            seen.add(file_name)
            names.append(file_name)
        if len(names) >= 8:
            break
    return names


def _kept_sources(sources: List[Dict[str, Any]], grades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    kept = []
    for grade in grades:
        if not grade.get("relevant"):
            continue
        idx = grade.get("index")
        if idx is None or idx < 0 or idx >= len(sources):
            continue
        kept.append(sources[idx])
    return kept


def _grade_hint(query: str, dropped: List[str], kept_count: int) -> str:
    dropped_text = ", ".join(dropped) if dropped else "unrelated notes"
    return (
        f"Previous search query: {query}. "
        f"Kept {kept_count} relevant chunks. "
        f"Dropped notes: {dropped_text}. "
        "Write a more specific search query targeting the missing information."
    )


async def run_rag_agent(
    query: str,
    conversation_history: Optional[List] = None,
    top_k: int = 5,
    filter_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run the Corrective-RAG loop and return answer, sources, and an agent trace.

    Returns:
        dict with keys answer, sources, context, agent_trace
    """
    retrieve_k = max(top_k, DEFAULT_RETRIEVE_K)
    search_query = query
    rewritten_query = query
    iterations = 0
    first_round_sources: List[Dict[str, Any]] = []
    kept: List[Dict[str, Any]] = []
    all_dropped: List[str] = []
    chunks_retrieved = 0
    used_fallback = False

    for round_idx in range(MAX_RETRIEVAL_ROUNDS):
        iterations = round_idx + 1
        is_retry = round_idx > 0

        if _should_rewrite(conversation_history, is_retry):
            retry_hint = None
            if is_retry:
                retry_hint = _grade_hint(search_query, all_dropped, len(kept))
            rewrite = await rewrite_query(
                query,
                conversation_history=conversation_history,
                retry_hint=retry_hint,
            )
            rewritten_query = rewrite["rewritten_query"]
            search_query = rewritten_query
        else:
            search_query = query
            rewritten_query = query
            logger.info("Skipping query rewrite (no conversation history)")

        rag_result = await query_documents(
            query=search_query,
            top_k=retrieve_k,
            filter_metadata=filter_metadata,
        )
        sources = rag_result.get("sources") or []
        chunks_retrieved = len(sources)
        if round_idx == 0:
            first_round_sources = sources

        if not sources:
            logger.info("No chunks retrieved")
            kept = []
            break

        grades = await grade_documents(search_query, sources)
        kept = _kept_sources(sources, grades)
        dropped = _dropped_file_names(sources, grades)
        for name in dropped:
            if name not in all_dropped:
                all_dropped.append(name)

        logger.info(
            f"Retrieval round {iterations}: retrieved={len(sources)} kept={len(kept)}"
        )

        if len(kept) >= MIN_RELEVANT_CHUNKS:
            break

        if round_idx + 1 < MAX_RETRIEVAL_ROUNDS:
            logger.info("Too few relevant chunks; retrying retrieval")
            continue

    if not kept:
        kept = first_round_sources[:FALLBACK_TOP_N]
        used_fallback = bool(kept)
        if used_fallback:
            logger.info(
                f"Grader rejected all chunks; falling back to top {len(kept)} vector hits"
            )

    if kept:
        neighbors = await get_neighbor_chunks(kept, max_neighbors=MAX_NEIGHBORS)
        if neighbors:
            kept = kept + neighbors

    context = format_sources_as_context(kept)
    if not kept:
        context = NO_NOTES_CONTEXT

    answer = await generate_response(
        query=query,
        context=context,
        conversation_history=conversation_history,
    )

    agent_trace = {
        "rewritten_query": rewritten_query,
        "iterations": iterations,
        "chunks_retrieved": chunks_retrieved,
        "chunks_kept": len(kept),
        "dropped_file_names": all_dropped[:8],
        "used_fallback": used_fallback,
    }

    logger.info(
        f"Agent complete: rewritten='{rewritten_query[:60]}' "
        f"iterations={iterations} kept={len(kept)} fallback={used_fallback}"
    )

    return {
        "answer": answer,
        "sources": kept,
        "context": context,
        "agent_trace": agent_trace,
    }
