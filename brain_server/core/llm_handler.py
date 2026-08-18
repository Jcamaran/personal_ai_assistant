# Ollama API integration
# Handles LLM calls to local Llama 3 model
import json
import os
import re
from typing import Any, Dict, List, Optional

import httpx

from .logger import setup_logger

DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:3b")  # Default to llama3.2 3B if not specified in .env
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
NO_NOTES_CONTEXT = "No relevant information found in your notes."

logger = setup_logger(__name__)


async def ollama_chat(
    messages: List[Dict[str, str]],
    *,
    json_mode: bool = False,
    temperature: float = 0.2,
    num_predict: int = 500,
    timeout: float = 60.0,
) -> str:
    """
    Send a chat completion request to Ollama and return the message content.

    Args:
        messages: Chat messages in Ollama format.
        json_mode: If True, request JSON-only output via Ollama format=json.
        temperature: Sampling temperature.
        num_predict: Max tokens to generate.
        timeout: HTTP timeout in seconds.

    Returns:
        str: Assistant message content.

    Raises:
        httpx.HTTPError: On transport or HTTP status errors.
        KeyError: If the response is missing the expected message field.
    """
    payload: Dict[str, Any] = {
        "model": DEFAULT_LLM_MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
        },
    }
    if json_mode:
        payload["format"] = "json"

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(f"{OLLAMA_HOST}/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()
        return data["message"]["content"]


def _format_history(conversation_history: Optional[List]) -> str:
    """Format conversation history into a prompt block."""
    if not conversation_history:
        return ""

    history_text = "\n\nPrevious conversation (for context):\n"
    for item in conversation_history:
        history_text += f"User: {item.get('query', '')}\n"
        history_text += f"Assistant: {item.get('answer', '')}\n\n"
    return history_text


def _parse_json_response(text: str) -> Dict[str, Any]:
    """Extract a JSON object from an LLM response, ignoring markdown fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in LLM response: {text[:200]}")

    return json.loads(cleaned[start:end + 1])


async def rewrite_query(
    query: str,
    conversation_history: Optional[List] = None,
    retry_hint: Optional[str] = None,
) -> Dict[str, str]:
    """
    Rewrite a user query into a standalone retrieval query.

    Resolves pronouns from conversation history and, on retry, targets
    information that the previous search missed.

    Returns:
        dict: {"rewritten_query": str, "reasoning": str}
    """
    history_text = _format_history(conversation_history)
    retry_block = ""
    if retry_hint:
        retry_block = (
            "\nThe previous search was too weak. Improve the query using this hint:\n"
            f"{retry_hint}\n"
        )

    prompt = f"""Rewrite the current question into a standalone search query for a personal Obsidian notes vault.
Replace pronouns (it, that, this, them) with the actual names or topics from conversation history.
Keep the rewritten query concise and specific. Do not answer the question.
{history_text}{retry_block}
Current question: {query}

Return JSON with keys rewritten_query and reasoning."""

    try:
        logger.info("Rewriting query for retrieval")
        raw = await ollama_chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You rewrite search queries for a personal notes database. "
                        "Return JSON only with keys rewritten_query and reasoning."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            json_mode=True,
            temperature=0.0,
            num_predict=150,
        )
        parsed = _parse_json_response(raw)
        rewritten = str(parsed.get("rewritten_query") or query).strip()
        reasoning = str(parsed.get("reasoning") or "").strip()
        if not rewritten:
            rewritten = query
        logger.info(f"Rewritten query: '{rewritten[:80]}'")
        return {"rewritten_query": rewritten, "reasoning": reasoning}
    except Exception as e:
        logger.warning(f"Query rewrite failed, using original query: {e}")
        return {"rewritten_query": query, "reasoning": f"rewrite failed: {e}"}


async def grade_documents(query: str, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Grade retrieved note chunks for relevance to the query in one batched call.

    Args:
        query: The (possibly rewritten) search query.
        sources: Retrieved source dicts with content_snippet and metadata.

    Returns:
        list: One dict per source: {"index": int, "relevant": bool, "reason": str}
    """
    if not sources:
        return []

    numbered = []
    for i, source in enumerate(sources):
        file_name = (source.get("metadata") or {}).get("file_name", "unknown")
        snippet = (source.get("content_snippet") or "")[:400]
        numbered.append(f"[{i}] From {file_name}:\n{snippet}")

    prompt = f"""Question: {query}

Note chunks:
{chr(10).join(numbered)}

For each chunk, decide if it contains facts that could help answer the question.
Be strict: mark unrelated chunks as not relevant.

Return JSON: {{"documents": [{{"index": 0, "relevant": true, "reason": "short"}}]}}
Include every index from 0 to {len(sources) - 1}."""

    fallback = [
        {"index": i, "relevant": True, "reason": "grader unavailable; kept by default"}
        for i in range(len(sources))
    ]

    try:
        logger.info(f"Grading {len(sources)} retrieved chunks")
        raw = await ollama_chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You grade whether note chunks help answer a question. "
                        "Return JSON only with a documents array."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            json_mode=True,
            temperature=0.0,
            num_predict=400,
        )
        parsed = _parse_json_response(raw)
        graded_raw = parsed.get("documents") or parsed.get("grades") or []
        by_index: Dict[int, Dict[str, Any]] = {}
        for item in graded_raw:
            try:
                idx = int(item.get("index"))
            except (TypeError, ValueError):
                continue
            if idx < 0 or idx >= len(sources):
                continue
            relevant = item.get("relevant")
            if isinstance(relevant, str):
                relevant = relevant.strip().lower() in {"true", "yes", "relevant", "1"}
            else:
                relevant = bool(relevant)
            by_index[idx] = {
                "index": idx,
                "relevant": relevant,
                "reason": str(item.get("reason") or ""),
            }

        grades = []
        for i in range(len(sources)):
            if i in by_index:
                grades.append(by_index[i])
            else:
                # Model omitted this index — keep it rather than dropping on sloppiness
                grades.append({
                    "index": i,
                    "relevant": True,
                    "reason": "index omitted by grader; kept by default",
                })

        kept = sum(1 for g in grades if g["relevant"])
        logger.info(f"Grader kept {kept}/{len(sources)} chunks")
        return grades
    except Exception as e:
        logger.warning(f"Document grading failed, keeping all chunks: {e}")
        return fallback


async def generate_response(query: str, context: str, conversation_history: Optional[List] = None) -> str:
    """     
    Generate a response from the LLM based on the query and retrieved context.
    
    Args:
        query (str): The user's query string.
        context (str): The relevant context retrieved from the knowledge base.
    
    Returns:
        str: The answer generated by the LLM.
        
    """
    history_text = _format_history(conversation_history)
    if conversation_history:
        logger.info(f"Using {len(conversation_history)} previous interactions for context")
    else:
        logger.info("No conversation history provided")

    notes_missing = not context or context.strip() == NO_NOTES_CONTEXT
    if notes_missing:
        context_block = NO_NOTES_CONTEXT
        extra_instruction = (
            "The notes did not contain relevant information for this question. "
            "Say that clearly. Do not invent facts that are not in the context."
        )
    else:
        context_block = context
        extra_instruction = (
            "Answer using only the notes above. When you use a fact, mention the note "
            "filename from the [From ...] labels. If the notes are incomplete, say so."
        )

    prompt = f"""You are an AI assistant with access to the user's personal Obsidian vault.

Context from their notes:
{context_block}
{history_text}
Current question: {query}

IMPORTANT: If this question refers to something from the previous conversation (like "it", "that", "tell me more"), use the conversation history above to understand what the user is referring to.
{extra_instruction}

Answer:"""
    try:
        logger.info(f"Generating response with context length: {len(context_block)} characters")
        return await ollama_chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant that answers from the user's personal notes. "
                        "Cite note filenames. Never invent details that are not in the provided context."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            json_mode=False,
            temperature=0.2,
            num_predict=500,
        )
    except httpx.HTTPError as e:
        logger.error(f"HTTP error generating response: {e}")
        return "I'm sorry, I couldn't connect to the LLM service."
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return "I'm sorry, I couldn't generate a response at this time."


async def check_ollama_health() -> bool:
    """Check if the Ollama LLM service is reachable and operational"""
    try:
        # Async health check using httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{OLLAMA_HOST}/api/tags")
            response.raise_for_status()
            models = response.json().get('models', [])
            
            # Check if our model is available
            model_names = [model['name'] for model in models]
            if DEFAULT_LLM_MODEL not in model_names:
                logger.warning(f"Specified LLM model '{DEFAULT_LLM_MODEL}' not found in Ollama. Available models: {model_names}")
                return False
                
            logger.info("Ollama health check successful")
            return True
            
    except httpx.HTTPError as e:
        logger.error(f"Ollama health check failed (HTTP): {e}")
        return False
    except Exception as e:
        logger.error(f"Ollama health check failed: {e}")
        return False
