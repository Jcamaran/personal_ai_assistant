# Ollama API integration for local LLM inference.
import json
import re
from typing import Any, Dict, List, Optional

import httpx

from . import config
from .logger import setup_logger

logger = setup_logger(__name__)

NO_NOTES_CONTEXT = "No relevant information found in your notes."

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers from the user's personal notes. "
    "Cite note filenames. Never invent details that are not in the provided context."
)


async def ollama_chat(
    messages: List[Dict[str, str]],
    *,
    json_mode: bool = False,
    temperature: Optional[float] = None,
    num_predict: Optional[int] = None,
    timeout: Optional[float] = None,
) -> str:
    """Send a chat completion request to Ollama and return the message content."""
    payload: Dict[str, Any] = {
        "model": config.LLM_MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": config.LLM_TEMPERATURE if temperature is None else temperature,
            "num_predict": config.LLM_MAX_TOKENS if num_predict is None else num_predict,
        },
    }
    if json_mode:
        payload["format"] = "json"

    async with httpx.AsyncClient(timeout=timeout or config.LLM_TIMEOUT_SECONDS) as client:
        response = await client.post(f"{config.OLLAMA_HOST}/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()
        return data["message"]["content"]


def _format_history(conversation_history: Optional[List]) -> str:
    """Format conversation history into a prompt block."""
    if not conversation_history:
        return ""

    lines = ["", "Previous conversation (for context):"]
    for item in conversation_history:
        lines.append(f"User: {item.get('query', '')}")
        lines.append(f"Assistant: {item.get('answer', '')}")
    return "\n".join(lines) + "\n"


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


def _build_prompt(query: str, context: str, conversation_history: Optional[List] = None) -> str:
    """Assemble the user prompt from retrieved context and prior turns."""
    history_text = _format_history(conversation_history)
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

    return (
        "You are an AI assistant with access to the user's personal Obsidian vault.\n\n"
        f"Context from their notes:\n{context_block}\n"
        f"{history_text}\n"
        f"Current question: {query}\n\n"
        "If this question refers to something from the previous conversation "
        '(like "it", "that", or "tell me more"), use the conversation history '
        "above to resolve the reference.\n"
        f"{extra_instruction}\n\n"
        "Answer:"
    )


async def rewrite_query(
    query: str,
    conversation_history: Optional[List] = None,
    retry_hint: Optional[str] = None,
) -> Dict[str, str]:
    """Rewrite a user query into a standalone retrieval query."""
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
    """Grade retrieved note chunks for relevance to the query in one batched call."""
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
    """Generate an answer from the LLM for a query with retrieved context."""
    prompt = _build_prompt(query, context, conversation_history)
    if conversation_history:
        logger.info(f"Using {len(conversation_history)} previous interactions for context")
    else:
        logger.info("No conversation history provided")

    try:
        logger.info(f"Generating response (context: {len(context)} chars)")
        return await ollama_chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
    except httpx.HTTPError as e:
        logger.error(f"HTTP error generating response: {e}")
        return "I'm sorry, I couldn't connect to the LLM service."
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return "I'm sorry, I couldn't generate a response at this time."


async def check_ollama_health() -> bool:
    """Return True if Ollama is reachable and the configured model is available."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{config.OLLAMA_HOST}/api/tags")
            response.raise_for_status()
            models = response.json().get("models", [])

            model_names = [model["name"] for model in models]
            if config.LLM_MODEL not in model_names:
                logger.warning(
                    f"Model '{config.LLM_MODEL}' not found in Ollama. "
                    f"Available: {model_names}"
                )
                return False
            return True

    except Exception as e:
        logger.error(f"Ollama health check failed: {e}")
        return False
