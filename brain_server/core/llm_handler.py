# Ollama API integration for local LLM inference.
import httpx
from typing import Optional, List
from . import config
from .logger import setup_logger

logger = setup_logger(__name__)

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using the user's "
    "personal notes."
)


def _build_prompt(query: str, context: str, conversation_history: Optional[List] = None) -> str:
    """Assemble the user prompt from retrieved context and prior turns."""
    history_text = ""
    if conversation_history:
        lines = ["", "Previous conversation (for context):"]
        for item in conversation_history:
            lines.append(f"User: {item['query']}")
            lines.append(f"Assistant: {item['answer']}")
        history_text = "\n".join(lines) + "\n"

    return (
        "You are an AI assistant with access to the user's personal Obsidian vault.\n\n"
        f"Context from their notes:\n{context}\n"
        f"{history_text}\n"
        f"Current question: {query}\n\n"
        "If this question refers to something from the previous conversation "
        "(like \"it\", \"that\", or \"tell me more\"), use the conversation history "
        "above to resolve the reference. Answer using both the context and the "
        "conversation history.\n\n"
        "Answer:"
    )


async def generate_response(query: str, context: str, conversation_history: Optional[List] = None) -> str:
    """Generate an answer from the LLM for a query with retrieved context."""
    prompt = _build_prompt(query, context, conversation_history)

    try:
        logger.info(f"Generating response (context: {len(context)} chars)")
        async with httpx.AsyncClient(timeout=config.LLM_TIMEOUT_SECONDS) as client:
            response = await client.post(
                f"{config.OLLAMA_HOST}/api/chat",
                json={
                    "model": config.LLM_MODEL,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "stream": False,
                    "options": {
                        "temperature": config.LLM_TEMPERATURE,
                        "num_predict": config.LLM_MAX_TOKENS,
                    },
                },
            )
            response.raise_for_status()
            return response.json()["message"]["content"]

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
