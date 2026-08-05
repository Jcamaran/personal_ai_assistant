# Central configuration for the brain server.
# All environment-driven settings live here so the rest of the code
# doesn't need to read os.environ directly.
import os
from dotenv import load_dotenv

load_dotenv()

# LLM
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:3b")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "500"))
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))

# Vector store
CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./vector_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "default_collection")

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# File watching / ingestion
WATCH_DIRECTORY = os.getenv("WATCH_DIRECTORY", "/data/obsidian")
EXCLUDED_FOLDERS = {".obsidian", ".trash", ".git", "node_modules", ".DS_Store"}
DEBOUNCE_SECONDS = int(os.getenv("DEBOUNCE_SECONDS", "3"))

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
