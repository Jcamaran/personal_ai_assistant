# Personal AI Assistant

A voice-activated assistant for querying a personal knowledge base (like an Obsidian vault). Hit a hotkey, ask a question out loud, and the assistant searches your notes and speaks the answer back. It keeps recent conversation context, so follow-up questions work naturally.

Everything runs locally: no cloud services, no data leaving your network.

## How it works

The project is split into two parts.

**Brain server** — the heavy lifter that runs on your main machine:

- FastAPI server exposing a REST API
- Ollama (llama3.2:3b by default) for answer generation
- ChromaDB for vector search over your documents
- A file watcher that re-indexes notes automatically when they change

**Edge client** — the interface you talk to (runs locally or on a Raspberry Pi):

- Records your voice on hotkey press (fixed duration or until you stop talking)
- Transcribes speech with faster-whisper
- Sends the question to the brain server
- Speaks the answer back with offline TTS
- Tracks conversation history for follow-ups

## API

| Method | Endpoint     | Description                                  |
|--------|--------------|----------------------------------------------|
| POST   | `/query`     | Ask a question; returns answer plus sources  |
| POST   | `/ingest`    | Chunk and index a document                   |
| GET    | `/documents` | List indexed documents with chunk counts     |
| DELETE | `/documents` | Remove a document from the index             |
| GET    | `/stats`     | Collection statistics                        |
| GET    | `/health`    | Ollama and ChromaDB connectivity check       |

Interactive docs are available at `http://localhost:8000/docs` while the server is running.

## Quick start

### Brain server

```powershell
# With Docker (recommended)
.\START_BRAIN_SERVER.bat

# Or manually
cd brain_server
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

See `brain_server/DOCKER_SETUP.md` for the full Docker walkthrough.

### Edge client

```powershell
cd edge_client
python main.py
```

Press the hotkey (default Ctrl+Shift+Space) and ask your question.

## Project structure

```
brain_server/          # Retrieval and generation
├── core/              # RAG pipeline, LLM handler, embeddings, config
├── data_sync/         # File watcher for auto-ingestion
├── utils/             # Bulk document ingestion
└── vector_db/         # ChromaDB data (gitignored)

edge_client/           # Voice interface
├── audio/             # Recording, STT, TTS, playback
├── api_client.py      # HTTP client for the brain server
└── main.py            # Hotkey-driven voice loop

shared/                # Pydantic models used by both sides
```

## Configuration

Brain server settings live in `brain_server/.env` (see `.env.example`). Key options:

```
OLLAMA_HOST=http://localhost:11434
LLM_MODEL=llama3.2:3b
COLLECTION_NAME=obsidian_notes
WATCH_DIRECTORY=C:/path/to/your/vault
```

Edge client settings go in `edge_client/.env`:

```
BRAIN_SERVER_URL=http://localhost:8000
ACTIVATION_HOTKEY=ctrl+shift+space
MAX_RECORDING_DURATION=5
RECORDING_MODE=fixed      # or 'auto' to stop recording on silence
WHISPER_MODEL=base
```

## Tech stack

- Python 3.10+
- FastAPI + Uvicorn
- Ollama for local LLM inference
- ChromaDB for vector storage
- faster-whisper for speech-to-text
- pyttsx3 for offline text-to-speech
