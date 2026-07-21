# Personal AI Assistant

This is a voice-activated AI assistant that can answer questions about your personal knowledge base (like Obsidian notes). Think of it as having a conversation with your own notes—just hit a hotkey, ask a question, and get an answer spoken back to you.

## What does it do?

You press a hotkey (default: Ctrl+Shift+Space), ask something like "what were my thoughts on project X?", and the assistant searches through your documents, finds relevant info, and speaks the answer back to you. It remembers the conversation context, so you can ask follow-up questions naturally.

## How it works

The project is split into two parts:

**Brain Server** - The heavy lifter that runs on your main computer:
- FastAPI server with REST endpoints
- Uses Ollama (llama3.2:3b) for natural language generation
- ChromaDB for vector search over your documents (1600+ chunks indexed so far)
- Watches your document folders and automatically updates the index when files change
- Maintains conversation history for context-aware responses

**Edge Client** - The interface you interact with (can run locally or on a Raspberry Pi):
- Records your voice when you press the hotkey
- Converts speech to text
- Sends the question to the brain server
- Gets the response and speaks it back to you
- Keeps track of conversation for follow-ups

## Current Status

- ✅ Brain server fully functional with async architecture
- ✅ RAG pipeline working with auto-ingestion
- ✅ File watcher automatically syncs document changes
- ✅ Docker setup available
- ✅ Edge client working locally for voice interaction
- 🔄 Raspberry Pi deployment (planned)

## Quick Start

### Running the Brain Server

```powershell
# Start the server
.\START_BRAIN_SERVER.bat

# Or manually:
cd brain_server
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The server will be available at `http://localhost:8000`

### Running the Edge Client

```powershell
cd edge_client
python main.py
```

Press Ctrl+Shift+Space to start recording, then ask your question.

## Project Structure

```
brain_server/          # The "thinking" part
├── core/              # RAG pipeline, LLM, embeddings
├── data_sync/         # File watcher for auto-ingestion
├── utils/             # Bulk document ingestion tools
└── vector_db/         # ChromaDB data

edge_client/           # The "talking" part
├── audio/             # Recording, STT, TTS, playback
└── api_client.py      # Talks to brain server

shared/                # Common data models
```

## Configuration

Create a `.env` file in the edge_client folder:

```
BRAIN_SERVER_URL=http://localhost:8000
ACTIVATION_HOTKEY=ctrl+shift+space
MAX_RECORDING_DURATION=5
```

## Tech Stack

- Python 3.10+
- FastAPI for the server
- Ollama for local LLM inference
- ChromaDB for vector storage
- Whisper for speech-to-text (or other STT providers)
- TTS provider for speech output

## Notes

This is a personal project for experimenting with local AI and RAG systems. The goal is to have a fully private assistant that runs entirely on your own hardware—no cloud services, no data leaving your network.
