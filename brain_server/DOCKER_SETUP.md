# Docker Setup Guide

## Prerequisites

1. **Install Docker Desktop**
   - Download from [docker.com](https://www.docker.com/products/docker-desktop)
   - Start Docker Desktop
   - Verify: `docker --version`

## Quick Start

### 1. Configure environment

```powershell
# Copy example env file
cp .env.example .env

# Edit .env and update:
# - WATCH_DIRECTORY with your Obsidian vault path
```

### 2. Update docker-compose.yml

Edit `docker-compose.yml` and replace the Obsidian vault path:
```yaml
volumes:
  - C:/Users/YOUR_USERNAME/path/to/obsidian/vault:/data/obsidian:ro
```

### 3. Start services

```powershell
# Build and start all services
docker-compose up --build

# Or run in detached mode (background)
docker-compose up -d
```

### 4. Pull the model

```powershell
# Access Ollama container
docker exec -it brain_ollama ollama pull llama3

# Verify model is downloaded
docker exec -it brain_ollama ollama list
```

### 5. Test the API

Open browser: http://localhost:8000/docs

Or use curl:
```powershell
# Health check
curl http://localhost:8000/health

# Ingest a file
curl -X POST http://localhost:8000/ingest -H "Content-Type: application/json" -d '{"file_path": "/data/obsidian/test-note.md"}'

# Query
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"query": "What are my notes about?"}'
```

## Useful commands

```powershell
# View logs
docker-compose logs -f

# View logs for specific service
docker-compose logs -f brain_server

# Stop services
docker-compose down

# Stop and remove volumes (clean slate)
docker-compose down -v

# Rebuild after code changes
docker-compose up --build

# Access container shell
docker exec -it brain_server bash
```

## Troubleshooting

### GPU Support (Optional)

If you have NVIDIA GPU and want to use it:
1. Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
2. Keep the `deploy` section in docker-compose.yml

If you DON'T have NVIDIA GPU:
1. Remove the entire `deploy` section from the `ollama` service in docker-compose.yml

### Port Conflicts

If port 8000 or 11434 is already in use:
```yaml
ports:
  - "8001:8000"  # Change external port
```

### Volume Mounting Issues on Windows

Use forward slashes in paths:
```yaml
- C:/Users/camar/Documents/vault:/data/obsidian:ro
```

## Architecture

```
┌─────────────────────────────────────────┐
│  Docker Compose                         │
│                                         │
│  ┌──────────────┐  ┌─────────────────┐ │
│  │   Ollama     │  │  Brain Server   │ │
│  │  (Llama 3)   │←─│  (FastAPI+RAG)  │ │
│  │  Port: 11434 │  │  Port: 8000     │ │
│  └──────────────┘  └─────────────────┘ │
│         ↓                  ↓            │
│  ┌──────────────┐  ┌─────────────────┐ │
│  │ Ollama Data  │  │   ChromaDB      │ │
│  │  (Volume)    │  │   (Volume)      │ │
│  └──────────────┘  └─────────────────┘ │
│                           ↓             │
│                  ┌─────────────────┐    │
│                  │    Watcher      │    │
│                  │  (Background)   │    │
│                  └─────────────────┘    │
└─────────────────────────────────────────┘
         ↑
    Obsidian Vault
    (Host Machine)
```

## Next steps

1. Verify all services are running: `docker-compose ps`
2. Check brain_server is healthy: `curl http://localhost:8000/health`
3. Start ingesting your Obsidian notes
4. Query your knowledge base!
