#!/bin/bash
# Startup script to pre-warm Ollama model

echo "Warming up Ollama model..."

# Send a dummy query to load model into memory
curl -s http://ollama:11434/api/chat -d '{
  "model": "llama3.2:3b",
  "messages": [{"role": "user", "content": "Hello"}],
  "stream": false,
  "options": {"num_predict": 10}
}' > /dev/null

echo "Model warmed up and ready!"
