# 🚀 Week 2 Implementation Plan
**Goal**: Build the Edge Client for Voice Interaction

---

## ✅ Week 1 Completed
- ✅ Brain server with FastAPI + async architecture
- ✅ RAG pipeline with ChromaDB (1667+ chunks indexed)
- ✅ LLM integration with Ollama (llama3.2:3b on GPU)
- ✅ File watcher with automatic ingestion
- ✅ **Update system working** (old chunks deleted, new chunks replace them)
- ✅ Docker containerization
- ✅ Shared package architecture

---

## 🎯 Week 2 Objectives

### **Primary Goal**: Build voice-enabled edge client
1. Local development on PC first (test without Raspberry Pi)
2. Audio input (microphone recording)
3. Speech-to-Text (STT) conversion
4. API client to communicate with brain server
5. Text-to-Speech (TTS) conversion
6. Audio output (speaker playback)
7. Deploy to Raspberry Pi

---

## 📅 Day-by-Day Breakdown

### **Day 1-2: Project Structure & API Client**

#### Create Edge Client Structure
```bash
cd "C:\Users\camar\OneDrive\Documents\Joa Projects\personal_ai_assistant"
mkdir edge_client
cd edge_client
mkdir audio
mkdir config
```

**Files to create:**
```
edge_client/
├── main.py                    # Main voice loop
├── api_client.py              # HTTP client for brain_server
├── requirements.txt           # Python dependencies
├── .env                       # Configuration
├── config/
│   └── __init__.py           # Config loader
└── audio/
    ├── __init__.py
    ├── recorder.py           # Microphone input
    ├── player.py             # Speaker output
    ├── stt.py                # Speech-to-Text
    └── tts.py                # Text-to-Speech
```

#### Task 1.1: Create `requirements.txt`
```txt
# API Communication
requests>=2.31.0
httpx>=0.27.0

# Audio Processing
pyaudio>=0.2.13
sounddevice>=0.4.6
soundfile>=0.12.1

# Speech Recognition
openai-whisper>=20231117
SpeechRecognition>=3.10.0

# Text-to-Speech (choose one)
pyttsx3>=2.90           # Local TTS (offline)
gTTS>=2.4.0             # Google TTS (requires internet)
# elevenlabs>=0.2.0     # Premium option

# Utilities
python-dotenv>=1.0.0
pydantic>=2.6.0

# Optional: Wake word detection
pvporcupine>=3.0.0      # Picovoice wake word
```

#### Task 1.2: Create `api_client.py`
**Purpose**: HTTP client to call brain server's `/query` and `/health` endpoints

**Key functions:**
- `check_health()` - Verify brain server is running
- `query_knowledge_base(query: str, top_k: int)` - Send query to RAG
- Uses `shared.models` for request/response schemas

#### Task 1.3: Create `.env` file
```env
# Brain Server Configuration
BRAIN_SERVER_URL=http://localhost:8000
BRAIN_SERVER_TIMEOUT=60

# Audio Configuration
SAMPLE_RATE=16000
CHANNELS=1
CHUNK_SIZE=1024

# Speech Recognition
STT_MODEL=base.en
STT_LANGUAGE=en

# Wake Word (optional)
WAKE_WORD_ENABLED=false
WAKE_WORD=hey assistant
```

#### Task 1.4: Test API Client (Local)
Test connecting to your running brain server from PC:
```python
python -c "from api_client import BrainClient; client = BrainClient(); print(client.check_health())"
```

---

### **Day 3-4: Audio Recording & STT**

#### Task 2.1: Create `audio/recorder.py`
**Purpose**: Record audio from microphone

**Key functions:**
- `record_audio(duration: int)` - Record for X seconds
- `listen_for_speech()` - Record until silence detected
- `save_audio(filename: str)` - Save WAV file

**Libraries**: `sounddevice` or `pyaudio`

#### Task 2.2: Create `audio/stt.py`
**Purpose**: Convert speech to text using Whisper

**Options:**
- **Local Whisper** (recommended for privacy):
  ```python
  import whisper
  model = whisper.load_model("base.en")
  result = model.transcribe("audio.wav")
  ```
- **Google Speech Recognition** (requires internet)
- **Azure Speech** (paid service)

**Key function:**
- `transcribe_audio(audio_data) -> str` - Returns transcribed text

#### Task 2.3: Test Recording & Transcription
```python
# Test script
from audio.recorder import record_audio
from audio.stt import transcribe_audio

print("Recording for 5 seconds...")
audio = record_audio(duration=5)
print("Transcribing...")
text = transcribe_audio(audio)
print(f"You said: {text}")
```

---

### **Day 5-6: TTS & Audio Playback**

#### Task 3.1: Create `audio/tts.py`
**Purpose**: Convert text to speech

**Options:**
1. **pyttsx3** (Local, offline, basic quality)
   ```python
   import pyttsx3
   engine = pyttsx3.init()
   engine.say("Hello, this is your AI assistant")
   engine.runAndWait()
   ```

2. **gTTS** (Google TTS, requires internet, better quality)
   ```python
   from gtts import gTTS
   tts = gTTS(text="Hello", lang='en')
   tts.save("response.mp3")
   ```

3. **ElevenLabs** (Premium, best quality, API key required)

**Key function:**
- `text_to_speech(text: str, output_file: str)` - Generate audio file

#### Task 3.2: Create `audio/player.py`
**Purpose**: Play audio through speakers

**Key functions:**
- `play_audio_file(filename: str)` - Play WAV/MP3
- `play_audio_stream(audio_data)` - Play from memory

**Libraries**: `sounddevice`, `pygame`, or `playsound`

#### Task 3.3: Test TTS & Playback
```python
from audio.tts import text_to_speech
from audio.player import play_audio

text = "Hello! I'm your AI assistant, how can I help you?"
audio_file = text_to_speech(text)
play_audio(audio_file)
```

---

### **Day 7: Main Voice Loop Integration**

#### Task 4.1: Create `main.py`
**Purpose**: Main application loop

**Pseudocode:**
```python
from api_client import BrainClient
from audio.recorder import record_audio
from audio.stt import transcribe_audio
from audio.tts import text_to_speech
from audio.player import play_audio
from shared.models import QueryRequest

def main():
    client = BrainClient()
    
    # Check brain server health
    if not client.check_health():
        print("❌ Brain server not accessible!")
        return
    
    print("✅ Connected to brain server")
    print("🎤 Listening... (say 'Hey Assistant' or press Enter)")
    
    while True:
        try:
            # Option 1: Wait for wake word
            # wait_for_wake_word()
            
            # Option 2: Press Enter to start
            input("Press Enter to speak...")
            
            # Record user query
            print("🎤 Recording...")
            audio = record_audio(duration=5)
            
            # Convert speech to text
            print("🔊 Transcribing...")
            query_text = transcribe_audio(audio)
            print(f"You asked: {query_text}")
            
            if not query_text:
                continue
            
            # Query brain server
            print("🧠 Thinking...")
            response = client.query(query_text, top_k=5)
            answer = response.answer
            
            print(f"Answer: {answer[:100]}...")
            
            # Convert answer to speech
            print("🗣️ Speaking...")
            audio_file = text_to_speech(answer)
            play_audio(audio_file)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
```

#### Task 4.2: Test Full Pipeline (PC)
Run the complete voice assistant on your PC:
```bash
python main.py
```

**Expected flow:**
1. Press Enter
2. Speak: "What is a RAG application?"
3. Hears transcription
4. Gets answer from brain server
5. Plays answer through speakers

---

## 🍓 Raspberry Pi Deployment (Optional This Week)

### **Setup Raspberry Pi**

#### Hardware Needed:
- Raspberry Pi 4 (4GB+ RAM recommended)
- USB Microphone or USB sound card with mic
- Speaker (3.5mm jack or USB)
- MicroSD card (32GB+)
- Power supply

#### Software Setup:
```bash
# 1. Install Raspberry Pi OS (64-bit recommended)
# 2. Update system
sudo apt-get update && sudo apt-get upgrade -y

# 3. Install Python dependencies
sudo apt-get install -y python3-pip python3-dev portaudio19-dev

# 4. Install PortAudio (required for pyaudio)
sudo apt-get install -y portaudio19-dev python3-pyaudio

# 5. Clone your project
cd ~
git clone <your-repo-url> personal_ai_assistant
cd personal_ai_assistant/edge_client

# 6. Install Python packages
pip3 install -r requirements.txt

# 7. Install Whisper (may take time on Pi)
pip3 install openai-whisper

# 8. Configure environment
nano .env  # Set BRAIN_SERVER_URL to your PC's IP
```

#### Network Configuration:
```bash
# Find your PC's IP address (on PC)
ipconfig  # Windows
# Look for IPv4 Address, e.g., 192.168.1.100

# Update .env on Raspberry Pi
BRAIN_SERVER_URL=http://192.168.1.100:8000
```

#### Test on Raspberry Pi:
```bash
# Test API connection
python3 -c "from api_client import BrainClient; print(BrainClient().check_health())"

# Test audio devices
python3 -c "import sounddevice; print(sounddevice.query_devices())"

# Run voice assistant
python3 main.py
```

---

## 🎯 Success Criteria

By end of Week 2, you should have:

- ✅ Edge client code structure created
- ✅ API client communicating with brain server
- ✅ Audio recording working
- ✅ Speech-to-Text converting voice to text
- ✅ Text-to-Speech generating audio responses
- ✅ Full voice loop working on PC
- ✅ (Optional) Deployed to Raspberry Pi and tested

---

## 🔄 Testing Checklist

### **Local Testing (PC)**
```bash
# 1. Test brain server connection
python -m edge_client.api_client

# 2. Test microphone
python -c "from audio.recorder import record_audio; record_audio(3)"

# 3. Test STT
python -c "from audio.stt import test_transcription; test_transcription()"

# 4. Test TTS
python -c "from audio.tts import text_to_speech; text_to_speech('Hello world')"

# 5. Run full pipeline
python main.py
```

### **Integration Test**
Ask these test questions:
1. "What is a RAG application?"
2. "Tell me about my notes on machine learning"
3. "What did I write about yesterday?"

---

## 📚 Useful Resources

### **Speech Recognition:**
- OpenAI Whisper: https://github.com/openai/whisper
- Whisper models: tiny (39M), base (74M), small (244M), medium (769M), large (1550M)
- Use `base.en` for balance of speed/accuracy on Raspberry Pi

### **Text-to-Speech:**
- pyttsx3: https://pyttsx3.readthedocs.io/
- gTTS: https://gtts.readthedocs.io/
- ElevenLabs: https://elevenlabs.io/

### **Audio Libraries:**
- PyAudio: https://people.csail.mit.edu/hubert/pyaudio/
- sounddevice: https://python-sounddevice.readthedocs.io/

### **Wake Word Detection:**
- Picovoice Porcupine: https://picovoice.ai/platform/porcupine/

---

## 💡 Tips & Best Practices

1. **Start Simple**: Get basic text input/output working before adding voice
2. **Test Incrementally**: Test each component separately before integration
3. **Use PC First**: Develop on PC, then deploy to Pi (faster iteration)
4. **Handle Errors**: Add error handling for network issues, audio failures
5. **Log Everything**: Add logging to debug issues on Raspberry Pi
6. **Optimize Later**: Get it working first, optimize performance later

---

## 🚧 Known Challenges

### **Raspberry Pi Performance:**
- Whisper models are slow on Pi (base.en takes ~10-20 seconds)
- Consider using cloud STT (Google, Azure) for faster response
- Or use smaller model (tiny) for speed

### **Network Latency:**
- Brain server must be accessible from Pi on same network
- Test with `ping` and `curl` before running app

### **Audio Setup:**
- USB microphones are easier than 3.5mm
- Test with `arecord` and `aplay` on Pi

---

## ⏭️ Week 3 Preview

If Week 2 goes well, Week 3 will focus on:
- Wake word detection ("Hey Assistant")
- Conversation context/memory
- Voice activity detection (automatic recording)
- Multi-turn conversations
- System integration (autostart on boot)
- Web dashboard for monitoring

---

**Ready to start? Begin with Day 1 Task 1.1! 🚀**
