# Audio Customer Support Agent

A modular audio-based customer support agent that uses Speech-to-Text (STT), Large Language Model (LLM) with Retrieval-Augmented Generation (RAG), and Text-to-Speech (TTS) technologies.

## Project Overview

This project provides a **blueprint** for implementing an audio customer support agent. Students implement the core functionality by completing TODO sections throughout the codebase.

### Pipeline Flow
```
Audio Input → STT → LLM Agent (with RAG) → TTS → Audio Output
```

## Architecture

### Core Components

1. **STT (Speech-to-Text)**: `src/stt/base_stt.py`
   - Generic STT service template with multiple implementation options
   - Supports both API services (Deepgram, AssemblyAI) and local models (Whisper, Wav2Vec2)

2. **LLM Agent**: `src/llm/agent.py`
   - Customer support agent using LangChain ReAct framework
   - Complete RAG system with 16 predefined customer support documents
   - ChromaDB integration for semantic search

3. **TTS (Text-to-Speech)**: `src/tts/base_tts.py`
   - Generic TTS service template with multiple implementation options
   - Supports API services (ElevenLabs, OpenAI) and local models (Edge TTS, Coqui TTS)

4. **Pipeline**: `src/pipeline.py`
   - Orchestrates the complete STT → LLM → TTS flow
   - Handles configuration and error management

5. **API Server**: `src/api/server.py`
   - FastAPI server with REST endpoints for testing
   - Health monitoring and debug endpoints

6. **Testing UI**: `streamlit_app.py`
   - Comprehensive testing interface with 4 tabs
   - Audio recording, playback, and health monitoring

## Project Structure

```
audio_support_agent/
├── src/
│   ├── stt/
│   │   ├── __init__.py
│   │   └── base_stt.py          # STT implementation template
│   ├── llm/
│   │   ├── __init__.py
│   │   └── agent.py             # LLM agent with complete RAG system
│   ├── tts/
│   │   ├── __init__.py
│   │   └── base_tts.py          # TTS implementation template  
│   ├── api/
│   │   ├── __init__.py
│   │   └── server.py            # FastAPI server
│   ├── utils/
│   │   ├── __init__.py
│   │   └── kb_test.py           # Knowledge base testing utility
│   ├── __init__.py
│   └── pipeline.py              # Main orchestrator
├── docs/
│   ├── ASSIGNMENT_GUIDE.md      # Implementation instructions
│   └── RAG_IMPLEMENTATION_GUIDE.md  # Detailed RAG guide
├── tests/                       # Test files
├── data/                        # ChromaDB storage (created automatically)
├── requirements.txt             # Dependencies
├── .env.example                # Environment template
├── streamlit_app.py            # Testing UI
└── README.md                   # This file
```

## Quick Start

### 1. Installation

```bash
# Navigate to the project
cd audio_support_agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install core dependencies
pip install fastapi uvicorn streamlit requests numpy
pip install langchain chromadb sentence-transformers

# Optional: For audio recording in UI
pip install sounddevice
```

### 2. Environment Setup

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys (based on chosen services)
# Example:
STT_API_KEY=your_deepgram_key_here
GROQ_API_KEY=your_groq_api_key_here
TTS_API_KEY=your_elevenlabs_key_here
```

### 3. Service Options

**You can choose from multiple options for each component:**

**STT Options:**
- **API Services**: Deepgram (free $200), AssemblyAI, Azure Speech, Google Cloud Speech
- **Local Models**: OpenAI Whisper, Wav2Vec2, Vosk, Coqui STT, SpeechRecognition

**LLM Options:**
- **API Services**: Groq (default in this project), OpenAI, Anthropic Claude
- **Local Models**: Ollama, Hugging Face Transformers

**TTS Options:**
- **API Services**: ElevenLabs (free 10k chars), OpenAI TTS, Azure Speech, Google Cloud TTS
- **Local Models**: Coqui TTS, Parler TTS, Bark, Edge TTS (free), Festival, eSpeak

**Quick Start Combinations:**
- **Completely Free**: Whisper + Local LLM + Edge TTS
- **Minimal Cost**: Whisper + OpenAI API + Edge TTS  
- **Full Cloud**: Deepgram + OpenAI + ElevenLabs

## Testing Your Implementation

### Two-Terminal Workflow

**Terminal 1 - Start API Server:**
```bash
python -m src.api.server
```

**Terminal 2 - Launch Testing UI:**
```bash
streamlit run streamlit_app.py
```

### Testing Interface Features

The Streamlit UI provides 4 main tabs:

1. **Text Chat**: Test LLM agent responses without audio processing
2. **Audio Chat**: Complete pipeline testing with audio recording/upload
3. **Health Monitor**: Real-time component status and debugging
4. **Documentation**: Built-in help and troubleshooting

### API Endpoints

Once the server is running, you can access:

- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Root Info**: http://localhost:8000/

**Manual API Testing:**
```bash
# Health check
curl http://localhost:8000/health

# Text chat
curl -X POST http://localhost:8000/chat/text \
  -H "Content-Type: application/json" \
  -d '{"text": "What is your return policy?"}'

# Audio upload (full pipeline)
curl -X POST http://localhost:8000/chat/audio \
  -F "audio=@test_audio.wav" --output response.mp3
```

## Development Utilities

### Knowledge Base Testing
```bash
# Test RAG implementation
python src/utils/kb_test.py
```
This utility shows:
- Knowledge base structure (16 customer support documents)
- Sample test queries
- Your RAG implementation results

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific component tests
pytest tests/test_stt.py -v
pytest tests/test_tts.py -v
pytest tests/test_llm.py -v
```

## What's Provided vs What You Implement

### Already Complete (Provided)

**RAG Knowledge Base:**
- 16 comprehensive customer support documents
- Automatic ChromaDB setup and document ingestion
- Embedding generation with sentence-transformers
- Persistent storage (survives server restarts)

**Infrastructure:**
- Complete FastAPI server with all endpoints
- Streamlit testing interface with 4 tabs
- Abstract base classes for all components
- Configuration management and environment setup
- Test utilities and documentation

**Pipeline Framework:**
- Complete orchestration logic structure
- Error handling and logging framework
- Health monitoring system

### Your Implementation Tasks

**Core Components (Required):**
1. **RAG Search Logic**: Complete `_rag_search()` method in `CustomerSupportAgent`
2. **STT Implementation**: Complete `STTService` class methods (initialize, transcribe, cleanup)
3. **TTS Implementation**: Complete `TTSService` class methods (initialize, synthesize, cleanup)
4. **Pipeline Integration**: Complete pipeline initialization and audio processing flow
5. **Server Configuration**: Configure startup with your chosen services

**See `docs/ASSIGNMENT_GUIDE.md` for detailed implementation instructions.**

## Troubleshooting

### Common Setup Issues

**Import Errors:**
- Ensure you're running from the project root directory
- Verify all dependencies are installed: `pip install -r requirements.txt`

**Server Won't Start:**
- Check if port 8000 is already in use
- Verify environment variables are set correctly

**Audio Issues:**
- For recording: Install `pip install sounddevice`
- For playback: Ensure audio files are in supported formats (WAV recommended)

**API Quota Issues:**
- Check your API key validity and usage limits
- Consider using local models for development/testing

### Debug Resources

1. **Health Endpoint**: Shows which components are working
2. **Server Logs**: Detailed error messages and processing info
3. **Test Utilities**: `kb_test.py` for RAG debugging
4. **Streamlit UI**: Real-time component monitoring

## Learning Objectives

By completing this assignment, you will learn:

1. **Async Python Programming**: Working with async/await patterns
2. **API Integration**: Multiple third-party service integration
3. **Modular Design**: Creating reusable, testable components
4. **RAG Implementation**: Building retrieval-augmented generation systems
5. **FastAPI Development**: Creating REST APIs for ML applications
6. **Error Handling**: Robust error handling in production systems
7. **Audio Processing**: Working with audio data in Python

## Documentation

- **Implementation Guide**: `docs/ASSIGNMENT_GUIDE.md` - Detailed coding instructions
- **RAG Guide**: `docs/RAG_IMPLEMENTATION_GUIDE.md` - Specific RAG implementation help
- **API Documentation**: Available at `/docs` when server is running

## Support

If you encounter issues:

1. Check the health endpoint to see component status
2. Review server logs for detailed error messages  
3. Use the test utilities to debug individual components
4. Verify API keys and service configurations
5. Check the troubleshooting section in documentation

## Getting Help

- Use the provided test utilities (`kb_test.py`, Streamlit UI)
- Check the comprehensive documentation in the `docs/` folder
- Review the sample implementations and code comments
- Test components individually before integration

Good luck with your implementation!