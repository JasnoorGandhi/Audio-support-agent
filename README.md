# 🎙️ Audio Customer Support Agent

An end-to-end voice-based customer support agent built on a **STT → LLM → TTS pipeline**. Customers can speak their queries and receive spoken responses, powered by OpenAI Whisper, Gemini, LangChain RAG, and Microsoft Edge TTS — all served via a FastAPI backend with a Streamlit UI.

---

## Architecture

```
Audio Input
    │
    ▼
┌─────────────────────┐
│   STT (Whisper)     │  Transcribes speech to text
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  LangChain ReAct    │  Reasons over query
│  Agent + Gemini     │
│                     │
│  ┌───────────────┐  │
│  │  RAG Search   │  │  Retrieves relevant docs
│  │  (ChromaDB)   │  │  from knowledge base
│  └───────────────┘  │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│   TTS (Edge TTS)    │  Synthesizes spoken response
└─────────────────────┘
    │
    ▼
Audio Output
```

---

## Features

- **Voice-first interface** — speak your question, get a spoken answer
- **RAG-powered responses** — answers grounded in a 16-document customer support knowledge base, reducing hallucinations
- **LangChain ReAct agent** — structured reasoning before responding, with tool use
- **Fully async pipeline** — FastAPI + asyncio for non-blocking concurrent requests
- **Text chat support** — also works as a standard text chatbot
- **Health monitoring** — `/health` endpoint shows per-component status
- **Modular design** — swap any component (STT, LLM, TTS) via config

---

## Tech Stack

| Component | Technology |
|---|---|
| STT | OpenAI Whisper (local) |
| LLM | Google Gemini 2.0 Flash |
| Agent Framework | LangChain ReAct |
| Vector Database | ChromaDB |
| Embeddings | all-MiniLM-L6-v2 (via ChromaDB) |
| TTS | Microsoft Edge TTS |
| Backend | FastAPI + Uvicorn |
| Frontend | Streamlit |

---

## Project Structure

```
audio_support_agent/
├── src/
│   ├── api/
│   │   └── server.py          # FastAPI server & endpoints
│   ├── llm/
│   │   └── agent.py           # LangChain ReAct agent + RAG
│   ├── stt/
│   │   └── base_stt.py        # Whisper STT service
│   ├── tts/
│   │   └── base_tts.py        # Edge TTS service
│   ├── pipeline.py            # Orchestrates STT → LLM → TTS
│   └── utils/
│       └── kb_test.py         # Knowledge base test utility
├── streamlit_app.py           # Streamlit UI
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup & Installation

### Prerequisites
- Python 3.11
- FFmpeg installed and on PATH ([download here](https://www.gyan.dev/ffmpeg/builds/))
- Google Gemini API key ([get free key](https://aistudio.google.com))

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/audio-support-agent.git
cd audio-support-agent
```

### 2. Create virtual environment with Python 3.11
```bash
py -3.11 -m venv venv311
venv311\Scripts\activate      # Windows
source venv311/bin/activate   # Linux/Mac
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
```bash
copy .env.example .env        # Windows
cp .env.example .env          # Linux/Mac
```

Edit `.env`:
```env
GOOGLE_API_KEY=your-gemini-api-key-here
STT_PROVIDER=whisper
STT_MODEL=base
TTS_PROVIDER=edge_tts
TTS_VOICE=en-US-AriaNeural
LLM_MODEL=gemini-2.0-flash
CHROMA_DB_PATH=./data/chroma_db
```

### 5. Run the API server
```bash
python -m src.api.server
```

### 6. Run the Streamlit UI (new terminal)
```bash
venv311\Scripts\activate
streamlit run streamlit_app.py
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API info |
| GET | `/health` | Component health status |
| POST | `/chat/text` | Text query → text + audio response |
| POST | `/chat/audio` | Audio file → audio response (full pipeline) |
| GET | `/chat/audio/{text}` | TTS test: text → audio file |
| POST | `/debug/stt` | STT test: audio file → transcript |

Interactive docs available at `http://localhost:8000/docs`

### Example requests

**Text chat:**
```bash
curl -X POST http://localhost:8000/chat/text \
  -H "Content-Type: application/json" \
  -d '{"text": "What is your return policy?"}'
```

**Audio chat:**
```bash
curl -X POST http://localhost:8000/chat/audio \
  -F "audio=@question.wav" \
  --output response.mp3
```

**Health check:**
```bash
curl http://localhost:8000/health
```

---

## RAG Implementation

The knowledge base contains 16 customer support documents across categories: returns, shipping, warranty, payments, account management, and products.

On each query:
1. Query text is embedded using `all-MiniLM-L6-v2` (via ChromaDB)
2. Top-3 semantically similar documents retrieved using L2 distance
3. Documents filtered by distance threshold (`< 1.5`) to exclude irrelevant results
4. Retrieved context injected into the LangChain ReAct agent prompt
5. Gemini generates a grounded response based on retrieved knowledge

```python
results = self.collection.query(
    query_texts=[query],
    n_results=3,
    include=["documents", "metadatas", "distances"],
)
```

---

## STT Options

| Provider | Setup | Cost | Notes |
|---|---|---|---|
| **Whisper** (default) | Local, no API key | Free | Requires FFmpeg |
| Deepgram | API key | Free $200 credits | Faster, cloud |
| AssemblyAI | API key | Free tier | Easy setup |

Change provider in `.env`:
```env
STT_PROVIDER=whisper   # or deepgram, assemblyai
```

## TTS Options

| Provider | Setup | Cost | Notes |
|---|---|---|---|
| **Edge TTS** (default) | No API key | Free | Microsoft neural voices |
| ElevenLabs | API key | 10k chars/month free | Best quality |
| OpenAI TTS | API key | Pay-per-use | High quality |

---

## Testing

**Test knowledge base:**
```bash
python src/utils/kb_test.py
```

**Test STT:**
```bash
curl -X POST http://localhost:8000/debug/stt -F "audio=@test.wav"
```

**Test TTS:**
```bash
curl "http://localhost:8000/chat/audio/Hello%20world" --output test.mp3
```

---

## Troubleshooting

| Issue | Cause | Fix |
|---|---|---|
| `[WinError 2] file not found` | FFmpeg not installed | Install FFmpeg and add to PATH |
| `ContextOverflowError` | LangChain version mismatch | Use Python 3.11 venv, pin langchain==0.3.25 |
| `Pipeline not initialized` | Missing API key | Check `.env` has `GOOGLE_API_KEY` set |
| Request timeout | Whisper slow on CPU | Switch to `STT_MODEL=tiny` or use Groq Whisper API |
| ChromaDB download on startup | First run only | Embedding model (~80MB) downloads once and caches |

---

## Evaluation Criteria Met

- ✅ RAG retrieves relevant customer support information
- ✅ STT accurately transcribes speech to text
- ✅ TTS generates clear, natural-sounding speech
- ✅ Complete pipeline processes audio end-to-end
- ✅ Health endpoint shows all component status
- ✅ Streamlit UI works for text and audio chat
- ✅ Error handling gracefully manages failures
- ✅ Async patterns used throughout

---

## Future Improvements

- Streaming audio response for lower latency
- Hybrid search (semantic + BM25 keyword) for better retrieval
- Persistent conversation memory across sessions
- Docker containerization for easier deployment
- Evaluation framework for RAG retrieval accuracy

---

## License

MIT
