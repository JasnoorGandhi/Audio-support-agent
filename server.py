"""
FastAPI Server for Audio Customer Support Agent

Endpoints:
  GET  /           — API info
  GET  /health     — Component health status
  POST /chat/text  — Text query -> text + audio response
  POST /chat/audio — Audio file -> audio response (full pipeline)
  GET  /chat/audio/{text} — TTS test: text -> audio file
  POST /debug/stt  — STT test: audio file -> transcript
"""

import os
import time
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.pipeline import AudioSupportPipeline, create_pipeline, PipelineConfig
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
# override=True so values in project .env replace empty/stale GROQ_* from OS env
load_dotenv(_PROJECT_ROOT / ".env", override=True)
load_dotenv(override=False)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Pydantic models
# ------------------------------------------------------------------

class TextRequest(BaseModel):
    text: str
    parameters: Optional[Dict[str, Any]] = {}


class HealthResponse(BaseModel):
    status: str
    components: Dict[str, bool]
    message: str


class TextResponse(BaseModel):
    response_text: str
    audio_available: bool
    processing_time_ms: int


# ------------------------------------------------------------------
# App setup
# ------------------------------------------------------------------

app = FastAPI(
    title="Audio Customer Support Agent API",
    description="REST API for the STT → LLM (RAG) → TTS pipeline",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline: Optional[AudioSupportPipeline] = None
# Set when create_pipeline() fails so /health and 503 responses explain why.
pipeline_init_error: Optional[str] = None


def _pipeline_503_detail() -> str:
    if pipeline_init_error:
        return f"Pipeline not initialized: {pipeline_init_error}"
    return (
        "Pipeline not initialized. Startup failed or is still in progress; "
        "check server logs. Ensure GROQ_API_KEY is set in .env for the LLM."
    )


# ------------------------------------------------------------------
# Startup / Shutdown
# ------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    """Initialize the pipeline on server start."""
    global pipeline, pipeline_init_error

    logger.info("Starting Audio Customer Support Agent API...")
    load_dotenv(_PROJECT_ROOT / ".env", override=True)

    # -----------------------------------------------------------------
    # Configuration — edit these or set environment variables
    # -----------------------------------------------------------------
    stt_config = {
        # Options: 'whisper' (local, free), 'deepgram', 'assemblyai'
        "provider": os.getenv("STT_PROVIDER", "whisper"),
        # Whisper model size: tiny | base | small | medium | large
        "model": os.getenv("STT_MODEL", "base"),
        # API key for cloud providers (not needed for whisper)
        "api_key": os.getenv("STT_API_KEY"),
    }
    def _groq_key() -> Optional[str]:
        for name in ("GROQ_API_KEY", "GROQ_KEY"):
            v = os.getenv(name)
            if v and str(v).strip():
                return str(v).strip()
        return None

    llm_config = {
        "api_key": _groq_key(),
        "model": os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
        "temperature": float(os.getenv("LLM_TEMPERATURE", "0.7")),
        "chroma_db_path": os.getenv("CHROMA_DB_PATH", "./data/chroma_db"),
    }

    tts_config = {
        # Options: 'edge_tts' (local, free), 'elevenlabs', 'openai'
        "provider": os.getenv("TTS_PROVIDER", "edge_tts"),
        # Edge TTS voice (see: edge-tts --list-voices)
        "voice": os.getenv("TTS_VOICE", "en-US-AriaNeural"),
        # API key and voice ID for cloud providers
        "api_key": os.getenv("TTS_API_KEY") or os.getenv("ELEVENLABS_API_KEY"),
        "voice_id": os.getenv("TTS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM"),
        "model": os.getenv("TTS_MODEL", "eleven_turbo_v2_5"),
    }

    try:
        pipeline = await create_pipeline(stt_config, llm_config, tts_config)
        pipeline_init_error = None
        health = await pipeline.health_check()
        logger.info(f"Pipeline ready: {health}")
    except Exception as e:
        pipeline_init_error = str(e)
        pipeline = None
        logger.error(f"Failed to initialize pipeline: {e}")
        logger.warning("Server running in degraded mode — fix config and restart.")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on server shutdown."""
    global pipeline, pipeline_init_error
    if pipeline:
        logger.info("Shutting down pipeline...")
        await pipeline.cleanup()
        pipeline = None
        pipeline_init_error = None


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@app.get("/", response_model=Dict[str, str])
async def root():
    return {
        "message": "Audio Customer Support Agent API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Return health status of all pipeline components."""
    global pipeline

    if not pipeline:
        return HealthResponse(
            status="unhealthy",
            components={
                "pipeline_initialized": False,
                "stt_ready": False,
                "llm_ready": False,
                "tts_ready": False,
            },
            message=pipeline_init_error or "Pipeline not initialized",
        )

    try:
        components = await pipeline.health_check()
        all_healthy = all(components.values())
        return HealthResponse(
            status="healthy" if all_healthy else "degraded",
            components=components,
            message="All components ready" if all_healthy else "Some components not ready",
        )
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return HealthResponse(
            status="error",
            components={},
            message=f"Health check failed: {e}",
        )


@app.post("/chat/text", response_model=TextResponse)
async def chat_text(request: TextRequest):
    """
    Process a text query through LLM (RAG) + TTS.
    Returns the text response and indicates audio availability.
    """
    global pipeline

    if not pipeline or not pipeline.is_initialized:
        raise HTTPException(status_code=503, detail=_pipeline_503_detail())

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        start = time.time()
        response_text, response_audio = await pipeline.process_text(
            request.text, **request.parameters
        )
        elapsed_ms = int((time.time() - start) * 1000)

        return TextResponse(
            response_text=response_text,
            audio_available=bool(response_audio),
            processing_time_ms=elapsed_ms,
        )
    except Exception as e:
        logger.error(f"Text chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/audio")
async def chat_audio(audio: UploadFile = File(...)):
    """
    Full pipeline: audio upload -> STT -> LLM (RAG) -> TTS -> audio response.
    Returns an MP3 audio file.
    """
    global pipeline

    if not pipeline or not pipeline.is_initialized:
        raise HTTPException(status_code=503, detail=_pipeline_503_detail())

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")

    try:
        response_audio = await pipeline.process_audio(audio_bytes)
        return Response(
            content=response_audio,
            media_type="audio/mpeg",
            headers={"Content-Disposition": "attachment; filename=response.mp3"},
        )
    except Exception as e:
        logger.error(f"Audio chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat/audio/{text}")
async def text_to_audio(text: str):
    """
    TTS test: convert text to speech.
    curl "http://localhost:8000/chat/audio/Hello%20world" --output test.mp3
    """
    global pipeline

    if not pipeline or not pipeline.is_initialized:
        raise HTTPException(status_code=503, detail=_pipeline_503_detail())

    if not pipeline.tts or not pipeline.tts.is_ready():
        raise HTTPException(status_code=503, detail="TTS not available")

    try:
        audio_bytes = await pipeline.tts.synthesize(text)
        return Response(
            content=audio_bytes,
            media_type="audio/mpeg",
            headers={"Content-Disposition": "attachment; filename=tts_output.mp3"},
        )
    except Exception as e:
        logger.error(f"TTS error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/debug/stt")
async def debug_stt(audio: UploadFile = File(...)):
    """
    STT test: transcribe an audio file.
    curl -X POST http://localhost:8000/debug/stt -F "audio=@test.wav"
    """
    global pipeline

    if not pipeline or not pipeline.is_initialized:
        raise HTTPException(status_code=503, detail=_pipeline_503_detail())

    if not pipeline.stt or not pipeline.stt.is_ready():
        raise HTTPException(status_code=503, detail="STT not available")

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")

    try:
        transcription = await pipeline.stt.transcribe(audio_bytes)
        return {"transcription": transcription, "char_count": len(transcription)}
    except Exception as e:
        logger.error(f"STT debug error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
