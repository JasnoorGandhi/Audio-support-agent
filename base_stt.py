"""
Speech-to-Text (STT) Service

Uses OpenAI Whisper (local, no API key needed) by default.
Alternatively supports Deepgram, AssemblyAI, or Azure via config.
"""

import io
import logging
import tempfile
import os
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class BaseSTT(ABC):
    """Abstract base class for STT implementations."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.is_initialized = False

    @abstractmethod
    async def initialize(self) -> None:
        pass

    @abstractmethod
    async def transcribe(self, audio_bytes: bytes, **kwargs) -> str:
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        pass

    def is_ready(self) -> bool:
        return self.is_initialized


class STTService(BaseSTT):
    """
    STT implementation using OpenAI Whisper (local, no API key needed).

    Config keys:
      provider  - 'whisper' (default), 'deepgram', or 'assemblyai'
      model     - Whisper model size: 'tiny', 'base', 'small', 'medium', 'large'
                  (default: 'base' — good balance of speed & accuracy)
      api_key   - Required only for deepgram / assemblyai providers
      language  - Optional language hint, e.g. 'en' (Whisper auto-detects if omitted)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.client = None
        self.provider = (config or {}).get("provider", "whisper")

    # ------------------------------------------------------------------
    # Initialize
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Load the Whisper model (or connect to API-based provider)."""
        if self.provider == "whisper":
            await self._init_whisper()
        elif self.provider == "deepgram":
            await self._init_deepgram()
        elif self.provider == "assemblyai":
            await self._init_assemblyai()
        else:
            raise ValueError(f"Unknown STT provider: {self.provider}")

        self.is_initialized = True
        logger.info(f"STT service initialized (provider: {self.provider})")

    async def _init_whisper(self):
        """Load the local Whisper model."""
        try:
            import whisper
            model_name = self.config.get("model", "base")
            logger.info(f"Loading Whisper model '{model_name}'... (this may take a moment)")
            self.client = whisper.load_model(model_name)
            logger.info("Whisper model loaded.")
        except ImportError:
            raise ImportError("openai-whisper not installed. Run: pip install openai-whisper")

    async def _init_deepgram(self):
        """Initialize Deepgram client."""
        try:
            from deepgram import DeepgramClient
            api_key = self.config.get("api_key") or os.getenv("DEEPGRAM_API_KEY")
            if not api_key:
                raise ValueError("DEEPGRAM_API_KEY not set.")
            self.client = DeepgramClient(api_key)
        except ImportError:
            raise ImportError("deepgram-sdk not installed. Run: pip install deepgram-sdk")

    async def _init_assemblyai(self):
        """Initialize AssemblyAI."""
        try:
            import assemblyai as aai
            api_key = self.config.get("api_key") or os.getenv("ASSEMBLYAI_API_KEY")
            if not api_key:
                raise ValueError("ASSEMBLYAI_API_KEY not set.")
            aai.settings.api_key = api_key
            self.client = aai
        except ImportError:
            raise ImportError("assemblyai not installed. Run: pip install assemblyai")

    # ------------------------------------------------------------------
    # Transcribe
    # ------------------------------------------------------------------

    async def transcribe(self, audio_bytes: bytes, **kwargs) -> str:
        """Transcribe audio bytes to text."""
        if not self.is_ready():
            raise RuntimeError("STT service not initialized. Call initialize() first.")

        if not audio_bytes:
            raise ValueError("Empty audio data provided.")

        if self.provider == "whisper":
            return await self._transcribe_whisper(audio_bytes, **kwargs)
        elif self.provider == "deepgram":
            return await self._transcribe_deepgram(audio_bytes, **kwargs)
        elif self.provider == "assemblyai":
            return await self._transcribe_assemblyai(audio_bytes, **kwargs)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    async def _transcribe_whisper(self, audio_bytes: bytes, **kwargs) -> str:
        """Transcribe using local Whisper model."""
        try:
            # Whisper needs a file path; write to a temp file
            suffix = kwargs.get("audio_format", ".wav")
            if not suffix.startswith("."):
                suffix = f".{suffix}"

            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            try:
                language = self.config.get("language", None)
                result = self.client.transcribe(
                    tmp_path,
                    language=language,
                    fp16=False,  # Safer default (works on CPU)
                )
                transcript = result.get("text", "").strip()
                logger.info(f"Whisper transcript: '{transcript[:80]}...'")
                return transcript
            finally:
                os.unlink(tmp_path)

        except Exception as e:
            logger.error(f"Whisper transcription error: {e}")
            raise

    async def _transcribe_deepgram(self, audio_bytes: bytes, **kwargs) -> str:
        """Transcribe using Deepgram API."""
        try:
            response = await self.client.listen.prerecorded.v("1").transcribe_file(
                {"buffer": audio_bytes},
                {"model": self.config.get("model", "nova-2"), "smart_format": True},
            )
            return (
                response["results"]["channels"][0]["alternatives"][0]["transcript"]
            )
        except Exception as e:
            logger.error(f"Deepgram transcription error: {e}")
            raise

    async def _transcribe_assemblyai(self, audio_bytes: bytes, **kwargs) -> str:
        """Transcribe using AssemblyAI."""
        try:
            transcriber = self.client.Transcriber()
            transcript = transcriber.transcribe(audio_bytes)
            return transcript.text or ""
        except Exception as e:
            logger.error(f"AssemblyAI transcription error: {e}")
            raise

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def cleanup(self) -> None:
        """Release model/connection resources."""
        self.client = None
        self.is_initialized = False
        logger.info("STT service cleaned up.")
