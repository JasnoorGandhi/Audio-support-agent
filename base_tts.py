"""
Text-to-Speech (TTS) Service

Uses Edge TTS (free Microsoft voices, no API key) by default.
Alternatively supports ElevenLabs, OpenAI TTS, or Azure via config.
"""

import io
import logging
import asyncio
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class BaseTTS(ABC):
    """Abstract base class for TTS implementations."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.is_initialized = False

    @abstractmethod
    async def initialize(self) -> None:
        pass

    @abstractmethod
    async def synthesize(self, text: str, **kwargs) -> bytes:
        pass

    @abstractmethod
    async def synthesize_stream(self, text: str, **kwargs) -> io.BytesIO:
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        pass

    def is_ready(self) -> bool:
        return self.is_initialized


class TTSService(BaseTTS):
    """
    TTS implementation using Edge TTS (free, no API key needed) by default.

    Config keys:
      provider  - 'edge_tts' (default), 'elevenlabs', or 'openai'
      voice     - Edge TTS voice name (default: 'en-US-AriaNeural')
      voice_id  - ElevenLabs voice ID
      api_key   - Required for elevenlabs / openai providers
      model     - TTS model (ElevenLabs: 'eleven_turbo_v2_5'; OpenAI: 'tts-1')
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.client = None
        self.voice_id = None
        self.model = None
        self.provider = (config or {}).get("provider", "edge_tts")

    # ------------------------------------------------------------------
    # Initialize
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Set up the TTS provider."""
        if self.provider == "edge_tts":
            await self._init_edge_tts()
        elif self.provider == "elevenlabs":
            await self._init_elevenlabs()
        elif self.provider == "openai":
            await self._init_openai()
        else:
            raise ValueError(f"Unknown TTS provider: {self.provider}")

        self.is_initialized = True
        logger.info(f"TTS service initialized (provider: {self.provider})")

    async def _init_edge_tts(self):
        """Initialize Edge TTS (no client needed, just set voice)."""
        try:
            import edge_tts  # noqa: F401
            self.voice = self.config.get("voice", "en-US-AriaNeural")
            self.client = "edge_tts"
            logger.info(f"Edge TTS ready (voice: {self.voice})")
        except ImportError:
            raise ImportError("edge-tts not installed. Run: pip install edge-tts")

    async def _init_elevenlabs(self):
        """Initialize ElevenLabs client."""
        try:
            import os
            from elevenlabs import ElevenLabs
            api_key = self.config.get("api_key") or os.getenv("ELEVENLABS_API_KEY")
            if not api_key:
                raise ValueError("ELEVENLABS_API_KEY not set.")
            self.client = ElevenLabs(api_key=api_key)
            self.voice_id = self.config.get("voice_id", "21m00Tcm4TlvDq8ikWAM")
            self.model = self.config.get("model", "eleven_turbo_v2_5")
        except ImportError:
            raise ImportError("elevenlabs not installed. Run: pip install elevenlabs")

    async def _init_openai(self):
        """Initialize OpenAI TTS client."""
        try:
            import os
            from openai import OpenAI
            api_key = self.config.get("api_key") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set.")
            self.client = OpenAI(api_key=api_key)
            self.voice = self.config.get("voice", "alloy")
            self.model = self.config.get("model", "tts-1")
        except ImportError:
            raise ImportError("openai not installed. Run: pip install openai")

    # ------------------------------------------------------------------
    # Synthesize
    # ------------------------------------------------------------------

    async def synthesize(self, text: str, **kwargs) -> bytes:
        """Convert text to speech and return audio bytes (MP3)."""
        if not self.is_ready():
            raise RuntimeError("TTS service not initialized. Call initialize() first.")

        if not text or not text.strip():
            raise ValueError("Text cannot be empty.")

        if self.provider == "edge_tts":
            return await self._synthesize_edge_tts(text, **kwargs)
        elif self.provider == "elevenlabs":
            return await self._synthesize_elevenlabs(text, **kwargs)
        elif self.provider == "openai":
            return await self._synthesize_openai(text, **kwargs)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    async def _synthesize_edge_tts(self, text: str, **kwargs) -> bytes:
        """Synthesize speech using Edge TTS (Microsoft voices, free)."""
        try:
            import edge_tts
            voice = kwargs.get("voice", self.voice)
            communicate = edge_tts.Communicate(text, voice)
            audio_bytes = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_bytes += chunk["data"]

            if not audio_bytes:
                raise RuntimeError("Edge TTS returned empty audio.")

            logger.info(f"Edge TTS synthesized {len(audio_bytes)} bytes for: '{text[:60]}...'")
            return audio_bytes
        except Exception as e:
            logger.error(f"Edge TTS synthesis error: {e}")
            raise

    async def _synthesize_elevenlabs(self, text: str, **kwargs) -> bytes:
        """Synthesize speech using ElevenLabs API."""
        try:
            audio_stream = self.client.text_to_speech.stream(
                text=text,
                voice_id=kwargs.get("voice_id", self.voice_id),
                model=kwargs.get("model", self.model),
            )
            audio_bytes = b""
            for chunk in audio_stream:
                if isinstance(chunk, bytes):
                    audio_bytes += chunk
            logger.info(f"ElevenLabs synthesized {len(audio_bytes)} bytes.")
            return audio_bytes
        except Exception as e:
            logger.error(f"ElevenLabs synthesis error: {e}")
            raise

    async def _synthesize_openai(self, text: str, **kwargs) -> bytes:
        """Synthesize speech using OpenAI TTS API."""
        try:
            response = self.client.audio.speech.create(
                model=kwargs.get("model", self.model),
                voice=kwargs.get("voice", self.voice),
                input=text,
            )
            audio_bytes = response.content
            logger.info(f"OpenAI TTS synthesized {len(audio_bytes)} bytes.")
            return audio_bytes
        except Exception as e:
            logger.error(f"OpenAI TTS synthesis error: {e}")
            raise

    # ------------------------------------------------------------------
    # Streaming synthesis
    # ------------------------------------------------------------------

    async def synthesize_stream(self, text: str, **kwargs) -> io.BytesIO:
        """Return synthesis result as a seekable BytesIO buffer."""
        if not self.is_ready():
            raise RuntimeError("TTS service not initialized.")

        audio_data = await self.synthesize(text, **kwargs)
        buffer = io.BytesIO(audio_data)
        buffer.seek(0)
        return buffer

    # ------------------------------------------------------------------
    # Voice listing
    # ------------------------------------------------------------------

    async def get_available_voices(self) -> List[Dict[str, Any]]:
        """Return a list of available voices for the current provider."""
        if not self.is_ready():
            raise RuntimeError("TTS service not initialized.")

        if self.provider == "edge_tts":
            import edge_tts
            voices = await edge_tts.list_voices()
            return [{"name": v["ShortName"], "locale": v["Locale"]} for v in voices]
        elif self.provider == "elevenlabs":
            voices = self.client.voices.get_all()
            return [{"voice_id": v.voice_id, "name": v.name} for v in voices.voices]
        else:
            return []

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def cleanup(self) -> None:
        """Release resources."""
        self.client = None
        self.is_initialized = False
        logger.info("TTS service cleaned up.")
