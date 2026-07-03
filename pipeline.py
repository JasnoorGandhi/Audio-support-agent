"""
Audio Customer Support Agent Pipeline

Orchestrates the complete STT -> LLM (with RAG) -> TTS flow.
"""

import asyncio
import logging
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

from src.stt.base_stt import BaseSTT, STTService
from src.llm.agent import BaseAgent, CustomerSupportAgent
from src.tts.base_tts import BaseTTS, TTSService

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the audio support pipeline."""
    stt_config: Dict[str, Any]
    llm_config: Dict[str, Any]
    tts_config: Dict[str, Any]
    enable_logging: bool = True


class AudioSupportPipeline:
    """
    Main pipeline: STT -> LLM (RAG) -> TTS.

    Usage:
        pipeline = AudioSupportPipeline(config)
        await pipeline.initialize()
        response_audio = await pipeline.process_audio(audio_bytes)
        response_text, audio = await pipeline.process_text("What is your return policy?")
        await pipeline.cleanup()
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.stt: Optional[BaseSTT] = None
        self.llm_agent: Optional[BaseAgent] = None
        self.tts: Optional[BaseTTS] = None
        self.is_initialized = False

        logging.basicConfig(
            level=logging.INFO if config.enable_logging else logging.CRITICAL,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

    # ------------------------------------------------------------------
    # Initialize
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Initialize all pipeline components concurrently where possible."""
        try:
            logger.info("Initializing Audio Support Pipeline...")

            # STT and TTS can initialize in parallel; LLM sets up ChromaDB first
            self.stt = STTService(self.config.stt_config)
            self.tts = TTSService(self.config.tts_config)
            self.llm_agent = CustomerSupportAgent(self.config.llm_config)

            # Run STT and TTS init concurrently, then LLM
            await asyncio.gather(
                self.stt.initialize(),
                self.tts.initialize(),
            )
            await self.llm_agent.initialize()

            # Verify all components are ready
            if not self.stt.is_ready():
                raise RuntimeError("STT failed to initialize.")
            if not self.tts.is_ready():
                raise RuntimeError("TTS failed to initialize.")
            if not self.llm_agent.is_initialized:
                raise RuntimeError("LLM agent failed to initialize.")

            self.is_initialized = True
            logger.info("Pipeline initialized successfully — all components ready.")

        except Exception as e:
            logger.error(f"Pipeline initialization failed: {e}")
            await self.cleanup()
            raise

    # ------------------------------------------------------------------
    # Process audio (full STT -> LLM -> TTS)
    # ------------------------------------------------------------------

    async def process_audio(self, audio_bytes: bytes, **kwargs) -> bytes:
        """
        Full pipeline: audio in -> text -> LLM -> audio out.

        Args:
            audio_bytes: Raw audio data (WAV/MP3/etc.)
        Returns:
            bytes: Response audio (MP3)
        """
        if not self.is_initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        # Step 1: STT
        logger.info("Step 1/3 — Converting speech to text...")
        text_input = await self.stt.transcribe(audio_bytes, **kwargs)
        logger.info(f"Transcript: '{text_input}'")

        if not text_input.strip():
            logger.warning("Empty transcription — returning fallback audio.")
            fallback = "I'm sorry, I couldn't hear your question. Could you please repeat that?"
            return await self.tts.synthesize(fallback)

        # Step 2: LLM + RAG
        logger.info("Step 2/3 — Querying LLM agent with RAG...")
        agent_response = await self.llm_agent.process_query(text_input, **kwargs)
        logger.info(f"Agent response: '{agent_response[:100]}...'")

        # Step 3: TTS
        logger.info("Step 3/3 — Synthesizing response audio...")
        response_audio = await self.tts.synthesize(agent_response, **kwargs)
        logger.info(f"Audio generated: {len(response_audio)} bytes.")

        return response_audio

    # ------------------------------------------------------------------
    # Process text (LLM -> TTS, skips STT)
    # ------------------------------------------------------------------

    async def process_text(self, text_input: str, **kwargs) -> Tuple[str, bytes]:
        """
        Process a text query through LLM + TTS (no STT needed).

        Args:
            text_input: Customer's question as text
        Returns:
            Tuple[str, bytes]: (agent_response_text, response_audio)
        """
        if not self.is_initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        logger.info(f"Processing text query: '{text_input}'")

        agent_response = await self.llm_agent.process_query(text_input, **kwargs)
        logger.info(f"Agent response: '{agent_response[:100]}...'")

        response_audio = await self.tts.synthesize(agent_response, **kwargs)
        logger.info(f"Audio generated: {len(response_audio)} bytes.")

        return agent_response, response_audio

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def health_check(self) -> Dict[str, bool]:
        """Return health status of all pipeline components."""
        return {
            "pipeline_initialized": self.is_initialized,
            "stt_ready": self.stt.is_ready() if self.stt else False,
            "llm_ready": self.llm_agent.is_initialized if self.llm_agent else False,
            "tts_ready": self.tts.is_ready() if self.tts else False,
        }

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def cleanup(self) -> None:
        """Gracefully clean up all components."""
        logger.info("Cleaning up pipeline...")
        errors = []

        for component, name in [
            (self.stt, "STT"),
            (self.llm_agent, "LLM agent"),
            (self.tts, "TTS"),
        ]:
            if component:
                try:
                    await component.cleanup()
                except Exception as e:
                    errors.append(f"{name}: {e}")

        self.stt = None
        self.llm_agent = None
        self.tts = None
        self.is_initialized = False

        if errors:
            logger.warning(f"Cleanup warnings: {'; '.join(errors)}")
        else:
            logger.info("Pipeline cleanup complete.")


# ------------------------------------------------------------------
# Factory function
# ------------------------------------------------------------------

async def create_pipeline(
    stt_config: Dict[str, Any],
    llm_config: Dict[str, Any],
    tts_config: Dict[str, Any],
    enable_logging: bool = True,
) -> AudioSupportPipeline:
    """
    Create and initialize a complete pipeline.

    Example:
        pipeline = await create_pipeline(
            stt_config={"provider": "whisper", "model": "base"},
            llm_config={"api_key": "sk-...", "model": "gpt-3.5-turbo"},
            tts_config={"provider": "edge_tts", "voice": "en-US-AriaNeural"},
        )
    """
    config = PipelineConfig(
        stt_config=stt_config,
        llm_config=llm_config,
        tts_config=tts_config,
        enable_logging=enable_logging,
    )
    pipeline = AudioSupportPipeline(config)
    await pipeline.initialize()
    return pipeline


# ------------------------------------------------------------------
# CLI test entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    import os

    async def main():
        print("=== Audio Support Pipeline — Quick Test ===\n")

        pipeline = await create_pipeline(
            stt_config={"provider": "whisper", "model": "base"},
            llm_config={
                "api_key": os.getenv("OPENAI_API_KEY"),
                "model": "gpt-3.5-turbo",
                "temperature": 0.7,
            },
            tts_config={"provider": "edge_tts", "voice": "en-US-AriaNeural"},
        )

        print("\nHealth check:", await pipeline.health_check())

        # Test text queries
        test_queries = [
            "What is your return policy?",
            "How long does shipping take?",
            "Do you offer a warranty on electronics?",
        ]

        for query in test_queries:
            print(f"\nQ: {query}")
            response_text, response_audio = await pipeline.process_text(query)
            print(f"A: {response_text}")
            print(f"   [Audio: {len(response_audio)} bytes]")

        await pipeline.cleanup()
        print("\nDone!")

    asyncio.run(main())
