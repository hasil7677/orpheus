import asyncio
import re
import threading
import time
from enum import Enum, auto

import numpy as np

from audio.postprocessor import TTSPostProcessor, UtteranceDenoiser
from audio.preprocessor import AudioPreprocessor
from config import AppConfig
from models.llm import LLMProvider
from models.stt import SpeechToTextProvider
from models.tts import TextToSpeechProvider
from models.vad import VoiceActivityDetector

_ASTERISK_ACTIONS = re.compile(r"\*[^*]*\*")


class PipelineState(Enum):
    IDLE = auto()
    LISTENING = auto()
    THINKING = auto()
    SPEAKING = auto()


class PipelineOrchestrator:
    """
    Flow: Mic chunk -> preprocess -> VAD -> [buffer] -> STT -> Groq LLM -> Kokoro TTS -> speaker.

    Concurrency: the caller feeds raw mic chunks into `mic_queue` from a
    PyAudio read thread; `run_loop` drains it on the asyncio event loop.
    Heavy per-utterance work (STT/LLM/TTS) runs as a cancellable background
    task so barge-in can interrupt it cleanly.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self.state = PipelineState.IDLE
        self.current_task: asyncio.Task | None = None

        self.mic_queue: asyncio.Queue = asyncio.Queue(maxsize=50)
        self.speaker_queue: asyncio.Queue = asyncio.Queue()

        # Cross-thread stop signal for whatever's *actively* being written to
        # the sound device right now. Cancelling current_task and draining
        # speaker_queue only stops audio that hasn't started playing yet —
        # the speaker thread (cli.py) must check this between small writes to
        # actually cut off a chunk that's already mid-playback.
        self.interrupt_speaker = threading.Event()

        print("Loading models...")
        self.preprocessor = AudioPreprocessor(config.audio, config.processing)
        self.vad = VoiceActivityDetector(config.vad, config.audio.sample_rate)
        self.stt = SpeechToTextProvider(config.models)
        self.llm = LLMProvider(config.models)
        self.tts = TextToSpeechProvider(config.models, config.audio.tts_sample_rate)
        self.denoiser = UtteranceDenoiser(config.audio.sample_rate, config.processing)
        self.tts_postprocessor = TTSPostProcessor(config.audio.tts_sample_rate, config.processing)
        print("All models loaded.")

        self.utterance_buffer: list[np.ndarray] = []
        self.silence_ms = 0
        self.conversation_history: list[dict] = []

    async def push_chunk(self, raw_chunk: np.ndarray) -> None:
        """Called from the PyAudio read side. Drops oldest on overflow (TDD 7.2)."""
        if self.mic_queue.full():
            try:
                self.mic_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        await self.mic_queue.put(raw_chunk)

    async def run_loop(self) -> None:
        debug_counter = 0
        while True:
            raw_chunk = await self.mic_queue.get()
            clean_chunk = self.preprocessor.process_chunk(raw_chunk)
            prob = self.vad.speech_probability(clean_chunk)
            is_speech = prob > self.config.vad.threshold

            if self.state == PipelineState.SPEAKING:
                debug_counter += 1
                if debug_counter % 10 == 0:  # ~every 320ms, not every 32ms chunk
                    print(f"    (while speaking: vad_prob={prob:.2f}, threshold={self.config.vad.threshold})")
            else:
                debug_counter = 0

            await self._handle_state_transition(clean_chunk, is_speech)

    async def _handle_state_transition(self, chunk: np.ndarray, is_speech: bool) -> None:
        if self.state == PipelineState.IDLE:
            if is_speech:
                self.state = PipelineState.LISTENING
                self.utterance_buffer = [chunk]
                self.silence_ms = 0
                print("Listening...")

        elif self.state == PipelineState.LISTENING:
            self.utterance_buffer.append(chunk)
            if is_speech:
                self.silence_ms = 0
            else:
                self.silence_ms += self.config.audio.chunk_ms
                if self.silence_ms >= self.config.vad.silence_timeout_ms:
                    await self._finalize_utterance()

        elif self.state == PipelineState.SPEAKING:
            if is_speech:
                self._handle_barge_in()
                self.state = PipelineState.LISTENING
                self.utterance_buffer = [chunk]
                self.silence_ms = 0

    async def _finalize_utterance(self) -> None:
        total_ms = len(self.utterance_buffer) * self.config.audio.chunk_ms
        buffer = self.utterance_buffer
        self.utterance_buffer = []

        if total_ms < self.config.vad.min_speech_ms:
            self.state = PipelineState.IDLE
            return

        self.state = PipelineState.THINKING
        audio_data = np.concatenate(buffer)
        self.current_task = asyncio.create_task(self._process_utterance(audio_data))

    def _handle_barge_in(self) -> None:
        print("Barge-in detected — stopping playback.")
        self.interrupt_speaker.set()
        if self.current_task and not self.current_task.done():
            self.current_task.cancel()
        while not self.speaker_queue.empty():
            try:
                self.speaker_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def _process_utterance(self, audio: np.ndarray) -> None:
        try:
            self.interrupt_speaker.clear()
            t0 = time.monotonic()

            denoised = await asyncio.to_thread(self.denoiser.process, audio)
            transcript = await asyncio.to_thread(self.stt.transcribe, denoised)
            t1 = time.monotonic()
            print(f"  [STT {1000 * (t1 - t0):.0f}ms] You said: \"{transcript}\"")

            if not transcript.strip():
                self.state = PipelineState.IDLE
                return

            token_stream = self.llm.generate_stream(transcript, self.conversation_history)
            first_token_timed = {"done": False}

            async def _timed_tokens():
                async for token in token_stream:
                    if not first_token_timed["done"]:
                        first_token_timed["done"] = True
                        t2 = time.monotonic()
                        print(f"  [LLM TTFT {1000 * (t2 - t1):.0f}ms]")
                    yield token

            # Stay in THINKING (barge-in-inert) until audio is actually about
            # to play. Flipping to SPEAKING any earlier means the user's own
            # continued/natural speech during LLM+TTS latency gets treated as
            # an interruption of a response that hasn't started yet.
            first_audio_timed = {"done": False}
            t_llm_start = time.monotonic()

            async for audio_chunk in self.tts.synthesize_stream(_timed_tokens()):
                if not first_audio_timed["done"]:
                    first_audio_timed["done"] = True
                    t3 = time.monotonic()
                    print(f"  [TTS TTFA {1000 * (t3 - t_llm_start):.0f}ms]")
                    self.state = PipelineState.SPEAKING
                polished = self.tts_postprocessor.process(audio_chunk)
                await self.speaker_queue.put(polished)

            self.state = PipelineState.IDLE

        except asyncio.CancelledError:
            # Cancellation only ever comes from _handle_barge_in(), which has
            # already synchronously moved state to LISTENING (to start
            # capturing the new utterance) before this handler runs. Only
            # reset to IDLE if that hasn't happened — otherwise this clobbers
            # the just-entered LISTENING state and silently drops the start
            # of whatever the user said right after interrupting.
            if self.state in (PipelineState.THINKING, PipelineState.SPEAKING):
                self.state = PipelineState.IDLE
            raise
        except Exception as exc:
            # Without this, any failure here (Groq network error, STT
            # exception, etc.) kills the background task silently —
            # _handle_state_transition has no recovery branch for THINKING/
            # SPEAKING, so the pipeline would stop responding to speech
            # entirely until the process is restarted.
            print(f"  [error] utterance processing failed: {exc!r}")
            self.state = PipelineState.IDLE
