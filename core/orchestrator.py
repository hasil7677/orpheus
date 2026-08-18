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

# A finished sentence ends in terminal punctuation, possibly behind a closing
# quote or bracket. Moonshine punctuates its output, so this is the whole
# signal the punctuation gate runs on.
_TURN_END = re.compile(r"[.!?][\"'’”)\]]*\s*$")
# ...except a trailing ellipsis, which is exactly how an ASR renders someone
# trailing off mid-thought — the one case where a terminal '.' means the
# opposite of "done".
_TRAILING_OFF = re.compile(r"(\.\.\.|…)[\"'’”)\]]*\s*$")


def looks_like_end_of_turn(transcript: str) -> bool:
    """Whether a partial transcript reads as a completed turn.

    Deliberately dumb and free: no model, no memory, no network. It is wrong in
    both directions — "New York" as an answer to "where do you live" has no
    period, and an ASR will happily punctuate a sentence the speaker was going
    to continue — which is why the caller keeps a hard ceiling on the wait.
    """
    text = transcript.strip()
    if not text or _TRAILING_OFF.search(text):
        return False
    return bool(_TURN_END.search(text))


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

        # Punctuation-gate state (endpoint_mode == "punctuation"). The check
        # itself runs in a worker thread so the mic keeps draining while STT
        # is busy; _next_check_ms is the silence_ms at which to ask again.
        self._endpoint_check: asyncio.Task | None = None
        self._next_check_ms = config.vad.silence_floor_ms
        # Most recent gate transcript, kept for the ceiling path. Valid only
        # until the next voiced chunk, which is exactly when it gets cleared.
        self._gate_transcript: tuple[str, float] | None = None
        # Serializes every call into Moonshine. A gate check that gets
        # cancelled when speech resumes leaves its thread running to
        # completion (asyncio.to_thread can't interrupt it), so without this
        # the next transcribe could enter the same ONNX session concurrently.
        self._stt_lock = threading.Lock()

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
                self._reset_endpoint_gate()
                print("Listening...")

        elif self.state == PipelineState.LISTENING:
            self.utterance_buffer.append(chunk)
            if is_speech:
                # Rolling reset: one voiced chunk puts the whole endpointing
                # decision back to zero, so breaths and filler never
                # accumulate toward a cut. Both modes depend on this.
                self.silence_ms = 0
                self._reset_endpoint_gate()
            else:
                self.silence_ms += self.config.audio.chunk_ms
                if self.config.vad.endpoint_mode == "fixed":
                    if self.silence_ms >= self.config.vad.silence_timeout_ms:
                        await self._finalize_utterance()
                else:
                    await self._advance_punctuation_gate()

        elif self.state == PipelineState.SPEAKING:
            if is_speech:
                self._handle_barge_in()
                self.state = PipelineState.LISTENING
                self.utterance_buffer = [chunk]
                self.silence_ms = 0
                self._reset_endpoint_gate()

    def _reset_endpoint_gate(self) -> None:
        """Abandon any in-flight turn-complete check and re-arm the floor."""
        if self._endpoint_check is not None and not self._endpoint_check.done():
            self._endpoint_check.cancel()
        self._endpoint_check = None
        self._next_check_ms = self.config.vad.silence_floor_ms
        self._gate_transcript = None

    async def _advance_punctuation_gate(self) -> None:
        """One chunk's worth of progress on the turn-complete decision.

        Called on every silent chunk while LISTENING. Three things can happen:
        the ceiling forces a cut, a finished check answers the question, or a
        new check gets launched. Everything else is waiting.
        """
        vad = self.config.vad

        if self.silence_ms >= vad.silence_ceiling_ms:
            # The gate said "not done" (or failed) all the way to the cap.
            # Cut anyway — a wrong cut costs a re-ask, a hang costs the call.
            # Reuse the last gate transcript if there is one: it covers the
            # same speech (only silence has been added since), and paying a
            # fresh STT pass here would put the full ASR cost on top of the
            # longest wait the endpointer can produce — the worst possible
            # place for it.
            cached = self._gate_transcript
            print(f"  [ENDPOINT ceiling {self.silence_ms}ms] cutting anyway"
                  f"{' (reusing gate transcript)' if cached else ''}")
            self._reset_endpoint_gate()
            if cached is not None:
                await self._finalize_utterance(transcript=cached[0], stt_ms=cached[1])
            else:
                await self._finalize_utterance()
            return

        check = self._endpoint_check
        if check is not None:
            if not check.done():
                return
            self._endpoint_check = None
            try:
                complete, transcript, stt_ms = check.result()
            except asyncio.CancelledError:
                return
            except Exception as exc:
                # Fall through to the ceiling rather than dropping the turn.
                print(f"  [ENDPOINT error] gate check failed: {exc!r}")
                self._next_check_ms = self.silence_ms + vad.endpoint_recheck_ms
                return

            if complete:
                # Only silence has been appended since the snapshot the gate
                # transcribed (any voiced chunk would have cancelled it), so
                # that transcript still describes the whole utterance and the
                # main path can skip re-transcribing it.
                print(f"  [ENDPOINT {self.silence_ms}ms] turn complete: "
                      f'"{transcript}"')
                await self._finalize_utterance(transcript=transcript, stt_ms=stt_ms)
                return

            print(f"  [ENDPOINT {self.silence_ms}ms] still going, waiting")
            if transcript.strip():
                self._gate_transcript = (transcript, stt_ms)
            self._next_check_ms = self.silence_ms + vad.endpoint_recheck_ms
            return

        buffered_ms = len(self.utterance_buffer) * self.config.audio.chunk_ms
        if (self.silence_ms >= self._next_check_ms
                and buffered_ms >= vad.min_speech_ms):
            audio = np.concatenate(self.utterance_buffer)
            self._endpoint_check = asyncio.create_task(
                asyncio.to_thread(self._turn_is_complete, audio)
            )

    def _turn_is_complete(self, audio: np.ndarray) -> tuple[bool, str, float]:
        """Worker-thread half of the gate: is this buffer a finished turn?

        Runs the same denoise+STT the main path would, so the transcript is
        reusable verbatim when the answer is yes — which is what keeps the
        gate free on the common case instead of costing a second ASR pass.
        """
        t0 = time.monotonic()
        with self._stt_lock:
            denoised = self.denoiser.process(audio)
            transcript = self.stt.transcribe(denoised)
        return (looks_like_end_of_turn(transcript), transcript,
                1000 * (time.monotonic() - t0))

    def _locked_transcribe(self, audio: np.ndarray) -> str:
        with self._stt_lock:
            return self.stt.transcribe(audio)

    async def _finalize_utterance(self, transcript: str | None = None,
                                  stt_ms: float = 0.0) -> None:
        """Close the utterance and hand it off.

        `transcript` is the punctuation gate's already-computed transcription
        of this same audio; when present the STT pass is skipped rather than
        repeated, and `stt_ms` is what that transcription actually cost.
        """
        total_ms = len(self.utterance_buffer) * self.config.audio.chunk_ms
        buffer = self.utterance_buffer
        self.utterance_buffer = []

        if total_ms < self.config.vad.min_speech_ms:
            self.state = PipelineState.IDLE
            return

        self.state = PipelineState.THINKING
        audio_data = np.concatenate(buffer)
        self.current_task = asyncio.create_task(
            self._process_utterance(audio_data, transcript, stt_ms)
        )

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

    async def _process_utterance(self, audio: np.ndarray,
                                 transcript: str | None = None,
                                 stt_ms: float = 0.0) -> None:
        try:
            self.interrupt_speaker.clear()
            t0 = time.monotonic()

            if transcript is None:
                denoised = await asyncio.to_thread(self.denoiser.process, audio)
                transcript = await asyncio.to_thread(self._locked_transcribe, denoised)
                stt_ms = 1000 * (time.monotonic() - t0)
            t1 = time.monotonic()
            print(f"  [STT {stt_ms:.0f}ms] You said: \"{transcript}\"")

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
