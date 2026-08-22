"""Barge-in regression test — no GPU/mic/speaker hardware or API key required.

Drives PipelineOrchestrator's real state machine (push_chunk/run_loop/
_handle_state_transition/_handle_barge_in) with fake VAD/STT/LLM/TTS
providers so it runs fast and deterministically. This verifies the
orchestrator-level contract: on barge-in, interrupt_speaker is set, the
in-flight task is cancelled, speaker_queue is drained, and the state machine
ends up back in LISTENING (not clobbered back to IDLE by the cancelled
task's own cleanup — see bug notes in core/orchestrator.py).

This does NOT verify that audio physically stops coming out of real
speakers — that depends on PortAudio/hardware behavior this environment
can't exercise. Run cli.py live and talk over it to confirm that part.

Run: .venv\\Scripts\\python.exe scripts\\smoke\\smoke_barge_in.py
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import AppConfig
from core.orchestrator import PipelineOrchestrator, PipelineState


class FakeVAD:
    """Ignores audio content; probability is scripted by the test."""

    def __init__(self, config, sample_rate):
        self.script: list[bool] = []

    def speech_probability(self, audio: np.ndarray) -> float:
        is_speech = self.script.pop(0) if self.script else False
        return 1.0 if is_speech else 0.0


class FakeSTT:
    def __init__(self, config):
        pass

    def transcribe(self, audio: np.ndarray) -> str:
        return "hello there"


class FakeDenoiser:
    def __init__(self, sample_rate, config):
        pass

    def process(self, audio: np.ndarray) -> np.ndarray:
        return audio


class FakeLLM:
    def __init__(self, config):
        pass

    async def generate_stream(self, user_text, history):
        yield "Hi"
        yield " there."


class FakeTTS:
    """Yields one audio chunk (enough to flip state to SPEAKING), then hangs
    on a controllable event so the test has a stable window to inject a
    barge-in before the utterance finishes on its own."""

    def __init__(self, config, sample_rate):
        self.hold = asyncio.Event()

    async def synthesize_stream(self, text_stream):
        async for _ in text_stream:
            pass
        yield np.zeros(160, dtype=np.float32)
        await self.hold.wait()
        yield np.zeros(160, dtype=np.float32)


async def _pump(n=5):
    for _ in range(n):
        await asyncio.sleep(0)


async def _wait_for_state(orch, state, timeout=2.0):
    elapsed = 0.0
    step = 0.01
    while orch.state != state:
        await asyncio.sleep(step)
        elapsed += step
        if elapsed > timeout:
            raise AssertionError(f"Timed out waiting for state={state}, still {orch.state}")


async def run_test():
    config = AppConfig()

    with patch("core.orchestrator.VoiceActivityDetector", FakeVAD), \
         patch("core.orchestrator.SpeechToTextProvider", FakeSTT), \
         patch("core.orchestrator.LLMProvider", FakeLLM), \
         patch("core.orchestrator.TextToSpeechProvider", FakeTTS), \
         patch("core.orchestrator.UtteranceDenoiser", FakeDenoiser):
        orch = PipelineOrchestrator(config)

    loop_task = asyncio.create_task(orch.run_loop())
    chunk_samples = config.audio.chunk_samples
    silence_chunk = np.zeros(chunk_samples, dtype=np.int16)

    try:
        # Speak for ~320ms (10 chunks) then go silent long enough to finalize.
        orch.vad.script = [True] * 10
        for _ in range(10):
            await orch.push_chunk(silence_chunk)
        await _pump()

        n_silence = (config.vad.silence_timeout_ms // config.audio.chunk_ms) + 2
        orch.vad.script = [False] * n_silence
        for _ in range(n_silence):
            await orch.push_chunk(silence_chunk)
        await _pump()

        await _wait_for_state(orch, PipelineState.SPEAKING)
        print("OK: reached SPEAKING state after simulated utterance.")

        assert orch.speaker_queue.qsize() >= 1, "expected first TTS chunk queued"
        task_before = orch.current_task
        assert task_before is not None and not task_before.done()

        # Barge in: push one more "speech" chunk while SPEAKING.
        orch.vad.script = [True]
        await orch.push_chunk(silence_chunk)
        await _pump(10)

        assert orch.interrupt_speaker.is_set(), "interrupt_speaker was not set on barge-in"
        assert orch.speaker_queue.empty(), "speaker_queue was not drained on barge-in"
        assert task_before.cancelled() or task_before.done(), "old utterance task was not cancelled"

        await _pump(10)  # let the cancelled task's cleanup fully run

        assert orch.state == PipelineState.LISTENING, (
            f"expected state LISTENING after barge-in (new utterance should still be "
            f"capturing), got {orch.state} — likely clobbered by the cancelled task's "
            f"cleanup resetting state to IDLE"
        )
        assert orch.utterance_buffer, "barge-in utterance buffer was dropped"
        print("OK: state is LISTENING (not clobbered back to IDLE) after barge-in.")

        print("\nBarge-in orchestrator-level test passed.")
    finally:
        loop_task.cancel()
        try:
            await loop_task
        except asyncio.CancelledError:
            pass


def main():
    asyncio.run(run_test())


if __name__ == "__main__":
    main()
