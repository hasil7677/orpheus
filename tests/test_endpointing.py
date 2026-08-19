"""Endpointing regression test — no GPU/mic/speaker hardware or API key required.

Drives the real PipelineOrchestrator state machine with fake VAD/STT/LLM/TTS/
TurnDetector providers, so all three endpoint modes can be exercised
deterministically and in about a second.

What it pins down:
  * "fixed" mode still cuts at exactly silence_timeout_ms (the v1.0 behaviour
    the committed latency table was measured with),
  * "punctuation" mode cuts early on a finished sentence, and reuses the
    gate's transcript instead of paying a second STT pass,
  * a mid-sentence pause longer than the old 650ms constant does NOT get cut,
  * a transcript that never gains punctuation still gets cut at the ceiling,
    so the turn can't hang,
  * "turn_detector" mode cuts early when the detector says so, and — unlike
    punctuation — always pays one fresh STT pass after the cut, since the
    detector judges raw audio and never produces a transcript to reuse,
  * a detector that never says "done" still gets cut at the ceiling.

What it does NOT show: whether Moonshine's punctuation, or smart-turn's
audio judgment, is any good on real speech at a real mic. That is what
benchmarks/eval_endpointing.py measures, and it needs recorded, hand-labeled
utterances to mean anything.

Run: .venv\\Scripts\\python.exe tests\\test_endpointing.py
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import AppConfig
from core.orchestrator import (
    PipelineOrchestrator,
    PipelineState,
    looks_like_end_of_turn,
)


class FakeVAD:
    """Ignores audio content; `speaking` is flipped by the test."""

    def __init__(self, config, sample_rate):
        self.speaking = False

    def speech_probability(self, audio: np.ndarray) -> float:
        return 1.0 if self.speaking else 0.0


class ScriptedSTT:
    """Returns whatever the test currently wants heard, and counts passes."""

    def __init__(self, config=None):
        self.transcript = ""
        self.calls = 0

    def transcribe(self, audio: np.ndarray) -> str:
        self.calls += 1
        return self.transcript


class FakeDenoiser:
    def __init__(self, sample_rate, config):
        pass

    def process(self, audio: np.ndarray) -> np.ndarray:
        return audio


class ScriptedTurnDetector:
    """Returns whatever the test currently wants decided, and counts calls.

    Constructed lazily by the orchestrator on first use (see
    core/orchestrator.py), so a test can't grab the instance before that —
    `default_complete` sets what the very first check decides.
    """

    default_complete = False

    def __init__(self, config=None):
        self.complete = ScriptedTurnDetector.default_complete
        self.calls = 0

    def is_turn_complete(self, audio: np.ndarray) -> bool:
        self.calls += 1
        return self.complete


class RecordingLLM:
    """Captures the text the pipeline decided to answer."""

    seen: list[str] = []

    def __init__(self, config):
        pass

    async def generate_stream(self, user_text, history):
        RecordingLLM.seen.append(user_text)
        yield "Sure."


class FakeTTS:
    def __init__(self, config, sample_rate):
        pass

    async def synthesize_stream(self, text_stream):
        async for _ in text_stream:
            pass
        yield np.zeros(160, dtype=np.float32)


CHUNK_MS = 32
SILENT_CHUNK = np.zeros(512, dtype=np.int16)  # 32ms @ 16kHz


def build(config) -> PipelineOrchestrator:
    return PipelineOrchestrator(config)


async def feed(orch, ms: int, speech: bool) -> None:
    """Push `ms` of audio, marked speech or silence, at loop speed."""
    orch.vad.speaking = speech
    for _ in range(ms // CHUNK_MS):
        await orch.push_chunk(SILENT_CHUNK)
        await asyncio.sleep(0.01)  # let run_loop + any gate thread land


async def feed_silence_until_cut(orch, max_ms: int) -> int | None:
    """Push silence until the utterance is finalized. Returns the silence at
    the moment of the cut, or None if `max_ms` went by without one."""
    orch.vad.speaking = False
    for _ in range(max_ms // CHUNK_MS):
        await orch.push_chunk(SILENT_CHUNK)
        await asyncio.sleep(0.01)
        if orch.state != PipelineState.LISTENING:
            return orch.silence_ms
    return None


async def _run(config, body):
    # Kept active for the whole run, not just construction: TurnDetector is
    # instantiated lazily on first actual use (core/orchestrator.py), which
    # can happen well after build() returns — a patch scoped to build() alone
    # unpatches before that first use and the real ~8MB model loads instead.
    with patch("core.orchestrator.VoiceActivityDetector", FakeVAD), \
         patch("core.orchestrator.SpeechToTextProvider", ScriptedSTT), \
         patch("core.orchestrator.LLMProvider", RecordingLLM), \
         patch("core.orchestrator.TextToSpeechProvider", FakeTTS), \
         patch("core.orchestrator.UtteranceDenoiser", FakeDenoiser), \
         patch("core.orchestrator.TurnDetector", ScriptedTurnDetector):
        orch = build(config)
        loop_task = asyncio.create_task(orch.run_loop())
        try:
            return await body(orch)
        finally:
            # Cancel the in-flight utterance too, not just the loop —
            # otherwise it outlives this orchestrator and its LLM/TTS calls
            # land in the middle of the next test.
            for task in (loop_task, orch.current_task):
                if task is not None:
                    task.cancel()
            for task in (loop_task, orch.current_task):
                if task is not None:
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass


async def test_fixed_mode_unchanged():
    config = AppConfig()
    assert config.vad.endpoint_mode == "fixed", "shipped default must stay 'fixed'"

    async def body(orch):
        orch.stt.transcript = "What is the capital of France?"
        await feed(orch, 320, speech=True)
        cut = await feed_silence_until_cut(orch, 1500)
        assert cut is not None, "fixed mode never finalized"
        # silence accrues one 32ms chunk at a time, so the cut lands on the
        # first multiple of chunk_ms at or past the timeout — 672, not 650
        expected = -(-config.vad.silence_timeout_ms // CHUNK_MS) * CHUNK_MS
        assert cut == expected, (
            f"fixed mode cut at {cut}ms, expected exactly {expected}ms"
        )
        return cut

    cut = await _run(config, body)
    print(f"OK: fixed mode cuts at {cut}ms, unchanged from v1.0.")


async def test_punctuation_cuts_early_and_reuses_transcript():
    config = AppConfig()
    config.vad.endpoint_mode = "punctuation"
    RecordingLLM.seen = []

    async def body(orch):
        orch.stt.transcript = "What is the capital of France?"
        await feed(orch, 320, speech=True)
        cut = await feed_silence_until_cut(orch, 1500)
        assert cut is not None, "punctuation mode never finalized"
        assert cut < config.vad.silence_timeout_ms, (
            f"cut at {cut}ms — no better than the {config.vad.silence_timeout_ms}ms "
            f"constant it is supposed to beat"
        )
        assert cut >= config.vad.silence_floor_ms, (
            f"cut at {cut}ms, below the {config.vad.silence_floor_ms}ms floor"
        )
        await asyncio.sleep(0.15)  # let the utterance task reach the LLM
        assert orch.stt.calls == 1, (
            f"STT ran {orch.stt.calls}x — the gate's transcript should have been "
            f"reused, not recomputed"
        )
        assert RecordingLLM.seen == ["What is the capital of France?"], (
            f"LLM saw {RecordingLLM.seen!r}"
        )
        return cut

    cut = await _run(config, body)
    print(f"OK: punctuation mode cuts at {cut}ms on a finished sentence, "
          f"one STT pass total.")


async def test_mid_sentence_pause_is_not_cut():
    """The whole point: a pause longer than 650ms that isn't a turn end."""
    config = AppConfig()
    config.vad.endpoint_mode = "punctuation"

    async def body(orch):
        orch.stt.transcript = "I was thinking"          # unfinished
        await feed(orch, 320, speech=True)
        cut = await feed_silence_until_cut(orch, 900)   # a long, real pause
        assert cut is None, f"cut mid-sentence at {cut}ms — this is the failure users hate"

        orch.stt.transcript = "I was thinking we should go."
        await feed(orch, 320, speech=True)              # speaker resumes
        assert orch.state == PipelineState.LISTENING
        cut = await feed_silence_until_cut(orch, 1500)
        assert cut is not None, "never finalized after the sentence was finished"
        return cut

    cut = await _run(config, body)
    print(f"OK: 900ms mid-sentence pause survived; cut at {cut}ms once the "
          f"sentence finished.")


async def test_ceiling_always_cuts():
    """Punctuation that never arrives must not hang the turn."""
    config = AppConfig()
    config.vad.endpoint_mode = "punctuation"
    config.vad.silence_ceiling_ms = 1200  # keep the test quick
    RecordingLLM.seen = []

    async def body(orch):
        orch.stt.transcript = "no punctuation ever"
        await feed(orch, 320, speech=True)
        cut = await feed_silence_until_cut(orch, 2500)
        assert cut is not None, "turn hung — ceiling did not fire"

        # The ceiling is the longest wait this mode can produce, so it is the
        # worst place to also pay a fresh ASR pass — the last gate transcript
        # covers the same speech and must be reused.
        calls_at_cut = orch.stt.calls
        await asyncio.sleep(0.2)
        assert orch.stt.calls == calls_at_cut, (
            f"STT ran again after the ceiling cut ({calls_at_cut} -> "
            f"{orch.stt.calls}); the gate transcript should have been reused"
        )
        assert RecordingLLM.seen == ["no punctuation ever"], (
            f"LLM saw {RecordingLLM.seen!r}"
        )

        assert cut >= config.vad.silence_ceiling_ms, (
            f"cut at {cut}ms, before the {config.vad.silence_ceiling_ms}ms ceiling"
        )
        assert cut < config.vad.silence_ceiling_ms + 200, (
            f"cut at {cut}ms, well past the ceiling"
        )
        return cut

    cut = await _run(config, body)
    print(f"OK: unpunctuated transcript still cut, at the {cut}ms ceiling.")


async def test_turn_detector_cuts_early_and_pays_stt_after():
    """turn_detector's key difference from punctuation: the decision comes
    from audio, not a transcript, so there is nothing to reuse — STT runs
    once, after the cut, same as fixed mode, just with a smarter cutoff."""
    config = AppConfig()
    config.vad.endpoint_mode = "turn_detector"
    ScriptedTurnDetector.default_complete = True
    RecordingLLM.seen = []

    async def body(orch):
        orch.stt.transcript = "What is the capital of France?"
        await feed(orch, 320, speech=True)
        cut = await feed_silence_until_cut(orch, 1500)
        assert cut is not None, "turn_detector mode never finalized"
        assert cut < config.vad.silence_timeout_ms, (
            f"cut at {cut}ms — no better than the {config.vad.silence_timeout_ms}ms "
            f"constant it is supposed to beat"
        )
        assert cut >= config.vad.silence_floor_ms, (
            f"cut at {cut}ms, below the {config.vad.silence_floor_ms}ms floor"
        )
        await asyncio.sleep(0.15)  # let the utterance task reach STT + the LLM
        assert orch.turn_detector.calls >= 1, "the gate never asked the detector"
        assert orch.stt.calls == 1, (
            f"STT ran {orch.stt.calls}x — should run exactly once, after the "
            f"cut, since the detector produced no transcript to reuse"
        )
        assert RecordingLLM.seen == ["What is the capital of France?"], (
            f"LLM saw {RecordingLLM.seen!r}"
        )
        return cut

    cut = await _run(config, body)
    print(f"OK: turn_detector mode cuts at {cut}ms on a decided-complete turn, "
          f"STT paid once after.")


async def test_turn_detector_ceiling_always_cuts():
    """A detector that never says 'done' must not hang the turn — and unlike
    punctuation mode there's no cached transcript to fall back on, so STT
    must still run once the ceiling forces the cut."""
    config = AppConfig()
    config.vad.endpoint_mode = "turn_detector"
    config.vad.silence_ceiling_ms = 1200  # keep the test quick
    ScriptedTurnDetector.default_complete = False
    RecordingLLM.seen = []

    async def body(orch):
        orch.stt.transcript = "never sounds done"
        await feed(orch, 320, speech=True)
        cut = await feed_silence_until_cut(orch, 2500)
        assert cut is not None, "turn hung — ceiling did not fire"
        assert cut >= config.vad.silence_ceiling_ms, (
            f"cut at {cut}ms, before the {config.vad.silence_ceiling_ms}ms ceiling"
        )
        await asyncio.sleep(0.15)
        assert orch.stt.calls == 1, (
            f"STT ran {orch.stt.calls}x — turn_detector mode never has a gate "
            f"transcript to reuse, so exactly one fresh pass is expected"
        )
        assert RecordingLLM.seen == ["never sounds done"], (
            f"LLM saw {RecordingLLM.seen!r}"
        )
        return cut

    cut = await _run(config, body)
    print(f"OK: never-complete turn still cut, at the {cut}ms ceiling, "
          f"STT paid once.")


def test_end_of_turn_heuristic():
    complete = [
        "What is the capital of France?",
        "Tell me a joke.",
        "Stop!",
        'He said "go home."',
        "Sure, that works.  ",
    ]
    incomplete = [
        "",
        "   ",
        "I was thinking",
        "and then the",
        "I was thinking...",   # trailing off is the opposite of finished
        "I was thinking…",
        "so, um",
    ]
    for text in complete:
        assert looks_like_end_of_turn(text), f"should read as finished: {text!r}"
    for text in incomplete:
        assert not looks_like_end_of_turn(text), f"should read as unfinished: {text!r}"
    print(f"OK: end-of-turn heuristic agrees on "
          f"{len(complete)} finished / {len(incomplete)} unfinished samples.")


async def run_all():
    test_end_of_turn_heuristic()
    await test_fixed_mode_unchanged()
    await test_punctuation_cuts_early_and_reuses_transcript()
    await test_mid_sentence_pause_is_not_cut()
    await test_ceiling_always_cuts()
    await test_turn_detector_cuts_early_and_pays_stt_after()
    await test_turn_detector_ceiling_always_cuts()
    print("\nEndpointing tests passed.")


if __name__ == "__main__":
    asyncio.run(run_all())
