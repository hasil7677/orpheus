"""Compare endpoint modes on labeled clips: how often each cuts you off, and
how long each makes you wait.

    .venv\\Scripts\\python.exe benchmarks\\eval_endpointing.py [clips_dir] [reps]

Defaults to benchmarks/clips_pauses at 1 rep. Reports, per mode:

  false cuts        turns finalized while the speaker still had audio left —
                    the failure people actually hate, and the reason a fixed
                    650ms timeout is worth replacing
  response-ready    ms from the true end of speech to the moment the LLM
                    would have been called. This is the endpointing wait plus
                    whatever STT was not already paid for during it, which is
                    the entire quantity this change moves.

Ground truth comes from a labels.json in clips_dir:

    {"my_clip_16k": {"speech_end_ms": 3120, "pauses": [[1180, 2050]]}}

`speech_end_ms` is the only required field — the last instant the speaker was
still talking. Without labels.json each clip's full duration is assumed to be
its speech end, which is right for tightly trimmed recordings and wrong for
anything with trailing room tone.

What runs and what doesn't
--------------------------
The real orchestrator, real preprocessor, real Silero VAD, real state machine,
real denoiser, real Moonshine, real punctuation gate. LLM and TTS are stubbed
out: they are downstream of the decision being measured, they add network
variance that would swamp it, and dropping them keeps this at roughly VAD+STT
memory instead of the pipeline's full ~5GB — which matters a lot on a 5.86GB
box. Two extra instrumentation hooks are attached from here: the stub LLM
timestamps when it was called, and _finalize_utterance is wrapped to timestamp
cuts. Nothing in core/ is modified.

Both modes run against the same clips in one process, so model load, machine
state, and audio are shared and the comparison is like-for-like.
"""

import asyncio
import ctypes
import json
import statistics
import sys
import time
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from config import AppConfig
from core.orchestrator import PipelineOrchestrator, PipelineState

CLIPS_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "benchmarks" / "clips_pauses"
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 1
MODES = ("fixed", "punctuation")

# filled by the stub LLM and the _finalize_utterance wrapper, per clip
EVENTS: list[tuple[str, float, str]] = []


class StubLLM:
    """Records when the pipeline was ready to answer, then says nothing."""

    def __init__(self, config):
        pass

    async def generate_stream(self, user_text, history):
        EVENTS.append(("ready", time.monotonic(), user_text))
        return
        yield  # unreachable; makes this an async generator


class StubTTS:
    def __init__(self, config, sample_rate):
        pass

    async def synthesize_stream(self, text_stream):
        async for _ in text_stream:
            pass
        return
        yield


class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def available_mb() -> float:
    """Available physical MB. Below ~700 on this box, STT latency stops
    describing the code and starts describing the pagefile — every number
    here is only as trustworthy as this figure."""
    if not hasattr(ctypes, "windll"):
        return -1.0
    status = _MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return -1.0
    return status.ullAvailPhys / 1e6


def load_16k_mono(path: Path) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != 16000:
        g = np.gcd(sr, 16000)
        audio = resample_poly(audio, 16000 // g, sr // g)
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak > 0:
        audio = audio / peak * 0.7  # consistent level into the VAD
    return audio.astype(np.float32)


def instrument(orch: PipelineOrchestrator) -> None:
    """Timestamp every finalize without touching core/."""
    original = orch._finalize_utterance

    async def wrapped(transcript=None, stt_ms=0.0):
        EVENTS.append(("cut", time.monotonic(), transcript or ""))
        await original(transcript, stt_ms)

    orch._finalize_utterance = wrapped


async def feed_clip(orch, config, audio: np.ndarray) -> float:
    """Push one clip at real-time cadence. Returns the monotonic time at which
    the last sample of actual speech was handed to the pipeline."""
    n = config.audio.chunk_samples
    tail_ms = config.vad.max_endpoint_wait_ms + 800
    lead = np.zeros(n * 3, dtype=np.float32)
    silence = np.zeros(int(config.audio.sample_rate * tail_ms / 1000), dtype=np.float32)
    stream = np.concatenate([lead, audio, silence])
    speech_end_sample = len(lead) + len(audio)

    t_speech_end = 0.0
    period = config.audio.chunk_ms / 1000.0
    next_t = time.monotonic()
    for start in range(0, len(stream) - n + 1, n):
        chunk = stream[start : start + n]
        await orch.push_chunk((np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16))
        if not t_speech_end and start + n >= speech_end_sample:
            t_speech_end = time.monotonic()
        next_t += period
        delay = next_t - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
    return t_speech_end or time.monotonic()


async def settle(orch, timeout: float = 20.0) -> None:
    """Wait for any in-flight turn to finish and the machine to go quiet."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        task_busy = orch.current_task is not None and not orch.current_task.done()
        if orch.state == PipelineState.IDLE and not task_busy:
            return
        await asyncio.sleep(0.05)


async def run_clip(orch, config, path: Path, audio: np.ndarray,
                   speech_end_ms: float | None) -> dict:
    EVENTS.clear()
    orch.state = PipelineState.IDLE
    orch.utterance_buffer = []
    orch.silence_ms = 0
    orch.current_task = None

    t_fed_end = await feed_clip(orch, config, audio)
    await settle(orch)

    # speech_end_ms from labels overrides where the clip physically ends —
    # a recording with trailing room tone ends after the speaker stopped.
    clip_ms = len(audio) / config.audio.sample_rate * 1000
    t_speech_end = t_fed_end - max(0.0, (clip_ms - speech_end_ms) / 1000) \
        if speech_end_ms is not None else t_fed_end

    cuts = [(t - t_speech_end) * 1000 for kind, t, _ in EVENTS if kind == "cut"]
    ready = [(t - t_speech_end) * 1000 for kind, t, _ in EVENTS if kind == "ready"]
    heard = [text for kind, _, text in EVENTS if kind == "ready"]

    return {
        "clip": path.stem,
        "false_cuts": sum(1 for c in cuts if c < 0),
        "cuts": cuts,
        "ready_ms": ready[-1] if ready else None,
        "transcript": heard[-1] if heard else "",
        "turns": len(ready),
    }


async def main():
    clips = [c for c in sorted(CLIPS_DIR.glob("*.wav")) if not c.stem.endswith("_24k")]
    if not clips:
        print(f"No WAVs found in {CLIPS_DIR}")
        print("Make the spliced-pause set with: "
              r".venv\Scripts\python.exe benchmarks\make_pause_clips.py")
        return

    labels_path = CLIPS_DIR / "labels.json"
    labels = json.loads(labels_path.read_text(encoding="utf-8")) if labels_path.exists() else {}

    try:
        shown = CLIPS_DIR.resolve().relative_to(REPO)
    except ValueError:
        shown = CLIPS_DIR.name

    print("=" * 72)
    print("orpheus endpointing eval - real VAD/STT/state machine, stubbed LLM+TTS")
    print(f"clips: {shown}  ({len(clips)} files x {REPS} reps x {len(MODES)} modes)")
    print(f"labels: {'labels.json' if labels else 'NONE - assuming each clip ends at its last sample'}")
    print(f"available physical memory at start: {available_mb():,.0f} MB")
    print("=" * 72)

    config = AppConfig()
    with patch("core.orchestrator.LLMProvider", StubLLM), \
         patch("core.orchestrator.TextToSpeechProvider", StubTTS):
        orch = PipelineOrchestrator(config)
    instrument(orch)
    loop_task = asyncio.create_task(orch.run_loop())

    print(f"\nfixed:       silence_timeout_ms={config.vad.silence_timeout_ms}")
    print(f"punctuation: floor={config.vad.silence_floor_ms} "
          f"recheck={config.vad.endpoint_recheck_ms} "
          f"ceiling={config.vad.silence_ceiling_ms}")

    results: dict[str, list[dict]] = {mode: [] for mode in MODES}
    try:
        # Modes alternate clip by clip rather than running in two blocks. This
        # box is RAM-bound and its STT latency drifts by several hundred ms
        # over a few minutes; running all of one mode first would hand that
        # drift to whichever mode went second and call it a result.
        for rep in range(REPS):
            for path in clips:
                audio = load_16k_mono(path)
                label = labels.get(path.stem, {})
                print(f"\n--- rep {rep + 1}: {path.stem} "
                      f"({available_mb():,.0f} MB available) ---")
                for mode in MODES:
                    config.vad.endpoint_mode = mode
                    row = await run_clip(orch, config, path, audio,
                                         label.get("speech_end_ms"))
                    results[mode].append(row)
                    flag = "  <-- FALSE CUT" if row["false_cuts"] else ""
                    ready = (f"{row['ready_ms']:6.0f}ms" if row["ready_ms"] is not None
                             else "  never")
                    print(f"  {mode:<12} ready {ready}  turns {row['turns']}{flag}")
                    if row["transcript"]:
                        print(f"      heard: \"{row['transcript']}\"")
    finally:
        loop_task.cancel()

    print("\n" + "=" * 72)
    print(f"{'mode':<14}{'n':>4}{'false cuts':>12}{'median':>10}{'p90':>9}{'max':>9}")
    print("-" * 58)
    summary = {}
    for mode in MODES:
        rows = results[mode]
        ready = sorted(r["ready_ms"] for r in rows if r["ready_ms"] is not None)
        false_cuts = sum(r["false_cuts"] for r in rows)
        if not ready:
            print(f"{mode:<14}{len(rows):>4}{false_cuts:>12}{'  no turns completed':>28}")
            continue
        p90 = ready[min(int(len(ready) * 0.9), len(ready) - 1)]
        summary[mode] = (statistics.median(ready), p90, max(ready), false_cuts)
        print(f"{mode:<14}{len(rows):>4}{false_cuts:>12}"
              f"{statistics.median(ready):>8.0f}ms{p90:>7.0f}ms{max(ready):>7.0f}ms")

    if len(summary) == 2:
        base, new = summary["fixed"], summary["punctuation"]
        print(f"\ndelta (punctuation - fixed): median {new[0] - base[0]:+.0f}ms  "
              f"p90 {new[1] - base[1]:+.0f}ms  max {new[2] - base[2]:+.0f}ms  "
              f"false cuts {new[3] - base[3]:+d}")
        print("\nA median win with a worse max is a regression in conversation: the "
              "tail is\nwhat gets noticed. And a median win bought with even one "
              "extra false cut is\nnot a win at all.")

    print("\nresponse-ready = end of speech -> LLM call. Under 'punctuation' the "
          "STT pass\nis paid inside the silence wait rather than after it, so the "
          "saving is larger\nthan the change in timeout alone.")


if __name__ == "__main__":
    asyncio.run(main())
