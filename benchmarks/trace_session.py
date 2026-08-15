"""Capture a session trace from the pipeline without a human at the mic.

Drives the REAL PipelineOrchestrator: real AudioPreprocessor, real Silero VAD,
real state machine, real UtteranceDenoiser, real Moonshine STT, real Groq LLM,
real Kokoro TTS. The ONLY substitution is the PyAudio mic thread — instead of
mic.read(), WAV chunks are pushed into orchestrator.push_chunk() at real-time
cadence (same chunk size, same int16 dtype), and speaker_queue is drained in
place of the PyAudio speaker thread.

All timings printed are the orchestrator's own instrumentation, unmodified.

Usage (from repo root):
    .venv\\Scripts\\python.exe benchmarks\\trace_session.py [clips_dir] [reps]

Defaults to benchmarks/clips at 5 reps. Any WAV in clips_dir is used (sorted
by name); files are converted to 16kHz mono automatically and peak-normalized,
so you can drop in phone recordings of your own voice.

Redirect to a log and feed it to analyze.py:
    .venv\\Scripts\\python.exe benchmarks\\trace_session.py > run.log 2>&1
    .venv\\Scripts\\python.exe benchmarks\\analyze.py run.log
"""

import asyncio
import ctypes
import ctypes.wintypes as wt
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from dotenv import load_dotenv

load_dotenv(REPO / ".env")

from config import AppConfig
from core.orchestrator import PipelineOrchestrator

CLIPS_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "benchmarks" / "clips"
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 5


class _PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = [
        ("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


def mem_mb():
    """(working set MB, private commit MB) for this process, or (-1, -1) off Windows.

    Worth watching: this pipeline commits ~5GB, so on a small-RAM box the
    latency numbers degrade badly once the machine starts paging.
    """
    if not hasattr(ctypes, "windll"):
        return -1.0, -1.0
    fn = ctypes.windll.kernel32.K32GetProcessMemoryInfo
    fn.argtypes = [wt.HANDLE, ctypes.POINTER(_PROCESS_MEMORY_COUNTERS_EX), wt.DWORD]
    fn.restype = wt.BOOL
    counters = _PROCESS_MEMORY_COUNTERS_EX()
    counters.cb = ctypes.sizeof(counters)
    if not fn(ctypes.windll.kernel32.GetCurrentProcess(),
              ctypes.byref(counters), counters.cb):
        return -1.0, -1.0
    return counters.WorkingSetSize / 1e6, counters.PrivateUsage / 1e6


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


async def feed_clip(orch, config, audio: np.ndarray):
    """Push one utterance plus trailing silence, mimicking cli.py's mic thread."""
    n = config.audio.chunk_samples
    tail_ms = config.vad.silence_timeout_ms + 500
    silence = np.zeros(int(config.audio.sample_rate * tail_ms / 1000), dtype=np.float32)
    stream = np.concatenate([np.zeros(n * 3, dtype=np.float32), audio, silence])

    period = config.audio.chunk_ms / 1000.0
    next_t = time.monotonic()
    for start in range(0, len(stream) - n + 1, n):
        chunk = stream[start : start + n]
        await orch.push_chunk((np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16))
        next_t += period
        delay = next_t - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)


async def drain_speaker(orch, stop: threading.Event):
    """Stand-in for cli.py's speaker thread: consume audio so TTS isn't blocked."""
    while not stop.is_set():
        try:
            await asyncio.wait_for(orch.speaker_queue.get(), timeout=0.2)
        except asyncio.TimeoutError:
            continue


async def wait_for_turn(orch, timeout: float = 90.0) -> bool:
    """Wait for the utterance task created at finalize to run to completion.

    Polls for the task object rather than watching state — a turn that
    completes quickly would otherwise be missed between polls.
    """
    t_end = time.monotonic() + timeout
    task = orch.current_task
    while task is None and time.monotonic() < t_end:
        await asyncio.sleep(0.01)
        task = orch.current_task
    if task is None:
        return False
    while not task.done() and time.monotonic() < t_end:
        await asyncio.sleep(0.02)
    await asyncio.sleep(0.05)  # let the final state transition settle
    return task.done()


async def main():
    config = AppConfig()
    clips = [c for c in sorted(CLIPS_DIR.glob("*.wav")) if not c.stem.endswith("_24k")]
    if not clips:
        print(f"No WAVs found in {CLIPS_DIR}")
        print("Generate the default set with: "
              r".venv\Scripts\python.exe benchmarks\make_utterances.py")
        return

    # print a repo-relative path — these logs get committed, and an absolute
    # one would bake this machine's directory layout into them
    try:
        shown = CLIPS_DIR.resolve().relative_to(REPO)
    except ValueError:
        shown = CLIPS_DIR.name

    print("=" * 72)
    print("orpheus session trace - real orchestrator, WAV-fed mic")
    print(f"clips: {shown}  ({len(clips)} files x {REPS} reps)")
    print("=" * 72)

    orch = PipelineOrchestrator(config)
    stop = threading.Event()
    loop_task = asyncio.create_task(orch.run_loop())
    drain_task = asyncio.create_task(drain_speaker(orch, stop))

    print("\nPipeline ready. Feeding utterances at real-time cadence.\n")

    turn = 0
    for _rep in range(REPS):
        for path in clips:
            turn += 1
            audio = load_16k_mono(path)
            print(f"--- turn {turn} ({path.stem}, {len(audio)/16000:.2f}s of speech) ---")
            orch.current_task = None
            await feed_clip(orch, config, audio)
            if not await wait_for_turn(orch):
                print("  [harness] turn did not complete")
            ws, priv = mem_mb()
            print(f"  [mem] working set {ws:,.0f} MB | private {priv:,.0f} MB | "
                  f"history {len(orch.conversation_history)} msgs")
            await asyncio.sleep(0.4)
            print()

    stop.set()
    loop_task.cancel()
    drain_task.cancel()
    print("=" * 72)
    print("done")


if __name__ == "__main__":
    asyncio.run(main())
