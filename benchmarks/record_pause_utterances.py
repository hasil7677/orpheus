"""Record real utterances with deliberate pauses/trailing-off/restarts, and
auto-label speech_end_ms from the waveform (RMS voice-activity scan, same
approach as make_pause_clips.py's gap finder — not hand-written).

Run from repo root (interactive — you have to speak):
    .venv\\Scripts\\python.exe benchmarks\\record_pause_utterances.py [seconds]

Writes benchmarks/clips_real/*.wav plus merges into
benchmarks/clips_real/labels.json (creates it if absent, updates existing
keys in place). Then evaluate:
    .venv\\Scripts\\python.exe benchmarks\\eval_endpointing.py benchmarks\\clips_real 3
"""

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "benchmarks" / "clips_real"
sys.path.insert(0, str(REPO))

import numpy as np
import pyaudio
import soundfile as sf
from dotenv import load_dotenv

load_dotenv(REPO / ".env")

from config import AppConfig

SECONDS = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
FRAME_MS = 10

PROMPTS = {
    "pause1": (
        'Say, with a real 1-2 second pause in the middle like you\'re '
        'thinking: "I was wondering if you could... (pause) ...help me '
        'understand how this whole voice pipeline actually works."'
    ),
    "pause2": (
        'Same idea, different sentence — pause mid-thought: '
        '"So the thing is... (pause) ...I don\'t really get how it decides '
        'when I\'m done talking."'
    ),
    "trailing1": (
        'Start a sentence and just trail off — do NOT finish it, go silent '
        'mid-thought: "So I was gonna ask you something but, actually, '
        'never mind, I—"'
    ),
    "restart1": (
        'Say "um" and restart mid-sentence: "What\'s the— um, actually, '
        'can you just tell me what the capital of France is."'
    ),
    "clean2": (
        'A normal, fully finished sentence (control): "Thanks, that was '
        'really helpful, I appreciate it."'
    ),
}


def trim_leading_silence(audio: np.ndarray, sr: int, pad_ms: int = 120) -> np.ndarray:
    """Trim only leading silence — trailing silence is what we're measuring."""
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak <= 0:
        return audio
    voiced = np.abs(audio) > peak * 0.05
    if not voiced.any():
        return audio
    pad = int(sr * pad_ms / 1000)
    first = max(0, int(np.argmax(voiced)) - pad)
    return audio[first:]


def last_voiced_end_ms(audio: np.ndarray, sr: int) -> int:
    """RMS-per-frame voice-activity scan; returns ms of the end of the last
    voiced frame. Same thresholding approach as make_pause_clips.py's
    widest_inner_gap, just scanning for the last True instead of the widest
    gap in the middle."""
    n = int(sr * FRAME_MS / 1000)
    usable = len(audio) // n * n
    if usable == 0:
        return 0
    frames = audio[:usable].reshape(-1, n)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    nonzero = rms[rms > 0]
    thresh = max(np.median(nonzero) * 0.15, 1e-4) if len(nonzero) else 1.0
    voiced = rms > thresh
    idx = np.where(voiced)[0]
    if len(idx) == 0:
        return 0
    return round(int(idx[-1] + 1) * n / sr * 1000)


def record(pa, config, device, seconds: float) -> np.ndarray:
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=config.audio.channels,
        rate=config.audio.sample_rate,
        input=True,
        input_device_index=device,
        frames_per_buffer=config.audio.chunk_samples,
    )
    n_chunks = int(seconds * 1000 / config.audio.chunk_ms)
    frames = []
    try:
        for i in range(n_chunks):
            raw = stream.read(config.audio.chunk_samples, exception_on_overflow=False)
            chunk = np.frombuffer(raw, dtype=np.int16)
            frames.append(chunk)
            level = int(np.abs(chunk).max() / 32768 * 40)
            remaining = seconds - i * config.audio.chunk_ms / 1000
            print(f"\r  [{'#' * level}{'.' * (40 - level)}] {remaining:4.1f}s ",
                  end="", flush=True)
    finally:
        stream.stop_stream()
        stream.close()
    print()
    return np.concatenate(frames) if frames else np.zeros(0, dtype=np.int16)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    config = AppConfig()
    pa = pyaudio.PyAudio()

    env_device = os.environ.get("AUDIO_INPUT_DEVICE", "").strip()
    device = int(env_device) if env_device else None
    name = (pa.get_device_info_by_index(device)["name"] if device is not None
            else pa.get_default_input_device_info()["name"])
    print(f"Mic: {name}")
    print(f"Recording {SECONDS:.0f}s per utterance at {config.audio.sample_rate}Hz mono.")
    print("Read each instruction fully before pressing Enter.\n")

    labels_path = OUT / "labels.json"
    labels = json.loads(labels_path.read_text(encoding="utf-8")) if labels_path.exists() else {}
    results = []
    try:
        for stem, instruction in PROMPTS.items():
            print(f"--- {stem} ---")
            print(f"  {instruction}")
            input("    press Enter, then speak: ")
            audio = record(pa, config, device, SECONDS)
            trimmed = trim_leading_silence(
                audio.astype(np.float32) / 32768.0, config.audio.sample_rate
            )
            peak = float(np.max(np.abs(trimmed))) if len(trimmed) else 0.0
            end_ms = last_voiced_end_ms(trimmed, config.audio.sample_rate)
            dur = len(trimmed) / config.audio.sample_rate

            path = OUT / f"{stem}_16k.wav"
            sf.write(path, trimmed, config.audio.sample_rate)
            labels[f"{stem}_16k"] = {"speech_end_ms": end_ms}
            results.append((stem, dur, peak, end_ms))

            warn = ""
            if peak >= 0.99:
                warn = "  <-- CLIPPING, move back from the mic and re-run"
            elif peak < 0.05:
                warn = "  <-- very quiet, check the mic and re-run"
            elif dur < 0.5:
                warn = "  <-- almost nothing captured, re-run"
            print(f"    saved {path.name}  clip {dur:.2f}s  peak {peak:.2f}  "
                  f"detected speech_end_ms={end_ms}{warn}\n")
    finally:
        pa.terminate()

    labels_path.write_text(json.dumps(labels, indent=2), encoding="utf-8")

    print("=" * 60)
    for stem, dur, peak, end_ms in results:
        print(f"  {stem:10s} clip {dur:5.2f}s  peak {peak:.2f}  speech_end_ms {end_ms}")
    print(f"\nWrote {len(results)} clips + updated {labels_path.relative_to(REPO)}")
    print("\nsanity-check the auto-detected speech_end_ms against what you actually said —")
    print("if a value looks wrong (e.g. cut off mid-word or way too long), edit labels.json by hand.")
    print("\nThen evaluate:")
    print(r"  .venv\Scripts\python.exe benchmarks\eval_endpointing.py benchmarks\clips_real 3")


if __name__ == "__main__":
    main()
