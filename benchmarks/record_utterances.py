"""Record the benchmark utterances in your own voice, through your real mic.

Opens the mic with exactly the same PyAudio parameters as cli.py's mic thread
(paInt16, 16kHz, mono, honouring AUDIO_INPUT_DEVICE), so the audio reaching the
VAD is what the live pipeline would see.

Run from repo root (interactive — you have to speak):
    .venv\\Scripts\\python.exe benchmarks\\record_utterances.py [seconds]

Writes benchmarks/clips_real/*.wav. Then benchmark against them:
    .venv\\Scripts\\python.exe benchmarks\\trace_session.py benchmarks\\clips_real 5

The prompts are the same sentences make_utterances.py synthesizes, so the
transcripts are directly comparable and any STT difference is attributable to
real-mic audio rather than to different words.
"""

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

SECONDS = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0

UTTERANCES = {
    "short": "What time is it?",
    "medium": "What is the capital of France?",
    "long": "Can you explain how a voice assistant pipeline works, briefly?",
}


def trim_silence(audio: np.ndarray, sr: int, pad_ms: int = 120) -> np.ndarray:
    """Trim leading/trailing silence so the clip is speech plus a little air.

    trace_session.py appends its own silence tail to trigger endpointing, so
    trailing room tone here would just inflate the measured utterance length.
    """
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak <= 0:
        return audio
    voiced = np.abs(audio) > peak * 0.05
    if not voiced.any():
        return audio
    pad = int(sr * pad_ms / 1000)
    first = max(0, int(np.argmax(voiced)) - pad)
    last = min(len(audio), len(audio) - int(np.argmax(voiced[::-1])) + pad)
    return audio[first:last]


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
            # crude live level meter so a dead mic is obvious immediately
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
    print("Speak naturally, at the distance and in the room you'd actually use.\n")

    results = []
    try:
        for stem, text in UTTERANCES.items():
            print(f'--- {stem} --- say: "{text}"')
            input("    press Enter, then speak: ")
            audio = record(pa, config, device, SECONDS)
            trimmed = trim_silence(audio.astype(np.float32) / 32768.0,
                                   config.audio.sample_rate)
            peak = float(np.max(np.abs(trimmed))) if len(trimmed) else 0.0
            dur = len(trimmed) / config.audio.sample_rate

            path = OUT / f"{stem}_16k.wav"
            sf.write(path, trimmed, config.audio.sample_rate)
            results.append((stem, dur, peak))

            warn = ""
            if peak >= 0.99:
                warn = "  <-- CLIPPING, move back from the mic and re-run"
            elif peak < 0.05:
                warn = "  <-- very quiet, check the mic and re-run"
            elif dur < 0.5:
                warn = "  <-- almost nothing captured, re-run"
            print(f"    saved {path.name}  {dur:.2f}s  peak {peak:.2f}{warn}\n")
    finally:
        pa.terminate()

    print("=" * 60)
    for stem, dur, peak in results:
        print(f"  {stem:7s} {dur:5.2f}s  peak {peak:.2f}")
    print(f"\nWrote {len(results)} clips to {OUT}")
    print("\nNow benchmark them:")
    print(r"  .venv\Scripts\python.exe benchmarks\trace_session.py "
          r"benchmarks\clips_real 5 > real1.log 2>&1")


if __name__ == "__main__":
    main()
