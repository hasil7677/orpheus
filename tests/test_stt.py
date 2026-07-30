"""Smoke test: Moonshine ONNX loads on GPU and transcribes audio.

Generates its own test clip via Kokoro TTS (no external fixture needed),
then feeds it through Moonshine and checks the transcript is sane.

Run: .venv\\Scripts\\python.exe tests\\test_stt.py
"""

import asyncio
import sys
from pathlib import Path

import numpy as np
from scipy.signal import resample

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import AppConfig
from models.stt import SpeechToTextProvider
from models.tts import TextToSpeechProvider

TEST_PHRASE = "The quick brown fox jumps over the lazy dog."


async def _make_test_clip(config: AppConfig) -> np.ndarray:
    tts = TextToSpeechProvider(config.models, config.audio.tts_sample_rate)

    async def token_stream():
        yield TEST_PHRASE

    chunks = [chunk async for chunk in tts.synthesize_stream(token_stream())]
    audio_24k = np.concatenate(chunks)

    n_samples_16k = int(len(audio_24k) * config.audio.sample_rate / config.audio.tts_sample_rate)
    return resample(audio_24k, n_samples_16k).astype(np.float32)


async def main():
    config = AppConfig()
    audio_16k = await _make_test_clip(config)

    stt = SpeechToTextProvider(config.models)
    transcript = stt.transcribe(audio_16k)
    print(f"Transcript: \"{transcript}\"")

    assert transcript, "Got an empty transcript from a clean synthesized clip"
    overlap = set(transcript.lower().split()) & set(TEST_PHRASE.lower().rstrip(".").split())
    assert len(overlap) >= 4, f"Transcript barely resembles the input phrase: {transcript!r}"

    print("\nSTT smoke test passed.")


if __name__ == "__main__":
    asyncio.run(main())
