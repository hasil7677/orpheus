"""Generate spoken utterances as 16kHz mono WAVs using the repo's own Kokoro
TTS, for driving the pipeline without a human at the mic.

Run from repo root:
    .venv\\Scripts\\python.exe benchmarks\\make_utterances.py

Writes benchmarks/clips/{short,medium,long}_16k.wav (fed to the VAD) plus
_24k.wav copies at Kokoro's native rate for listening.

Synthesized speech is cleaner than a real mic in a real room, so STT on these
clips is a best case. To benchmark against your own voice instead, record a few
WAVs and point the harness at them:
    .venv\\Scripts\\python.exe benchmarks\\trace_session.py path\\to\\my_clips 5
"""

import asyncio
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "benchmarks" / "clips"
sys.path.insert(0, str(REPO))

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from config import AppConfig
from models.tts import TextToSpeechProvider

UTTERANCES = {
    "short": "What time is it?",
    "medium": "What is the capital of France?",
    "long": "Can you explain how a voice assistant pipeline works, briefly?",
}


async def main():
    OUT.mkdir(parents=True, exist_ok=True)
    config = AppConfig()
    tts = TextToSpeechProvider(config.models, config.audio.tts_sample_rate)

    for name, text in UTTERANCES.items():
        chunks = []
        # a different voice than the assistant default, so it reads as "the user"
        async for samples, _sr in tts.kokoro.create_stream(
            text, voice="am_michael", speed=1.0, lang="en-us"
        ):
            chunks.append(samples.astype(np.float32))
        audio24 = np.concatenate(chunks)
        # Kokoro is 24kHz; the mic path is 16kHz mono
        audio16 = resample_poly(audio24, 16000, 24000).astype(np.float32)
        sf.write(OUT / f"{name}_16k.wav", audio16, 16000)
        sf.write(OUT / f"{name}_24k.wav", audio24, 24000)
        print(f'{name:7s} {len(audio16)/16000:5.2f}s  "{text}"  -> {name}_16k.wav')


if __name__ == "__main__":
    asyncio.run(main())
