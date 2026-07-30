"""Benchmark: does flushing TTS on the first clause boundary (new behavior)
actually beat waiting for a full sentence (old behavior) for time-to-first-
audio? Uses the real Kokoro model (local/assets/) so the numbers are real,
not simulated. Not a pass/fail test — prints a comparison.

Run: .venv\\Scripts\\python.exe tests\\bench_tts_ttfa.py
"""

import asyncio
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import AppConfig
from models.tts import TextToSpeechProvider

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")

SAMPLE_RESPONSES = [
    "Sure, I can help with that. Let me walk you through the steps.",
    "Well, that depends on a few things. What are you trying to build?",
    "Yes, absolutely — the meeting is at three, and I'll send a reminder.",
]


async def _instant_token_stream(text: str):
    for word in text.split(" "):
        yield word + " "


async def old_synthesize_stream(tts: TextToSpeechProvider, text_stream):
    """Pre-fix behavior: only flush on full sentence boundaries."""
    buffer = ""
    async for token in text_stream:
        buffer += token
        parts = _SENTENCE_BOUNDARY.split(buffer)
        if len(parts) > 1:
            for sentence in parts[:-1]:
                async for chunk in tts._synthesize_sentence(sentence):
                    yield chunk
            buffer = parts[-1]
    if buffer.strip():
        async for chunk in tts._synthesize_sentence(buffer):
            yield chunk


async def time_to_first_audio(stream) -> float:
    t0 = time.monotonic()
    async for _ in stream:
        return time.monotonic() - t0
    return -1.0


async def main():
    config = AppConfig()
    print("Loading Kokoro...")
    tts = TextToSpeechProvider(config.models, config.audio.tts_sample_rate)
    print("Loaded.\n")

    old_times, new_times = [], []
    for text in SAMPLE_RESPONSES:
        old_t = await time_to_first_audio(old_synthesize_stream(tts, _instant_token_stream(text)))
        new_t = await time_to_first_audio(tts.synthesize_stream(_instant_token_stream(text)))
        old_times.append(old_t)
        new_times.append(new_t)
        print(f'"{text}"')
        print(f"  old (full-sentence flush):  {old_t * 1000:.0f}ms")
        print(f"  new (clause flush):         {new_t * 1000:.0f}ms")
        print()

    avg_old = sum(old_times) / len(old_times)
    avg_new = sum(new_times) / len(new_times)
    print(f"Average old: {avg_old * 1000:.0f}ms | Average new: {avg_new * 1000:.0f}ms | "
          f"Improvement: {(avg_old - avg_new) * 1000:.0f}ms ({(1 - avg_new / avg_old) * 100:.0f}%)")


if __name__ == "__main__":
    asyncio.run(main())
