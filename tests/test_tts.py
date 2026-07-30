"""Smoke test: Kokoro-onnx loads on GPU and synthesizes audio.

Run: .venv\\Scripts\\python.exe tests\\test_tts.py

Requires kokoro-v1.0.onnx and voices-v1.0.bin in local/assets/
(see models/tts.py for download links).
"""

import asyncio
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import AppConfig
from models.tts import TextToSpeechProvider

OUT_PATH = Path(__file__).parent / "_tts_smoke_output.wav"


async def fake_token_stream():
    for token in ["Hello, ", "this is ", "a test ", "of the local voice pipeline."]:
        yield token


async def main():
    config = AppConfig()
    tts = TextToSpeechProvider(config.models, config.audio.tts_sample_rate)

    try:
        active_providers = tts.kokoro.sess.get_providers()
        print(f"ONNX Runtime providers: {active_providers}")
        if "CUDAExecutionProvider" not in active_providers:
            print("WARNING: CUDAExecutionProvider not active — falling back to CPU.")
    except AttributeError:
        print("Could not introspect ONNX Runtime providers (internal API may have changed) — continuing.")

    chunks = []
    async for chunk in tts.synthesize_stream(fake_token_stream()):
        chunks.append(chunk)

    assert chunks, "No audio produced"
    audio = np.concatenate(chunks)
    assert len(audio) > 0, "Concatenated audio is empty"

    sf.write(str(OUT_PATH), audio, config.audio.tts_sample_rate)
    print(f"Synthesized {len(audio) / config.audio.tts_sample_rate:.2f}s of audio -> {OUT_PATH}")
    print("\nTTS smoke test passed.")


if __name__ == "__main__":
    asyncio.run(main())
