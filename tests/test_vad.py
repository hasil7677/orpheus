"""Smoke test: Silero VAD loads and produces sane probabilities.

Run: .venv\\Scripts\\python.exe tests\\test_vad.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import AppConfig
from models.vad import VoiceActivityDetector


def main():
    config = AppConfig()
    vad = VoiceActivityDetector(config.vad, config.audio.sample_rate)

    silence = np.zeros(config.audio.chunk_samples, dtype=np.float32)
    assert vad.is_speech(silence) is False, "Silence was flagged as speech"
    print("Silence chunk correctly flagged as non-speech.")

    vad.reset()
    rng = np.random.default_rng(0)
    loud_noise = (rng.uniform(-1, 1, config.audio.chunk_samples) * 0.8).astype(np.float32)
    prob_is_bool = isinstance(vad.is_speech(loud_noise), bool)
    assert prob_is_bool, "is_speech did not return a bool"
    print("VAD ran on noise input without error.")

    print("\nVAD smoke test passed.")


if __name__ == "__main__":
    main()
