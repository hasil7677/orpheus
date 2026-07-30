import numpy as np
import torch
from silero_vad import load_silero_vad

from config import VADConfig


class VoiceActivityDetector:
    """Silero VAD wrapper. Runs on CPU — it's <1ms per chunk, no GPU needed."""

    def __init__(self, config: VADConfig, sample_rate: int):
        self.config = config
        self.sample_rate = sample_rate
        self.model = load_silero_vad()

    def speech_probability(self, audio: np.ndarray) -> float:
        chunk_tensor = torch.from_numpy(audio)
        return self.model(chunk_tensor, self.sample_rate).item()

    def is_speech(self, audio: np.ndarray) -> bool:
        """
        Contract:
        1. Input: float32 numpy array, exactly one chunk (32ms) length.
        2. Output: boolean based on config.threshold.
        3. Silero's internal recurrent state is maintained by the model object
           across calls automatically as long as you keep calling the same
           instance in order (streaming use) — do not reload per-chunk.
        """
        return self.speech_probability(audio) > self.config.threshold

    def reset(self) -> None:
        """Reset internal recurrent state between separate audio streams/sessions."""
        self.model.reset_states()
