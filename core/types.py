from dataclasses import dataclass

import numpy as np


@dataclass
class AudioChunk:
    """Raw audio data passed around the system."""

    data: np.ndarray  # float32 array, normalized [-1.0, 1.0]
    sample_rate: int
    is_speech: bool = False


@dataclass
class Utterance:
    """A complete thought spoken by the user."""

    audio_data: np.ndarray
    transcript: str = ""
    duration_ms: int = 0
