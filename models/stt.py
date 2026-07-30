import re

import gpu_setup  # noqa: F401 — must run before onnxruntime is imported (below)

import numpy as np
from moonshine_onnx import MoonshineOnnxModel, load_tokenizer

from config import ModelConfig

_HALLUCINATION_PATTERNS = re.compile(r"^\s*[\[\(].*[\]\)]\s*$")
_MIN_AUDIO_SECONDS = 0.1  # Moonshine asserts on shorter clips


class SpeechToTextProvider:
    """Moonshine ONNX wrapper. Uses onnxruntime's CUDA execution provider when available."""

    def __init__(self, config: ModelConfig):
        self.model = MoonshineOnnxModel(model_name=config.stt_model)
        self.tokenizer = load_tokenizer()

    def transcribe(self, audio: np.ndarray) -> str:
        """
        Contract:
        1. Input: variable-length float32 array (the complete utterance), 16kHz mono.
        2. Output: cleaned string transcript, empty string if nothing usable.
        """
        if len(audio) < int(_MIN_AUDIO_SECONDS * 16000):
            return ""

        # Moonshine wants shape [batch, samples], not a flat 1D array.
        batched = audio[None, ...].astype(np.float32)
        tokens = self.model.generate(batched)
        text = self.tokenizer.decode_batch(tokens)[0]

        text = text.strip()
        if not text or _HALLUCINATION_PATTERNS.match(text):
            return ""
        return text
