from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import WhisperFeatureExtractor

from config import VADConfig

MODELS_DIR = Path(__file__).parent.parent / "assets"
TURN_DETECTOR_MODEL_PATH = MODELS_DIR / "smart-turn-v3.2-cpu.onnx"

_DOWNLOAD_HELP = (
    "Turn-detector model not found. Download it into assets/:\n"
    "  smart-turn-v3.2-cpu.onnx -> https://huggingface.co/pipecat-ai/"
    "smart-turn-v3/resolve/main/smart-turn-v3.2-cpu.onnx"
)

_SAMPLE_RATE = 16000
_CHUNK_SECONDS = 8
_N_SAMPLES = _CHUNK_SECONDS * _SAMPLE_RATE


class TurnDetector:
    """pipecat-ai smart-turn-v3.2-cpu: Whisper-Tiny encoder + linear
    classifier, ONNX, int8, ~8MB. Forced onto CPUExecutionProvider — it
    never competes with Kokoro for the 4GB VRAM budget.

    Takes the buffered utterance audio directly; no transcript involved,
    unlike the punctuation gate. Model card / benchmarks:
    https://huggingface.co/pipecat-ai/smart-turn-v3
    """

    def __init__(self, config: VADConfig):
        if not TURN_DETECTOR_MODEL_PATH.exists():
            raise FileNotFoundError(_DOWNLOAD_HELP)
        self.threshold = config.turn_detector_threshold
        self._feature_extractor = WhisperFeatureExtractor(chunk_length=_CHUNK_SECONDS)
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(TURN_DETECTOR_MODEL_PATH),
            sess_options=so,
            providers=["CPUExecutionProvider"],
        )

    def turn_probability(self, audio: np.ndarray) -> float:
        """
        Contract:
        1. Input: float32 audio, 16kHz mono, the buffered utterance so far
           (any length).
        2. Output: probability in [0, 1] that the speaker has finished
           their turn — >`self.threshold` means "done".

        The model only ever looks at the most recent 8s and expects that
        audio at the *end* of a fixed 8s window, silence-padded at the
        front for shorter buffers — a right-padded window (transformers'
        own default) makes the model's output nearly insensitive to
        content, since a short real utterance ends up buried near the
        start of the window instead of at the boundary the model was
        trained to read. Verified empirically: real speech cut mid-sentence
        vs at its true end should give a large probability swing.
        """
        tail = audio[-_N_SAMPLES:].astype(np.float32)
        if len(tail) < _N_SAMPLES:
            tail = np.concatenate(
                [np.zeros(_N_SAMPLES - len(tail), dtype=np.float32), tail]
            )
        inputs = self._feature_extractor(
            tail, sampling_rate=_SAMPLE_RATE, return_tensors="np", do_normalize=True
        )
        input_features = inputs.input_features.astype(np.float32)
        logit = self._session.run(None, {"input_features": input_features})[0][0][0]
        return float(1.0 / (1.0 + np.exp(-logit)))

    def is_turn_complete(self, audio: np.ndarray) -> bool:
        return self.turn_probability(audio) > self.threshold
