import numpy as np
from scipy.signal import butter, sosfilt

from config import AudioConfig, AudioProcessingConfig


class AudioPreprocessor:
    """Fast per-chunk cleanup gating VAD/STT.

    Deliberately minimal: DC offset removal + high-pass filter only.
    Anything heavier (spectral noise reduction) runs once per utterance
    in AudioPostProcessor.denoise_utterance, not here, since it needs more
    context than a single ~32ms chunk and would blow the per-chunk latency
    budget this class is meant to respect.
    """

    def __init__(self, audio_config: AudioConfig, processing_config: AudioProcessingConfig):
        self.audio_config = audio_config
        self.hp_sos = butter(
            4, processing_config.highpass_cutoff_hz, btype="high",
            fs=audio_config.sample_rate, output="sos",
        )

    def process_chunk(self, chunk: np.ndarray) -> np.ndarray:
        if chunk.dtype == np.int16:
            chunk = chunk.astype(np.float32) / 32768.0
        else:
            chunk = chunk.astype(np.float32)

        chunk = chunk - np.mean(chunk)
        chunk = sosfilt(self.hp_sos, chunk).astype(np.float32)
        return chunk
