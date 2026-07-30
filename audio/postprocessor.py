import numpy as np
import noisereduce as nr
import pyloudnorm as pyln
from scipy.signal import butter, sosfilt

from config import AudioProcessingConfig


class UtteranceDenoiser:
    """Noise reduction run ONCE on the full buffered utterance, right before STT.

    Spectral gating needs more context than a single VAD chunk (~32ms) to build
    a useful noise profile, and is too slow to run per-chunk anyway. Call this
    after the orchestrator has concatenated the full speech buffer.
    """

    def __init__(self, sample_rate: int, config: AudioProcessingConfig):
        self.sample_rate = sample_rate
        self.config = config

    def process(self, audio: np.ndarray) -> np.ndarray:
        if not self.config.enable_noise_reduction or len(audio) < 2048:
            return audio
        return nr.reduce_noise(
            y=audio, sr=self.sample_rate, stationary=True, prop_decrease=0.75,
        ).astype(np.float32)


class TTSPostProcessor:
    """Polish TTS output.

    Called per streamed chunk, not once per utterance — so anything with
    internal state (IIR filters) or that needs a full-signal view to be
    accurate (LUFS measurement, silence trimming) will produce audible
    artifacts if run here: filters reset to zero state each call (clicks at
    chunk boundaries), and per-chunk LUFS targets cause volume pumping
    between chunks. Kokoro's raw output doesn't need correction, so the
    default path is just a safety clip. Set
    AudioProcessingConfig.tts_apply_effects=True to opt into the full
    de-ess/compress/LUFS chain if you're processing a fully-buffered
    utterance (not a live per-chunk stream) and want it polished/corrected.
    """

    def __init__(self, sample_rate: int, config: AudioProcessingConfig):
        self.sr = sample_rate
        self.config = config
        self.meter = pyln.Meter(sample_rate)
        self.hp_sos = butter(4, 60, btype="high", fs=sample_rate, output="sos")
        self.lp_sos = butter(4, min(12000, sample_rate // 2 - 100), btype="low", fs=sample_rate, output="sos")

    def process(self, audio: np.ndarray) -> np.ndarray:
        if len(audio) == 0:
            return audio

        if not self.config.tts_apply_effects:
            return np.clip(audio, -0.99, 0.99).astype(np.float32)

        audio = audio - np.mean(audio)
        audio = sosfilt(self.hp_sos, audio).astype(np.float32)
        audio = sosfilt(self.lp_sos, audio).astype(np.float32)
        audio = self._de_ess(audio)
        audio = self._compress(audio, threshold_db=-18, ratio=3.0)
        audio = self._normalize_lufs(audio, target=self.config.target_lufs)
        audio = self._trim_silence(audio)
        audio = np.clip(audio, -0.99, 0.99)
        return audio.astype(np.float32)

    def _de_ess(self, audio: np.ndarray, threshold_db: float = -20) -> np.ndarray:
        high = min(8000, self.sr // 2 - 100)
        low = min(4000, high - 100)
        sos = butter(4, [low, high], btype="band", fs=self.sr, output="sos")
        sibilant = sosfilt(sos, audio).astype(np.float32)

        threshold = 10 ** (threshold_db / 20)
        compressed = np.copy(sibilant)
        mask = np.abs(sibilant) > threshold
        if np.any(mask):
            excess = np.abs(sibilant[mask]) - threshold
            compressed[mask] = np.sign(sibilant[mask]) * (threshold + excess / 6.0)

        return (audio - sibilant + compressed).astype(np.float32)

    def _compress(self, audio: np.ndarray, threshold_db: float = -18, ratio: float = 3.0) -> np.ndarray:
        threshold = 10 ** (threshold_db / 20)
        result = np.copy(audio)
        mask = np.abs(audio) > threshold
        if np.any(mask):
            excess = np.abs(audio[mask]) - threshold
            result[mask] = np.sign(audio[mask]) * (threshold + excess / ratio)
        return result.astype(np.float32)

    def _normalize_lufs(self, audio: np.ndarray, target: float) -> np.ndarray:
        try:
            loudness = self.meter.integrated_loudness(audio)
            if np.isinf(loudness):
                return audio
            return pyln.normalize.loudness(audio, loudness, target).astype(np.float32)
        except Exception:
            peak = np.max(np.abs(audio))
            if peak > 0:
                return (audio / peak * 0.95).astype(np.float32)
            return audio

    def _trim_silence(self, audio: np.ndarray, threshold_db: float = -40, pad_ms: int = 50) -> np.ndarray:
        threshold = 10 ** (threshold_db / 20)
        pad_samples = int(self.sr * pad_ms / 1000)

        above = np.where(np.abs(audio) > threshold)[0]
        if len(above) == 0:
            return audio

        start = max(0, above[0] - pad_samples)
        end = min(len(audio), above[-1] + pad_samples)
        return audio[start:end]
