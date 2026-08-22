"""Turn-detector smoke test — needs the real ONNX model on disk (assets/), no
GPU or API key. Loads real recorded clips from benchmarks/clips_real/ if
present, else synthesizes silence-only sanity checks.

Run: .venv\\Scripts\\python.exe tests\\test_turn_detector.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import AppConfig
from models.turn_detector import TURN_DETECTOR_MODEL_PATH, TurnDetector

REPO = Path(__file__).parent.parent
CLIPS = REPO / "benchmarks" / "clips_real"

# smart-turn-v3.2-cpu.onnx is a downloaded asset (assets/*.onnx is
# gitignored — see README's "Kokoro model files" step and
# models/turn_detector.py's _DOWNLOAD_HELP), not GPU/API-key gated but
# absent on a fresh checkout or CI runner. Skip cleanly instead of failing
# when it isn't present, rather than every test here raising FileNotFoundError.
pytestmark = pytest.mark.skipif(
    not TURN_DETECTOR_MODEL_PATH.exists(),
    reason=(
        "smart-turn-v3.2-cpu.onnx not present in assets/ (gitignored, "
        "download separately per README) — skipping turn-detector tests "
        "that need the real model."
    ),
)


def test_probability_is_in_range():
    config = AppConfig()
    detector = TurnDetector(config.vad)
    silence = np.zeros(16000, dtype=np.float32)  # 1s of nothing
    p = detector.turn_probability(silence)
    assert 0.0 <= p <= 1.0, f"probability {p} out of [0, 1]"
    print(f"OK: silence scores {p:.3f} (in-range).")


def test_short_vs_long_buffer_both_run():
    """Buffers shorter and longer than the model's 8s window must not crash —
    this is the left-pad-vs-truncate path exercising both branches."""
    config = AppConfig()
    detector = TurnDetector(config.vad)
    short = np.zeros(1600, dtype=np.float32)  # 0.1s, needs left-padding
    long = np.zeros(16000 * 12, dtype=np.float32)  # 12s, needs truncation
    p_short = detector.turn_probability(short)
    p_long = detector.turn_probability(long)
    assert 0.0 <= p_short <= 1.0 and 0.0 <= p_long <= 1.0
    print(f"OK: short buffer -> {p_short:.3f}, long buffer -> {p_long:.3f}, "
          f"neither crashed.")


def test_mid_sentence_scores_lower_than_finished():
    """The core sanity check: cutting a real finished utterance mid-sentence
    should score lower than the buffer at its true end. This is the property
    that silently failed with right-padding (transformers' feature-extractor
    default) during development — both scored ~identically regardless of
    where the cut landed, because a short real utterance ended up buried
    away from the boundary the model was trained to read. Confirms the
    left-pad fix in TurnDetector.turn_probability is still in effect.

    Uses labels.json's speech_end_ms, not the raw clip length — these
    recordings run a fixed ~10s window and keep several seconds of trailing
    room tone after speech actually ends, so slicing at the raw file length
    would feed the model mostly silence and defeat the point of the check.
    """
    clip = CLIPS / "clean2_16k.wav"
    labels_path = CLIPS / "labels.json"
    if not clip.exists() or not labels_path.exists():
        print("SKIP: benchmarks/clips_real/clean2_16k.wav or labels.json not "
              "present (record via benchmarks/record_pause_utterances.py to "
              "run this check).")
        return
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    speech_end_ms = labels["clean2_16k"]["speech_end_ms"]

    config = AppConfig()
    detector = TurnDetector(config.vad)
    audio, sr = sf.read(clip, dtype="float32")
    assert sr == 16000
    finished = audio[: int(speech_end_ms / 1000 * sr)]
    mid_sentence = audio[: int(speech_end_ms / 1000 * sr / 2)]
    p_mid = detector.turn_probability(mid_sentence)
    p_finished = detector.turn_probability(finished)
    assert p_finished > p_mid, (
        f"finished utterance ({p_finished:.3f}) should score higher than a "
        f"mid-sentence cut ({p_mid:.3f}) — did the left-padding regress?"
    )
    print(f"OK: mid-sentence {p_mid:.3f} < finished {p_finished:.3f}.")


def run_all():
    test_probability_is_in_range()
    test_short_vs_long_buffer_both_run()
    test_mid_sentence_scores_lower_than_finished()
    print("\nTurn-detector tests passed.")


if __name__ == "__main__":
    run_all()
